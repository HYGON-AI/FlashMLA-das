#include "fwd.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
#include <tuple>

#include "kerutils/supplemental/torch_tensors.h"
#include "gfx93/prefill/sparse/dsa_mls/dispatch.h"

namespace gfx93::decode::sparse_bf16_dsa {

static constexpr float LOG_2_E = 1.44269504f;

struct LocalArch {
    int num_sms;
    std::string arch_name;

    LocalArch() {
        auto* props = at::cuda::getCurrentDeviceProperties();
        num_sms = props->multiProcessorCount;
        arch_name = props->gcnArchName;
    }

    bool is_gfx93x() const {
        const auto base = arch_name.substr(0, arch_name.find(':'));
        return base == "gfx936" || base == "gfx938";
    }

    bool is_gfx938() const {
        const auto base = arch_name.substr(0, arch_name.find(':'));
        return base == "gfx938";
    }
};

static int int64_stride_to_int(int64_t stride) {
    TORCH_CHECK(stride <= std::numeric_limits<int>::max(), "DSA BF16 sparse decode stride exceeds int32 limit: ", stride);
    return static_cast<int>(stride);
}

static int default_num_splits(int b, int s_q, int topk, int extra_topk) {
    if (extra_topk > 0) {
        return 2;
    }

    const int64_t decode_tasks = static_cast<int64_t>(b) * s_q;
    if (topk == 512) {
        return decode_tasks <= 8 ? 8 : 1;
    }
    if (topk == 1024) {
        if (decode_tasks <= 4) return 16;
        if (decode_tasks <= 8) return 8;
        return 1;
    }
    if (topk > 1024) {
        if (decode_tasks <= 2) return 32;
        if (decode_tasks <= 4) return 16;
        if (decode_tasks <= 8) return 8;
        if (decode_tasks <= 64) return 4;
        return 2;
    }
    return 1;
}

static void check_optional_extra(
    const std::optional<at::Tensor>& extra_kv,
    const std::optional<at::Tensor>& extra_indices,
    const std::optional<at::Tensor>& extra_topk_length) {
    if (extra_kv.has_value()) {
        TORCH_CHECK(extra_indices.has_value(), "extra_indices_in_kvcache must be provided when extra_k_cache is provided");
    } else {
        TORCH_CHECK(!extra_indices.has_value(), "extra_indices_in_kvcache must not be provided when extra_k_cache is not provided");
        TORCH_CHECK(!extra_topk_length.has_value(), "extra_topk_length must not be provided when extra_k_cache is not provided");
    }
}

std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
run(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& attn_sink,
    std::optional<at::Tensor>& tile_scheduler_metadata,
    std::optional<at::Tensor>& num_splits,
    const std::optional<at::Tensor>& extra_kv,
    const std::optional<at::Tensor>& extra_indices,
    const std::optional<at::Tensor>& extra_topk_length,
    int d_v,
    float sm_scale) {
    LocalArch arch;
    TORCH_CHECK(arch.is_gfx938(), "DSA BF16 sparse decode is only supported on gfx938");

    KU_CHECK_NDIM(q, 4);
    KU_CHECK_NDIM(kv, 4);
    KU_CHECK_NDIM(indices, 3);
    if (extra_kv.has_value()) KU_CHECK_NDIM(extra_kv, 4);
    if (extra_indices.has_value()) KU_CHECK_NDIM(extra_indices, 3);

    const int b = q.size(0);
    const int s_q = q.size(1);
    const int h_q = q.size(2);
    const int d_qk = q.size(3);
    const int page_block_size = kv.size(1);
    const int h_kv = kv.size(2);
    const int topk = indices.size(2);
    const bool has_extra = extra_kv.has_value() && extra_indices.has_value() &&
                           extra_kv->numel() > 0 && extra_indices->numel() > 0 &&
                           extra_indices->size(2) > 0;
    const int extra_topk = has_extra ? extra_indices->size(2) : 0;

    TORCH_CHECK(b > 0 && s_q > 0 && h_q > 0, "Invalid q shape for DSA BF16 sparse decode");
    TORCH_CHECK(h_kv == 1, "DSA BF16 sparse decode only supports h_kv == 1");
    TORCH_CHECK(h_q == 64 || h_q == 128, "DSA BF16 sparse decode only supports h_q == 64 or 128");
    TORCH_CHECK(d_qk == 512, "DSA BF16 sparse decode only supports d_qk == 512 for now");
    TORCH_CHECK(d_v == 512, "DSA BF16 sparse decode only supports d_v == 512");
    TORCH_CHECK(topk > 0, "topk must be positive");
    if (has_extra) {
        TORCH_CHECK(extra_kv->size(1) > 0, "extra page_block_size must be positive");
        TORCH_CHECK(extra_kv->size(2) == h_kv, "extra_kv h_kv must match kv h_kv");
        TORCH_CHECK(extra_kv->size(3) == d_qk, "extra_kv d_qk must match q d_qk");
    }
    check_optional_extra(extra_kv, extra_indices, extra_topk_length);

    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(topk_length);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(tile_scheduler_metadata);
    KU_CHECK_DEVICE(num_splits);
    KU_CHECK_DEVICE(extra_kv);
    KU_CHECK_DEVICE(extra_indices);
    KU_CHECK_DEVICE(extra_topk_length);

    KU_CHECK_DTYPE(q, torch::kBFloat16);
    KU_CHECK_DTYPE(kv, torch::kBFloat16);
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(tile_scheduler_metadata, torch::kInt32);
    KU_CHECK_DTYPE(num_splits, torch::kInt32);
    KU_CHECK_DTYPE(extra_kv, torch::kBFloat16);
    KU_CHECK_DTYPE(extra_indices, torch::kInt32);
    KU_CHECK_DTYPE(extra_topk_length, torch::kInt32);

    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_CONTIGUOUS(topk_length);
    KU_CHECK_CONTIGUOUS(attn_sink);
    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_indices);
    KU_CHECK_CONTIGUOUS(extra_topk_length);

    KU_CHECK_SHAPE(q, b, s_q, h_q, d_qk);
    KU_CHECK_SHAPE(kv, kv.size(0), page_block_size, h_kv, d_qk);
    KU_CHECK_SHAPE(indices, b, s_q, topk);
    KU_CHECK_SHAPE(topk_length, b);
    KU_CHECK_SHAPE(attn_sink, h_q);
    if (has_extra) {
        KU_CHECK_SHAPE(extra_indices, b, s_q, extra_topk);
        KU_CHECK_SHAPE(extra_topk_length, b);
    }

    at::Tensor indices_for_dsa = indices.unsqueeze(2);
    at::Tensor extra_indices_for_dsa;
    if (has_extra) {
        extra_indices_for_dsa = extra_indices->unsqueeze(2);
    }

    c10::cuda::CUDAGuard device_guard{q.device()};
    auto opts = q.options();
    at::Tensor out = torch::empty({b, s_q, h_q, d_v}, opts);
    at::Tensor lse = torch::empty({b, h_q, s_q}, opts.dtype(at::kFloat));
    at::Tensor scores_memory = torch::empty({2, b, h_kv, s_q * h_q}, opts.dtype(at::kFloat));
    at::Tensor scores_max = scores_memory.select(0, 0);
    at::Tensor scores_sum = scores_memory.select(0, 1);

    if (!num_splits.has_value()) {
        const int split = default_num_splits(b, s_q, topk, extra_topk);
        num_splits = torch::empty({1}, opts.dtype(torch::kInt32));
        num_splits->fill_(split);
    }
    KU_CHECK_DTYPE(num_splits, torch::kInt32);
    KU_CHECK_DEVICE(num_splits);
    KU_CHECK_CONTIGUOUS(num_splits);
    TORCH_CHECK(num_splits->numel() == 1, "DSA BF16 sparse decode expects num_splits to be a scalar tensor");
    const int requested_num_splits = num_splits->item<int>();
    TORCH_CHECK(requested_num_splits >= 1 && requested_num_splits <= 64, "DSA BF16 sparse decode requires 1 <= num_splits <= 64");
    if (requested_num_splits == 1) {
        if (has_extra) {
            TORCH_CHECK(topk <= 256, "DSA BF16 sparse decode with extra_kv and num_splits == 1 supports topk <= 256");
            TORCH_CHECK(extra_topk <= 1024, "DSA BF16 sparse decode with extra_kv and num_splits == 1 supports extra_topk <= 1024");
        } else {
            TORCH_CHECK(topk <= 1024, "DSA BF16 sparse decode with num_splits == 1 supports topk <= 1024");
        }
    }

    Flash_fwd_mla_params_dsa params;
    std::memset(&params, 0, sizeof(params));
    params.layout = 1;
    params.b = b;
    params.h = h_kv;
    params.h_k = h_kv;
    params.h_h_k_ratio = 1;
    params.mtp = 1;
    params.ngroups = h_q / h_kv;
    params.topk = topk;
    params.extra_topk = has_extra ? extra_topk : 0;
    params.d = d_qk;
    params.d_v = d_v;
    params.scale_softmax = sm_scale;
    params.scale_softmax_log2 = sm_scale * LOG_2_E;
    params.topk_length = ku::get_optional_tensor_ptr<int>(topk_length);
    params.extra_topk_length = ku::get_optional_tensor_ptr<int>(extra_topk_length);
    params.attn_sink = ku::get_optional_tensor_ptr<float>(attn_sink);
    params.q_ptr = q.data_ptr();
    params.k_ptr = kv.data_ptr();
    params.v_ptr = kv.data_ptr();
    params.extra_k_ptr = has_extra ? extra_kv->data_ptr() : nullptr;
    params.extra_v_ptr = has_extra ? extra_kv->data_ptr() : nullptr;
    params.o_ptr = out.data_ptr();
    params.sparse_indices = reinterpret_cast<int*>(indices_for_dsa.data_ptr());
    params.extra_sparse_indices = has_extra ? reinterpret_cast<int*>(extra_indices_for_dsa.data_ptr()) : nullptr;
    params.softmax_lse_ptr = lse.data_ptr<float>();
    params.scores_max_ptr = scores_max.data_ptr<float>();
    params.scores_sum_ptr = scores_sum.data_ptr<float>();
    params.page_block_size = page_block_size;
    params.extra_page_block_size = has_extra ? extra_kv->size(1) : 0;
    params.is_causal = false;

    params.q_batch_stride = int64_stride_to_int(q.stride(0));
    params.q_token_stride = int64_stride_to_int(q.stride(1));
    params.q_row_stride = int64_stride_to_int(q.stride(2));
    params.q_head_stride = int64_stride_to_int(q.stride(2));
    params.k_batch_stride = int64_stride_to_int(kv.stride(0));
    params.k_row_stride = int64_stride_to_int(kv.stride(1));
    params.k_head_stride = int64_stride_to_int(kv.stride(2));
    params.v_batch_stride = params.k_batch_stride;
    params.v_row_stride = params.k_row_stride;
    params.v_head_stride = params.k_head_stride;
    params.extra_k_batch_stride = has_extra ? int64_stride_to_int(extra_kv->stride(0)) : 0;
    params.extra_k_row_stride = has_extra ? int64_stride_to_int(extra_kv->stride(1)) : 0;
    params.extra_v_batch_stride = params.extra_k_batch_stride;
    params.extra_v_row_stride = params.extra_k_row_stride;
    params.sparse_indices_batch_stride = int64_stride_to_int(indices_for_dsa.stride(0));
    params.sparse_indices_row_stride = int64_stride_to_int(indices_for_dsa.stride(1));
    params.sparse_indices_head_stride = int64_stride_to_int(indices_for_dsa.stride(2));
    params.sparse_indices_topk_stride = int64_stride_to_int(indices_for_dsa.stride(3));
    params.extra_sparse_indices_batch_stride = has_extra ? int64_stride_to_int(extra_indices_for_dsa.stride(0)) : 0;
    params.extra_sparse_indices_row_stride = has_extra ? int64_stride_to_int(extra_indices_for_dsa.stride(1)) : 0;
    params.extra_sparse_indices_head_stride = has_extra ? int64_stride_to_int(extra_indices_for_dsa.stride(2)) : 0;
    params.extra_sparse_indices_topk_stride = has_extra ? int64_stride_to_int(extra_indices_for_dsa.stride(3)) : 0;
    params.o_batch_stride = int64_stride_to_int(out.stride(0));
    params.o_row_stride = int64_stride_to_int(out.stride(1));
    params.o_head_stride = int64_stride_to_int(out.stride(2));
    params.seqlen_q = s_q * params.ngroups;
    params.seqlen_k = kv.size(0) * kv.size(1);
    params.max_seqlen = s_q;
    params.is_bf16 = true;
    params.is_e4m3 = false;
    params.is_int8 = false;
    params.cu_count = arch.num_sms;
    params.seqlenq_ngroups_swapped = true;
    params.is_seqlens_k_cumulative = false;
    params.splitkv_use_fp32_as_accum = false;
    constexpr int64_t kBufferLoadPaddedTokenLimit = 32LL * 64 * 1024;
    const int64_t padded_k_tokens = static_cast<int64_t>(kv.size(0)) * page_block_size;
    const int64_t padded_extra_k_tokens = has_extra ? static_cast<int64_t>(extra_kv->size(0)) * extra_kv->size(1) : 0;
    params.decode_use_c_load = padded_k_tokens > kBufferLoadPaddedTokenLimit ||
                               padded_extra_k_tokens > kBufferLoadPaddedTokenLimit;
    params.num_splits = requested_num_splits;
    params.partition_size = topk + params.extra_topk;
    if (params.num_splits > 1) {
        params.partition_size = std::max(64, (params.partition_size + params.num_splits - 1) / params.num_splits);
        params.partition_size = ((params.partition_size + 63) / 64) * 64;
    }

    at::Tensor out_accum;
    at::Tensor lse_accum;
    if (params.num_splits > 1) {
        lse_accum = torch::empty({params.num_splits, b, h_kv, params.seqlen_q}, opts.dtype(at::kFloat));
        out_accum = torch::empty({params.num_splits, b, s_q, h_q, d_v}, opts);
        params.softmax_lse_ptr = lse_accum.data_ptr<float>();
        params.oaccum_ptr = out_accum.data_ptr();
    }

    hipStream_t stream = reinterpret_cast<hipStream_t>(at::cuda::getCurrentCUDAStream().stream());
    gfx93::fwd::dsa_mls::run_dsa_prefill_nopage_64_dispatch<BFloat16, 512, 512>(params, stream);

    return {out, lse, tile_scheduler_metadata, num_splits};
}

}  // namespace gfx93::decode::sparse_bf16_dsa
