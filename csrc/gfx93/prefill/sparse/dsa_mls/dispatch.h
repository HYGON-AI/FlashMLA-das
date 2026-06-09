#pragma once

#include <algorithm>
#include <stdexcept>

#include <hip/hip_runtime.h>

#include "legacy/include/flash.h"
#include "legacy/include/kernel_traits.h"
#include "legacy/include/static_switch.h"
#include "legacy/src/flash_fwd_b16_mla.h"

#include "legacy/src/flash_fwd_reduce.h"

namespace gfx93::fwd::dsa_mls {

static inline bool dsa_decode_single_kernel_supports(const Flash_fwd_mla_params_dsa& params) {
    if (params.extra_sparse_indices != nullptr && params.extra_topk > 0) {
        return params.topk <= 256 && params.extra_topk <= 1024;
    }
    return params.topk <= 1024;
}

template<typename Kernel_traits, const bool Tail, typename Params>
void run_dsa_mla_splitkv_reduce(Params& params, hipStream_t stream) {
    static_assert(Kernel_traits::kHeadDimV == 512,
                  "run_dsa_mla_splitkv_reduce only supports hdimv == 512");
    using Element = typename Kernel_traits::Element;
    using SplitkvAccumType = typename Kernel_traits::SplitkvAccumType;

    Flash_fwd_mla_reduce_params reduce_params;
    reduce_params.softmax_lse_ptr = params.softmax_lse_ptr;
    reduce_params.oaccum_ptr = params.oaccum_ptr;
    reduce_params.o_ptr = params.o_ptr;
    reduce_params.cu_seqlens_k = params.cu_seqlens_k;
    reduce_params.num_splits = params.num_splits;
    reduce_params.partition_size = params.partition_size;
    reduce_params.h = params.h;
    reduce_params.ngroups = params.ngroups;
    reduce_params.seqlen_q = params.seqlen_q;
    reduce_params.layout = params.layout;
    reduce_params.topk_length = params.topk_length;
    reduce_params.attn_sink = params.attn_sink;
    reduce_params.extra_topk_length = params.extra_topk_length;
    reduce_params.topk = params.topk;
    reduce_params.extra_topk = params.extra_topk;

    if (params.num_splits > 1) {
        dim3 block(256);
        dim3 grid(params.b * params.h * params.seqlen_q, 4);
        constexpr int MAX_NUM_SPLITS = 64;
        if (params.num_splits > MAX_NUM_SPLITS) {
            printf("\x1b[31mnum_splits %d is larger than limit %d, and thus won't execute the kernel\033[0m\n",
                   params.num_splits, MAX_NUM_SPLITS);
            return;
        }
        if (params.num_splits == 2) {
            ::flash_mla_splitkv_reduce_kernel<SplitkvAccumType, Element, 2, true, Tail, Kernel_traits::kHeadDimV>
                <<<grid, block, 0, stream>>>(reduce_params);
        } else if (params.num_splits == 4) {
            ::flash_mla_splitkv_reduce_kernel<SplitkvAccumType, Element, 4, true, Tail, Kernel_traits::kHeadDimV>
                <<<grid, block, 0, stream>>>(reduce_params);
        } else if (params.num_splits == 8) {
            ::flash_mla_splitkv_reduce_kernel<SplitkvAccumType, Element, 8, true, Tail, Kernel_traits::kHeadDimV>
                <<<grid, block, 0, stream>>>(reduce_params);
        } else if (params.num_splits == 16) {
            ::flash_mla_splitkv_reduce_kernel<SplitkvAccumType, Element, 16, true, Tail, Kernel_traits::kHeadDimV>
                <<<grid, block, 0, stream>>>(reduce_params);
        } else if (params.num_splits == 32) {
            ::flash_mla_splitkv_reduce_kernel<SplitkvAccumType, Element, 32, true, Tail, Kernel_traits::kHeadDimV>
                <<<grid, block, 0, stream>>>(reduce_params);
        } else if (params.num_splits == 64) {
            ::flash_mla_splitkv_reduce_kernel<SplitkvAccumType, Element, 64, true, Tail, Kernel_traits::kHeadDimV>
                <<<grid, block, 0, stream>>>(reduce_params);
        } else {
            printf("\x1b[31mnum_splits %d is not supported yet, and thus won't execute the kernel\033[0m\n",
                   params.num_splits);
        }
    }
}

template<typename T, int Headdim, int HeaddimV>
static inline void run_dsa_prefill_nopage_64_dispatch(Flash_fwd_mla_params_dsa& params, hipStream_t stream) {
    constexpr int kBlockM = 64;
    constexpr int kBlockN = 64;
    constexpr int WARP_M = 16;

    dim3 dimBlock;
    dimBlock.x = std::min((kBlockM / WARP_M) * 64, 1024);
    dimBlock.y = 1;
    dimBlock.z = 1;

    dim3 dimGrid;
    dimGrid.x = (params.seqlen_q + kBlockM - 1) / kBlockM;
    dimGrid.y = 1;
    dimGrid.z = params.b;

    using Kernel_traits = Flash_fwd_kernel_traits<
        Headdim, HeaddimV, kBlockM, kBlockN, 32, WARP_M, 64, 2,
        false, false, T, T>;

    constexpr bool Is_dropout = false;
    constexpr bool IsEvenMNConst = false;

    constexpr int REUSE_KV = 1;
    const bool has_extra = params.extra_sparse_indices != nullptr && params.extra_topk > 0;
    if (params.num_splits == 1) {
        BOOL_SWITCH(params.mtp > 1, Is_MTP, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                BOOL_SWITCH(has_extra, Has_extra, [&] {
                    if (dsa_decode_single_kernel_supports(params)) {
                        BOOL_SWITCH(params.decode_use_c_load, DecodeCLoad, [&] {
                            flash::flash_fwd_mla_decode_kernel_gfx938_dsa_nopage_64<
                                Kernel_traits, true, Is_dropout, false, Is_causal,
                                IsEvenMNConst, true, false, Is_MTP, 0, DecodeCLoad, Has_extra, Flash_fwd_mla_params_dsa>
                                <<<dimGrid, dimBlock, 21 * 1024, stream>>>(params);
                        });
                    } else {
                        flash::flash_fwd_mla_decode_kernel_gfx938_dsa_nopage_64_splitkv_direct<
                            Kernel_traits, true, Is_dropout, false, Is_causal,
                            IsEvenMNConst, true, false, Is_MTP, 0, Has_extra, Flash_fwd_mla_params_dsa>
                            <<<dimGrid, dimBlock, 21 * 1024, stream>>>(params);
                    }
                });
            });
        });
    } else if (params.num_splits != 0) {
        if (!params.splitkv_use_fp32_as_accum) {
            throw std::runtime_error("DSA splitkv path requires official fp32 accum/combine");
        }
        dimGrid.y = params.splitkv_use_meta_dispatch ? 1 : params.num_splits;
        dimGrid.z = params.splitkv_use_meta_dispatch ? params.num_sm_parts : params.b;
        BOOL_SWITCH(params.mtp > 1, Is_MTP, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                BOOL_SWITCH(has_extra, Has_extra, [&] {
                    if (params.splitkv_use_meta_dispatch) {
                        flash::flash_fwd_mla_decode_kernel_gfx938_dsa_nopage_64_splitkv_meta_dispatch<
                            Kernel_traits, true, Is_dropout, false, Is_causal,
                            IsEvenMNConst, true, false, Is_MTP, 0, Has_extra, Flash_fwd_mla_params_dsa>
                            <<<dimGrid, dimBlock, 21 * 1024, stream>>>(params);
                    } else {
                        flash::flash_fwd_mla_decode_kernel_gfx938_dsa_nopage_64_splitkv<
                            Kernel_traits, true, Is_dropout, false, Is_causal,
                            IsEvenMNConst, true, false, Is_MTP, 0, Has_extra, Flash_fwd_mla_params_dsa>
                            <<<dimGrid, dimBlock, 21 * 1024, stream>>>(params);
                    }
                });
            });
        });
    } else {
        BOOL_SWITCH(params.mtp > 1, Is_MTP, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                if (params.topk == 2048) {
                    constexpr bool CanUseFastTopk2048 = Headdim == 576 && HeaddimV == 512;
                    if constexpr (CanUseFastTopk2048) {
                        if (params.seqlen_k < params.topk) {
                            flash::flash_fwd_mla_decode_kernel_gfx938_dsa_prefill_topk2048_fast_nopage_64<
                                Kernel_traits, true, Is_dropout, false, Is_causal,
                                IsEvenMNConst, true, false, Is_MTP, 0, true, Flash_fwd_mla_params_dsa>
                                <<<dimGrid, dimBlock, 21 * 1024, stream>>>(params);
                        } else {
                            flash::flash_fwd_mla_decode_kernel_gfx938_dsa_prefill_topk2048_fast_nopage_64<
                                Kernel_traits, true, Is_dropout, false, Is_causal,
                                IsEvenMNConst, true, false, Is_MTP, 0, false, Flash_fwd_mla_params_dsa>
                                <<<dimGrid, dimBlock, 21 * 1024, stream>>>(params);
                        }
                    } else {
                        flash::flash_fwd_mla_decode_kernel_gfx938_dsa_prefill_nopage_64<
                            Kernel_traits, true, Is_dropout, false, Is_causal,
                            IsEvenMNConst, true, false, Is_MTP, 0, Flash_fwd_mla_params_dsa>
                            <<<dimGrid, dimBlock, 21 * 1024, stream>>>(params);
                    }
                } else {
                    flash::flash_fwd_mla_decode_kernel_gfx938_dsa_prefill_nopage_64_topk1024<
                        Kernel_traits, true, Is_dropout, false, Is_causal,
                        IsEvenMNConst, true, false, Is_MTP, 0, Flash_fwd_mla_params_dsa>
                        <<<dimGrid, dimBlock, 21 * 1024, stream>>>(params);
                }
            });
        });
    }
}

}  // namespace gfx93::fwd::dsa_mls
