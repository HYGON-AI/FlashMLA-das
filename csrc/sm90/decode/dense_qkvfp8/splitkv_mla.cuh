#include <cutlass/cutlass.h>

#include "utils.h"

#include "params.h"
#include "config.h"
#include "traits.h"
#include "softmax.h"
using namespace cute;

namespace sm90 {

template<typename T>
__device__ void
compute_attn_1rowblock_splitkv_mla_qkvfp8_gfx938(const DenseAttnDecodeParams params, 
                                        const int bidb, const int bidh, const int m_block,
                                        const int n_split_idx, const int seqlen_k,
                                        const int n_block_min, const int n_block_max, const bool NoSplit)
{
    constexpr int kBlockM = T::kBlockM;
    constexpr int kBlockN = T::kBlockN;
    constexpr int kHeadDim = T::kHeadDim;
    constexpr int kHeadDimV = T::kHeadDimV;
    const int tidx = threadIdx.x;

    
}

template<typename T>
__global__ void __launch_bounds__(T::NUM_THREADS, 1)
flash_fwd_splitkv_mla_qkvfp8_kernel(const DenseAttnDecodeParams params) {
    const int m_block = blockIdx.x;
    const int bidh = blockIdx.y;
    const int partition_idx = blockIdx.z;

    DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
    if (sched_meta.begin_req_idx >= params.b) return;
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
        constexpr int kBlockN = T::PAGE_BLOCK_SIZE;
        const int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
        int seqlen_k = __ldg(params.seqlens_k_ptr + batch_idx);
        const int start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
        int end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : cute::ceil_div(seqlen_k, kBlockN);
        const bool is_no_split = batch_idx == sched_meta.begin_req_idx ? !sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);
        
        if (batch_idx > sched_meta.begin_req_idx) {
            __syncthreads(); 
        }
        compute_attn_1rowblock_splitkv_mla_qkvfp8_gfx938<T>(params, batch_idx, bidh, m_block, n_split_idx, 
            seqlen_k, start_block_idx, end_block_idx, is_no_split
        );

    }
}


template<typename InputT>
void run_flash_splitkv_mla_qkvfp8_kernel(DenseAttnDecodeParams &params) {
    FLASH_ASSERT(params.d == Config::HEAD_DIM_K);
    FLASH_ASSERT(params.d_v == Config::HEAD_DIM_V);

    
 
    constexpr size_t smem_size = 65536;

    // Use cudaLaunchKernelEx to enable PDL (Programmatic Dependent Launch)

    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        using T = Traits<InputT, Is_causal>;
        const int num_m_block = cute::ceil_div(params.q_seq_per_hk, T::BLOCK_SIZE_M);
        auto mla_kernel = &flash_fwd_splitkv_mla_qkvfp8_kernel<T>;
        mla_kernel<<<dim3(num_m_block, params.h_k, params.num_sm_parts), T::NUM_THREADS, smem_size, params.stream>>>(params);
    });
    // cudaLaunchConfig_t mla_kernel_config = {
    //     dim3(num_m_block, params.h_k, params.num_sm_parts),
    //     dim3(T::NUM_THREADS, 1, 1),
    //     smem_size,
    //     params.stream,
    //     mla_kernel_attributes,
    //     1
    // };
    // cudaLaunchKernelEx(&mla_kernel_config, mla_kernel, params, tma_params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}
