#include <cutlass/cutlass.h>

#include "utils.h"

#include "params.h"
#include "config.h"
#include "traits.h"

using namespace cute;

namespace sm90 {

// Here we use MAX_INIT_VAL_SM to initialize sM, and MAX_INIT_VAL for masking
// The reason is that, we need to calculate new_max = max(sM(row_idx), cur_max*scale_softmax_log2)
// so we must guarantee that MAX_INIT_VAL*scale_softmax_log2 < MAX_INIT_VAL_SM
static constexpr float MAX_INIT_VAL_SM = -1e30f;
static constexpr float MAX_INIT_VAL = -1e33f;


template<typename T>
__global__ void __launch_bounds__(T::NUM_THREADS, 1)
flash_fwd_splitkv_mla_kernel(const DenseAttnDecodeParams params) {

}


template<typename InputT>
void run_flash_splitkv_mla_kernel(DenseAttnDecodeParams &params) {
    FLASH_ASSERT(params.d == Config::HEAD_DIM_K);
    FLASH_ASSERT(params.d_v == Config::HEAD_DIM_V);

    using T = Traits<InputT>;
    auto shape_Q = make_shape(params.q_seq_per_hk, params.d, params.h_k, params.b);
 
    auto mla_kernel = &flash_fwd_splitkv_mla_kernel<T>;
    constexpr size_t smem_size = sizeof(typename T::SharedMemoryPlan);

    // Use cudaLaunchKernelEx to enable PDL (Programmatic Dependent Launch)
    const int num_m_block = cute::ceil_div(params.q_seq_per_hk, T::BLOCK_SIZE_M);

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
