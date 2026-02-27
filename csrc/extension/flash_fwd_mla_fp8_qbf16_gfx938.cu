#include "flash_fwd_mla_kernel_fp8.h"

template void run_mha_fwd_splitkv_mla_fp8<cutlass::bfloat16_t,cutlass::bfloat16_t,576>(Flash_fwd_mla_params &params, cudaStream_t stream, bool is_with_cat);