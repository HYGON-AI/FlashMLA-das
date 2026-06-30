#include "../splitkv_mla.cuh"

namespace gfx93 {

template void run_flash_splitkv_mla_kernel_impl<cutlass::bfloat16_t, 576, 512>(DenseAttnDecodeParams &params);

}
