#ifndef FLASH_MLA_DISABLE_FP16
#include "../splitkv_mla.cuh"

namespace gfx93 {

template void run_flash_splitkv_mla_kernel_impl<cutlass::half_t, 512, 512>(DenseAttnDecodeParams &params);

}
#endif
