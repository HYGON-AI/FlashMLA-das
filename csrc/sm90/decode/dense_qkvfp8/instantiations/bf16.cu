#include "../splitkv_mla.cuh"
#include "../splitkv_mla.h"

namespace sm90 {

template void run_flash_splitkv_mla_qkvfp8_kernel<cutlass::float_e4m3_t>(DenseAttnDecodeParams &params);

}
