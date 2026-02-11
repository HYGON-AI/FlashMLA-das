#pragma once

#include "params.h"

namespace sm90 {

template<typename InputT>
void run_flash_splitkv_mla_kvfp8_kernel(DenseAttnDecodeParams_fp8 &params);

}
