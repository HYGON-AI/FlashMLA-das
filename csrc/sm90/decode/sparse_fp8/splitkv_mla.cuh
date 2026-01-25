#pragma once

#include "splitkv_mla.h"

// #include <cuda_fp8.h>
// #include <math_constants.h>
// #include <cutlass/barrier.h>
// #include <cutlass/arch/barrier.h>
// #include <cutlass/arch/reg_reconfig.h>
// #include <cutlass/cluster_launch.hpp>

#include <kerutils/kerutils.cuh>

#include "utils.h"
#include "components/dequant.h"
#include "components/helpers.h"
#include "config.h"
using namespace cute;

namespace sm90::decode::sparse_fp8 {

static constexpr float MAX_INIT_VAL = -1e30;    // Prevent (-inf) - (-inf) = nan



template<ModelType MODEL_TYPE, int NUM_HEADS>
__device__ void KernelTemplate<MODEL_TYPE, NUM_HEADS>::devfunc(const SparseAttnDecodeParams &params) {

}

template<typename Kernel>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1)
flash_fwd_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams params) {
    Kernel::devfunc(params);
}

template<ModelType MODEL_TYPE, int NUM_HEADS>
void KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(const SparseAttnDecodeParams &params) {
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % TOPK_BLOCK_SIZE == 0);
    KU_ASSERT(params.d_qk == HEAD_DIM_K);
    KU_ASSERT(params.d_v == HEAD_DIM_V);
    KU_ASSERT(params.h_q % BLOCK_M == 0);
    if constexpr (MODEL_TYPE == ModelType::MODEL1) {
        constexpr int BYTES_PER_TOKEN = HEAD_DIM_NOPE + 2*HEAD_DIM_ROPE + 8;
        KU_ASSERT(params.stride_kv_row == BYTES_PER_TOKEN, "Each page block in KV cache must be contiguous for head64 sparse fp8 decoding attention in MODEL1");  // Each block must be contiguous
        if (params.extra_kv != nullptr) {
            KU_ASSERT(params.stride_extra_kv_row == BYTES_PER_TOKEN, "Each page block in extra KV cache must be contiguous for head64 sparse fp8 decoding attention in MODEL1");  // Each block must be contiguous
        }
    } else {
        KU_ASSERT(params.extra_kv == nullptr, "V3.2 does not support extra KV cache");
        KU_ASSERT(params.topk_length == nullptr, "V3.2 does not support dynamic topk length");
        KU_ASSERT(params.stride_kv_row == 656);  // number of bytes per token (512 fp8 + 4 float32 + 64 bfloat16)
    }
}

template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params) {
    KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(params);
}

}
