#pragma once

#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <kerutils/kerutils.cuh>

#include "defines.h"
#include "params.h"

namespace sm90::fwd {

using namespace cute;

template<int D_QK, bool HAVE_TOPK_LENGTH>
class KernelTemplate {
public:

static constexpr int D_Q = D_QK;
static constexpr int D_K = D_QK;
static constexpr int D_V = 512;

static constexpr int B_H = 64;
static constexpr int B_TOPK = 64;    // TopK block size
static constexpr int NUM_THREADS = 128*3;
static constexpr float MAX_INIT_VAL = -1e30;    // We use this number as the initial value for mi (max logits)

template<int NUM_TILES>
using SmemLayoutQTiles = decltype(coalesce(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutOTiles = decltype(coalesce(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTiles = decltype(coalesce(tile_to_shape(
    GMMA::Layout_SW128_Atom<bf16, GMMA::Major::K>{},
    Shape<Int<B_TOPK>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTilesTransposed = decltype(composition(
	SmemLayoutKTiles<NUM_TILES>{},
	Layout<Shape<Int<64*NUM_TILES>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}
));

using SmemLayoutQ = SmemLayoutQTiles<D_Q/64>;
using SmemLayoutO = SmemLayoutOTiles<D_V/64>;
using SmemLayoutK = SmemLayoutKTiles<D_Q/64>;
using SmemLayoutV = SmemLayoutKTilesTransposed<D_V/64>;
using SmemLayoutHalfV = SmemLayoutKTilesTransposed<D_V/64/2>;

using SmemLayoutS = decltype(coalesce(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<B_TOPK>>{}
), Shape<_1, _1>{}));

struct SharedMemoryPlan {
    union {
        array_aligned<bf16, cosize_v<SmemLayoutQ>> q;
        array_aligned<bf16, cosize_v<SmemLayoutO>> o;
    } q_o;
    array_aligned<bf16, cosize_v<SmemLayoutK>> k[2];
    array_aligned<bf16, cosize_v<SmemLayoutS>> s[D_QK == 576 ? 1 : 2];  // For V3.2 (whose D_QK is 576), we overlap sS[0] with k's RoPE part to save shared memory; For MODEL1 (whose D_QK is 512), we allocate two buffers

    bool is_kv_valid[2][B_TOPK];
    float2 sM[32];
    float2 sL[64];   // For reduction across WG0/1 in epilogue
    float final_max_logits[64], final_lse[64];
    // transac_bar_t bar_q, bar_k0_free[2], bar_k0_ready[2], bar_k1_free[2], bar_k1_ready[2], bar_is_kv_valid_ready;
};





static __device__ __forceinline__ void
devfunc(const SparseAttnFwdParams &params);

static void run(const SparseAttnFwdParams &params);

};


};
