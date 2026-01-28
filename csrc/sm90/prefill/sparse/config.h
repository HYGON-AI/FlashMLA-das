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

static constexpr int kNWarps = 4;
static constexpr int B_H = 16;
static constexpr int B_TOPK = 64;    // TopK block size
static constexpr int NUM_THREADS = kNWarps * 64;
static constexpr float MAX_INIT_VAL = -1e30;    // We use this number as the initial value for mi (max logits)

using Element = cutlass::bfloat16_t;
using elem_type = Element;
using ElementAccum = float;
using index_t = int64_t;
static constexpr int kBlockM = B_H;
static constexpr int kBlockN = B_TOPK;
static constexpr int kHeadDim = D_QK;
static constexpr int kHeadDimV = D_V;

using ValLayoutMNK = Layout<Shape<_1, _1, _1>>;
// 没打开?
// #if defined(__gfx936__) || defined(__gfx938__) || 1
    // using MMA_Atom_Arch = std::conditional_t<
    //     std::is_same_v<elem_type, cutlass::half_t>,
    //     MMA_Atom<GFX928_16x16x32_F32F16F16F32_NT>,
    //     MMA_Atom<GFX928_16x16x32_F32BF16BF16F32_NT>
    // >;
using MMA_Atom_Arch = std::conditional_t<
    std::is_same_v<elem_type, cutlass::half_t>,
    MMA_Atom<GFX928_16x16x32_F32F16F16F32_NN>,
    MMA_Atom<GFX928_16x16x32_F32BF16BF16F32_NN>
>;
using TiledMma = TiledMMA<
    MMA_Atom_Arch,
    Layout<Shape<_1, Int<kNWarps>, _1>>,  // 1x4x1 or 1x8x1 thread group
    ValLayoutMNK>;
// #endif

using MMA_Atom_Arch_16x32 = std::conditional_t<
    std::is_same_v<elem_type, cutlass::half_t>,
    MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
    MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
>;

using TiledMma_O = TiledMMA<
    MMA_Atom_Arch_16x32,
    Layout<Shape<_1, Int<kNWarps>, _1>>,  // 1x4x1 or 1x8x1 thread group
    ValLayoutMNK>;

using SmemLayoutAtomQ = 
        Layout<Shape<Int<16>, Int<32>>, Stride<Int<32>, _1>>;
using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

using SmemLayoutAtomK = decltype(composition(
    Swizzle<3, 3, 3>{},
    Layout<Shape<Int<8>, Int<32>>, Stride<Int<32>, _1>>{}));
using SmemLayoutK = decltype(tile_to_shape(
    SmemLayoutAtomK{},
    Shape<Int<kBlockN>, Int<16 * 32>>{}));
using SmemLayoutAtomV = SmemLayoutAtomK;
using SmemLayoutV = decltype(tile_to_shape(
    SmemLayoutAtomV{},
    Shape<Int<kBlockN>, Int<kHeadDimV>>{}));
using SmemLayoutVtransposed = decltype(
    composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));
using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

using SmemLayoutAtomP = Layout<Shape<Int<4*16*16>>, Stride<Int<1>>>;
using SmemLayoutP = decltype(tile_to_shape(
    SmemLayoutAtomP{},
    Shape<Int<4*16*16>>{}));

using SmemLayoutRow = Layout<Shape<_128>, Stride<_1>>; 

using SmemLayoutK_place_holder = decltype(tile_to_shape(
    SmemLayoutAtomK{},
    Shape<Int<kBlockN>, Int<4 * 32>>{}));

struct SharedMemoryPlan {
    union {
        struct {
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;  // Double buffer
        };
        struct {
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutK_place_holder>> smem_place_holder;  // Double buffer
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>> smem_p;
            cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutRow>> smem_row_sum;
            cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutRow>> smem_row_max;
        };
        struct {
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
        };
    };
    // transac_bar_t bar_q, bar_k0_free[2], bar_k0_ready[2], bar_k1_free[2], bar_k1_ready[2], bar_is_kv_valid_ready;
};





static __device__ __forceinline__ void
devfunc(const SparseAttnFwdParams &params);

static void run(const SparseAttnFwdParams &params);

};


};
