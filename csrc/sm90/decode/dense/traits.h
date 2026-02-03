#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/barrier.h>

#include "config.h"

using namespace cute;

template<typename InputT_, bool Is_causal_>
struct Traits {
    using InputT = InputT_;
    static constexpr bool Is_causal = Is_causal_;
    static constexpr int BLOCK_SIZE_M = Config::BLOCK_SIZE_M;
    static constexpr int PAGE_BLOCK_SIZE = Config::PAGE_BLOCK_SIZE;
    static constexpr int HEAD_DIM_K = Config::HEAD_DIM_K;
    static constexpr int HEAD_DIM_V = Config::HEAD_DIM_V;

    static constexpr int NUM_THREADS = 256;

    static_assert(std::is_same_v<InputT, cutlass::bfloat16_t> || std::is_same_v<InputT, cutlass::half_t>);

    static constexpr int kBlockM = BLOCK_SIZE_M;
    static constexpr int kBlockN = PAGE_BLOCK_SIZE;
    static constexpr int kHeadDim = HEAD_DIM_K;
    static constexpr int kHeadDimV = HEAD_DIM_V;
    static constexpr int kNWarps = 4;

    using Element = InputT;
    using elem_type = Element;
    using ElementAccum = float;
    
    using SmemLayoutRow = Layout<Shape<_128>, Stride<_1>>; 
    using SmemLayoutAtomK = decltype(composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<Int<8>, Int<32>>, Stride<Int<32>, _1>>{}));
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<16 * 32>>{}));
      
    using SmemLayoutK_place_holder = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<15 * 32>>{}));
    using SmemLayoutAtomV = SmemLayoutAtomK;
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<kBlockN>, Int<kHeadDimV>>{}));
    
    using SmemLayoutAtomP = Layout<Shape<Int<4*16*16>>, Stride<Int<1>>>;
    using SmemLayoutP = decltype(tile_to_shape(
        SmemLayoutAtomP{},
        Shape<Int<4*16*16>>{}));
    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));
    
    using SmemLayoutAtomQ = decltype(composition(
            Swizzle<3, 3, 3>{},
            Layout<Shape<Int<8>, Int<64>>, Stride<Int<64>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
            SmemLayoutAtomQ{},
            Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using ValLayoutMNK = Layout<Shape<_1, _1, _1>>;     
// #if defined(__gfx936__) ||  defined(__gfx938__)
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x16x32_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x16x32_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<_1, Int<kNWarps>, _1>>,  // 1x4x1 or 1x8x1 thread group
        ValLayoutMNK>;
// #elif defined(__gfx928__)
//     using MMA_Atom_Arch = std::conditional_t<
//         std::is_same_v<elem_type, cutlass::half_t>,
//         MMA_Atom<GFX928_16x16x32_F32F16F16F32_NT>,
//         MMA_Atom<GFX928_16x16x32_F32BF16BF16F32_NT>
//     >;
//     using TiledMma = TiledMMA<
//         MMA_Atom_Arch,
//         Layout<Shape<_1, Int<kNWarps>, _1>>,  // 1x4x1 or 1x8x1 thread group
//         ValLayoutMNK>;
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
    using GmemLayoutAtomQ = Layout<Shape <_32, _8>,  
                Stride< _8, _1>>;
    using GmemTiledCopyQ = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
            GmemLayoutAtomQ{},
            Layout<Shape<_1, _8>>{})); 



    struct SharedMemoryPlan {
        union {
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;  // Double buffer

            };
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutK_place_holder>> smem_temp;  // Double buffer
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>> smem_p;
                cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutRow>> smem_row_sum;
                cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutRow>> smem_row_max;
            };
            struct {
                cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
            };

        };
    };

};



