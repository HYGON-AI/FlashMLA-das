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
    static_assert(std::is_same_v<InputT, cutlass::float_e4m3_t>);
    static constexpr int kBlockM = BLOCK_SIZE_M;
    static constexpr int kBlockN = PAGE_BLOCK_SIZE;
    static constexpr int kHeadDim = HEAD_DIM_K;
    static constexpr int kHeadDimV = HEAD_DIM_V;
    static constexpr int kNWarps = 4;
    using Element = InputT;
    using elem_type = Element;
    using ElementAccum = float;

    using SmemLayoutRow = Layout<Shape<_128>, Stride<_1>>; 
    static constexpr int kSwizzle = 3;

    using SmemLayoutAtomK = decltype(composition(
        Swizzle<kSwizzle, 4, 3>{},
        Layout<Shape<Int<8>, Int<64>>, Stride<Int<64>, _1>>{}));

    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<8 * 64>>{}));


    struct SharedMemoryPlan {
        
    };

};



