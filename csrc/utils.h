#pragma once
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <cstdint>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include "defines.h"
#define CHECK_CUDA(call)                                                                                  \
    do {                                                                                                  \
        cudaError_t status_ = call;                                                                       \
        if (status_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
            exit(1);                                                                              \
        }                                                                                                 \
    } while(0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())


#define FLASH_ASSERT(cond)                                                                                \
    do {                                                                                                  \
        if (not (cond)) {                                                                                 \
            fprintf(stderr, "Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                 \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)


#define FLASH_DEVICE_ASSERT(cond)                                                                         \
    do {                                                                                                  \
        if (not (cond)) {                                                                                 \
            printf("Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                          \
            asm volatile("s_trap 0 \n\t");                                                                                 \
        }                                                                                                 \
    } while(0)

#define println(fmt, ...) { print(fmt, ##__VA_ARGS__); print("\n"); }

template<typename T>
__inline__ __host__ __device__ T ceil_div(const T &a, const T &b) {
    return (a + b - 1) / b;
}

#ifndef TRAP_ONLY_DEVICE_ASSERT
#define TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        asm("trap;"); \
} while (0)
#endif

#ifndef TRAP_ONLY_DEVICE_ASSERT
#define TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        asm("trap;"); \
} while (0)
#endif


struct RingBufferState {
    uint32_t cur_block_idx = 0u;

    __device__ __forceinline__
    void update() {
        cur_block_idx += 1;
    }    

    template<uint32_t NUM_STAGES>
    __device__ __forceinline__
    std::pair<uint32_t, bool> get() const {
        uint32_t stage_idx = cur_block_idx % NUM_STAGES;
        bool phase = (cur_block_idx / NUM_STAGES) & 1;
        return {stage_idx, phase};
    }

    __device__ __forceinline__
    RingBufferState offset_by(const int offset) const {
        // Must guarantee no underflow
        uint32_t new_block_idx = static_cast<uint32_t>(static_cast<int>(cur_block_idx) + offset);
        RingBufferState new_state;
        new_state.cur_block_idx = new_block_idx;
        return new_state;
    }
};

namespace flash {

using namespace cute;
////////////////////////////////////////////////////////////////////////////////////////////////////


template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 64 || THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4 || THREADS == 2);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor(x, OFFSET, 64));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<1> {
    // static_assert(THREADS == 64 || THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4 || THREADS == 2);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        return x;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<32> {
template<typename T, typename Operator> 
static __device__ __forceinline__ T run(T x, Operator &op) {
     x = op(x, __shfl_xor(x, 16, 64));
    return x;
}
};

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0, int begin_k=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
}

template<int row, int col, int r_row, typename Tensor0, typename Tensor1>
__forceinline__ __device__  void __ds_read_m32x16_row_col_rrow(Tensor0& src, Tensor1& dst)
{

    auto lds = reinterpret_cast<__fp16 *>(src.data().get());
    auto layout  = src.layout();
    constexpr short offset = layout(0, row, col) * 2;

    auto d = __builtin_amdgcn_ds_read_m32x16f16((__attribute__((address_space(3))) __fp16*)(lds), offset);

    uint16_t * d_ptr = reinterpret_cast<uint16_t*>(&d);
    uint16_t * dst_ptr = reinterpret_cast<uint16_t*>(&(dst(0, r_row, col)));

    dst_ptr[0] = d_ptr[0];
    dst_ptr[1] = d_ptr[1];
    dst_ptr[2] = d_ptr[2];
    dst_ptr[3] = d_ptr[3];
    dst_ptr[4] = d_ptr[4];
    dst_ptr[5] = d_ptr[5];
    dst_ptr[6] = d_ptr[6];
    dst_ptr[7] = d_ptr[7];
}

template<int row, int col, typename Tensor0, typename Tensor1>
__forceinline__ __device__  void __ds_read_m32x16_row_col(Tensor0& src, Tensor1& dst)
{

    auto lds = reinterpret_cast<__fp16 *>(src.data().get());
    auto layout  = src.layout();
    constexpr short offset = layout(0, row, col) * 2;

    auto d = __builtin_amdgcn_ds_read_m32x16f16((__attribute__((address_space(3))) __fp16*)(lds), offset);

    uint16_t * d_ptr = reinterpret_cast<uint16_t*>(&d);
    uint16_t * dst_ptr = reinterpret_cast<uint16_t*>(&(dst(0, row, col)));

    dst_ptr[0] = d_ptr[0];
    dst_ptr[1] = d_ptr[1];
    dst_ptr[2] = d_ptr[2];
    dst_ptr[3] = d_ptr[3];
    dst_ptr[4] = d_ptr[4];
    dst_ptr[5] = d_ptr[5];
    dst_ptr[6] = d_ptr[6];
    dst_ptr[7] = d_ptr[7];
}

inline __device__ float fp8e4m3_to_fp32(const fp8& input) {
  const uint32_t w = (uint32_t)input << 24;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
  uint32_t renorm_shift = __clz(nonsign);
  renorm_shift = renorm_shift > 4 ? renorm_shift - 4 : 0;
  uint32_t result = sign | ((nonsign << renorm_shift >> 4) + ((0x78 - renorm_shift) << 23));

  union {
    uint32_t as_bits;
    float as_value;
  } fp32 = {result};
  return fp32.as_value;
}

template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    // static_assert(decltype(size<0>(acc_layout))::value == 4 || decltype(size<0>(acc_layout))::value == 8);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_1>{});   // (_4,_1,_2):(_1,_0,_4) -> ((_1,_4),_1,_2):((_0,_1),_0,_4)

    return make_layout(make_layout(get<1>(l)), make_layout(get<1>(get<0>(l)), get<2>(l)));  // (1, (4, 2)):((_0),(_1,_4))
};


template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    if constexpr (std::is_same_v<To_type, From_type>)
    {
        return tensor;
    }
    
    constexpr int numel = decltype(size(tensor))::value;
    Tensor tensor_To_type = make_tensor<To_type>(layout(tensor));
    cutlass::Array<To_type, numel> *result_ptr = reinterpret_cast<cutlass::Array<To_type, numel> *>(tensor_To_type.data());
    #if defined(__gfx938__)
        {
            if constexpr (std::is_same_v<To_type, cutlass::bfloat16_t>) {
                cutlass::NumericArrayConverter<To_type, From_type, numel, cutlass::FloatRoundStyle::round_to_nearest> convert_op;
                *result_ptr = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
                } 
                else if constexpr (std::is_same_v<To_type, cutlass::float_e4m3_t>) {
                
                    cutlass::NumericArrayConverter<To_type, From_type, numel,cutlass::FloatRoundStyle::round_to_nearest> convert_op;
                    *result_ptr = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
                }
                else {
                    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
                    *result_ptr = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
                }
                return tensor_To_type;
        }
       
    #else
        {
            if constexpr (std::is_same_v<To_type, cutlass::bfloat16_t>) {
            cutlass::NumericArrayConverter<To_type, From_type, numel, cutlass::FloatRoundStyle::round_toward_zero> convert_op;
                *result_ptr = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
            } else {
                cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
                *result_ptr = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
            }
            return tensor_To_type;
        }
        

    #endif
    // cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // // HACK: this requires tensor to be "contiguous"
    // auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    // return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}


template <class TiledMma, class TiledMma_O,
        typename Engine0, typename Layout0,
        typename Engine1, typename Layout1
        >
__forceinline__ __device__ auto convert_layout_acc_Aregs(const TiledMma& tiled_mma, const TiledMma_O& tiled_mma_o, Tensor<Engine0, Layout0> const& tOrP,
    Tensor<Engine1, Layout1> const& sAcc)
{
    using Value_type = typename Engine0::value_type;

    int tid = threadIdx.x % 64;
    int warp_id = threadIdx.x / 64;

    sAcc((tid % 16 ) * 8  + (tid / 16) + (warp_id % 2) * 4 + (warp_id / 2) * 16 * 32) = tOrP(0, 0, 0);
    sAcc((tid % 16 ) * 8  + (tid / 16) + 1 * 16 * 8 + (warp_id % 2) * 4 + (warp_id / 2) * 16 * 32) = tOrP(1, 0, 0);
    sAcc((tid % 16 ) * 8  + (tid / 16) + 2 * 16 * 8 + (warp_id % 2) * 4 + (warp_id / 2) * 16 * 32) = tOrP(2, 0, 0);
    sAcc((tid % 16 ) * 8  + (tid / 16) + 3 * 16 * 8 + (warp_id % 2) * 4 + (warp_id / 2) * 16 * 32) = tOrP(3, 0, 0);
    __syncthreads();

    using SmemLayoutAtomP = Layout<Shape<Int<16>, Int<64>>, Stride<Int<64>, _1>>;
    using SmemLayoutP = decltype(tile_to_shape(
        SmemLayoutAtomP{},
        Shape<Int<16>, Int<64>>{}));
    Tensor sP_tmp = make_tensor(sAcc.data(),SmemLayoutP{});    
    auto thr_mma = tiled_mma_o.get_thread_slice(tid);
    Tensor tSrACC  = thr_mma.partition_fragment_A(sP_tmp);  

    tSrACC(0, 0, 0) = sAcc(tid * 8 + 0);
    tSrACC(1, 0, 0) = sAcc(tid * 8 + 1);
    tSrACC(2, 0, 0) = sAcc(tid * 8 + 2);
    tSrACC(3, 0, 0) = sAcc(tid * 8 + 3);

    tSrACC(0, 0, 1) = sAcc(tid * 8 + 0 + 4);
    tSrACC(1, 0, 1) = sAcc(tid * 8 + 1 + 4);
    tSrACC(2, 0, 1) = sAcc(tid * 8 + 2 + 4);
    tSrACC(3, 0, 1) = sAcc(tid * 8 + 3 + 4);

    tSrACC(0, 0, 2) = sAcc(tid * 8 + 0 + 16*32);
    tSrACC(1, 0, 2) = sAcc(tid * 8 + 1 + 16*32);
    tSrACC(2, 0, 2) = sAcc(tid * 8 + 2 + 16*32);
    tSrACC(3, 0, 2) = sAcc(tid * 8 + 3 + 16*32);

    tSrACC(0, 0, 3) = sAcc(tid * 8 + 0 + 4 + 16*32);
    tSrACC(1, 0, 3) = sAcc(tid * 8 + 1 + 4 + 16*32);
    tSrACC(2, 0, 3) = sAcc(tid * 8 + 2 + 4 + 16*32);
    tSrACC(3, 0, 3) = sAcc(tid * 8 + 3 + 4 + 16*32);


    return tSrACC;
}

__forceinline__ __device__ bool
is_positive_infinity(const float& f_val)
{
    union Fp32{
        uint32_t as_bits;
        float as_value;
    };
    Fp32 fp32;
    fp32.as_value = f_val;
    Fp32 inf_tmp;
    inf_tmp.as_value = INFINITY;

    return fp32.as_bits == inf_tmp.as_bits;
}

template <
        bool Is_even_MN=true, 
        bool Is_even_K=true, 
        bool Is_load_Q=false,
        class SrcEngine, class SrcLayout,
        class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
lds_direct_copy(
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst,
     int k_idx_, const int row_stride, 
     const int max_MN=0)
{
    #if defined(__gfx936__) || defined(__gfx938__)
    {
        if constexpr (Is_load_Q) {
            // // 32x64
            constexpr int warp_size = 64;
            int tidx = threadIdx.x;
            int warp_id = __builtin_amdgcn_readfirstlane(tidx / warp_size);
            int lane = tidx % warp_size;
            constexpr int element_size = 2;
            int k_idx = __builtin_amdgcn_readfirstlane(k_idx_);
            const int offset_s = 0;

            struct PtrWrapper {
                uint32_t former;
                uint32_t latter;
            };
            PtrWrapper glob_ptr;
            *(uint64_t*)&glob_ptr = reinterpret_cast<uint64_t>(src.data().get());
            // glob_ptr.latter |= 0x40000000; // 62 bit: cache swizzle;  48~61: Stride
        
            uint32x4_t global_addr = {0};
            global_addr[0] = (glob_ptr.former);
            global_addr[1] = (glob_ptr.latter);
            global_addr[2] = 0x80000000;
            global_addr[3] = 0x00020000;

            constexpr int elements_per_thread = 8;
            constexpr int bytes_per_warp = warp_size * 8 * element_size;
            int mma_k = 16*128;
            
            int row = lane % 16;
            int col = lane / 16;
            
            int row_offset = row ;
            int col_offset = (col + warp_id  * 4) * elements_per_thread + k_idx * 128;
            int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
            if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;


            if (!Is_even_K && col_offset >= 576) offset_v = -1;
            int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_idx * mma_k * element_size;
    
            asm volatile(
                "s_mov_b32 m0, %1 \n\t"
                "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
                "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
            :);    



        }
        else {
            constexpr int warp_size = 64;
            int tidx = threadIdx.x;
            int warp_id = __builtin_amdgcn_readfirstlane(tidx / warp_size);
            int lane = tidx % warp_size;
            constexpr int element_size = 2;
            int k_idx = __builtin_amdgcn_readfirstlane(k_idx_);
            const int offset_s = 0;
    
            // global addr
            // uint32x4_t global_addr = {0};
            // *(uint64_t*)&global_addr = reinterpret_cast<uint64_t>(src.data().get());
            // global_addr[1] += 0x41000000; // 62 bit: cache swizzle;  48~61: Stride
            // global_addr[2] = 0xfffffffe;
            // global_addr[3] = 0x00020000;
            struct PtrWrapper {
                uint32_t former;
                uint32_t latter;
            };
            PtrWrapper glob_ptr;
            *(uint64_t*)&glob_ptr = reinterpret_cast<uint64_t>(src.data().get());
            // glob_ptr.latter |= 0x40000000; // 62 bit: cache swizzle;  48~61: Stride
        
            uint32x4_t global_addr = {0};
            global_addr[0] = (glob_ptr.former);
            global_addr[1] = (glob_ptr.latter);
            global_addr[2] = 0x80000000;
            global_addr[3] = 0x00020000;
    
    
    
            constexpr int elements_per_thread = 8;
            constexpr int bytes_per_warp = warp_size * 8 * element_size;
            int mma_k = 32*64;
        
    
            // int row = lane / 4;
            // int col = lane % 4;
            // int swizzle_col = ((row / 2) ^ (col  )) * 4 + (col % 4);
            // 此处待优化，后8行，行号需要交换
            int virtual_row = lane / 8;
            int virtual_col = lane % 8;
            int swizzle_col = virtual_row ^ virtual_col;
            int row = lane / 4;
            // 8->9 9->8
            row = (row >= 8 ) ^ row;
            // row = row >= 8 ? (swizzle_col / 4) > 0 ? row + 1 : row - 1 : row;
            int col = swizzle_col % 4;
            int row_offset = row +  (warp_id * 16) ;
            int col_offset = col * elements_per_thread + k_idx * 32;
            int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
            if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;
            int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_idx * mma_k * element_size;
    
            asm volatile(
                "s_mov_b32 m0, %1 \n\t"
                "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
                "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
            :);    
        }

    }
    #endif
}

template <bool Is_even_K=true, 
          bool Is_even_MN=true, 
          bool Use_cache_swizzle = true,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout
        //   class IdxEngine, class IdxLayout
          >
CUTE_HOST_DEVICE
void
lds_direct_copy_for_prefill_sparse_mla(
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst,
     int row_offset,
     int col,
     int k_idx_, const int row_stride, int max_MN=0)
{
    constexpr int warp_size = 64;
    int tidx = threadIdx.x;
    int warp_id = __builtin_amdgcn_readfirstlane(tidx / warp_size);
    int lane = tidx % warp_size;
    constexpr int element_size = 2;
    int k_idx = __builtin_amdgcn_readfirstlane(k_idx_);
    const int offset_s = 0;

    // global addr
    // uint32x4_t global_addr = {0};
    // *(uint64_t*)&global_addr = reinterpret_cast<uint64_t>(src.data().get());
    // global_addr[1] += 0x41000000; // 62 bit: cache swizzle;  48~61: Stride
    // global_addr[2] = 0xfffffffe;
    // global_addr[3] = 0x00020000;
    struct PtrWrapper {
        uint32_t former;
        uint32_t latter;
    };
    PtrWrapper glob_ptr;
    *(uint64_t*)&glob_ptr = reinterpret_cast<uint64_t>(src.data().get());
    glob_ptr.latter |= ((row_stride * 2) << 16); // 62 bit: cache swizzle;  48~61: Stride

    uint32x4_t global_addr = {0};
    global_addr[0] = (glob_ptr.former);
    global_addr[1] = (glob_ptr.latter);
    global_addr[2] = max_MN;
    global_addr[3] = 0x00020000;



    constexpr int elements_per_thread = 8;
    constexpr int bytes_per_warp = warp_size * 8 * element_size;
    int mma_k = 32*64;

    int col_offset = col * elements_per_thread + k_idx * 32;
    int offset_v = (col_offset) * element_size; // bytes
    // int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
    // if (!Is_even_MN && (row_offset >= max_MN || row_offset < 0)) offset_v = -1;
    int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + (k_idx % 4) * mma_k * element_size;
    typedef uint32_t uint32x2_t __attribute__((ext_vector_type(2)));
    uint32x2_t index_offset = {0};
    index_offset[0] = row_offset == -1 ? max_MN : row_offset;
    index_offset[1] = offset_v;
    asm volatile(
        "s_mov_b32 m0, %1 \n\t"
        "buffer_load_dwordx4 %0, %2, %3 , idxen offen  offset:0, lds \n" ::"v"(index_offset),
        "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
    :);    

}


}