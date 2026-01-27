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
#include "softmax.h"
using namespace cute;

namespace sm90::decode::sparse_fp8 {
#define CUDART_L2E_F            1.442695041F

static constexpr float MAX_INIT_VAL = -1e30;    // Prevent (-inf) - (-inf) = nan

template<ModelType MODEL_TYPE, int NUM_HEADS>
__device__ void KernelTemplate<MODEL_TYPE, NUM_HEADS>::compute_attn_1rowblock_splitkv_sparse_mla_fp8(const SparseAttnDecodeParams &params, const DecodingSchedMeta& sched_meta, int batch_idx)
{
    using Element = cutlass::bfloat16_t;
    using index_t = int64_t;
    const int tidx = threadIdx.x;
    const int lane_idx = tidx % 64;
    const int warp_idx = tidx / 64;
    const int head_block_idx = NUM_M_BLOCKS == 1 ? 0 : blockIdx.x;
    const int s_q_idx = blockIdx.y;
    extern __shared__ char shared_memory[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(shared_memory);
    struct MainloopArgs {
        int start_block_idx, end_block_idx;
        bool is_no_split;

        // The following fields are only valid for MODEL1
        int topk_length, extra_topk_length, num_orig_kv_blocks;
    };

    auto get_cur_req_info = [&](int batch_idx) -> MainloopArgs {
        MainloopArgs args;
        int total_topk_padded;
        if constexpr (MODEL_TYPE == ModelType::V32) {
            total_topk_padded = params.topk;
        } else {
            int topk_length = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
            int orig_topk_padded = max(ku::ceil(topk_length, (int)TOPK_BLOCK_SIZE), (int)TOPK_BLOCK_SIZE);
            int extra_topk_length = params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
            total_topk_padded = orig_topk_padded + ku::ceil(extra_topk_length, (int)TOPK_BLOCK_SIZE);
            args.topk_length = topk_length;
            args.extra_topk_length = extra_topk_length;
            args.num_orig_kv_blocks = orig_topk_padded / TOPK_BLOCK_SIZE;
        }

        args.start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
        args.end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : total_topk_padded / TOPK_BLOCK_SIZE;
        args.is_no_split = batch_idx == sched_meta.begin_req_idx ? !sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);

        return args;
    };

    const index_t row_offset_q = batch_idx * params.stride_q_b + head_block_idx * BLOCK_M * params.stride_q_h_q + s_q_idx * params.stride_q_s_q;
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q) + row_offset_q),
                            Shape<Int<BLOCK_M>, Int<HEAD_DIM_K>>{},
                            make_stride(params.stride_q_h_q, _1{}));
    const index_t row_offset_k = 0;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<uint8_t *>(params.kv) + row_offset_k),
                            Shape<Int<TOPK_BLOCK_SIZE>, Int<HEAD_DIM_K>>{},
                            make_stride(params.stride_kv_row, _1{}));
    Tensor sV = make_tensor(make_smem_ptr(plan.smem_v.data()), SmemLayoutV{});
    Tensor sK = make_tensor(make_smem_ptr(plan.smem_v.data()), SmemLayoutK{});
    Tensor sP = make_tensor(make_smem_ptr(plan.smem_p.data()), SmemLayoutP{});    
    Tensor sVt = make_tensor(sV.data(), SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), SmemLayoutVtransposedNoSwizzle{});
    Tensor sRow_max_reduce_buffer = make_tensor(make_smem_ptr(plan.smem_row_max.data()), SmemLayoutRow{});    
    Tensor sRow_sum_reduce_buffer = make_tensor(make_smem_ptr(plan.smem_row_sum.data()), SmemLayoutRow{});

    const index_t row_offset_topk =  batch_idx * params.stride_indices_b + s_q_idx * params.stride_indices_s_q; // todo
    int* gIndices = reinterpret_cast<int *>(params.indices) + row_offset_topk;
    int* gExtraIndices = params.extra_indices + batch_idx*params.stride_extra_indices_b + s_q_idx*params.stride_extra_indices_s_q; // (extra_topk) : (1)
    TiledMMA tiled_mma = TiledMma{}; 
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    TiledMMA tiled_mma_16x16x32 = TiledMma_16_16_32{}; 
    auto thr_mma_16x16x32 = tiled_mma_16x16x32.get_thread_slice(tidx);
    TiledMMA tiled_mma_o = TiledMma_O{}; 
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);

    // load Q
    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);
    Tensor tSrQ = thr_mma.partition_fragment_A(gQ);
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));
    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true>(gmem_tiled_copy_Q, tSgQ, tSrQ, tQcQ, tQpQ, params.h_q - head_block_idx * BLOCK_M);
    __syncthreads();
    // zhj debug
    // if (head_block_idx == 0)
    // {
    //     printf("tidx = %d, %.2f %.2f %.2f %.2f \n", tidx, float(tSrQ(0)), float(tSrQ(1)), float(tSrQ(2)), float(tSrQ(3)));
    // }
    Tensor tSrK  = thr_mma.partition_fragment_B(gK); 
    auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_16x16x32);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tOsV = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_o);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
    Tensor tOrVt  = thr_mma_o.partition_fragment_B(sVt);
    Tensor tOrVt_copy_view = smem_thr_copy_V.retile_D(tOrVt);

    const auto gK_data = gK.data();
    typedef unsigned int __hip_fp8x4_storage_t;
    typedef unsigned short int __hip_fp8x2_storage_t;
    typedef unsigned char __hip_fp8_storage_t;
    typedef  __fp16  __fp16x8_t __attribute__((ext_vector_type(8)));
    
    union Fp8_storage{
        __fp16x8_t data_128;
        __hip_fp8x4_storage_t fp8_array[4];
    };

    union bf16_storage{
        uint32x4_t data_128;
        uint16_t data_array[8];
    };


    Tensor acc_o = partition_fragment_C(tiled_mma_o, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{});
    clear(acc_o);
    flash::Softmax<size<1>(acc_o)> softmax;

    struct IsOrigBlock {};
    struct IsExtraBlock {};
    auto process_one_block = [&](int block_idx, auto is_extra_block_t) {
        static constexpr bool IS_EXTRA_BLOCK = std::is_same_v<decltype(is_extra_block_t), IsExtraBlock>;
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{}); 
        clear(acc_s);
        int col_idx = lane_idx / 16;
        int token_index = gIndices[block_idx * TOPK_BLOCK_SIZE + (lane_idx % 16) + warp_idx * 16];
        int page_block_size = params.page_block_size;
        int block_index = token_index == -1 ? 0 : (int)((uint32_t)token_index/(uint32_t)page_block_size);   // Use uint32_t division and mod to improve performance        const int token_indexrel_idx_in_block = (token_index + page_block_size) % page_block_size;
        int rel_idx_in_block = (uint32_t)token_index % (uint32_t)page_block_size;   // NOTE When token_index is -1 (UINT_MAX), UINT_MAX%page_block_size < page_block_size, so there will be no illegal-memory-access error
        const index_t offset_k = block_index * params.stride_kv_block;
        uint8_t* gK_base = (uint8_t*)params.kv + offset_k + rel_idx_in_block*params.stride_kv_row;
        float* scale_ptr = (float*)(gK_base + 512);
        float scales[4];
        if (token_index == -1)
        {
            scales[0] = 0.0f;
            scales[1] = 0.0f;
            scales[2] = 0.0f;
            scales[3] = 0.0f;
        }
        else
        {
            for (int i = 0; i < 4; i++)
            {
                scales[i] = scale_ptr[i];
            }
        }
        // zhj debug
        // if (head_block_idx == 0 && threadIdx.x < 64)
        // {
        //     printf("tidx = %d, %.2f %.2f %.2f %.2f %d offset_k = %d token_indexrel_idx_in_block = %d params.stride_kv_row = %d %p params.kv  = %p \n", tidx, float(scales[0]), float(scales[1]), float(scales[2]), float(scales[3]), 
        //     token_index,
        //     offset_k,
        //     token_indexrel_idx_in_block,
        //     params.stride_kv_row,
        //     gK_base,
        //     params.kv 
        //     );
        // }
        Fp8_storage data[4];
        for (int k_idx = 4; k_idx < 8; k_idx++) 
        {
            if (token_index == -1) {
                data[k_idx - 4].data_128 = {0};
            } else {
                data[k_idx - 4].data_128 = *((__fp16x8_t*)(gK_base + col_idx * 16 + k_idx * 64));
            }
        }

        for (int k_idx = 4; k_idx < 8; k_idx++)
        {
            for (int j = 0; j < 16; j+=4) {
                
                #if defined(__gfx938__)
                auto res1 =  __builtin_amdgcn_cvt_pk_f32_fp8(data[k_idx - 4].fp8_array[j/4], false);
                auto res2 =  __builtin_amdgcn_cvt_pk_f32_fp8(data[k_idx - 4].fp8_array[j/4], true);

                auto f1 = res1[0];
                auto f2 = res1[1];
                auto f3 = res2[0];
                auto f4 = res2[1];
                #else
                auto fp8x2_low = *reinterpret_cast<__hip_fp8x2_storage_t*>(&data[k_idx - 4].fp8_array[j / 4]);
                auto fp8x2_high = *(reinterpret_cast<__hip_fp8x2_storage_t*>(&(data[k_idx - 4].fp8_array[j / 4])) + 1);
                auto f1 =  flash::fp8e4m3_to_fp32(static_cast<__hip_fp8_storage_t>((fp8x2_low << 8) >> 8));
                auto f2 =  flash::fp8e4m3_to_fp32(static_cast<__hip_fp8_storage_t>((fp8x2_low >> 8)));
                auto f3 =  flash::fp8e4m3_to_fp32(static_cast<__hip_fp8_storage_t>((fp8x2_high << 8) >> 8));
                auto f4 =  flash::fp8e4m3_to_fp32(static_cast<__hip_fp8_storage_t>((fp8x2_high) >> 8));
                #endif

                f1 *= scales[k_idx / 2];
                f2 *= scales[k_idx / 2];
                f3 *= scales[k_idx / 2];
                f4 *= scales[k_idx / 2];

                // if (block0)
                // {
                //     printf(" tidx = %d  %.4f %.4f %.4f %.4f \n", threadIdx.x, f1, f2, f3, f4);
                // }

                cutlass::NumericConverter<Element, float, cutlass::FloatRoundStyle::round_toward_zero> convert_;
                auto rst0 = convert_(f1);
                auto rst1 = convert_(f2);
                auto rst2 = convert_(f3);
                auto rst3 = convert_(f4);
                tSrK(j, 0, k_idx) = rst0;
                tSrK(j + 1, 0, k_idx) = rst1;
                tSrK(j + 2, 0, k_idx) = rst2;
                tSrK(j + 3, 0, k_idx) = rst3;

            }
            // cute::copy(smem_tiled_copy_K, tSrK(_, _, k_idx), tOsV(_, _, k_idx % 4));
            // __builtin_amdgcn_sched_barrier(0);

            #pragma unroll
            for (int j = 0; j < 8; j++) {
                tOsV(j, 0, (k_idx - 4) * 2) =  tSrK(j, 0, k_idx);
            }
            #pragma unroll     
            for (int j = 8; j < 16; j++) {
                tOsV(j - 8, 0, (k_idx - 4) * 2 + 1) =  tSrK(j, 0, k_idx);
            }    
            // __builtin_amdgcn_sched_barrier(0);

            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
        }
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);
        flash::__ds_read_m32x16_row_col_rrow<0, 0, 2>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow<0, 1, 2>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow<0, 2, 2>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow<0, 3, 2>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow<1, 0, 3>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow<1, 1, 3>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow<1, 2, 3>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow<1, 3, 3>(tOsVt, tOrVt_copy_view);
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);

        // __ds_read_m64x16_row_col_rrow<0, 0, 4>(tOsVt, tOrVt_copy_view);
        for (int k_idx = 0; k_idx < 4; k_idx++)
        {
            if (token_index == -1) {
                data[k_idx].data_128 = {0};
            } else {
                data[k_idx].data_128 = *((__fp16x8_t*)(gK_base + col_idx * 16 + k_idx * 64));
            }        
        }
        for (int k_idx = 0; k_idx < 4; k_idx++)
        {
            for (int j = 0; j < 16; j+=4) {

                #if defined(__gfx938__)
        
                auto res1 =  __builtin_amdgcn_cvt_pk_f32_fp8(data[k_idx].fp8_array[j/4], false);
                auto res2 =  __builtin_amdgcn_cvt_pk_f32_fp8(data[k_idx].fp8_array[j/4], true);

                auto f1 = res1[0];
                auto f2 = res1[1];
                auto f3 = res2[0];
                auto f4 = res2[1];
                #else
                auto fp8x2_low = *reinterpret_cast<__hip_fp8x2_storage_t*>(&data[k_idx].fp8_array[j / 4]);
                auto fp8x2_high = *(reinterpret_cast<__hip_fp8x2_storage_t*>(&(data[k_idx].fp8_array[j / 4])) + 1);
                auto f1 =  flash::fp8e4m3_to_fp32(static_cast<__hip_fp8_storage_t>((fp8x2_low << 8) >> 8));
                auto f2 =  flash::fp8e4m3_to_fp32(static_cast<__hip_fp8_storage_t>((fp8x2_low >> 8)));
                auto f3 =  flash::fp8e4m3_to_fp32(static_cast<__hip_fp8_storage_t>((fp8x2_high << 8) >> 8));
                auto f4 =  flash::fp8e4m3_to_fp32(static_cast<__hip_fp8_storage_t>((fp8x2_high) >> 8));
                #endif

                f1 *= scales[k_idx / 2];
                f2 *= scales[k_idx / 2];
                f3 *= scales[k_idx / 2];
                f4 *= scales[k_idx / 2];

                // if (block0)
                // {
                //     printf(" tidx = %d  %.4f %.4f %.4f %.4f \n", threadIdx.x, f1, f2, f3, f4);
                // }

                cutlass::NumericConverter<Element, float, cutlass::FloatRoundStyle::round_toward_zero> convert_;
                auto rst0 = convert_(f1);
                auto rst1 = convert_(f2);
                auto rst2 = convert_(f3);
                auto rst3 = convert_(f4);
                tSrK(j, 0, k_idx) = rst0;
                tSrK(j + 1, 0, k_idx) = rst1;
                tSrK(j + 2, 0, k_idx) = rst2;
                tSrK(j + 3, 0, k_idx) = rst3;

            }
            // for (int j = 0; j < 16; j++) {
            //     tOsV(j % 8, 0, (k_idx % 4) * 2 + ( j / 8) ) = tSrK(j, 0, k_idx);
            // }
            // __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                tOsV(j, 0, k_idx * 2) =  tSrK(j, 0, k_idx);
            }
            #pragma unroll     
            for (int j = 8; j < 16; j++) {
                tOsV(j - 8, 0, k_idx * 2 + 1) =  tSrK(j, 0, k_idx);
            }   
            // __builtin_amdgcn_sched_barrier(0);  
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
        }

        __syncthreads();
        flash::__ds_read_m32x16_row_col_rrow<0, 0, 0>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow<0, 1, 0>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow<0, 2, 0>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow<0, 3, 0>(tOsVt, tOrVt_copy_view);

        {
            bf16_storage bf16_data0;
            bf16_storage bf16_data1;
            bf16_data0.data_128 = *((uint32x4_t*)(gK_base + col_idx * 16 * 2 + 512 + 16));
            bf16_data1.data_128 = *((uint32x4_t*)(gK_base + col_idx * 16 * 2 + 8 * 2 + 512 + 16));
            for (int j = 0; j < 8; j++) {
                auto rst = cutlass::bfloat16_t::bitcast(bf16_data0.data_array[j]);
                tSrK(j, 0, 8) = rst;
            }
            for (int j = 8; j < 16; j++) {
                auto rst = cutlass::bfloat16_t::bitcast(bf16_data1.data_array[j - 8]);
                tSrK(j, 0, 8) = rst;
            }
            cute::gemm(tiled_mma, tSrQ(_, _, 8), tSrK(_, _, 8), acc_s);    

        }
        // zhj debug
        // if (head_block_idx == 0)
        // {
        //     printf("tidx = %d, %.2f %.2f %.2f %.2f \n", tidx, float(acc_s(0)), float(acc_s(1)), float(acc_s(2)), float(acc_s(3)));
        // }
        asm volatile("s_waitcnt lgkmcnt(0) \n\t s_barrier\n\t");

        Tensor cS = make_identity_tensor(Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{});
        Tensor tScS = thr_mma.partition_C(cS);

        for (int i = 0; i < size(acc_s); ++i) {
            {
                int idx = int(get<1>(tScS(i))) + block_idx * TOPK_BLOCK_SIZE;
                idx = gIndices[idx] ;
                if (idx == -1) acc_s(i) = -INFINITY;
            }

        }
        block_idx == 0
        ? softmax.template softmax_rescale_o_prefill</*Is_first=*/true,  /*Check_inf=*/Is_causal>(acc_s, acc_o, sRow_max_reduce_buffer, params.sm_scale_div_log2)
        :   softmax.template softmax_rescale_o_prefill</*Is_first=*/false, /*Check_inf=*/Is_causal>(acc_s, acc_o, sRow_max_reduce_buffer, params.sm_scale_div_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);
        Tensor tOrP = flash::convert_layout_acc_Aregs(tiled_mma, tiled_mma_o, rP, sP);

        {
            // __ds_read_m32x16_row_col<0, 0>(tOsVt, tOrVt_copy_view);
            flash::__ds_read_m32x16_row_col<1, 0>(tOsVt, tOrVt_copy_view);
            // __ds_read_m32x16_row_col<2, 0>(tOsVt, tOrVt_copy_view);

            // __ds_read_m32x16_row_col<0, 1>(tOsVt, tOrVt_copy_view);
            flash::__ds_read_m32x16_row_col<1, 1>(tOsVt, tOrVt_copy_view);
            // __ds_read_m32x16_row_col<2, 1>(tOsVt, tOrVt_copy_view);
            cute::gemm(tiled_mma_o, tOrP(_, _, 0), tOrVt(_, _, 0), acc_o);
            cute::gemm(tiled_mma_o, tOrP(_, _, 1), tOrVt(_, _, 1), acc_o);
            // __ds_read_m32x16_row_col<0, 2>(tOsVt, tOrVt_copy_view);
            flash::__ds_read_m32x16_row_col<1, 2>(tOsVt, tOrVt_copy_view);
            // __ds_read_m32x16_row_col<2, 2>(tOsVt, tOrVt_copy_view);
            
            
            // __ds_read_m32x16_row_col<0, 3>(tOsVt, tOrVt_copy_view);
            flash::__ds_read_m32x16_row_col<1, 3>(tOsVt, tOrVt_copy_view);
            // __ds_read_m32x16_row_col<2, 3>(tOsVt, tOrVt_copy_view);
            
            
            cute::gemm(tiled_mma_o, tOrP(_, _, 2), tOrVt(_, _, 2), acc_o);
            cute::gemm(tiled_mma_o, tOrP(_, _, 3), tOrVt(_, _, 3), acc_o);
        }
    };

    MainloopArgs args = get_cur_req_info(batch_idx);
    for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; block_idx++) {
        process_one_block(block_idx, IsOrigBlock{});
    }

    if (args.is_no_split) {
        int start_head_idx = head_block_idx*BLOCK_M;
        Tensor lse = softmax.template normalize_softmax_lse<false>(acc_o, sRow_sum_reduce_buffer, params.sm_scale);
        const index_t row_offset_o = batch_idx * params.stride_o_b + start_head_idx * params.stride_o_h_q + s_q_idx * params.stride_o_s_q ;
        Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.out) + row_offset_o),
                                        Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
                                        make_stride(params.stride_o_h_q, _1{}));
        if (params.attn_sink != nullptr) {
            float rAttn_sink = __ldg((float*)params.attn_sink + start_head_idx + lane_idx % 16); 
            float lse_exp2 = __builtin_amdgcn_exp2f(lse[lane_idx % 16] * CUDART_L2E_F);
            float rAttn_sink_exp2 = __builtin_amdgcn_exp2f(rAttn_sink * CUDART_L2E_F);
            float o_scale = lse_exp2 / (lse_exp2 + rAttn_sink_exp2);
            for (int i = 0; i < size(acc_o); i++)
            {
                acc_o(i) *= o_scale;
            }
        }
        
        float* gSoftmaxLse = (float*)params.lse + batch_idx * params.stride_lse_b + start_head_idx + s_q_idx * params.stride_lse_s_q;	// (BLOCK_M) : (1)
        {
            auto rO = flash::convert_type<Element>(acc_o);
            int row, col;
            const int warpId = tidx / 64;
            const int laneId = tidx % 64;
            for (int mi = 0; mi < size<1>(acc_o); ++mi) {
                row = mi * BLOCK_M + laneId % 16;
                if (row < params.h_q) {
                    for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                        // col = (laneId / 16) + ni * 128 + warpId * 32 ;
                        // 为了使用global_loadx4指令, V矩阵吸入lds的时候 N方向发生了了交换
                        /*
                        ------------------- N 方向----------------------
                        |0 1 ... 7 16 ... 31 40 ... 47 56... 64 8 .. 15 32 ... 39
                        |
                        |
                        k
                        方向
                        |
                        |
                        |
                        */
                        col = (laneId / 16) + ni * 128 + (warpId % 2) * 8 + (warpId / 2) * 64;
                        for (int i = 0; i < 4; i ++) {
                            for (int j = 0; j < 2; j++) {
                                gO(row, col) = rO(i * 2 + j, mi, ni);
                                col += 4;
                            }
                            col += 8;
                        }
                        // for (int ei = 0; ei < size<0>(acc_o); ++ei) {
                        //     gO(row, col) = rO(ei, mi, ni);
                        //     col += 4;
                        // }
                    }
                    gSoftmaxLse[row] = lse(mi);
                }


                // if (s_q_idx == 1)
                // {
                //     printf(" %.2f \n", lse(mi));
                // }

                
                // gMax_logits[row] = softmax.row_max(mi) * params.sm_scale_div_log2;
            
            }
        }


    } else {
        int start_head_idx = head_block_idx*BLOCK_M;
        Tensor lse = softmax.template normalize_softmax_lse<false, true>(acc_o, sRow_sum_reduce_buffer, params.sm_scale);
        int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
        int split_idx = __ldg(params.num_splits_ptr+batch_idx) + n_split_idx;
        float* oaccum_ptr = (float*)params.o_accum + split_idx*params.stride_o_accum_split + s_q_idx*params.stride_o_accum_s_q + start_head_idx*params.stride_o_accum_h_q;	// (BLOCK_M, HEAD_DIM_V) : (params.stride_o_accum_h_q, 1)
        Tensor gOaccum = make_tensor(make_gmem_ptr(oaccum_ptr), make_layout(
            Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
            make_stride(params.stride_o_accum_h_q, _1{})
        ));
        float* gSoftmaxLseAccum = (float*)params.lse_accum + split_idx*params.stride_lse_accum_split + s_q_idx*params.stride_lse_accum_s_q + start_head_idx;	// (BLOCK_M) : (1)
        {
            // auto rO = flash::convert_type<Element>(acc_o);
            int row, col;
            const int warpId = tidx / 64;
            const int laneId = tidx % 64;
            for (int mi = 0; mi < size<1>(acc_o); ++mi) {
                row = mi * BLOCK_M + laneId % 16;
                if (row < params.h_q) {
                    for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                        // col = (laneId / 16) + ni * 128 + warpId * 32 ;
                        // for (int ei = 0; ei < size<0>(acc_o); ++ei) {
                        //     gOaccum(row, col) = acc_o(ei, mi, ni);
                        //     col += 4;
                        // }
                        col = (laneId / 16) + ni * 128 + (warpId % 2) * 8 + (warpId / 2) * 64;
                        for (int i = 0; i < 4; i ++) {
                            for (int j = 0; j < 2; j++) {
                                gOaccum(row, col) = acc_o(i * 2 + j, mi, ni);
                                col += 4;
                            }
                            col += 8;
                        }
                    }

                    gSoftmaxLseAccum[row] = lse(mi);
                }

                // gMax_logits[row] = softmax.row_max(mi) * params.sm_scale_div_log2;
            
            }
        }
    }

}


template<ModelType MODEL_TYPE, int NUM_HEADS>
__device__ void KernelTemplate<MODEL_TYPE, NUM_HEADS>::devfunc(const SparseAttnDecodeParams &params) {
    const int partition_idx = blockIdx.z;
    DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];

    if (sched_meta.begin_req_idx >= params.b) return;
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
        // if (threadIdx.x == 0)
        // {
        //     printf(" batch_idx = %d end_req_idx = %d \n ", batch_idx, sched_meta.end_req_idx);
        // }
        if (batch_idx > sched_meta.begin_req_idx) {
            __syncthreads();  
        }
        compute_attn_1rowblock_splitkv_sparse_mla_fp8(params, sched_meta, batch_idx);
    
    }
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
    auto mla_kernel = &flash_fwd_splitkv_mla_fp8_sparse_kernel<KernelTemplate<MODEL_TYPE, NUM_HEADS>>;
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    // zhj debug
    // printf("NUM_M_BLOCKS = %d smem_size = %d \n",NUM_M_BLOCKS, smem_size);
    mla_kernel<<<dim3(NUM_M_BLOCKS, params.s_q, params.num_sm_parts), NUM_THREADS, smem_size, params.stream>>>(params);

}

template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params) {
    KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(params);
}

}
