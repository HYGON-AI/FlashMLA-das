#pragma once

#include "config.h"
#include "utils.h"
#include "softmax.h"
#include "../../helpers.h"

namespace sm90::fwd {
#define CUDART_L2E_F            1.442695041F

using namespace cute;

template<int D_QK, bool HAVE_TOPK_LENGTH>
__device__ void KernelTemplate<D_QK, HAVE_TOPK_LENGTH>::devfunc(const SparseAttnFwdParams &params) {
    extern __shared__ char smem_[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(smem_);

    const int tidx = threadIdx.x;
    static constexpr int kBlockM = B_H;
    static constexpr int kBlockN = B_TOPK;
    static constexpr int kHeadDim = D_QK;
    static constexpr int kHeadDimV = D_V;
    const int warp_idx = tidx / 64;
    const int s_q_idx = blockIdx.x;
    const int bidh = blockIdx.y;
    const int lane_idx = tidx % 64;
    const index_t row_offset_q = s_q_idx * static_cast<index_t>(params.stride_q_s_q) + bidh * kBlockM * params.stride_q_h_q;
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q) + row_offset_q),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                make_stride(params.stride_q_h_q, _1{}));

    const index_t row_offset_k = 0 * params.stride_kv_h_kv; 

    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.kv) + row_offset_k),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.stride_kv_s_kv, _1{}));
    const index_t row_offset_topk = s_q_idx * params.stride_indices_s_q;
    int* gIndices = reinterpret_cast<int *>(params.indices) + row_offset_topk;
    Tensor sQ = make_tensor(make_smem_ptr(plan.smem_q.data()), SmemLayoutQ{});    
    Tensor sV = make_tensor(make_smem_ptr(plan.smem_v.data()), SmemLayoutV{});
    Tensor sK = make_tensor(make_smem_ptr(plan.smem_v.data()), SmemLayoutK{}); 
    Tensor sP = make_tensor(make_smem_ptr(plan.smem_p.data()), SmemLayoutP{});    

    Tensor sVt = make_tensor(sV.data(), SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), SmemLayoutVtransposedNoSwizzle{});

    Tensor sRow_max_reduce_buffer = make_tensor(make_smem_ptr(plan.smem_row_max.data()), SmemLayoutRow{});  
    Tensor sRow_sum_reduce_buffer = make_tensor(make_smem_ptr(plan.smem_row_sum.data()), SmemLayoutRow{});    

    TiledMMA tiled_mma = TiledMma{}; 
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    TiledMMA tiled_mma_o = TiledMma_O{}; 
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);

    flash::lds_direct_copy<false, true, true>(gQ, sQ, 0, params.stride_q_h_q, params.h_q - bidh * kBlockM);                                                            
    flash::lds_direct_copy<false, true, true>(gQ, sQ, 1, params.stride_q_h_q, params.h_q - bidh * kBlockM);                                                            
    flash::lds_direct_copy<false, true, true>(gQ, sQ, 2, params.stride_q_h_q, params.h_q - bidh * kBlockM);                                                            
    flash::lds_direct_copy<false, true, true>(gQ, sQ, 3, params.stride_q_h_q, params.h_q - bidh * kBlockM); 
    if constexpr (D_QK == 576)
    {
        flash::lds_direct_copy<false, false, true>(gQ, sQ, 4, params.stride_q_h_q, params.h_q - bidh * kBlockM);                                                            
    }                                                           
    
    auto smem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);                                                               
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);                                                               
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);

    // asm volatile("s_waitcnt vmcnt(0) \n\t s_barrier\n\t");
    if constexpr (D_QK == 576)
    {
        asm volatile("s_waitcnt vmcnt(4) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 0), tSrQ_copy_view(_, _, 0));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 1), tSrQ_copy_view(_, _, 1));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 2), tSrQ_copy_view(_, _, 2));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 3), tSrQ_copy_view(_, _, 3));
        asm volatile("s_waitcnt vmcnt(3) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 4), tSrQ_copy_view(_, _, 4));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 5), tSrQ_copy_view(_, _, 5));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 6), tSrQ_copy_view(_, _, 6));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 7), tSrQ_copy_view(_, _, 7));                                                                
        asm volatile("s_waitcnt vmcnt(2) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 8), tSrQ_copy_view(_, _, 8));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 9), tSrQ_copy_view(_, _, 9));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 10), tSrQ_copy_view(_, _, 10));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 11), tSrQ_copy_view(_, _, 11));
        asm volatile("s_waitcnt vmcnt(1) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 12), tSrQ_copy_view(_, _, 12));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 13), tSrQ_copy_view(_, _, 13));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 14), tSrQ_copy_view(_, _, 14));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 15), tSrQ_copy_view(_, _, 15));                                                        
        asm volatile("s_waitcnt vmcnt(0) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 16), tSrQ_copy_view(_, _, 16));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 17), tSrQ_copy_view(_, _, 17));
    }
    else
    {
        asm volatile("s_waitcnt vmcnt(3) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 0), tSrQ_copy_view(_, _, 0));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 1), tSrQ_copy_view(_, _, 1));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 2), tSrQ_copy_view(_, _, 2));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 3), tSrQ_copy_view(_, _, 3));
        asm volatile("s_waitcnt vmcnt(2) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 4), tSrQ_copy_view(_, _, 4));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 5), tSrQ_copy_view(_, _, 5));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 6), tSrQ_copy_view(_, _, 6));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 7), tSrQ_copy_view(_, _, 7));                                                                
        asm volatile("s_waitcnt vmcnt(1) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 8), tSrQ_copy_view(_, _, 8));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 9), tSrQ_copy_view(_, _, 9));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 10), tSrQ_copy_view(_, _, 10));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 11), tSrQ_copy_view(_, _, 11));
        asm volatile("s_waitcnt vmcnt(0) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 12), tSrQ_copy_view(_, _, 12));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 13), tSrQ_copy_view(_, _, 13));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 14), tSrQ_copy_view(_, _, 14));
        cute::copy(smem_tiled_copy_Q, tSsQ(_, _, 15), tSrQ_copy_view(_, _, 15));                                                        
    }


    __syncthreads();

    const int topk_length = HAVE_TOPK_LENGTH ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int num_topk_blocks = HAVE_TOPK_LENGTH ? ku::ceil_div(topk_length, (int)B_TOPK) : (int)((unsigned int)params.topk/(unsigned int)B_TOPK);

    auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);
    Tensor tSrK  = thr_mma.partition_fragment_B(sK); 
    Tensor tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);
    Tensor tSrK_smem  = thr_mma.partition_fragment_B(gK); 

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_o);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
    Tensor tOrVt  = thr_mma_o.partition_fragment_B(sVtNoSwizzle);  
    Tensor tOrVt_copy_view = smem_thr_copy_V.retile_D(tOrVt);


    Tensor acc_o = partition_fragment_C(tiled_mma_o, Shape<Int<kBlockM>, Int<kHeadDimV>>{});
    clear(acc_o);
    flash::Softmax<size<1>(acc_o)> softmax;
    auto calc_row_and_col = [&](const int block_idx) -> std::tuple<int, int> {
        // 计算swizzle后的全局显存访存地址
        int virtual_row = lane_idx / 8;
        int virtual_col = lane_idx % 8;
        int swizzle_col = virtual_row ^ virtual_col;
        int row = lane_idx / 4;
        row = (row >= 8 ) ^ row;
        int col = swizzle_col % 4;
        int warp_id = tidx / 64;
        int row_offset = block_idx * kBlockN + row +  (warp_idx * 16) ;
        // row_offset = row_offset < params.topk ? gIndices[row_offset] : -1;
        row_offset = gIndices[row_offset];
        return {row_offset, col};
    };

    for (int block_idx = 0; block_idx < num_topk_blocks; block_idx++)
    {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s);
        auto [row_offset, col] = calc_row_and_col(block_idx);
        if constexpr (D_QK == 576)
        {
            for (int i = 16; i < 18; i++)
            {
                flash::lds_direct_copy_for_prefill_sparse_mla<true, false, false>(gK, sK, row_offset, col, i, params.stride_kv_s_kv, params.s_kv);
            }

            asm volatile("s_waitcnt vmcnt(1) \n s_barrier"); 
            cute::copy(smem_tiled_copy_K, tSsK(_, _, 0), tSrK_copy_view(_, _, 0));
            cute::gemm(tiled_mma, tSrQ(_, _, 0 + 16), tSrK(_, _, 0), acc_s);
            asm volatile("s_waitcnt vmcnt(0) \n s_barrier"); 
            flash::lds_direct_copy_for_prefill_sparse_mla<true, false, false>(gK, sK, row_offset, col, 0, params.stride_kv_s_kv, params.s_kv);
            cute::copy(smem_tiled_copy_K, tSsK(_, _, 1), tSrK_copy_view(_, _, 1));
            cute::gemm(tiled_mma, tSrQ(_, _, 1 + 16), tSrK(_, _, 1), acc_s);
        }
        else
        {
            flash::lds_direct_copy_for_prefill_sparse_mla<true, false, false>(gK, sK, row_offset, col, 0, params.stride_kv_s_kv, params.s_kv);
        }


        for (int i = 1; i < 4; i++) {
            flash::lds_direct_copy_for_prefill_sparse_mla<true, false, false>(gK, sK, row_offset, col, i, params.stride_kv_s_kv, params.s_kv);
        }

        int k_idx = 0;
        asm volatile("s_waitcnt vmcnt(3) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx % 4), tSrK_copy_view(_, _, k_idx));
        cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
        
        k_idx++;
        asm volatile("s_waitcnt vmcnt(2) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx % 4), tSrK_copy_view(_, _, k_idx));
        cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);

        k_idx++;
        asm volatile("s_waitcnt vmcnt(1) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx % 4), tSrK_copy_view(_, _, k_idx));
        cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
        
        k_idx++;
        asm volatile("s_waitcnt vmcnt(0) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx % 4), tSrK_copy_view(_, _, k_idx));
        flash::__ds_read_m32x16_row_col_rrow<0, 0, 0>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow<0, 1, 0>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow<0, 2, 0>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow<0, 3, 0>(tOsVt, tOrVt_copy_view);
        cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
        asm volatile("s_waitcnt lgkmcnt(0) \n\t s_barrier\n\t");
        flash::lds_direct_copy_for_prefill_sparse_mla<true, false, false>(gK, sK, row_offset, col, 4, params.stride_kv_s_kv, params.s_kv);
        flash::lds_direct_copy_for_prefill_sparse_mla<true, false, false>(gK, sK, row_offset, col, 5, params.stride_kv_s_kv, params.s_kv);
        flash::lds_direct_copy_for_prefill_sparse_mla<true, false, false>(gK, sK, row_offset, col, 6, params.stride_kv_s_kv, params.s_kv);
        flash::lds_direct_copy_for_prefill_sparse_mla<true, false, false>(gK, sK, row_offset, col, 7, params.stride_kv_s_kv, params.s_kv);

        k_idx++;
        asm volatile("s_waitcnt vmcnt(3) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx % 4), tSrK_copy_view(_, _, k_idx));
        cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
        
        k_idx++;
        asm volatile("s_waitcnt vmcnt(2) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx % 4), tSrK_copy_view(_, _, k_idx));
        cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);

        k_idx++;
        asm volatile("s_waitcnt vmcnt(1) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx % 4), tSrK_copy_view(_, _, k_idx));
        cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
        
        k_idx++;
        asm volatile("s_waitcnt vmcnt(0) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx % 4), tSrK_copy_view(_, _, k_idx));
        flash::__ds_read_m32x16_row_col_rrow<0, 0, 1>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow<0, 1, 1>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow<0, 2, 1>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow<0, 3, 1>(tOsVt, tOrVt_copy_view);
        cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
        asm volatile("s_waitcnt lgkmcnt(0) \n\t s_barrier\n\t");

        flash::lds_direct_copy_for_prefill_sparse_mla<true, false, false>(gK, sK, row_offset, col, 8, params.stride_kv_s_kv, params.s_kv);
        flash::lds_direct_copy_for_prefill_sparse_mla<true, false, false>(gK, sK, row_offset, col, 9, params.stride_kv_s_kv, params.s_kv);
        flash::lds_direct_copy_for_prefill_sparse_mla<true, false, false>(gK, sK, row_offset, col, 10, params.stride_kv_s_kv, params.s_kv);
        flash::lds_direct_copy_for_prefill_sparse_mla<true, false, false>(gK, sK, row_offset, col, 11, params.stride_kv_s_kv, params.s_kv);

        k_idx++;
        asm volatile("s_waitcnt vmcnt(3) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx % 4), tSrK_copy_view(_, _, k_idx));
        cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
        
        k_idx++;
        asm volatile("s_waitcnt vmcnt(2) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx % 4), tSrK_copy_view(_, _, k_idx));
        cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);

        k_idx++;
        asm volatile("s_waitcnt vmcnt(1) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx % 4), tSrK_copy_view(_, _, k_idx));
        cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
        
        k_idx++;
        asm volatile("s_waitcnt vmcnt(0) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx % 4), tSrK_copy_view(_, _, k_idx));
        flash::__ds_read_m32x16_row_col_rrow<0, 0, 2>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow<0, 1, 2>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow<0, 2, 2>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow<0, 3, 2>(tOsVt, tOrVt_copy_view);
        cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
        asm volatile("s_waitcnt lgkmcnt(0) \n\t s_barrier\n\t");

        flash::lds_direct_copy_for_prefill_sparse_mla<true, false, false>(gK, sK, row_offset, col, 12, params.stride_kv_s_kv, params.s_kv);
        flash::lds_direct_copy_for_prefill_sparse_mla<true, false, false>(gK, sK, row_offset, col, 13, params.stride_kv_s_kv, params.s_kv);
        flash::lds_direct_copy_for_prefill_sparse_mla<true, false, false>(gK, sK, row_offset, col, 14, params.stride_kv_s_kv, params.s_kv);
        flash::lds_direct_copy_for_prefill_sparse_mla<true, false, false>(gK, sK, row_offset, col, 15, params.stride_kv_s_kv, params.s_kv);

        k_idx++;
        asm volatile("s_waitcnt vmcnt(3) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx % 4), tSrK_copy_view(_, _, k_idx));
        cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
        
        k_idx++;
        asm volatile("s_waitcnt vmcnt(2) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx % 4), tSrK_copy_view(_, _, k_idx));
        cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);

        k_idx++;
        asm volatile("s_waitcnt vmcnt(1) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx % 4), tSrK_copy_view(_, _, k_idx));
        cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
        
        k_idx++;
        asm volatile("s_waitcnt vmcnt(0) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx % 4), tSrK_copy_view(_, _, k_idx));
        cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
        asm volatile("s_barrier\n\t");

        // if (block0())
        // {
        //     printf(" %.2f %.2f %.2f \n ", acc_s(0), acc_s(1), acc_s(2));
        // }
        Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor tScS = thr_mma.partition_C(cS);
        auto is_valid_token = [&](const int idx) -> bool {
            int offs = int(get<1>(tScS(idx))) + block_idx * kBlockN;
            int t = gIndices[offs];
            bool is_cur_token_valid = t >= 0 && t < params.s_kv;
            if constexpr (HAVE_TOPK_LENGTH) {
                is_cur_token_valid = is_cur_token_valid && (offs < topk_length);
            }
            return is_cur_token_valid;
        };

        {
            for (int i = 0; i < size(acc_s); ++i) {
                // idx = idx < params.topk ? gIndices[idx] : -1;
                if (!is_valid_token(i)) acc_s(i) = -INFINITY;
            }
        }

        block_idx == 0 ?
            softmax.template softmax_rescale_o_prefill</*Is_first=*/true, /*Check_inf=*//*Is_local=*/false>(acc_s, acc_o, sRow_max_reduce_buffer, params.sm_scale_div_log2):
            softmax.template softmax_rescale_o_prefill</*Is_first=*/false, /*Check_inf=*//*Is_local=*/false>(acc_s, acc_o, sRow_max_reduce_buffer, params.sm_scale_div_log2);
        // if (block0())
        // {
        //     printf(" %.2f %.2f %.2f %.2f %.2f %.2f \n ", acc_s(0), acc_s(1), acc_s(2), acc_s(3), softmax.row_max(0), params.sm_scale_div_log2);
        // }

        Tensor rP = flash::convert_type<Element>(acc_s);
        Tensor tOrP = flash::convert_layout_acc_Aregs(tiled_mma, tiled_mma_o, rP, sP);
        
        {

            flash::__ds_read_m32x16_row_col_rrow<0, 0, 3>(tOsVt, tOrVt_copy_view);
            
            
            
            // __ds_read_m32x16_row_col<0, 0>(tOsVt, tOrVt_copy_view);
            // __ds_read_m32x16_row_col<1, 0>(tOsVt, tOrVt_copy_view);
            // __ds_read_m32x16_row_col<2, 0>(tOsVt, tOrVt_copy_view);

            // __ds_read_m32x16_row_col<0, 1>(tOsVt, tOrVt_copy_view);
            // __ds_read_m32x16_row_col<1, 1>(tOsVt, tOrVt_copy_view);
            // __ds_read_m32x16_row_col<2, 1>(tOsVt, tOrVt_copy_view);
            cute::gemm(tiled_mma_o, tOrP(_, _, 0), tOrVt(_, _, 0), acc_o);
            flash::__ds_read_m32x16_row_col_rrow<0, 1, 3>(tOsVt, tOrVt_copy_view);
            cute::gemm(tiled_mma_o, tOrP(_, _, 1), tOrVt(_, _, 1), acc_o);
            // __ds_read_m32x16_row_col<0, 2>(tOsVt, tOrVt_copy_view);
            // __ds_read_m32x16_row_col<1, 2>(tOsVt, tOrVt_copy_view);
            // __ds_read_m32x16_row_col<2, 2>(tOsVt, tOrVt_copy_view);
            
            
            // __ds_read_m32x16_row_col<0, 3>(tOsVt, tOrVt_copy_view);
            // __ds_read_m32x16_row_col<1, 3>(tOsVt, tOrVt_copy_view);
            // __ds_read_m32x16_row_col<2, 3>(tOsVt, tOrVt_copy_view);
            
            flash::__ds_read_m32x16_row_col_rrow<0, 2, 3>(tOsVt, tOrVt_copy_view);
            cute::gemm(tiled_mma_o, tOrP(_, _, 2), tOrVt(_, _, 2), acc_o);
            flash::__ds_read_m32x16_row_col_rrow<0, 3, 3>(tOsVt, tOrVt_copy_view);
            cute::gemm(tiled_mma_o, tOrP(_, _, 3), tOrVt(_, _, 3), acc_o);

            // for (int i = 0; i < size(tOrP); i++)
            // {
            //     tOrP(i) = Element(1.0f);
            // }
            // cute::copy(smem_tiled_copy_V, tOsVt(_, 0, 0), tOrVt_copy_view(_, 0, 0));
            // for (int i = 0; i < 4; i++) {
            //     cute::copy(smem_tiled_copy_V, tOsVt(_, _, i), tOrVt_copy_view(_, _, i));
                
            //     // if (tOrVt(_, _, i) )

            //     cute::gemm(tiled_mma_o, tOrP(_, _, i), tOrVt(_, _, i), acc_o);
            // }

            // for (int i = 0; i < 8 * 2 * 16; i++)
            // {

            // }

            // asm volatile("s_barrier"); 
            
            // if (thread0()) {
            //     for (int i = 0; i < 64; i++) {
            //         for (int j = 0; j < 512; j++) {
            //             printf(" %.2f  ", float(sK(i, j)));
            //         }
            //         printf("\n");
            //     }
            // }

            //             if (block0())
            // {
            //     print("tidx %d acc_s %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n", 
            //         tidx,  acc_o(0), acc_o(1), acc_o(2), acc_o(3), 
            //         acc_o(4), acc_o(5), acc_o(6), acc_o(7),     
            //         acc_o(8), acc_o(9), acc_o(10), acc_o(11), 
            //         acc_o(12), acc_o(13), acc_o(14), acc_o(15)
            //     );
            // }
        }
        // asm volatile("s_barrier\n\t");
    }

    Tensor lse = softmax.template normalize_softmax_lse_prefill<false>(acc_o, sRow_sum_reduce_buffer, params.sm_scale);
    const index_t row_offset_o = s_q_idx * static_cast<index_t>(params.h_q * params.d_v) + bidh * kBlockM * params.d_v;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.out) + row_offset_o),
                                Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                make_stride(params.d_v, _1{}));
    // lse = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    const index_t row_offset_lse = s_q_idx * params.h_q + bidh * kBlockM;
    float* gLSE = reinterpret_cast<float *>(params.lse) + row_offset_lse;
    // const index_t row_offset_lse = m_block * params.h_q;
    float* gMax_logits = reinterpret_cast<float *>(params.max_logits) + row_offset_lse;
    
    if (params.attn_sink != nullptr) {
        float rAttn_sink = __ldg((float*)params.attn_sink + bidh * kBlockM + lane_idx % 16); 
        if (flash::is_positive_infinity(rAttn_sink))
        {
            for (int i = 0; i < size(acc_o); i++)
            {
                acc_o(i) = 0.0f;
            } 
        }
        else
        {
            if (!flash::is_positive_infinity(lse(0)))
            {
                float lse_exp2 = __builtin_amdgcn_exp2f(lse[0] * CUDART_L2E_F);
                float rAttn_sink_exp2 = __builtin_amdgcn_exp2f(rAttn_sink * CUDART_L2E_F);
                float o_scale = lse_exp2 / (lse_exp2 + rAttn_sink_exp2);
                for (int i = 0; i < size(acc_o); i++)
                {
                    acc_o(i) *= o_scale;
                }
            }
        }
    }
            //                 if (block0())
            // {
            //     print("tidx %d acc_s %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n", 
            //         tidx,  acc_o(0), acc_o(1), acc_o(2), acc_o(3), 
            //         acc_o(4), acc_o(5), acc_o(6), acc_o(7),     
            //         acc_o(8), acc_o(9), acc_o(10), acc_o(11), 
            //         acc_o(12), acc_o(13), acc_o(14), acc_o(15)
            //     );
            // }


    {
        // store O and gLSE
        auto rO = flash::convert_type<Element>(acc_o);
        int row, col;
        const int warpId = tidx / 64;
        const int laneId = tidx % 64;
        for (int mi = 0; mi < size<1>(acc_o); ++mi) {
            row = mi * kBlockM + laneId % 16;
            if (row < params.h_q) {
                for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                    col = (laneId / 16) + ni * 128 + warpId * 32 ;
                    for (int ei = 0; ei < size<0>(acc_o); ++ei) {
                        gO(row, col) = rO(ei, mi, ni);
                        col += 4;
                    }
                }
                gLSE[row] = lse(mi);
                gMax_logits[row] = topk_length == 0 ? -INFINITY : softmax.row_max(mi) * params.sm_scale;
            }
        }
    }


}

template<typename Kernel>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1)
sparse_attn_fwd_kernel(const SparseAttnFwdParams params) {
    Kernel::devfunc(params);
}

template<int D_QK, bool HAVE_TOPK_LENGTH>
void KernelTemplate<D_QK, HAVE_TOPK_LENGTH>::run(const SparseAttnFwdParams &params) {
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % (2*B_TOPK) == 0);   // To save some boundry checkings
    KU_ASSERT(params.topk > 0);
    KU_ASSERT(params.h_q % B_H == 0);
    auto kernel = &sparse_attn_fwd_kernel<KernelTemplate<D_QK, HAVE_TOPK_LENGTH>>;
    constexpr size_t smem_size = 65536;
    dim3 grid(params.s_q, params.h_q/B_H, 1);
    kernel<<<grid, NUM_THREADS, smem_size, params.stream>>>(params);
    KU_CHECK_KERNEL_LAUNCH();
}

template<int D_QK, bool HAVE_TOPK_LENGTH>
void run_fwd_phase1_kernel(const SparseAttnFwdParams& params) {
    KernelTemplate<D_QK, HAVE_TOPK_LENGTH>::run(params);
}

}
