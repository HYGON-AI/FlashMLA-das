#include <cutlass/cutlass.h>

#include "utils.h"

#include "params.h"
#include "config.h"
#include "traits.h"
#include "softmax.h"
using namespace cute;

namespace sm90 {

template<typename T>
__device__ void
compute_attn_1rowblock_splitkv_mla_gfx936(const DenseAttnDecodeParams params, 
                                        const int bidb, const int bidh, const int m_block,
                                        const int n_split_idx, const int seqlen_k,
                                        const int n_block_min, const int n_block_max, const bool NoSplit)
{
    extern __shared__ char shared_memory[];
    using SharedMemoryPlan = typename T::SharedMemoryPlan;
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(shared_memory);

    const int tidx = threadIdx.x;
    constexpr int kBlockM = T::BLOCK_SIZE_M;
    constexpr int kBlockN = T::PAGE_BLOCK_SIZE;
    constexpr int kHeadDim = T::HEAD_DIM_K;
    constexpr int kHeadDimV = T::HEAD_DIM_V;
    using Element = T::InputT;
    using index_t = int64_t;

    const index_t row_offset_q = bidb * params.q_batch_stride + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                make_stride(params.q_row_stride, _1{}));
    const index_t row_offset_k = (bidh) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.k_row_stride, _1{}));

    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                                    Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                                    make_stride(params.k_row_stride, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(plan.smem_q.data()), typename T::SmemLayoutQ{});    
    Tensor sV = make_tensor(make_smem_ptr(plan.smem_v.data()), typename T::SmemLayoutV{});

    Tensor sK = make_tensor(make_smem_ptr(plan.smem_v.data()), typename T::SmemLayoutK{}); 

    Tensor sP = make_tensor(make_smem_ptr(plan.smem_p.data()), typename T::SmemLayoutP{});    
    Tensor sVt = make_tensor(sV.data(), typename T::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename T::SmemLayoutVtransposedNoSwizzle{});

    Tensor sRow_max_reduce_buffer = make_tensor(make_smem_ptr(plan.smem_row_max.data()), typename T::SmemLayoutRow{});  
    Tensor sRow_sum_reduce_buffer = make_tensor(make_smem_ptr(plan.smem_row_sum.data()), typename T::SmemLayoutRow{});  
    typename T::TiledMma tiled_mma; 
    auto thr_mma = tiled_mma.get_thread_slice(tidx);


    typename T::TiledMma_O tiled_mma_o; 
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);

    #if 1
    typename T::GmemTiledCopyQ gmem_tiled_copy_Q;
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQgQ)));
    if (tidx < 128)
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, false>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, tQpQ, 
            params.q_seq_per_hk - m_block * kBlockM);
    __syncthreads();
    
    auto smem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);

    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    __syncthreads();    
    #else
    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);
    Tensor tSrQ = thr_mma.partition_fragment_A(gQ);
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true>(gmem_tiled_copy_Q, tSgQ, tSrQ, tQcQ, tQpQ, 
        params.q_seq_per_hk - m_block * kBlockM);
    __syncthreads();

    #endif
    
        // if (block0() && tidx < 64)
        // {
        //     printf(" %.3f %.3f \n", float(tSrQ(0)), float(tSrQ(1)));
        // }
    auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSgK = smem_thr_copy_K.partition_S(gK);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);
    Tensor tSrK  = thr_mma.partition_fragment_B(sK); 
    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_o);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
    Tensor tOrVt  = thr_mma_o.partition_fragment_B(sVtNoSwizzle);  

    constexpr int n_masking_steps = !T::Is_causal ? 1 : cute::ceil_div(kBlockM, kBlockN) + 1; 
    const int *block_table = params.block_table + bidb * params.block_table_batch_stride;
    int n_block = n_block_max - 1;

    Tensor acc_o = partition_fragment_C(tiled_mma_o, Shape<Int<kBlockM>, Int<kHeadDimV>>{});
    clear(acc_o);
    flash::Softmax<size<1>(acc_o)> softmax;
    Tensor tOrVt_copy_view = smem_thr_copy_V.retile_D(tOrVt);
    Tensor tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);

    // Tensor tOrVt_copy_view = smem_thr_copy_V.retile_D(tOrVt);
    // Tensor tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);
    // Tensor tKcK_smem = smem_thr_copy_K.partition_S(cK);
    Tensor tKpK_smem = make_tensor<bool>(make_shape(size<2>(tSgK)));
    Tensor tSrK_smem  = thr_mma.partition_fragment_B(gK); 
    constexpr static int k0_lds_loops = 15;     
    constexpr static int k0_loops = size<2>(tSrK_smem);
    constexpr static int k1_loops = size<2>(tOrVt);
    constexpr static int STAGE = 15;
    for (int masking_step = 0; masking_step < n_masking_steps && n_block >= n_block_min; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s);
        // asm volatile("s_barrier\n\t");
        // 这个也做过循环2类似的修改，但是性能不如现在的好，所以保持不变
        int cur_block_table;
        const int *cur_block_table_ptr = block_table + n_block;
        // cur_block_table = block_table[n_block - 1];
        asm volatile("s_load_dword %1, %0, 0x0\n\t"
                    "s_waitcnt lgkmcnt(0)\n\t":
                    "+s"(cur_block_table_ptr),
             "=s"(cur_block_table));
        index_t offset_k = cur_block_table * params.k_batch_stride;
        gK.data() = gK.data() + (offset_k);
       
        #pragma unroll
        for (int i = 0; i < STAGE; i++) {
            flash::lds_direct_copy<false, true>(gK, sK, i, params.k_row_stride, seqlen_k - n_block * kBlockN);
        }

        constexpr static int BUFFER_SIZE = 3;
        uint128_t buffer[BUFFER_SIZE];
        flash::buffer_load_copy<false, true, true, true>(gK, buffer[0], 15, params.k_row_stride, offset_k, seqlen_k - n_block * kBlockN);
        flash::buffer_load_copy<false, true, true, true>(gK, buffer[1], 16, params.k_row_stride, offset_k, seqlen_k - n_block * kBlockN);
        flash::buffer_load_copy<false, true, true, true>(gK, buffer[2], 17, params.k_row_stride, offset_k, seqlen_k - n_block * kBlockN);


        // if constexpr (STAGE == 15) 
        {
            int k_idx = 0;

            // k_idx++;
            asm volatile("s_waitcnt vmcnt(14 + 3) \n\t s_barrier\n\t");
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);

            k_idx++;
            asm volatile("s_waitcnt vmcnt(13 + 3) \n\t s_barrier\n\t");
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);

            k_idx++;
            asm volatile("s_waitcnt vmcnt(12 + 3) \n\t s_barrier\n\t");
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
            
            k_idx++;
            asm volatile("s_waitcnt vmcnt(11 + 3) \n\t s_barrier\n\t");
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);

            k_idx++;
            asm volatile("s_waitcnt vmcnt(10 + 3) \n\t s_barrier\n\t");
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);

            k_idx++;
            asm volatile("s_waitcnt vmcnt(9+ 3) \n\t s_barrier\n\t");
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);

            k_idx++;
            asm volatile("s_waitcnt vmcnt(8+ 3) \n\t s_barrier\n\t");
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);

            k_idx++;
            asm volatile("s_waitcnt vmcnt(7+ 3) \n\t s_barrier\n\t");
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
            // asm volatile("s_barrier\n\t");

            asm volatile("s_waitcnt vmcnt(6+ 3) \n\t s_barrier\n\t");
            k_idx++;
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
            // asm volatile("s_barrier\n\t");

            asm volatile("s_waitcnt vmcnt(5 + 3) \n\t s_barrier\n\t");
            k_idx++;
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
            // asm volatile("s_barrier\n\t");

            asm volatile("s_waitcnt vmcnt(4 + 3) \n\t s_barrier\n\t");
            k_idx++;
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);

            k_idx++;
            asm volatile("s_waitcnt vmcnt(3 + 3) \n\t s_barrier\n\t");
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);

            asm volatile("s_waitcnt vmcnt(2 + 3) \n\t s_barrier\n\t");
            k_idx++;
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
            // asm volatile("s_barrier\n\t");

            asm volatile("s_waitcnt vmcnt(1 + 3) \n\t s_barrier\n\t");
            k_idx++;
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
            // asm volatile("s_barrier\n\t");

            asm volatile("s_waitcnt vmcnt(0 + 3) \n\t s_barrier\n\t");
            k_idx++;
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
            // asm volatile("s_barrier\n\t");

        }
        
        
        asm volatile("s_waitcnt vmcnt(2) \n\t \n\t");
        flash::buffer_to_tensor(buffer[0], tSrK_smem, 15);
        cute::gemm(tiled_mma, tSrQ(_, _, 15), tSrK_smem(_, _, 15), acc_s);
        asm volatile("s_waitcnt vmcnt(1) \n\t \n\t");
        flash::buffer_to_tensor(buffer[1], tSrK_smem, 16);
        cute::gemm(tiled_mma, tSrQ(_, _, 16), tSrK_smem(_, _, 16), acc_s);
        asm volatile("s_waitcnt vmcnt(0) \n\t \n\t");
        flash::buffer_to_tensor(buffer[2], tSrK_smem, 17);
        cute::gemm(tiled_mma, tSrQ(_, _, 17), tSrK_smem(_, _, 17), acc_s);
        // asm volatile("s_barrier\n\t");
        // if (block0() && tidx < 64)
        // {
        //     printf(" %.3f %.3f \n", acc_s(0), acc_s(1));
        // }
        Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor tScS = thr_mma.partition_C(cS);
        for (int i = 0; i < size(acc_s); ++i) {
            if constexpr (!T::Is_causal) {
                if (int(get<1>(tScS(i))) >= int(seqlen_k - n_block * kBlockN)) acc_s(i) = -INFINITY;
            } else {
                // Ensure seqlen_k - 1 - (n_block * kBlockN + col) >= (seqlen_q - 1 - (m_block * kBlockM + row)) / ngroups
                // col <= seqlen_k - 1 - n_block * kBlockN - (seqlen_q - 1 - (m_block * kBlockM + row)) / ngroups
                int row = int(get<0>(tScS(i)));
                int col_limit_right = seqlen_k - 1 - n_block * kBlockN - (params.q_seq_per_hk - 1 - (m_block * kBlockM + row)) / params.q_head_per_hk;
                if (int(get<1>(tScS(i))) > col_limit_right) acc_s(i) = -INFINITY;
            }
        }

       
        // We have key_padding_mask so we'll need to Check_inf
        if constexpr (n_masking_steps == 1) {
            softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/T::Is_causal>(acc_s, acc_o, sRow_max_reduce_buffer, params.scale_softmax_log2);
        }
        else {
            const bool is_first_masking_step = masking_step == 0;
            is_first_masking_step
                ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/T::Is_causal>(acc_s, acc_o, sRow_max_reduce_buffer, params.scale_softmax_log2)
                :   softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/T::Is_causal>(acc_s, acc_o, sRow_max_reduce_buffer, params.scale_softmax_log2);

        }


        Tensor rP = flash::convert_type<Element>(acc_s);

        // Tensor tOrP = convert_layout_acc_Aregs(tiled_mma_o, rP, sP);
        Tensor tOrP = flash::convert_layout_acc_Aregs_dense(tiled_mma, tiled_mma_o, rP, sP);
        __syncthreads();


        flash::lds_direct_copy<false, true>(gK, sK, 15, params.k_row_stride, seqlen_k - n_block * kBlockN);
        // asm_ds_write(buffer[0], tVsV, 15);
        // asm volatile("s_waitcnt lgkmcnt(0) \n\t s_barrier\n\t");
        asm volatile("s_waitcnt vmcnt(0) \n\t s_barrier\n\t");

        gK.data() = gK.data() + (-offset_k);


        
        #pragma unroll 
        for (int i = 0; i < k1_loops; i++) {
            cute::copy(smem_tiled_copy_V, tOsVt(_, _, i), tOrVt_copy_view(_, _, i));
            cute::gemm(tiled_mma_o, tOrP(_, _, i), tOrVt(_, _, i), acc_o);
        }
        // asm volatile("s_barrier\n\t");

    }

    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s);
        // asm volatile("s_barrier\n\t");
        int cur_block_table;
        const int *cur_block_table_ptr = block_table + n_block;
        // cur_block_table = block_table[n_block - 1];
        asm volatile("s_load_dword %1, %0, 0x0\n\t"
                    "s_waitcnt lgkmcnt(0)\n\t":
                    "+s"(cur_block_table_ptr),
             "=s"(cur_block_table));
        index_t offset_k = cur_block_table * params.k_batch_stride;
    
        gK.data() = gK.data() + (offset_k);
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            flash::lds_direct_copy<true, true>(gK, sK, i, params.k_row_stride, seqlen_k - n_block * kBlockN);
        }
        constexpr static int BUFFER_SIZE = 2;
        uint128_t buffer[BUFFER_SIZE];
        // buffer_load_copy<true, true, true, true>(gK, buffer[0], 15, params.k_row_stride, offset_k, seqlen_k - n_block * kBlockN);
        flash::buffer_load_copy<true, true, true, true>(gK, buffer[0], 16, params.k_row_stride, offset_k, seqlen_k - n_block * kBlockN);
        flash::buffer_load_copy<true, true, true, true>(gK, buffer[1], 17, params.k_row_stride, offset_k, seqlen_k - n_block * kBlockN);


        // if constexpr (STAGE == 15)
        {
            int k_idx = 0;

            // k_idx++;
            asm volatile("s_waitcnt vmcnt(14 + 3) \n\t s_barrier\n\t");
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);

            k_idx++;
            asm volatile("s_waitcnt vmcnt(13 + 3) \n\t s_barrier\n\t");
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);

            k_idx++;
            asm volatile("s_waitcnt vmcnt(12 + 3) \n\t s_barrier\n\t");
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
            
            k_idx++;
            asm volatile("s_waitcnt vmcnt(11 + 3) \n\t s_barrier\n\t");
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);

            k_idx++;
            asm volatile("s_waitcnt vmcnt(10 + 3) \n\t s_barrier\n\t");
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);

            k_idx++;
            asm volatile("s_waitcnt vmcnt(9+ 3) \n\t s_barrier\n\t");
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);

            k_idx++;
            asm volatile("s_waitcnt vmcnt(8+ 3) \n\t s_barrier\n\t");
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);

            k_idx++;
            asm volatile("s_waitcnt vmcnt(7+ 3) \n\t s_barrier\n\t");
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
            // asm volatile("s_barrier\n\t");

            asm volatile("s_waitcnt vmcnt(6+ 3) \n\t s_barrier\n\t");
            k_idx++;
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
            // asm volatile("s_barrier\n\t");

            asm volatile("s_waitcnt vmcnt(5 + 3) \n\t s_barrier\n\t");
            k_idx++;
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
            // asm volatile("s_barrier\n\t");

            asm volatile("s_waitcnt vmcnt(4 + 3) \n\t s_barrier\n\t");
            k_idx++;
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);

            k_idx++;
            asm volatile("s_waitcnt vmcnt(3 + 3) \n\t s_barrier\n\t");
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);

            asm volatile("s_waitcnt vmcnt(2 + 3) \n\t s_barrier\n\t");
            k_idx++;
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
            // asm volatile("s_barrier\n\t");

            asm volatile("s_waitcnt vmcnt(1 + 3) \n\t s_barrier\n\t");
            k_idx++;
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
            // asm volatile("s_barrier\n\t");

            asm volatile("s_waitcnt vmcnt(0 + 3) \n\t s_barrier\n\t");
            k_idx++;
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
            asm volatile("s_waitcnt vmcnt(0 + 2) \n\t s_barrier\n\t");
            k_idx++;
            cute::copy(smem_tiled_copy_K, tSsK(_, _, k_idx), tSrK_copy_view(_, _, k_idx));
            __builtin_amdgcn_sched_barrier(0);
            flash::__ds_read_m32x16_row_col<3, 0>(tOsVt, tOrVt_copy_view);
            flash::__ds_read_m32x16_row_col<3, 1>(tOsVt, tOrVt_copy_view);
            flash::__ds_read_m32x16_row_col<3, 2>(tOsVt, tOrVt_copy_view);
            flash::__ds_read_m32x16_row_col<3, 3>(tOsVt, tOrVt_copy_view);
            __builtin_amdgcn_sched_barrier(0);
            cute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
            // asm volatile("s_barrier\n\t");
        }

        asm volatile("s_waitcnt vmcnt(1) \n\t \n\t");
        flash::buffer_to_tensor(buffer[0], tSrK_smem, 16);
        cute::gemm(tiled_mma, tSrQ(_, _, 16), tSrK_smem(_, _, 16), acc_s);
        asm volatile("s_waitcnt vmcnt(0) \n\t \n\t");
        flash::buffer_to_tensor(buffer[1], tSrK_smem, 17);
        cute::gemm(tiled_mma, tSrQ(_, _, 17), tSrK_smem(_, _, 17), acc_s);
        // asm volatile("s_barrier\n\t");
        asm volatile("s_waitcnt lgkmcnt(0) \n\t s_barrier\n\t");

        gK.data() = gK.data() + (-offset_k);
        // We have key_padding_mask so we'll need to Check_inf
        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*//*Is_local=*/false>(acc_s, acc_o, sRow_max_reduce_buffer, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        // Tensor tOrP = convert_layout_acc_Aregs(tiled_mma_o, rP, sP);
        Tensor tOrP = flash::convert_layout_acc_Aregs_dense(tiled_mma, tiled_mma_o, rP, sP);
        flash::__ds_read_m32x16_row_col<0, 0>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col<1, 0>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col<2, 0>(tOsVt, tOrVt_copy_view);

        flash::__ds_read_m32x16_row_col<0, 1>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col<1, 1>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col<2, 1>(tOsVt, tOrVt_copy_view);
        cute::gemm(tiled_mma_o, tOrP(_, _, 0), tOrVt(_, _, 0), acc_o);
        cute::gemm(tiled_mma_o, tOrP(_, _, 1), tOrVt(_, _, 1), acc_o);
        flash::__ds_read_m32x16_row_col<0, 2>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col<1, 2>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col<2, 2>(tOsVt, tOrVt_copy_view);
        
        
        flash::__ds_read_m32x16_row_col<0, 3>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col<1, 3>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col<2, 3>(tOsVt, tOrVt_copy_view);
        
        
        cute::gemm(tiled_mma_o, tOrP(_, _, 2), tOrVt(_, _, 2), acc_o);
        cute::gemm(tiled_mma_o, tOrP(_, _, 3), tOrVt(_, _, 3), acc_o);

        // asm volatile("s_barrier\n\t");
    }
    using ElementAccum = float;
    if (NoSplit)
    {
        using ElementO = Element;
        const index_t row_offset_o = bidb * params.o_batch_stride + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_lse = (bidb * params.h_k + bidh) * params.q_seq_per_hk  + m_block * kBlockM;

        constexpr bool Split = false;
        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + ( row_offset_o)),
                                    Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                    make_stride(Split ? kHeadDimV : params.o_row_stride, _1{}));

        Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(acc_o, sRow_sum_reduce_buffer, params.scale_softmax);

        // if (block0() && tidx < 64)
        // {
        //     printf(" %.3f %.3f \n", float(acc_o(0)), float(acc_o(1)));
        // }
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + (row_offset_lse)),
                                   Shape<Int<kBlockM>>{}, Stride<_1>{});
           
        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma_o.partition_C(caccO);  
        Tensor rO = flash::convert_type<ElementO>(acc_o);      
        if (get<1>(taccOcO(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO(0, mi, 0));
                if (row < params.q_seq_per_hk - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
            }
        }

        {
            // using result_type = cutlass::Array<bfloat16_t, 2>;
            // int tidx = threadIdx.x;
            int col = 0;
            int warpid = tidx / 64;
            for (int m = 0; m < 1; m++) {
                const int row = tidx % 16;
                if (row < params.q_seq_per_hk - m_block * kBlockM) {
                    for (int n = 0; n < size<2>(acc_o); n++) {
                        col = (tidx % 64 / 16) + warpid * 32 + n * 128;
                        for (int ei = 0; ei < 8; ei ++) {
                            gOaccum(row, col) = rO(ei, m, n);
                            col += 4;
                        }   
                    }
                }
            }
        }
    }
    else
    {
        using ElementO = float;
        int split_idx = params.num_splits_ptr[bidb] + n_split_idx;        
        constexpr bool Split = true;
        const index_t row_offset_oaccum =  ((split_idx*params.h_k + bidh)*params.q_seq_per_hk + m_block * kBlockM)*T::HEAD_DIM_V;	// (BLOCK_SIZE_M, HEAD_DIM_V) : (HEAD_DIM_V, 1)
        const index_t row_offset_lseaccum = (split_idx*params.h_k + bidh)*params.q_seq_per_hk +  m_block * kBlockM;
        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (row_offset_oaccum)),
                                    Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                    make_stride(Split ? kHeadDimV : params.o_row_stride, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + (row_offset_lseaccum)),
                                    Shape<Int<kBlockM>>{}, Stride<_1>{});
        
        Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(acc_o, sRow_sum_reduce_buffer, params.scale_softmax);

           
        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma_o.partition_C(caccO);  

        if (get<1>(taccOcO(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO(0, mi, 0));
                if (row < params.q_seq_per_hk - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
            }
        }
        {
            // using result_type = cutlass::Array<bfloat16_t, 2>;
            // int tidx = threadIdx.x;
            int col = 0;
            int warpid = tidx / 64;
            for (int m = 0; m < 1; m++) {
                const int row = tidx % 16;
                if (row < params.q_seq_per_hk - m_block * kBlockM) {
                    for (int n = 0; n < size<2>(acc_o); n++) {
                        col = (tidx % 64 / 16) + warpid * 32 + n * 128;
                        for (int ei = 0; ei < 8; ei ++) {
                            gOaccum(row, col) = acc_o(ei, m, n);
                            col += 4;
                        }   
                    }
                }
            }
        }
    }

}

template<typename T>
__global__ void __launch_bounds__(T::NUM_THREADS, 1)
flash_fwd_splitkv_mla_kernel(const DenseAttnDecodeParams params) {
    const int m_block = blockIdx.x;
    const int bidh = blockIdx.y;
    const int partition_idx = blockIdx.z;

    DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
    if (sched_meta.begin_req_idx >= params.b) return;
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
        constexpr int kBlockN = T::PAGE_BLOCK_SIZE;
        const int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
        int seqlen_k = __ldg(params.seqlens_k_ptr + batch_idx);
        const int start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
        int end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : cute::ceil_div(seqlen_k, kBlockN);
        const bool is_no_split = batch_idx == sched_meta.begin_req_idx ? !sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);
        
        if (batch_idx > sched_meta.begin_req_idx) {
            __syncthreads(); 
        }
        compute_attn_1rowblock_splitkv_mla_gfx936<T>(params, batch_idx, bidh, m_block, n_split_idx, 
            seqlen_k, start_block_idx, end_block_idx, is_no_split
        );

    }
}


template<typename InputT>
void run_flash_splitkv_mla_kernel(DenseAttnDecodeParams &params) {
    FLASH_ASSERT(params.d == Config::HEAD_DIM_K);
    FLASH_ASSERT(params.d_v == Config::HEAD_DIM_V);

    
 
    constexpr size_t smem_size = 65536;

    // Use cudaLaunchKernelEx to enable PDL (Programmatic Dependent Launch)

    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        using T = Traits<InputT, Is_causal>;
        const int num_m_block = cute::ceil_div(params.q_seq_per_hk, T::BLOCK_SIZE_M);
        auto mla_kernel = &flash_fwd_splitkv_mla_kernel<T>;
        mla_kernel<<<dim3(num_m_block, params.h_k, params.num_sm_parts), T::NUM_THREADS, smem_size, params.stream>>>(params);

    });
    // cudaLaunchConfig_t mla_kernel_config = {
    //     dim3(num_m_block, params.h_k, params.num_sm_parts),
    //     dim3(T::NUM_THREADS, 1, 1),
    //     smem_size,
    //     params.stream,
    //     mla_kernel_attributes,
    //     1
    // };
    // cudaLaunchKernelEx(&mla_kernel_config, mla_kernel, params, tma_params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}
