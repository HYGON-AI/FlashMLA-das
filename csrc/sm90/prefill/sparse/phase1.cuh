#pragma once

#include "config.h"

#include "utils.h"
#include "../../helpers.h"

namespace sm90::fwd {

using namespace cute;

template<int D_QK, bool HAVE_TOPK_LENGTH>
__device__ void KernelTemplate<D_QK, HAVE_TOPK_LENGTH>::devfunc(const SparseAttnFwdParams &params) {

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

    auto shape_Q = make_shape(params.h_q, params.d_qk, params.s_q);
    
}

template<int D_QK, bool HAVE_TOPK_LENGTH>
void run_fwd_phase1_kernel(const SparseAttnFwdParams& params) {
    KernelTemplate<D_QK, HAVE_TOPK_LENGTH>::run(params);
}

}
