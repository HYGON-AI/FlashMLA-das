#include "fwd.h"

#include <stdexcept>

#include "dsa_mls/fwd.h"
#include "phase1.h"

namespace gfx93 {

void run_fwd_kernel(const SparseAttnFwdParams& params) {
    if (gfx93::fwd::dsa_mls::should_run(params)) {
        gfx93::fwd::dsa_mls::run(params);
        return;
    }

    const bool have_topk_length = params.topk_length != nullptr;

    // Dispatch based on d_qk dimension and presence of topk_length
    if (params.d_qk == 512) {
        if (have_topk_length) {
            gfx93::fwd::run_fwd_phase1_kernel<512, true>(params);
        } else {
            gfx93::fwd::run_fwd_phase1_kernel<512, false>(params);
        }
    } else if (params.d_qk == 576) {
        if (have_topk_length) {
            gfx93::fwd::run_fwd_phase1_kernel<576, true>(params);
        } else {
            gfx93::fwd::run_fwd_phase1_kernel<576, false>(params);
        }
    } else {
        throw std::runtime_error("Unsupported d_qk value in sparse attention fwd kernel");
    }
}

}  // namespace gfx93
