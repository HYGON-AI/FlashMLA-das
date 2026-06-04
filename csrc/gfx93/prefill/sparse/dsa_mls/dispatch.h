#pragma once

#include <algorithm>

#include <hip/hip_runtime.h>

#include "legacy/include/flash.h"
#include "legacy/include/kernel_traits.h"
#include "legacy/include/static_switch.h"
#include "legacy/src/flash_fwd_b16_mla.h"

namespace gfx93::fwd::dsa_mls {

template<typename T, int Headdim, int HeaddimV>
void run_dsa_prefill_nopage_64_dispatch(Flash_fwd_mla_params_dsa& params, hipStream_t stream) {
    constexpr int kBlockM = 64;
    constexpr int kBlockN = 64;
    constexpr int WARP_M = 16;

    dim3 dimBlock;
    dimBlock.x = std::min((kBlockM / WARP_M) * 64, 1024);
    dimBlock.y = 1;
    dimBlock.z = 1;

    dim3 dimGrid;
    dimGrid.x = (params.seqlen_q + kBlockM - 1) / kBlockM;
    dimGrid.y = 1;
    dimGrid.z = params.b;

    using Kernel_traits = Flash_fwd_kernel_traits<
        Headdim, HeaddimV, kBlockM, kBlockN, 32, WARP_M, 64, 2,
        false, false, T, T>;

    constexpr bool Is_dropout = false;
    constexpr bool IsEvenMNConst = false;

    BOOL_SWITCH(params.mtp > 1, Is_MTP, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            if (params.topk == 2048) {
                flash::flash_fwd_mla_decode_kernel_gfx938_dsa_prefill_nopage_64<
                    Kernel_traits, true, Is_dropout, false, Is_causal,
                    IsEvenMNConst, true, false, Is_MTP, 0, Flash_fwd_mla_params_dsa>
                    <<<dimGrid, dimBlock, 21 * 1024, stream>>>(params);
            } else {
                flash::flash_fwd_mla_decode_kernel_gfx938_dsa_prefill_nopage_64_topk1024<
                    Kernel_traits, true, Is_dropout, false, Is_causal,
                    IsEvenMNConst, true, false, Is_MTP, 0, Flash_fwd_mla_params_dsa>
                    <<<dimGrid, dimBlock, 21 * 1024, stream>>>(params);
            }
        });
    });
}

}  // namespace gfx93::fwd::dsa_mls
