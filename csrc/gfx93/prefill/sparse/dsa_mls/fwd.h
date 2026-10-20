#pragma once

#include "../../../../params.h"

namespace gfx93::fwd::dsa_mls {

bool can_run(const SparseAttnFwdParams& params);
bool should_run(const SparseAttnFwdParams& params);
void run(const SparseAttnFwdParams& params);

}  // namespace gfx93::fwd::dsa_mls
