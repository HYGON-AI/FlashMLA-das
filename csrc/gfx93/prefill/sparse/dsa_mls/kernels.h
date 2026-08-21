#pragma once

// DSA MLS kernel entry points still live in the imported legacy MLA kernel
// implementation. Keep this wrapper as the single include boundary while the
// implementation is being split into the gfx93-style module layout.
#include "legacy/src/flash_fwd_b16_mla.h"
