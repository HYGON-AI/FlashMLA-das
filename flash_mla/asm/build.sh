#!/usr/bin/env bash
set -euo pipefail

target=amdgcn-amd-amdhsa
base=flash_fwd_mla_fp8_gfx938-hip-${target}-gfx938

clang -x assembler -target ${target} -mcode-object-version=4 -mcpu=gfx938:sramecc+ -c -o ${base}.o ${base}.s
clang -target ${target} ${base}.o -o ${base}.co
