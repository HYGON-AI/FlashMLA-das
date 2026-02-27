clang -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=gfx938:sramecc+ -c -o flash_fwd_mla_fp8_gfx938-hip-amdgcn-amd-amdhsa-gfx938.o flash_fwd_mla_fp8_gfx938-hip-amdgcn-amd-amdhsa-gfx938.s 
clang -target amdgcn-amd-amdhsa  flash_fwd_mla_fp8_gfx938-hip-amdgcn-amd-amdhsa-gfx938.o -o flash_fwd_mla_fp8_gfx938-hip-amdgcn-amd-amdhsa-gfx938.co
