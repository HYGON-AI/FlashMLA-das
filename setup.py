import os
from pathlib import Path
from datetime import datetime
import subprocess
from typing import Optional
from get_version import get_version

from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    IS_WINDOWS,
    CUDA_HOME
)


def is_flag_set(flag: str) -> bool:
    return os.getenv(flag, "FALSE").lower() in ["true", "1", "y", "yes"]

def get_features_args():
    bf16_type = os.getenv("FLASH_MLA_BF16_TYPE", "0")
    assert bf16_type == "0" or bf16_type == "1", "bf16_type must be 0 or 1"
    bf16_mode_names = {"0": "round_toward_zero", "1": "round_half_ulp_truncate"}
    print(f"Using BFloat16 rounding mode: {bf16_mode_names.get(bf16_type, 'unknown')}")
    features_args = []
    if is_flag_set("FLASH_MLA_DISABLE_FP16"):
        features_args.append("-DFLASH_MLA_DISABLE_FP16")
    features_args.append(f"-DFLASH_MLA_BF16_TYPE={bf16_type}")
    return features_args

def get_arch_flags():
    arch_flags = []
    arch_flags.append("--offload-arch=gfx938;gfx936")
    return arch_flags

# def get_nvcc_thread_args():
#     # nvcc_threads = os.getenv("NVCC_THREADS") or "32"
#     return ["--threads", nvcc_threads]

# subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])

this_dir = os.path.dirname(os.path.abspath(__file__))

if False:
    cxx_args = ["/O2", "/std:c++20", "/DNDEBUG", "/W0"]
else:
    cxx_args = ["-O3", "-std=c++20", "-DNDEBUG", "-Wno-deprecated-declarations", "-DDCU_ASM", "-Wno-return-type", ]


ext_modules = []
ext_modules.append(
    CUDAExtension(
        name="flash_mla.cuda",
        sources=[
            # API
            "csrc/api/api.cpp",

            # # Misc kernels for decoding
            "csrc/gfx9/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu",
            "csrc/gfx9/decode/combine/combine.cu",

            # # gfx93 dense decode
            "csrc/gfx93/decode/dense/instantiations/fp16.cu",
            "csrc/gfx93/decode/dense/instantiations/bf16.cu",

            ## gfx93 dense qkvfp8 decode
            "csrc/gfx93/decode/dense_qkvfp8/instantiations/fp8e4m3.cu",

            ## gfx93 dense kvfp8 decode
            "csrc/gfx93/decode/dense_kvfp8/instantiations/kvfp8.cu",
            
            # # gfx93 sparse decode
            "csrc/gfx93/decode/sparse_fp8/instantiations/model1_persistent_h16.cu",
            "csrc/gfx93/decode/sparse_fp8/instantiations/model1_persistent_h64.cu",
            "csrc/gfx93/decode/sparse_fp8/instantiations/model1_persistent_h128.cu",
            "csrc/gfx93/decode/sparse_fp8/instantiations/v32_persistent_h16.cu",
            "csrc/gfx93/decode/sparse_fp8/instantiations/v32_persistent_h64.cu",
            "csrc/gfx93/decode/sparse_fp8/instantiations/v32_persistent_h128.cu",

            # # gfx93 sparse prefill
            "csrc/gfx93/prefill/sparse/fwd.cu",
            "csrc/gfx93/prefill/sparse/instantiations/phase1_k512.cu",
            "csrc/gfx93/prefill/sparse/instantiations/phase1_k512_topklen.cu",
            "csrc/gfx93/prefill/sparse/instantiations/phase1_k576.cu",
            "csrc/gfx93/prefill/sparse/instantiations/phase1_k576_topklen.cu",

            "csrc/extension/flash_fwd_mla_bf16_gfx936.cu",
            "csrc/extension/flash_fwd_mla_fp16_gfx936.cu",
            "csrc/extension/flash_fwd_mla_fp8_gfx938.cu",
            "csrc/extension/flash_fwd_mla_fp8_qbf16_gfx938.cu",
            "csrc/extension/flash_fwd_mla_metadata.cu",

        ],
        extra_compile_args={
            "cxx": cxx_args + get_features_args(),
            "nvcc": [
                "-O3",
                "-std=c++20",
                "-DNDEBUG",
                "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
                "-ffast-math",
                "-ftemplate-backtrace-limit=0",
                "-Rpass-analysis=kernel-resource-usage",
                "-DDCU_ASM",
                "--save-temps",
                "-w",
                "-mllvm -enable-num-vgprs-512=true",
                "-mllvm -allow-cse-cross-bb-convergent-call=true",
                "-mllvm -full-vectorize-slp=true",
            ] + get_features_args() + get_arch_flags()
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "csrc" / "kerutils" / "include",   # TODO Remove me
            Path(this_dir) / "csrc" / "gfx93",
            Path(this_dir) / "csrc" / "cutlass" / "include",
        ],
    )
)

def _find_rocm_home() -> Optional[str]:
    rocm_home = os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH')
    if rocm_home is None:
        try:
            pipe_hipcc = subprocess.Popen(
                ["which hipcc | xargs readlink -f"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            hipcc, _ = pipe_hipcc.communicate()
            rocm_home = os.path.dirname(os.path.dirname(hipcc.decode(*()).rstrip('\r\n')))
            if os.path.basename(rocm_home) == 'hip':
                rocm_home = os.path.dirname(rocm_home)
        except Exception:
            rocm_home = '/opt/rocm'
            if not os.path.exists(rocm_home):
                rocm_home = None
    if rocm_home and torch.version.hip is None:
        print(f"No ROCm runtime is found, using ROCM_HOME='{rocm_home}'")
    return rocm_home
ROCM_HOME = _find_rocm_home()

pytorch_dep = 'torch'
if os.getenv('PYTORCH_VERSION'):
    pytorch_dep += "==" + os.getenv('PYTORCH_VERSION')

setup(
    name="flash_mla",   
    version=get_version(ROCM_HOME),
    packages=find_packages(include=['flash_mla']),
    ext_modules=ext_modules,
    package_data={"flash_mla":["asm/*.co"]},
    cmdclass={"build_ext": BuildExtension},
    install_requires=[pytorch_dep],
)

