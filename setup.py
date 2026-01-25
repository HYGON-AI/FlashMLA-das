import os
from pathlib import Path
from datetime import datetime
import subprocess

from setuptools import setup, find_packages

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    IS_WINDOWS,
    CUDA_HOME
)


def is_flag_set(flag: str) -> bool:
    return os.getenv(flag, "FALSE").lower() in ["true", "1", "y", "yes"]

def get_features_args():
    features_args = []
    if is_flag_set("FLASH_MLA_DISABLE_FP16"):
        features_args.append("-DFLASH_MLA_DISABLE_FP16")
    return features_args

def get_arch_flags():
    return ["--offload-arch=gfx938"]

def get_nvcc_thread_args():
    # nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return ["--threads", nvcc_threads]

# subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])

this_dir = os.path.dirname(os.path.abspath(__file__))

if IS_WINDOWS:
    cxx_args = ["/O2", "/std:c++20", "/DNDEBUG", "/W0"]
else:
    cxx_args = ["-O3", "-std=c++17", "-DNDEBUG", "-Wno-deprecated-declarations", "-DDCU_ASM", "-Wno-return-type", ]


ext_modules = []
ext_modules.append(
    CUDAExtension(
        name="flash_mla.cuda",
        sources=[
            # API
            "csrc/api/api.cpp",

            # # Misc kernels for decoding
            "csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu",
            "csrc/smxx/decode/combine/combine.cu",

            # # sm90 dense decode
            "csrc/sm90/decode/dense/instantiations/fp16.cu",
            "csrc/sm90/decode/dense/instantiations/bf16.cu",

            # # sm90 sparse decode
            "csrc/sm90/decode/sparse_fp8/instantiations/model1_persistent_h64.cu",
            "csrc/sm90/decode/sparse_fp8/instantiations/model1_persistent_h128.cu",
            "csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h64.cu",
            "csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h128.cu",

            # # sm90 sparse prefill
            "csrc/sm90/prefill/sparse/fwd.cu",
            "csrc/sm90/prefill/sparse/instantiations/phase1_k512.cu",
            "csrc/sm90/prefill/sparse/instantiations/phase1_k512_topklen.cu",
            "csrc/sm90/prefill/sparse/instantiations/phase1_k576.cu",
            "csrc/sm90/prefill/sparse/instantiations/phase1_k576_topklen.cu",

        ],
        extra_compile_args={
            "cxx": cxx_args + get_features_args(),
            "nvcc": [
                "-O3",
                "-std=c++17",
                "-DNDEBUG",
                "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
                "-ffast-math",
                "-ftemplate-backtrace-limit=0",
                "-Rpass-analysis=kernel-resource-usage",
                "-DDCU_ASM",
                "-w"
            ] + get_features_args() + get_arch_flags()
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "csrc" / "kerutils" / "include",   # TODO Remove me
            Path(this_dir) / "csrc" / "sm90",
            "/public/home/zhanghj/work/dev/cutlass_3.2.1-mla/include",
            # "/public/home/zhanghj/work/dev/cutlass_3.2.1-mla/tools/util/include",
        ],
    )
)



setup(
    name="flash_mla",
    version="1.0.0",
    packages=find_packages(include=['flash_mla']),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
