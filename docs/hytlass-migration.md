# HYTLASS Release Migration

FlashMLA uses the unmodified HYTLASS `release` commit
`fa662ebf2d8aba6a8881603a448102d1ed8d9950` in `csrc/hytlass`.
The previous customized CUTLASS submodule is no longer used by the build.
The migration follows the integration pattern supplied by the dependency
maintainer in FlashAttention MR 196 (source commit `b59b7dee`): use release
headers and keep application-specific register layouts in application code.
No FlashAttention source file is imported by this change.

## Integration

- Headers and namespaces change from CUTLASS/CuTe to HYTLASS/HuTe.
- The old fork-specific build define is removed. The local operation
  implementations use the HIP compiler's native conditions.
- All 35 active `TiledMMA` instantiations used unit value tiling in the previous
  fork. They now use the release API's two-argument form; a unit value layout
  must not be passed as the new API's permutation argument.
- `csrc/hytlass_ext.h` defines 21 application-specific operation tags and their
  original MMA/copy layouts. Seven operations inherit the release implementation.
  Fourteen retain application-specific definitions, including layout-only tags
  and two shared-memory copy interfaces with different register types from the
  release. The extension does not override traits for HYTLASS's own tags.
- FP16/BF16 composite operations use the release's native MMAC helpers. FP8 LIT
  operations retain the original instruction flags and packing. The shared-memory
  copies use the release compiler's HCU builtin names with the original register ABI.

## Retained Source

The local extension is derived from customized CUTLASS commit
`33e45645599f371c2382e67920085498f5905955`, with its NVIDIA notices and
BSD-3-Clause license retained. The source files are:

| Original path under `include/cute/` | Retained contribution |
| --- | --- |
| `arch/mma_gfx928.hpp` | Composite operations and layout tags |
| `arch/mma_gfx938.hpp` | FP8 LIT operation and layout tags |
| `arch/copy_gfx928.hpp` | Shared-memory copy register interfaces |
| `atom/mma_traits_gfx928.hpp` | FP16/BF16 and mixed-value MMA layouts |
| `atom/mma_traits_gfx938.hpp` | FP8 MMA layouts |
| `atom/copy_traits_gfx928.hpp` | Shared-memory copy layouts |

Only the referenced operations and traits are retained, rather than a second
copy of the dependency. Existing inherited copyright and root licenses remain
in place. See [Third-Party Notices](../THIRD_PARTY_NOTICES.md).

## Validation

The migration was prepared on Windows without an HCU compiler or device.
Source checks cover the pinned submodule, header resolution, original register
layouts, release inheritance ABI, template argument changes, and unchanged
assembly artifacts. They do not replace the following HCU checks.

Compile the register ABI checks with the same compiler used for the extension:

```bash
for arch in gfx928 gfx936 gfx938; do
  aicc -std=c++20 --offload-arch="$arch" -fsyntax-only \
    -Icsrc -Icsrc/hytlass/include tests/test_hytlass_layouts.cu || exit 1
done
```

The check instantiates every local MMA tag against the release API and verifies
its shape, wave size, and register payload sizes. It also checks the copy layouts'
source and destination register sizes.

Then build and run the existing correctness tests on supported HCU hardware:

```bash
git submodule sync --recursive
git submodule update --init --recursive
python setup.py build_ext --inplace
python tests/test_flash_mla_dense_decoding.py
python tests/test_flash_mla_sparse_decoding.py
FLASH_MLA_DECODE_BF16=1 python tests/test_flash_mla_sparse_decoding.py
python tests/test_flash_mla_sparse_prefill.py
```

Validate the BF16/FP16, FP8 KV, FP8 QKV, and DSA paths and compare performance
with the previous build before integrating the migration into a release.
The `.s` and `.co` files in `flash_mla/asm` are unchanged by this migration;
their compiler-provenance cleanup and reproducible-build review remain separate.
