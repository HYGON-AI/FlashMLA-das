# Source Comparison Against the Application Baseline

The open-source application specifies DeepSeek FlashMLA `main` at
`48c6dc426f045cb7743b18f5c7329f35f1b7ed79`. This commit is the starting
point for the source comparison, rather than asking maintainers to classify
every file without using the available Git history.

## Fixed Scope

- Baseline: `48c6dc426f045cb7743b18f5c7329f35f1b7ed79`.
- Original reviewed commit: `2a291d5b4e3bedbe6b6f04276a63824e791a82b8`.
- Corresponding GitLab source tip: `1287a010c13f7cbc0f859b6eaac957545d482836`.
  The two source trees are identical.
- Later source checked: `b28f2d398a9b6dc4d81f7ef210a2a264347ae881`.
- Evidence inventory date: 2026-10-20.

The baseline/reviewed union contains 331 paths: 18 unchanged, 23 modified,
200 added, and 90 deleted. The modified count includes a file-mode change.
These are path-level differences, not counts of original works.

## Original Source-Review Queue

All 195 paths from the original source-review table were reconstructed through
141 first-parent commits after the baseline and checked against the reviewed
tree. Copy and rename detection used a 40% similarity threshold and searched
unchanged files as copy sources.

| Evidence category | Files | Meaning |
| --- | ---: | --- |
| Baseline copy/rename lineage; every edge at least 80% | 29 | Record the baseline paths and subsequent local adaptation. |
| Baseline lineage candidate; at least one edge below 80% | 14 | Inspect the copy/rename edge before accepting the mapping. |
| Historical DeepSeek content candidate | 14 | Shared source content provides a review lead beyond the baseline snapshot. |
| Generated version metadata | 1 | `flash_mla/version.py` is produced by `get_version.py`. |
| Post-baseline addition without a deterministic source match | 137 | Branch introduction is established; external origin and ownership remain unresolved. |

The 80% boundary only separates review queues. Neither this boundary nor Git's
copy/rename detection constitutes an ownership determination. A baseline path
history does not establish the authorship of every line subsequently added.

The 137 unmatched paths comprise 112 under `dsa_mls/legacy`, four extension
instantiations, and 21 other source paths. They are not automatically classified
as Hygon-original merely because the baseline lacks those paths.

## Recorded Examples

- Commit `f298a27184e603d78ffe9ca37841e57641719314` moves numerous dense/sparse
  paths from `csrc/sm90/` to `csrc/gfx93/`, including an identical `traits.h`
  move and a 99% `splitkv_mla.cuh` move. A whole-baseline diff can otherwise
  display their later modified versions as added paths.
- `csrc/api/dense_decode_kvfp8.h` and `dense_decode_qkvfp8.h` trace back to
  the baseline `csrc/api/dense_decode.h` through 93% copy edges.
- `csrc/gfx9/decode/combine/combine.h` traces back to the baseline
  `csrc/smxx/decode/combine/combine.h` through an 83% edge.
- The historical content search finds 137 distinct nontrivial normalized lines
  shared by `csrc/extension/flash_api.h` and a historical DeepSeek
  `csrc/flash_api.cpp` blob. The previously verified upstream source references
  are documented in [Third-Party Notices](../THIRD_PARTY_NOTICES.md).

Historical content candidates compare distinct whitespace-normalized lines of
at least 20 characters, excluding comment, include, pragma and end-if lines.
Shared lines locate inspection candidates; they are not an exact import commit
or a license determination. No reviewed blob among these 195 paths exactly
matches an upstream historical blob. For additions without baseline lineage,
checking their introduction blobs also did not establish an exact upstream copy.

## Remaining Review

Use the recorded baseline and historical candidates before requesting new
maintainer evidence. Confirm the unresolved import or generation details and
the actual scope of local changes. Preserve upstream notices and apply the
appropriate treatment after the source relationship is established; do not
batch-add copyright headers to the original 195-item queue.

The detailed evidence exports contain each path, blob, reference path,
introduction commit, copy/rename chain, candidate blob and outstanding question.
They are review artifacts held by the maintainer. The original review is kept
fixed: two later source edits in `flash_fwd_mla_kernel_fp8.h` and `asm/build.sh`
replace constructed target strings with equivalent literals and are tracked
separately from the original source-matching statistics.

This inventory is not a formal compliance rescan, HCU build validation, or a
claim that all source ownership has been confirmed.
