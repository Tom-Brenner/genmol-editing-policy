# Experiment runbook — decision log

Record chemistry-policy and integration decisions here before changing ambiguous
behavior. Silent changes to reward, eligibility, core verification, or sampling
are not allowed.

## Pinned revisions

| Artifact | Value |
| --- | --- |
| GenMol source | `add09fc83b7255bd09c797e527c0f4b51f5fb7c1` |
| GenMol V1 checkpoint URL revision | `cdaf0c1b45c281315ab10389836bc46093cdbb65` |
| ChEMBL release | 37 |
| ChEMBL chemreps SHA-256 | `ea6181ce8dc7af41974e35b92e1febb0c9dcbe2c62f7ccc4a5d983ac19f696e7` |
| Manifest seed | `20260919` |

## Decisions

### 2026-09-20 — Generation API: targeted SAFE-span remask

**Decision:** Use GenMol's native fragment-remask pathway (`SAFEConverter` +
`Slicer` + mask replacement of one `.`-delimited SAFE fragment), not
`fragment_completion`.

**Why:** Lead-optimization and PMO workflows in the pinned GenMol repo call
`mask_modification` / `remask`. The experiment is a remasking site-selection
ablation. `fragment_completion` is for motif-extension / scaffold decoration
from an attachment-point fragment and would change the generation interface.

**How:** `FrozenGenMol.complete_span` reimplements `Sampler.remask` with a
caller-chosen fragment index and pinned `num_insert_mask` from
`configs/genmol_sampling.yaml` (midpoint of GenMol's Uniform{5,...,15}).

### 2026-09-20 — Neutralization / standardization

**Decision:** Use RDKit `MolStandardize` cleanup: disconnect metals → normalize →
reionize → `Uncharger` → largest fragment only (reject if still disconnected) →
sanitize → strip atom maps → canonical SMILES.

**Why:** Fully specified, deterministic, and recorded with RDKit version in
provenance. Reject salts/mixtures rather than silently keeping a component.

### 2026-09-20 — Eligible pendant fragment definition

**Decision:** Protected core = Bemis–Murcko scaffold atoms of the standardized
parent. An eligible action is a connected set of non-core atoms joined to the
core by exactly one cuttable bond (non-ring single bond). Retain at most four
actions ordered by `(fragment_heavy_atoms desc, attachment_atom_idx asc,
fragment_canonical_smiles asc)`.

### 2026-09-20 — SAFE span alignment

**Superseded 2026-09-21:** GenMol's default `Slicer` (all non-ring single bonds)
does not uniquely align to Murcko pendants. Use `PendantBondSlicer` instead
(cuts only core↔pendant attachment bonds). Align each RDKit pendant to a SAFE
fragment by element multiset + RDKit cut-piece atom-set membership. Reject the
parent if any eligible action lacks a unique matching span. Store
`slicer_cut_bonds` on each action for reproducible remask encoding.

### 2026-09-20 — Exact core verifier

**Decision:** Atom-map the parent before Murcko extraction; persist core atom-map
numbers and internal core bonds. After decoding a candidate, require a unique
substructure match of the mapped core SMARTS/query that preserves element,
aromaticity, formal charge, and all original internal core bonds on the matched
atoms. A second copy of the core SMARTS elsewhere does not satisfy the check
unless it is the mapped match. Mapping failure → infeasible.

### 2026-09-21 — `--bf16` with fp16 fallback on Pascal GPUs

**Decision:** `--bf16 true` means *prefer* bf16. Runtime selection:

- CUDA compute capability ≥ 8.0 and `is_bf16_supported()` → **bf16**
- other CUDA GPUs (e.g. GTX 1080 Ti / sm_61) → **fp16**
- no CUDA → **fp32**

`--bf16 false` always forces float32.

### 2026-09-21 — Local dry-run before VM (`--bf16 false`)

**Decision:** `python scripts/dry_run_local.py --bf16 false` remains the
mock/CPU path. Real local GPU smoke test is
`python scripts/test_one_molecule.py --bf16 true` with checkpoint + SAFE.

### 2026-09-21 — Batched rollouts, independent per-completion seeds

**Decision:** For a given parent+action, missing rollouts are stacked into one
GenMol diffusion batch. Each JSONL row still records its own
`requested_seed` / `observed_seed` from `seed_for(...)`. Sampling noise comes
from a `torch.Generator` created by `generator_for_seed(seed, device)` — not
from a single `set_global_seeds` call covering the batch. `wall_seconds` is
batch-wall / N (effective seconds per completion). MCG (`gamma != 0`) stays
sequential because stock `Sampler.generate` indexes row 0.

**Why:** Phase 0 mean wall was 0.137 s/sample at batch size 1; L4 is idle on
short remasks. Independent generators keep the pre-registered seed schedule
without coupling Gumbel draws across candidates.

### 2026-09-21 — Manifest 1000/200/200

**Decision:** Core-disjoint split is 1,000 train / 200 validation / 200 test
(manifest seed `20260919`). JSONL + `seeds.json` + `hashes.json` are committed
so the VM gets them via `git pull`. Bulk ChEMBL chemreps stay on the VM via
`python scripts/download_chembl.py` (FTP), not rsync from the laptop.
