# Rollout-value learning for GenMol remasking

Frozen GenMol V1 generates replacements for eligible peripheral fragments. A
small RDKit-feature MLP predicts each edit site's expected rollout reward. The
learned site-selection policy is compared with random and largest-fragment
baselines on a core-disjoint ChEMBL holdout.

See `plan.md` and `implementation.md`. Chemistry-policy decisions live in
`RUNBOOK.md`.

## What is (and is not) in this repo

**Included:** experiment code (`src/`, `scripts/`, `configs/`, `tests/`), pinned
GenMol **source** under `vendor/genmol/` (commit
`add09fc83b7255bd09c797e527c0f4b51f5fb7c1`), small dry-run fixtures, and docs.

**Not included (download on the VM):**

| Artifact | Why excluded | How to obtain |
| --- | --- | --- |
| GenMol V1 `model.ckpt` (~1.4 GB) | NVIDIA Open Model License + size | `python scripts/download_genmol_ckpt.py` after accepting the license on Hugging Face |
| ChEMBL 37 `chembl_37_chemreps.txt.gz` (~279 MB) | Bulk data | `python scripts/download_chembl.py` (SHA-256 verified) |
| Train/val/test manifests & rollouts | Generated outputs | `scripts/prepare_manifest.py`, `generate_labels.py`, … |
| Value-model weights / provenance dumps | Run artifacts | Written under `artifacts/` and `results/` during phases |

Do **not** commit checkpoints, chemreps, or JSONL results.

## GCP VM (recommended)

Step-by-step create/SSH/setup/phases: see **[`GCP_SETUP.md`](GCP_SETUP.md)** (modeled on the DiffSinger GCP runbook; uses `glen` gcloud config).

Plan target (updated machine): one L4 GPU.

```bash
# Example shape — confirm zone/price before create
gcloud compute instances create genmol-rollout \
  --machine-type=g2-standard-8 \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  ...
```

**Disk:** **100 GB** boot disk is enough (image + CUDA PyTorch env dominate;
checkpoint + ChEMBL + JSONL are &lt;3 GB). Use **150 GB** for comfortable cache
headroom.

**Note:** this image ships Python 3.12; create a **Python 3.10** conda env for
GenMol as below.

## Clone and environment

```bash
git clone git@github.com:Tom-Brenner/genmol-editing-policy.git
cd genmol-editing-policy

conda create -y -n genmol-rollout python=3.10
conda activate genmol-rollout

# System libs sometimes required by RDKit / SAFE (see GenMol README)
# sudo apt-get update && sudo apt-get install -y libsm6 libxext6 libxrender-dev

python -m pip install --upgrade pip
python -m pip install 'setuptools>=65,<82'  # pkg_resources; setuptools 82+ dropped it
python -m pip install -r vendor/genmol/env/requirements.txt
python -m pip install -e vendor/genmol
python -m pip install -e ".[dev]"

# If `import safe` fails on wandb/pkg_resources, apply the GenMol README SAFE
# __init__ patch (comment stock imports; export SAFEConverter, decode, encode).
```

## Large-file downloads

```bash
mkdir -p vendor/genmol/checkpoints data/raw artifacts

# 1) GenMol V1 checkpoint (pinned HF revision). Accept the NVIDIA Open Model
#    License on https://huggingface.co/nvidia/NV-GenMol-89M-v1 first.
python scripts/download_genmol_ckpt.py
# Writes vendor/genmol/checkpoints/model.ckpt and artifacts/genmol-v1.sha256*

# 2) ChEMBL 37 chemical representations (pinned SHA-256 in configs/experiment.yaml)
python scripts/download_chembl.py
# Writes data/raw/chembl_37_chemreps.txt.gz

# Optional: refresh a spot-check molecule (not a bulk substitute).
# Production ChEMBL REST often 500s; the script falls back to staging, then
# the committed fixture / chemreps dump. Smoke tests do not need this —
# data/fixtures/CHEMBL521.tsv is in git. Fully offline:
# python scripts/download_chembl_molecule.py --chembl-id CHEMBL521 --offline
python scripts/download_chembl_molecule.py --chembl-id CHEMBL521
```

Record `nvidia-smi`, CUDA/PyTorch versions, and `pip freeze` in
`artifacts/provenance.json` (see `artifacts/provenance.template.json`).

## Local checks (before / on the VM)

```bash
# Mock path (no checkpoint / no ChEMBL bulk): float32 only
python scripts/dry_run_local.py --bf16 false

# Real GenMol remask + SAFE alignment (needs checkpoint)
python scripts/test_one_molecule.py --bf16 true --parent-tsv data/fixtures/CHEMBL521.tsv

# Unit tests (RDKit + precision helpers)
python -m pytest tests/ -q
```

`--bf16 true` means *prefer* bf16; on GPUs without hardware bf16 (e.g. GTX 1080
Ti) the adapter falls back to **fp16**. On L4 (Ampere), bf16 is used when
requested. `--bf16 false` forces float32.

## Experiment phases (after downloads)

```bash
# Phase 0 — throughput / compatibility (100 parents × 10 completions)
python scripts/probe_genmol.py --config configs/experiment.yaml --bf16 true
python scripts/audit_records.py results/probes.jsonl --sample 50

# Phase 1 — core-disjoint 1000/200/200 manifest (committed under manifests/)
python scripts/prepare_manifest.py --config configs/experiment.yaml --seed 20260919
chmod a-w manifests/test.jsonl

# Phase 1–2 — labels then value model
python scripts/generate_labels.py --split validation --rollouts 8
python scripts/generate_labels.py --split train --rollouts 8
python scripts/train_value.py --config configs/experiment.yaml

# Phase 3 — locked test (3 policies × 16)
python scripts/evaluate_test.py --config configs/experiment.yaml --rollouts 16
chmod -R a-w results/test

# Phase 4 — seed confirmation (learned vs largest_fragment)
python scripts/confirm_seeds.py --policies learned largest_fragment --rollouts 16
```

Freeze GenMol sampling settings in `configs/genmol_sampling.yaml` before test
generation. Do not tune thresholds on test results.

## Layout

```text
configs/             experiment + GenMol sampling (frozen before test)
src/rollout_policy/  library code
scripts/             phase entry points + download helpers
vendor/genmol/       pinned GenMol source (no checkpoints in Git)
data/fixtures/       small dry-run / smoke TSVs
data/raw/            ChEMBL bulk (gitignored; download script)
manifests/           core-disjoint splits (generated; test chmod a-w)
results/             JSONL rollouts (generated; test chmod a-w after Phase 3)
artifacts/           provenance, checksums, MLP checkpoint (mostly gitignored)
```

## License / attribution

- Experiment code: see repository license if present.
- GenMol code/weights: Apache-2.0 code; weights under NVIDIA Open Model License.
- ChEMBL: CC BY-SA 3.0 — cite [ChEMBL](https://www.ebi.ac.uk/chembl/) in reports.
