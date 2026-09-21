# Implementation handoff: rollout-value learning for GenMol remasking

## Mission and non-negotiable scope

Implement `plan.md` exactly: a frozen GenMol V1 sampler generates replacements for eligible peripheral fragments; a small RDKit-feature MLP predicts each edit site's expected rollout reward; the learned site-selection policy is compared with random and largest-fragment policies. This is **not** a GenMol training/fine-tuning project and must not add docking, cLogP, activity predictors, a GNN, online reinforcement learning, or extra objectives.

Before changing an ambiguous detail, stop and record the decision in `RUNBOOK.md`. Do not make silent chemistry-policy changes. In particular:

- the only optimized property is RDKit QED;
- every valid parent has 2–4 eligible pendant fragments;
- the split is by Bemis–Murcko core *before* any generation;
- train, validation, and test parents are disjoint by canonicalized core;
- all rollout outputs, seeds, failures, and timing measurements are retained;
- test parents and test output directories are write-protected after the manifest is frozen;
- the model checkpoint, sampling settings, fingerprint settings, and reward definition are frozen before test generation.

The implementation must call an exact-core verifier. A substructure/SMARTS-presence test alone is not sufficient: it can accept a newly generated copy of a core at a different position.

## Pinned external sources

Use these immutable inputs unless the project owner deliberately changes this document and records why.

| Input | Exact endpoint / revision | Integrity action |
| --- | --- | --- |
| ChEMBL 37 chemical representations | `https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/chembl_37_chemreps.txt.gz` | SHA-256 must be `ea6181ce8dc7af41974e35b92e1febb0c9dcbe2c62f7ccc4a5d983ac19f696e7` |
| ChEMBL 37 checksums | `https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/checksums.txt` | Save beside the raw data |
| ChEMBL 37 release metadata/license | `https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/README` | Include CC BY-SA 3.0 attribution in reports |
| Individual ChEMBL molecule, JSON | `https://www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL25.json` | Use only for spot checks, not bulk extraction |
| ChEMBL REST collection, JSON | `https://www.ebi.ac.uk/chembl/api/data/molecule.json?limit=1000&offset=0` | Paginated API alternative; do not use it for this experiment |
| GenMol source | `https://github.com/NVIDIA-BioNeMo/genmol.git` at `add09fc83b7255bd09c797e527c0f4b51f5fb7c1` | Record `git rev-parse HEAD` |
| GenMol V1 checkpoint | `https://huggingface.co/nvidia/NV-GenMol-89M-v1/resolve/cdaf0c1b45c281315ab10389836bc46093cdbb65/model.ckpt` | Download, then record a locally computed SHA-256 |

The ChEMBL bulk file is a tab-separated set of chemical representations, including ChEMBL ID and SMILES. Inspect and record its header before selecting the canonical-SMILES column; do not assume a column ordering. Do not substitute an unpinned `latest` URL, PubChem, ZINC, a cached third-party CSV, or ChEMBL 38+ without explicitly re-registering the dataset.

The fingerprint policy is also an experimental input:

```yaml
# configs/experiment.yaml
chembl_release: 37
fingerprint:
  kind: morgan_bit_vector
  radius: 2                 # ECFP4
  n_bits: 2048
  use_chirality: false
constraints:
  min_tanimoto_to_parent: 0.50
  # abs(Δheavy) ≤ round(fraction × heavy_atoms(parent))
  max_heavy_atom_delta_fraction: 0.3333333333333333
  exact_core_required: true
reward:
  primary_property: qed
  success_qed_delta: 0.05
  reward_qed_delta_cap: 0.10
rollouts:
  labels_per_action: 8
  test_per_selected_action: 16
```

`0.50`, `1/3`, and `0.10` are pre-registered study choices, not natural chemical constants. The implementation must retain raw similarity, heavy-atom counts/deltas, per-parent heavy-atom budget, and uncapped QED delta so later descriptive sensitivity analyses are possible. It must not tune these values on test results.

## Directory layout and immutable artifacts

Create this layout. `data/raw`, `manifests`, `artifacts`, and `results` must be excluded from Git but accompanied by checksums and small metadata files committed to Git.

```text
genmol-rollout-policy/
├── configs/
│   ├── experiment.yaml
│   └── genmol_sampling.yaml
├── data/
│   ├── raw/chembl_37_chemreps.txt.gz
│   └── processed/parents.parquet
├── manifests/
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── test.jsonl                 # chmod a-w after creation
│   ├── hashes.json
│   └── seeds.json
├── src/rollout_policy/
│   ├── chembl.py
│   ├── eligibility.py
│   ├── core_check.py
│   ├── genmol_adapter.py
│   ├── scoring.py
│   ├── features.py
│   ├── model.py
│   └── io.py
├── scripts/
│   ├── download_chembl.py
│   ├── prepare_manifest.py
│   ├── probe_genmol.py
│   ├── generate_labels.py
│   ├── train_value.py
│   ├── evaluate_test.py
│   └── confirm_seeds.py
├── artifacts/
│   ├── provenance.json
│   └── value_model.pt
└── results/
    ├── probes.jsonl
    ├── labels/{train,validation}.jsonl
    ├── test/                      # chmod a-w after Phase 3
    └── confirmation/
```

Each JSONL generation record must include: `phase`, `split`, `parent_id`, input canonical SMILES, parent atom-mapped SMILES, canonical core identifier, action ID, selected SAFE span, candidate index, requested seed, observed seed if GenMol exposes one, output SAFE, output SMILES or null, validity indicator, exact-core indicator, feasibility, Tanimoto, heavy-atom delta, parent/candidate heavy-atom counts, `max_heavy_atom_delta_allowed`, parent/candidate QED, raw QED delta, capped reward, wall seconds, and exception text or null.

Write records atomically: append a complete JSON line and `fsync`, or write a batch temp file then rename it. On resume, validate records and generate only missing `(phase, parent_id, action_id, candidate_index, seed)` keys.

## Environment setup on the L4 VM

Do all GPU work on one reproducibly provisioned Linux VM. Record the VM region, machine type, image, NVIDIA driver, CUDA, PyTorch, Python, `nvidia-smi`, and `pip freeze` in `artifacts/provenance.json`. Check available GPU memory before running batches.

```bash
sudo apt-get update
sudo apt-get install -y git curl wget build-essential libsm6 libxext6 libxrender-dev

# Install micromamba or conda first if absent, then:
conda create -y -n genmol-rollout python=3.10
conda activate genmol-rollout

git clone https://github.com/NVIDIA-BioNeMo/genmol.git vendor/genmol
git -C vendor/genmol checkout add09fc83b7255bd09c797e527c0f4b51f5fb7c1
python -m pip install --upgrade pip
python -m pip install -r vendor/genmol/env/requirements.txt
python -m pip install -e vendor/genmol

# Project-only dependencies. Pin the resolved versions into requirements.lock.
python -m pip install "rdkit" "scikit-learn==1.2.2" pandas pyarrow \
  requests pyyaml "torch" "lightning==2.5.1"
python -m pip freeze | sort > artifacts/pip-freeze.txt
```

Download the checkpoint only after reviewing and accepting its NVIDIA Open Model License. The Hugging Face model revision is intentionally pinned, rather than using `main`.

```bash
mkdir -p vendor/genmol/checkpoints artifacts data/raw
curl --fail --location --retry 3 \
  -o vendor/genmol/checkpoints/model.ckpt \
  https://huggingface.co/nvidia/NV-GenMol-89M-v1/resolve/cdaf0c1b45c281315ab10389836bc46093cdbb65/model.ckpt
sha256sum vendor/genmol/checkpoints/model.ckpt | tee artifacts/genmol-v1.sha256
git -C vendor/genmol rev-parse HEAD | tee artifacts/genmol-source.commit

curl --fail --location --retry 3 \
  -o data/raw/chembl_37_chemreps.txt.gz \
  https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/chembl_37_chemreps.txt.gz
echo "ea6181ce8dc7af41974e35b92e1febb0c9dcbe2c62f7ccc4a5d983ac19f696e7  data/raw/chembl_37_chemreps.txt.gz" \
  | sha256sum --check
```

Run the official fragment example once from the pinned GenMol checkout before writing custom generation logic. The reference uses `genmol.sampler.Sampler` and its fragment-completion interface. Keep GenMol sampling settings in `configs/genmol_sampling.yaml`; never edit a vendor source file to encode experiment settings.

## Data preparation

Do not generate from the entire bulk file. Standardize all candidate parents deterministically, persist both original source SMILES and standardized SMILES, then select exactly 500 tasks using a seed recorded in `manifests/seeds.json`.

Required parent filters:

1. parse and sanitize with RDKit;
2. reject mixtures/disconnected molecules, salts, and sanitization failures;
3. apply one fully specified neutralization/normalization procedure and record its RDKit version and rules;
4. require 15–45 heavy atoms, QED in `[0.30, 0.85]`, and a non-empty Bemis–Murcko core;
5. find 2–4 eligible pendant fragments: each is connected to the protected core through exactly one cuttable bond and has a verified SAFE remasking span;
6. if more than four qualify, retain the first four under a deterministic ordering and persist the ordering key;
7. reject any parent/action for which the selected SAFE span cannot be round-tripped and aligned to the RDKit fragment.

Perform a group split by canonicalized Murcko core: 350 train, 75 validation, 75 test. Shuffle core groups using the manifest seed before assigning whole groups; never split a core group. Check that no core identifier occurs in more than one split.

### Download stub

```python
# scripts/download_chembl.py
from __future__ import annotations
import hashlib
from pathlib import Path
from urllib.request import urlretrieve

URL = ("https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/"
       "chembl_37/chembl_37_chemreps.txt.gz")
EXPECTED_SHA256 = "ea6181ce8dc7af41974e35b92e1febb0c9dcbe2c62f7ccc4a5d983ac19f696e7"
DESTINATION = Path("data/raw/chembl_37_chemreps.txt.gz")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if not DESTINATION.exists():
        urlretrieve(URL, DESTINATION)
    actual = sha256(DESTINATION)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"Checksum mismatch: {actual} != {EXPECTED_SHA256}")
    print(f"Verified {DESTINATION} ({actual})")

if __name__ == "__main__":
    main()
```

### Eligibility contract stub

```python
# src/rollout_policy/eligibility.py
from dataclasses import dataclass
from rdkit import Chem

@dataclass(frozen=True)
class EligibleAction:
    action_id: str
    fragment_atom_indices: tuple[int, ...]
    core_attachment_atom: int
    fragment_attachment_atom: int
    bond_type: str
    canonical_order_key: tuple
    safe_span: tuple[int, int]

def enumerate_eligible_actions(parent: Chem.Mol) -> list[EligibleAction]:
    """Return only pendant fragments attached to the protected core by one cuttable bond.

    TODO: Implement SAFE-span mapping by round-tripping the exact parent and verifying
    that `safe_span` decodes to this RDKit fragment. Never infer span identity only from
    a repeated fragment SMARTS.
    """
    raise NotImplementedError("Implement and unit-test RDKit ↔ SAFE fragment alignment")

def verify_action_roundtrip(parent: Chem.Mol, action: EligibleAction) -> None:
    """Raise if decoded SAFE span and RDKit atom set are not exactly the same fragment."""
    raise NotImplementedError
```

The SAFE alignment is the highest-risk integration point. Build it before model training and manually inspect at least 50 `(parent, action, SAFE span)` examples. Do not replace the unimplemented functions with a string search or an “is substructure” test.

## Generation and scoring

The adapter isolates vendor-specific behavior. First establish the actual method names and accepted parameters using the pinned repository’s official fragment example. The expected pattern is `Sampler(checkpoint_path)` followed by fragment completion; do not claim a seed is honored until a reproducibility test shows it is.

```python
# src/rollout_policy/genmol_adapter.py
from dataclasses import dataclass
import random
import numpy as np
import torch
from genmol.sampler import Sampler

@dataclass(frozen=True)
class Completion:
    output_safe: str | None
    error: str | None

class FrozenGenMol:
    def __init__(self, checkpoint: str, sampling_kwargs: dict):
        self.sampler = Sampler(checkpoint)
        self.sampling_kwargs = sampling_kwargs

    def complete(self, masked_safe: str, seed: int) -> Completion:
        # The generator must remain frozen. Add no gradients and no checkpoint mutation.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try:
            # Replace this call only after confirming the pinned sampler API.
            outputs = self.sampler.fragment_completion(
                masked_safe, num_samples=1, **self.sampling_kwargs
            )
            return Completion(output_safe=outputs[0] if outputs else None, error=None)
        except Exception as exc:
            return Completion(output_safe=None, error=f"{type(exc).__name__}: {exc}")
```

```python
# src/rollout_policy/scoring.py
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, QED

def morgan_fp(mol: Chem.Mol):
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=2048, useChirality=False
    )

def reward(parent: Chem.Mol, candidate: Chem.Mol, exact_core_ok: bool) -> dict:
    # Candidate must already have passed RDKit sanitization.
    similarity = DataStructs.TanimotoSimilarity(morgan_fp(parent), morgan_fp(candidate))
    parent_n_heavy = parent.GetNumHeavyAtoms()
    heavy_atom_delta = candidate.GetNumHeavyAtoms() - parent_n_heavy
    max_delta = round((1.0 / 3.0) * parent_n_heavy)
    parent_qed, candidate_qed = QED.qed(parent), QED.qed(candidate)
    raw_delta = candidate_qed - parent_qed
    feasible = (
        exact_core_ok
        and similarity >= 0.50
        and abs(heavy_atom_delta) <= max_delta
    )
    value = max(0.0, min(raw_delta / 0.10, 1.0)) if feasible else 0.0
    return {
        "feasible": feasible, "exact_core_ok": exact_core_ok,
        "tanimoto": similarity, "heavy_atom_delta": heavy_atom_delta,
        "parent_qed": parent_qed, "candidate_qed": candidate_qed,
        "qed_delta_raw": raw_delta, "reward": value,
    }
```

Implement `exact_core_ok` in `core_check.py` with explicit atom identity tracking:

1. atom-map the original parent before extracting its Murcko core;
2. persist the original core atom-map numbers and all core internal bonds;
3. after SAFE decoding, derive candidate atom correspondence from the verified round-trip transformation;
4. require that the corresponding candidate atoms have the same element, aromaticity/charge/stereochemical policy, and all original internal core bonds;
5. treat ambiguity or mapping failure as `False`.

Unit tests must cover: a true retained core; a generated copy of the same core elsewhere; a candidate with one core-bond mutation; a mapping failure; and an invalid SMILES. The latter four must be infeasible.

## Rollout labels, features, and MLP

For every eligible train action, generate exactly eight independent samples. Generate the eight samples for every eligible validation action too, once, and reuse those records for validation policy comparison and MLP selection. The action label is the mean of all eight rewards, including zero rewards from invalid outputs.

```python
# src/rollout_policy/features.py
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

def action_features(parent: Chem.Mol, core: Chem.Mol, fragment: Chem.Mol,
                    bond_type: str, is_ring: bool, is_terminal: bool) -> np.ndarray:
    """Fixed descriptors only. Fit scaler on train actions only."""
    parent_heavy = parent.GetNumHeavyAtoms()
    return np.asarray([
        Descriptors.MolWt(parent), Descriptors.MolLogP(parent),
        rdMolDescriptors.CalcNumHBD(parent), rdMolDescriptors.CalcNumHBA(parent),
        parent.GetNumHeavyAtoms(), core.GetNumHeavyAtoms(), fragment.GetNumHeavyAtoms(),
        fragment.GetNumHeavyAtoms() / parent_heavy,
        int(bond_type == "SINGLE"), int(bond_type == "DOUBLE"),
        int(bond_type == "AROMATIC"), int(is_ring), int(is_terminal),
    ], dtype=np.float32)
```

```python
# src/rollout_policy/model.py
import torch
from torch import nn

class ValueMLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
```

`train_value.py` must:

1. read only train actions and fit feature preprocessing only there;
2. use the action-level mean reward as the target;
3. use a weighted Huber loss, with the declared weight calculation recorded in configuration;
4. use validation MSE and validation top-1 action value for early stopping/model selection;
5. save the selected checkpoint, scaler, ordered feature names, all hyperparameters, and the validation report;
6. never read a test manifest or test generation file.

The “two-layer MLP” means two hidden linear layers (64 and 32 here) plus a scalar output layer. If widths, dropout, loss weights, or learning rate are changed, configure the finite candidate set before training and select only on validation.

## Phases and commands

### Phase 0: compatibility / throughput

Run 1,000 completions from 100 probe molecules not present in the 500-task manifest. Measure complete elapsed time: SAFE conversion, generation, decoding, core check, and scoring. Continue only if validity is at least 25%, a 50-record manual audit finds no mapping/core errors, and wall time is at most 20 seconds per completion.

```bash
python scripts/download_chembl.py
python scripts/probe_genmol.py --config configs/experiment.yaml --n-parents 100 --n-completions 10
python scripts/audit_records.py results/probes.jsonl --sample 50
```

### Phase 1–2: manifest, labels, and model

```bash
python scripts/prepare_manifest.py --config configs/experiment.yaml --seed 20260919
chmod a-w manifests/test.jsonl

# Phase 1 validation labels, then Phase 2 train labels.
python scripts/generate_labels.py --split validation --rollouts 8
python scripts/generate_labels.py --split train --rollouts 8
python scripts/train_value.py --config configs/experiment.yaml
```

Do not claim there are exactly 14,000 train/validation label completions. With 2–4 actions, the specified workload is `425 × (2–4) × 8 = 6,800–13,600`; it is 10,200 at three actions per parent.

### Phase 3: locked test

For each test parent, select one action under each policy:

- random: deterministically choose via the recorded per-parent random-policy seed;
- largest-fragment: maximum fragment heavy atoms; canonical action order breaks ties;
- learned: eligible action with highest frozen MLP prediction; canonical action order breaks ties.

Generate 16 candidates for every selected action. This is exactly `75 × 3 × 16 = 3,600` completion attempts.

```bash
python scripts/evaluate_test.py --config configs/experiment.yaml --rollouts 16
chmod -R a-w results/test
```

Calculate the primary parent-level success outcome before inspecting molecule examples. Use a paired bootstrap over 75 parents for learned minus largest-fragment success. Candidate success requires sanitization, exact core preservation, Tanimoto ≥0.50, absolute heavy-atom change ≤ round((1/3) × parent heavy atoms), and QED delta ≥0.05.

### Phase 4: seed-sensitivity confirmation

Repeat only the frozen learned policy and **largest-fragment** policy, not a baseline selected after seeing test performance. Use the same 75 test parents and 16 new samples per selected action: `75 × 2 × 16 = 2,400` attempts. This is a seed-sensitivity replication, not an independent test set.

```bash
python scripts/confirm_seeds.py \
  --policies learned largest_fragment --rollouts 16 --seed-schedule confirmation-v1
```

Across probes, train/validation labels, test, and confirmation, the specified completion count is 13,800–20,600, depending on 2–4 eligible actions per non-test parent. If budget planning reserves 21,000, label it explicitly as a retry/operational allowance rather than the nominal workload.

## Verification checklist

Before reporting a result, another LLM must verify all of the following:

- [ ] Raw ChEMBL file checksum and release number match this document.
- [ ] Checkpoint and GenMol source revision hashes are in `provenance.json`.
- [ ] Every split is core-disjoint; test paths were write-protected before test generation.
- [ ] Every label includes eight attempts, not only valid or best-of-eight outputs.
- [ ] Per-rollout rewards include invalid outputs as zero.
- [ ] Core verifier passed the adversarial unit tests and the required manual audit.
- [ ] MLP scaler/model were fitted from train actions only.
- [ ] No test result influenced model, feature, threshold, baseline, or sampling-setting selection.
- [ ] Test includes three policies and 3,600 attempts; confirmation includes learned versus largest-fragment and 2,400 fresh attempts.
- [ ] Report all attempts and sanitized-only statistics separately, plus raw JSONL, metric code, paired-bootstrap seed, confidence intervals, validity change, and cost/throughput.
