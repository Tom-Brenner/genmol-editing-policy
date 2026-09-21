# Experiment V2 handoff: larger, endpoint-aligned edit-site models

This document is an implementation handoff for a new experiment. It does **not**
change or reinterpret the locked Phase 3 result described in `paper-skeleton.md`.
The new experiment uses fresh manifests with 30,000 training parents, 5,000
validation parents, and 5,000 test parents. Model selection must use only the new
training and validation sets; the new test set is opened once after the policy,
features, architecture, and thresholds are frozen.

The four model changes are:

1. train against the per-completion event used by the primary test endpoint;
2. add fragment chemistry and attachment-environment features;
3. compare a regularized shallow MLP and LightGBM tabular/ranking baselines;
4. add a compact, action-conditioned, edge-aware GATv2 model.

The code below is intentionally stub-level. Preserve existing audit fields and
tests while implementing it; do not copy a stub without checking its imports and
the current function signatures.

## 0. Create a separate V2 experiment namespace

Do not overwrite the original manifests, labels, model, or locked test results.
Create `configs/experiment_v2.yaml` from `configs/experiment.yaml` and give it
separate paths:

```yaml
paths:
  manifests_dir: "manifests_v2"
  parents_parquet: "data/processed/parents_v2.parquet"
  artifacts_dir: "artifacts_v2"
  results_dir: "results_v2"

split:
  n_train: 30000
  n_validation: 5000
  n_test: 5000
  manifest_seed: 20261001  # choose and freeze before scanning

label:
  target: "completion_success"
  schema_version: 2

model_selection:
  primary_metric: "validation_top1_success_rate"
  # Freeze this finite candidate set before opening the new test set.
  candidates: ["shallow_mlp", "lightgbm_regressor", "lightgbm_ranker", "gatv2"]
```

Update the docstring in `scripts/prepare_manifest.py`; it currently says
`1000/200/200`. The split sizes themselves are already read from configuration.
The current pool target at approximately line 150 is `4 * need`, which becomes
160,000 eligible parents. Do not assume that this is the right margin. Choose a
pool target from the action-count/core-frequency audit in section 5 and expose it
as configuration:

```yaml
manifest:
  target_eligible_pool: 80000  # placeholder; replace from the audit
  scan_all: false
```

Then replace the hard-coded expression in `scripts/prepare_manifest.py`:

```python
manifest_cfg = cfg.get("manifest", {})
target_pool = int(manifest_cfg.get("target_eligible_pool", need * 4))
if target_pool < need:
    raise ValueError("target_eligible_pool must be >= requested split total")
```

Retain the core-disjoint assertions in `assign_core_disjoint_splits`. Run a dry
split simulation during the audit because 40,000 eligible parents do not imply
that 40,000 can be packed into exact core-disjoint quotas.

Several scripts currently honor `paths.artifacts_dir` for checkpoints but still
write reports to a literal `artifacts/...` path. Before running V2, route these
through the configured directory so old reports are not overwritten:

```python
def artifact_path(cfg: dict, filename: str) -> Path:
    return resolve_path(cfg["paths"]["artifacts_dir"]) / filename
```

Apply this to the experiment outputs in `prepare_manifest.py`,
`generate_labels.py`, `train_value.py`, `evaluate_test.py`,
`confirm_seeds.py`, `probe_genmol.py`, and `dry_run_local.py`. Global download
metadata may remain shared if its hashes are immutable.

At the observed 0.137 seconds/completion, 35,000 train+validation parents with
four actions and eight labels/action would require 1.12 million label
completions, approximately 43 L4 GPU-hours before overhead. Test cost does not
scale with the number of eligible actions after selection:
`5,000 parents * 3 policies * 16 = 240,000` completions. Recalculate the budget
after measuring the new mean action count.

Treat the original protocol documents as historical records. Do not edit
`plan.md` or `paper-skeleton.md` so that they appear to have preregistered V2.
Use this handoff (or a separately frozen `plan-v2.md`) as the new specification.
Add V2-specific commands and cost estimates to `RUNBOOK.md`/`GCP_SETUP.md`, and
remove the old “no GNN” assumption only in clearly marked V2 sections of
`implementation.md` or a new `implementation-v2.md`.

## 1. Align training supervision with the primary endpoint

### Why

The current action target is mean clipped reward:

```text
mean(feasible * clip(delta_QED / 0.10, 0, 1))
```

The primary test event is different:

```text
valid
and exact core
and Tanimoto >= 0.50
and heavy-atom delta within budget
and delta_QED >= 0.05
```

For an action with per-completion event probability `p`, the probability of at
least one success in 16 independent completions is `1 - (1 - p)^16`. This is
strictly increasing in `p`, so selecting the action with the largest predicted
per-completion probability is also the correct ranking for the any-of-16
endpoint. The model should therefore predict `p`, not mean clipped QED reward.

Keep `reward`, `rhat`, and their variance in artifacts for backward-compatible
analysis. Add endpoint-aligned fields rather than replacing them.

### 1.1 Factor a shared per-completion predicate

**File:** `src/rollout_policy/scoring.py`

**Placement:** immediately before `parent_success`; rewrite `parent_success` as
`any(completion_success(...))`. This prevents training and evaluation from
silently acquiring different definitions.

```python
def completion_success(
    row: dict[str, Any],
    *,
    success_qed_delta: float = 0.05,
    min_tanimoto: float = 0.50,
    max_heavy_atom_delta: int | None = None,
    max_heavy_atom_delta_fraction: float | None = DEFAULT_HEAVY_ATOM_DELTA_FRACTION,
    constraints: dict | None = None,
) -> bool:
    """The exact per-completion event whose OR defines parent_success."""
    constraints = constraints or {}
    fraction = (
        float(constraints["max_heavy_atom_delta_fraction"])
        if "max_heavy_atom_delta_fraction" in constraints
        else max_heavy_atom_delta_fraction
    )
    if not row.get("valid") or not _exact_core_flag(row):
        return False
    tanimoto = row.get("tanimoto")
    heavy_delta = row.get("heavy_atom_delta")
    qed_delta = row.get("qed_delta_raw")
    if tanimoto is None or float(tanimoto) < min_tanimoto:
        return False
    if heavy_delta is None or qed_delta is None:
        return False

    if max_heavy_atom_delta is not None:
        budget = int(max_heavy_atom_delta)
    elif row.get("max_heavy_atom_delta_allowed") is not None:
        budget = int(row["max_heavy_atom_delta_allowed"])
    else:
        n_heavy = row.get("parent_n_heavy_atoms")
        if n_heavy is None:
            return False
        budget = max_heavy_atom_delta_allowed(
            int(n_heavy),
            fraction=float(fraction or DEFAULT_HEAVY_ATOM_DELTA_FRACTION),
        )
    return abs(int(heavy_delta)) <= budget and float(qed_delta) >= success_qed_delta


def parent_success(rows: list[dict[str, Any]], **kwargs: Any) -> bool:
    return any(completion_success(row, **kwargs) for row in rows)
```

Preserve the existing explicit keyword arguments on `parent_success`; the
`**kwargs` form above only abbreviates the stub. In production, retain a typed
signature so misspelled thresholds fail loudly.

### 1.2 Aggregate binomial counts for each action

**File:** `scripts/generate_labels.py`

**Placement:** extend `aggregate_action_labels` around lines 28-51 and pass
`cfg` from `main`. Sort each group by `candidate_index` before creating lists so
the stored outcome order is deterministic.

```python
from rollout_policy.scoring import completion_success


def aggregate_action_labels(records: list[dict], cfg: dict) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        groups[(row["parent_id"], row["action_id"])].append(row)

    labels = []
    for (parent_id, action_id), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: int(r["candidate_index"]))
        rewards = [float(r["reward"]) for r in rows]
        success_flags = [
            int(
                completion_success(
                    r,
                    success_qed_delta=float(cfg["reward"]["success_qed_delta"]),
                    min_tanimoto=float(cfg["constraints"]["min_tanimoto_to_parent"]),
                    constraints=cfg["constraints"],
                )
            )
            for r in rows
        ]
        n_trials = len(success_flags)
        n_successes = sum(success_flags)
        labels.append(
            {
                # Existing fields must remain.
                "parent_id": parent_id,
                "action_id": action_id,
                "core_id": rows[0]["core_id"],
                "n_rollouts": n_trials,
                "rewards": rewards,
                "rhat": float(np.mean(rewards)),
                "reward_variance": float(np.var(rewards, ddof=1))
                if n_trials > 1
                else 0.0,
                "n_valid": sum(bool(r.get("valid")) for r in rows),
                "n_exact_core": sum(bool(r.get("exact_core")) for r in rows),
                # New schema-v2 fields.
                "success_flags": success_flags,
                "n_trials": n_trials,
                "n_successes": n_successes,
                "p_success_observed": n_successes / max(n_trials, 1),
                "label_schema_version": 2,
            }
        )
    return labels
```

Add `import numpy as np`, call `aggregate_action_labels(records, cfg)`, and add
the total successes and mean observed success probability to the summary JSON.
Apply the same aggregation through a shared library function in
`scripts/dry_run_local.py`; do not maintain a second, drifting implementation.
A suitable home is a new `src/rollout_policy/labels.py`.

### 1.3 Use an action-level binomial likelihood

**File:** `src/rollout_policy/model.py`

**Placement:** after `weighted_huber_loss`. The model returns an unconstrained
logit. Use `sigmoid(logit)` only for reporting and policy prediction.

```python
import torch.nn.functional as F


def binomial_nll(
    logits: torch.Tensor,
    successes: torch.Tensor,
    trials: torch.Tensor,
) -> torch.Tensor:
    """Negative log likelihood, excluding the parameter-free combinatorial term."""
    if torch.any(successes < 0) or torch.any(successes > trials):
        raise ValueError("binomial counts must satisfy 0 <= successes <= trials")
    per_action = trials * F.softplus(logits) - successes * logits
    # Normalize by trials so a changed rollout count does not change loss scale.
    return per_action.sum() / trials.sum().clamp_min(1.0)
```

This is equivalent to trial-weighted binary cross-entropy with target
`successes / trials`. Do not reuse inverse reward-variance weights: binomial
sampling variance is already represented by the likelihood.

**File:** `scripts/train_value.py`

**Placement:** extend `_load_action_table` and replace the `rhat` target block
around lines 109-140:

```python
rows.append(
    {
        # existing identity and feature fields
        "parent_id": lab["parent_id"],
        "action_id": lab["action_id"],
        "successes": int(lab["n_successes"]),
        "trials": int(lab["n_trials"]),
        "p_success_observed": float(lab["p_success_observed"]),
        "fragment_heavy_atoms": int(action["fragment_heavy_atoms"]),
    }
)

k_train = torch.tensor([r["successes"] for r in train_rows], dtype=torch.float32)
n_train = torch.tensor([r["trials"] for r in train_rows], dtype=torch.float32)
ds = TensorDataset(torch.from_numpy(x_train_s), k_train, n_train)

for xb, kb, nb in loader:
    logits = model(xb.to(device))
    loss = binomial_nll(logits, kb.to(device), nb.to(device))
```

At inference:

```python
probabilities = torch.sigmoid(model(x)).cpu().numpy()
```

For validation, group rows by `parent_id`, choose the highest predicted
probability, and report the selected action's observed `n_successes / n_trials`.
Use this `validation_top1_success_rate` for early stopping/model selection.
Also report binomial NLL, Brier score, calibration, and agreement with the
largest-fragment action. Do not early-stop on the new test set.

The observed proportion is noisy when there are only eight trials. A
beta-binomial model can be a later sensitivity analysis, but the ordinary
binomial objective should be the preregistered primary implementation.

### 1.4 Tests for endpoint alignment

Add `tests/test_labels.py`:

```python
def test_completion_success_matches_single_row_parent_success():
    for row in [successful_row(), low_qed_row(), invalid_row(), oversized_row()]:
        assert completion_success(row) == parent_success([row])


def test_action_aggregation_keeps_old_and_new_targets():
    labels = aggregate_action_labels(eight_mock_rows(), experiment_cfg())
    assert labels[0]["n_trials"] == 8
    assert labels[0]["n_successes"] == sum(labels[0]["success_flags"])
    assert labels[0]["p_success_observed"] == labels[0]["n_successes"] / 8
    assert "rhat" in labels[0]


def test_binomial_nll_prefers_correct_logit():
    good = binomial_nll(torch.tensor([4.0]), torch.tensor([8.0]), torch.tensor([8.0]))
    bad = binomial_nll(torch.tensor([-4.0]), torch.tensor([8.0]), torch.tensor([8.0]))
    assert good < bad
```

Retain the existing QED-threshold and heavy-atom-budget tests in
`tests/test_core_check.py`.

## 2. Add fragment and attachment-environment features

### Why

Six of the current 13 values are constant for all actions of a parent, and
fragment heavy-atom fraction is redundant with fragment heavy-atom count within
that parent. The action-varying information is mostly size, ring/terminal flags,
and attachment type. A deeper MLP cannot infer missing atom types or functional
groups.

The V2 tabular representation should add:

- fragment MolWt, LogP, TPSA, HBD, HBA, heteroatom count, formal charge,
  rotatable bonds, ring count, aromatic atom count, and heavy-atom count;
- fragment-to-parent fractions for heavy atoms, TPSA, and MolWt;
- core-side and fragment-side attachment atom: element, degree, total valence,
  formal charge, aromatic, ring, hybridization, and hydrogen count;
- hashed radius-2 Morgan environments centered separately on the two attachment
  atoms (for example 128 bits each);
- selected cut-bond properties and existing parent/core descriptors.

The current eligible cut is always a non-ring single bond
(`_is_cuttable_bond` in `src/rollout_policy/eligibility.py`), so the existing
`bond_double` and `bond_aromatic` columns are normally constant zero. Keep them
only if needed for schema compatibility; the local environments carry more
information. Audit every feature for zero variance on training data;
`is_terminal` may also be constant under the current simple-pendant definition.

### 2.1 Refactor the feature API around the parent and action

**File:** `src/rollout_policy/features.py`

**Placement:** replace the positional `fragment`, `bond_type`, `is_ring`, and
`is_terminal` arguments with an `EligibleAction`. The manifest already stores
the required atom indices. Construct an induced fragment without the dummy atom
present in `fragment_smiles`.

```python
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rollout_policy.eligibility import EligibleAction

FEATURE_SCHEMA_VERSION = 2
ATTACHMENT_FP_BITS = 128


def induced_fragment(parent: Chem.Mol, atom_indices: tuple[int, ...]) -> Chem.Mol:
    smiles = Chem.MolFragmentToSmiles(
        parent,
        atomsToUse=list(atom_indices),
        canonical=True,
        isomericSmiles=False,
    )
    fragment = Chem.MolFromSmiles(smiles)
    if fragment is None:
        raise ValueError("failed to construct action fragment")
    return fragment


def atom_environment_features(atom: Chem.Atom, prefix: str) -> tuple[list[str], list[float]]:
    elements = (6, 7, 8, 9, 15, 16, 17, 35, 53)
    hybridizations = (
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
    )
    names = [f"{prefix}_element_{z}" for z in elements] + [
        f"{prefix}_element_other",
        f"{prefix}_degree",
        f"{prefix}_total_valence",
        f"{prefix}_formal_charge",
        f"{prefix}_is_aromatic",
        f"{prefix}_is_in_ring",
        f"{prefix}_total_h",
    ] + [f"{prefix}_hybridization_{str(h)}" for h in hybridizations] + [
        f"{prefix}_hybridization_other"
    ]
    z = atom.GetAtomicNum()
    values = [float(z == allowed) for allowed in elements]
    values += [float(z not in elements)]
    values += [
        float(atom.GetDegree()),
        float(atom.GetTotalValence()),
        float(atom.GetFormalCharge()),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(atom.GetTotalNumHs()),
    ]
    hybridization = atom.GetHybridization()
    values += [float(hybridization == h) for h in hybridizations]
    values += [float(hybridization not in hybridizations)]
    return names, values


def centered_morgan_bits(parent: Chem.Mol, atom_idx: int, prefix: str):
    fp = AllChem.GetMorganFingerprintAsBitVect(
        parent,
        radius=2,
        nBits=ATTACHMENT_FP_BITS,
        fromAtoms=[int(atom_idx)],
        useChirality=False,
    )
    return (
        [f"{prefix}_morgan_{i}" for i in range(ATTACHMENT_FP_BITS)],
        [float(fp.GetBit(i)) for i in range(ATTACHMENT_FP_BITS)],
    )


def action_features(
    parent: Chem.Mol,
    core: Chem.Mol,
    action: EligibleAction,
) -> np.ndarray:
    validate_action_indices(parent, action)
    fragment = induced_fragment(parent, action.fragment_atom_indices)
    core_atom = parent.GetAtomWithIdx(action.core_attachment_atom)
    frag_atom = parent.GetAtomWithIdx(action.fragment_attachment_atom)

    # Build values and names in one place; production code should return a
    # FeatureVector(names, values, schema_version) or validate against a single
    # generated FEATURE_NAMES tuple to prevent ordering drift.
    names: list[str] = []
    values: list[float] = []
    add_parent_core_features(names, values, parent, core)
    add_fragment_features(names, values, parent, fragment, action)
    for helper_args in (
        (core_atom, "core_attachment"),
        (frag_atom, "fragment_attachment"),
    ):
        new_names, new_values = atom_environment_features(*helper_args)
        names.extend(new_names)
        values.extend(new_values)
    for atom_idx, prefix in (
        (action.core_attachment_atom, "core_attachment"),
        (action.fragment_attachment_atom, "fragment_attachment"),
    ):
        new_names, new_values = centered_morgan_bits(parent, atom_idx, prefix)
        names.extend(new_names)
        values.extend(new_values)
    assert tuple(names) == FEATURE_NAMES
    return np.asarray(values, dtype=np.float32)
```

The fragment helper can follow this pattern:

```python
def add_fragment_features(names, values, parent, fragment, action) -> None:
    parent_mw = float(Descriptors.MolWt(parent))
    parent_tpsa = float(rdMolDescriptors.CalcTPSA(parent))
    fragment_mw = float(Descriptors.MolWt(fragment))
    fragment_tpsa = float(rdMolDescriptors.CalcTPSA(fragment))
    fragment_heavy = int(fragment.GetNumHeavyAtoms())
    ordered = [
        ("fragment_molwt", fragment_mw),
        ("fragment_logp", float(Crippen.MolLogP(fragment))),
        ("fragment_tpsa", fragment_tpsa),
        ("fragment_hbd", float(Lipinski.NumHDonors(fragment))),
        ("fragment_hba", float(Lipinski.NumHAcceptors(fragment))),
        (
            "fragment_heteroatoms",
            float(sum(a.GetAtomicNum() not in (1, 6) for a in fragment.GetAtoms())),
        ),
        (
            "fragment_formal_charge",
            float(sum(a.GetFormalCharge() for a in fragment.GetAtoms())),
        ),
        ("fragment_rotatable_bonds", float(Lipinski.NumRotatableBonds(fragment))),
        ("fragment_rings", float(rdMolDescriptors.CalcNumRings(fragment))),
        (
            "fragment_aromatic_atoms",
            float(sum(a.GetIsAromatic() for a in fragment.GetAtoms())),
        ),
        ("fragment_heavy_atoms", float(fragment_heavy)),
        ("fragment_heavy_fraction", fragment_heavy / parent.GetNumHeavyAtoms()),
        ("fragment_molwt_fraction", fragment_mw / max(parent_mw, 1e-8)),
        ("fragment_tpsa_fraction", fragment_tpsa / max(parent_tpsa, 1e-8)),
        ("is_ring_fragment", float(action.is_ring)),
        ("is_terminal_fragment", float(action.is_terminal)),
    ]
    names.extend(name for name, _ in ordered)
    values.extend(value for _, value in ordered)
```

The parent/core helper omitted above should retain the existing ordered
descriptors. All helpers must contain explicit, ordered names and values.
Never generate names by iterating over an unordered set or dictionary.

Validate persisted indices before using them:

```python
def validate_action_indices(parent: Chem.Mol, action: EligibleAction) -> None:
    n_atoms = parent.GetNumAtoms()
    indices = set(action.fragment_atom_indices)
    if not indices or any(i < 0 or i >= n_atoms for i in indices):
        raise ValueError(f"invalid fragment atom indices for {action.action_id}")
    if action.fragment_attachment_atom not in indices:
        raise ValueError("fragment attachment atom is not in selected fragment")
    if action.core_attachment_atom in indices:
        raise ValueError("core attachment atom is inside selected fragment")
    bond = parent.GetBondBetweenAtoms(
        action.core_attachment_atom, action.fragment_attachment_atom
    )
    if bond is None:
        raise ValueError("persisted attachment bond is missing")
```

### 2.2 Update all feature call sites

The following locations currently call the old `action_features` signature:

- `features_from_action_record` in `src/rollout_policy/features.py`;
- `_load_action_table` in `scripts/train_value.py`;
- `predict_values` in `scripts/evaluate_test.py`;
- the duplicate predictor in `scripts/confirm_seeds.py`;
- feature/training paths in `scripts/dry_run_local.py`.

Convert manifest dictionaries with `action_from_dict` before feature extraction:

```python
action_obj = action_from_dict(action)
features = action_features(mol, core, action_obj)
```

Save `feature_schema_version`, ordered `feature_names`, and `n_features` in
every checkpoint. At load time, compare checkpoint names with runtime names and
fail rather than silently accepting reordered columns:

```python
if blob["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
    raise ValueError("feature schema mismatch")
if tuple(blob["feature_names"]) != FEATURE_NAMES:
    raise ValueError("feature ordering mismatch")
```

In `scripts/train_value.py`, remove the hard-coded `features[6]` used by
`largest_select` around lines 237-239. Store
`fragment_heavy_atoms` directly on each row and select by that named value:

```python
def largest_select(rows):
    return max(rows, key=lambda r: (r["fragment_heavy_atoms"], r["action_id"]))
```

Match the production tie-break in `select_largest_fragment`; ideally call that
policy function on the reconstructed actions rather than duplicating it.

### 2.3 Feature tests

Add `tests/test_features.py` covering:

- deterministic names and values for a fixed molecule/action;
- `len(FEATURE_NAMES) == len(action_features(...))`;
- changing the selected pendant changes fragment and attachment features;
- invalid or stale atom indices fail;
- train-only scaler fitting;
- checkpoint feature-schema mismatch fails.

## 3. Compare shallow MLP and LightGBM baselines

Use one common action-table builder and one prediction interface for all models.
The existing duplicate loaders in `evaluate_test.py` and `confirm_seeds.py`
should be replaced with a shared module such as
`src/rollout_policy/predictors.py`.

### 3.1 Predictor contract

```python
from typing import Protocol, Sequence


class ActionPredictor(Protocol):
    model_type: str

    def predict_actions(
        self,
        parent: dict,
        actions: Sequence[EligibleAction],
    ) -> np.ndarray:
        """Return one per-completion success probability per action."""
```

`scripts/evaluate_test.py` and `scripts/confirm_seeds.py` should call only this
interface and continue passing the resulting values to `select_action`. Record
`model_type`, checkpoint hash, feature schema, and predicted probability in
`selections.json`.

### 3.2 Regularized shallow MLP

**File:** `src/rollout_policy/model.py`

Do not weaken `ValueMLP`'s existing two-hidden-layer guard if backward
compatibility matters. Add a new configurable class:

```python
class TabularMLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_sizes: tuple[int, ...] = (64,),
        dropout: float = 0.15,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for width in hidden_sizes:
            layers += [
                nn.Linear(in_dim, width),
                nn.ReLU(),
                nn.LayerNorm(width),
                nn.Dropout(dropout),
            ]
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logits
```

Start with one 64-unit hidden layer, weight decay, dropout, and the binomial
loss. Freeze the candidate set before validation comparison:

```yaml
models:
  shallow_mlp:
    hidden_sizes: [64]
    dropout: 0.15
    weight_decay: 0.0001
```

### 3.3 LightGBM regressor and grouped ranker

Add LightGBM to `pyproject.toml`, preferably as an optional model dependency:

```toml
[project.optional-dependencies]
dev = ["pytest>=7.0"]
models = [
  "lightgbm>=4.0,<5",
  "torch-geometric>=2.5,<3",
]
```

Create `scripts/train_tabular_models.py`. It should import the same action-table
builder used by the MLP and must never load `manifests_v2/test.jsonl`.

Pointwise baseline:

```python
from lightgbm import LGBMRegressor

regressor = LGBMRegressor(
    objective="regression",
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1e-3,
    reg_lambda=1e-2,
    random_state=model_seed,
    n_jobs=-1,
)
regressor.fit(
    x_train,
    k_train / n_train,
    sample_weight=n_train,
    eval_set=[(x_val, k_val / n_val)],
    eval_sample_weight=[n_val],
    callbacks=[lightgbm.early_stopping(50)],
)
```

Clip regressor outputs to `[0, 1]` for probability reporting. Selection only
depends on their ordering.

For `LGBMRanker`, rows must be sorted so each parent's actions are contiguous,
and `group` is the number of actions per parent:

```python
from lightgbm import LGBMRanker

train_rows = sorted(train_rows, key=lambda r: (r["parent_id"], r["action_id"]))
val_rows = sorted(val_rows, key=lambda r: (r["parent_id"], r["action_id"]))
x_train, k_train, train_group = ranking_arrays(train_rows)
x_val, k_val, val_group = ranking_arrays(val_rows)

max_trials = max(r["trials"] for r in train_rows)
if any(r["trials"] != max_trials for r in train_rows + val_rows):
    raise ValueError("ranker relevance counts require a common rollout count")

ranker = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    label_gain=list(range(max_trials + 1)),  # linear gain for 0..n successes
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=50,
    reg_lambda=1e-2,
    random_state=model_seed,
    n_jobs=-1,
)
ranker.fit(
    x_train,
    k_train.astype(np.int32),
    group=train_group,
    eval_set=[(x_val, k_val.astype(np.int32))],
    eval_group=[val_group],
    eval_at=[1],
    callbacks=[lightgbm.early_stopping(50)],
)
```

`ranking_arrays` must assert that `sum(group) == len(rows)` and that no parent
appears in two disjoint blocks. `LGBMRanker` scores are ranking scores, not
calibrated probabilities; do not label them as probabilities in artifacts.
If calibrated probabilities are required, fit calibration on a separate part
of the training set, not on test.

Serialize LightGBM model text plus a JSON sidecar containing:

```json
{
  "model_type": "lightgbm_ranker",
  "feature_schema_version": 2,
  "feature_names": [],
  "label_schema_version": 2,
  "training_manifest_sha256": "...",
  "validation_manifest_sha256": "...",
  "best_iteration": 0,
  "model_seed": 0
}
```

### 3.4 Fair model selection

For every candidate, report on the same validation labels:

- selected-action observed success rate, the primary selection metric;
- mean binomial NLL/Brier score where outputs are probabilities;
- agreement with largest-fragment;
- success stratified by number of actions;
- parameter count/model size and inference time.

Pick one learned policy using the preregistered validation rule, freeze it, and
then run the new test once. If all models are reported on test, declare them all
as preregistered comparisons and correct the inferential plan for multiple
comparisons; do not choose the best test result after inspection.

## 4. Add an action-conditioned edge-aware GATv2

The graph model must receive a different representation for each candidate
action. A parent-only graph cannot distinguish which fragment is proposed for
remasking.

Use PyTorch Geometric `GATv2Conv` with edge features. A compact default is:

- atom projection to 64 channels;
- three GATv2 layers, four heads by 16 channels each;
- residual connections, LayerNorm, ReLU, and dropout;
- mean pools for whole molecule, protected core, and selected fragment;
- concatenate the richer global tabular vector;
- 64-unit output head producing one binomial logit.

Depending on exact input dimensions, this is approximately 35,000-50,000
parameters. Record the exact value with
`sum(p.numel() for p in model.parameters() if p.requires_grad)`.

### 4.1 Build one graph per parent/action

Create `src/rollout_policy/graph_features.py`.

```python
from torch_geometric.data import Data
from rollout_policy.chembl import murcko_core_atom_indices


def action_to_graph(
    parent_record: dict,
    action: EligibleAction,
    global_features: np.ndarray,
) -> Data:
    mol = Chem.MolFromSmiles(parent_record["smiles"])
    if mol is None:
        raise ValueError("invalid parent SMILES")
    validate_action_indices(mol, action)
    core_atoms = murcko_core_atom_indices(mol)
    if core_atoms is None:
        raise ValueError("parent has no uniquely usable Murcko core")

    node_rows = [atom_feature_vector(atom) for atom in mol.GetAtoms()]
    fragment_set = set(action.fragment_atom_indices)
    fragment_mask = torch.tensor(
        [i in fragment_set for i in range(mol.GetNumAtoms())], dtype=torch.bool
    )
    core_mask = torch.tensor(
        [i in core_atoms for i in range(mol.GetNumAtoms())], dtype=torch.bool
    )
    core_attachment_mask = torch.zeros(mol.GetNumAtoms(), dtype=torch.bool)
    fragment_attachment_mask = torch.zeros(mol.GetNumAtoms(), dtype=torch.bool)
    core_attachment_mask[action.core_attachment_atom] = True
    fragment_attachment_mask[action.fragment_attachment_atom] = True

    src: list[int] = []
    dst: list[int] = []
    edge_rows: list[list[float]] = []
    selected_pair = {
        (action.core_attachment_atom, action.fragment_attachment_atom),
        (action.fragment_attachment_atom, action.core_attachment_atom),
    }
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        base = bond_feature_vector(bond)
        for u, v in ((a, b), (b, a)):
            src.append(u)
            dst.append(v)
            edge_rows.append(base + [float((u, v) in selected_pair)])

    return Data(
        x=torch.tensor(node_rows, dtype=torch.float32),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(edge_rows, dtype=torch.float32),
        fragment_mask=fragment_mask,
        core_mask=core_mask,
        core_attachment_mask=core_attachment_mask,
        fragment_attachment_mask=fragment_attachment_mask,
        global_features=torch.tensor(global_features, dtype=torch.float32).unsqueeze(0),
        parent_id=parent_record["parent_id"],
        action_id=action.action_id,
    )
```

`atom_feature_vector` should encode element, degree, formal charge,
hybridization, aromaticity, ring membership, chirality, and hydrogens.
`bond_feature_vector` should encode bond type, conjugation, ring membership, and
stereochemistry. Define fixed name tuples and schema versions for both.

Add both directed edges for every RDKit bond. Validate that the selected
attachment edge occurs exactly twice after direction expansion.

### 4.2 GATv2 model

Create `src/rollout_policy/model_gat.py`:

```python
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import scatter


def masked_mean(h: torch.Tensor, mask: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    weights = mask.to(h.dtype).unsqueeze(-1)
    total = scatter(h * weights, batch, dim=0, reduce="sum")
    count = scatter(weights, batch, dim=0, reduce="sum").clamp_min(1.0)
    return total / count


class ActionGATv2(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    add_self_loops=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        pooled_dim = hidden_dim * 3 + global_dim
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data) -> torch.Tensor:
        h = self.node_proj(data.x)
        for conv, norm in zip(self.layers, self.norms):
            update = conv(h, data.edge_index, data.edge_attr)
            h = norm(h + self.dropout(torch.relu(update)))
        whole = global_mean_pool(h, data.batch)
        core = masked_mean(h, data.core_mask, data.batch)
        fragment = masked_mean(h, data.fragment_mask, data.batch)
        global_features = data.global_features.view(whole.shape[0], -1)
        pooled = torch.cat([whole, core, fragment, global_features], dim=-1)
        return self.head(pooled).squeeze(-1)  # binomial logits
```

The attachment masks may either be appended to node features before
`node_proj`, or separately pooled and concatenated. Do not create the masks and
then leave them unused. The selected-edge flag in `edge_attr` is what makes the
attention explicitly action-conditioned at the attachment.

### 4.3 Dataset, training, and inference

Create `src/rollout_policy/datasets.py` with an `ActionGraphDataset` that joins
manifest actions to action-label records. Each `Data` object should carry:

```python
data.successes = torch.tensor([label["n_successes"]], dtype=torch.float32)
data.trials = torch.tensor([label["n_trials"]], dtype=torch.float32)
```

Use `torch_geometric.loader.DataLoader`, randomize only training batches, and
train with the same `binomial_nll`. Validation must group predictions by parent
and compute selected-action observed success rate. Parent actions may be in
different batches, so aggregate all validation predictions before selecting.

Create `scripts/train_gat.py` rather than overloading every branch in
`train_value.py`. It may reuse shared action-table, metric, early-stopping, and
artifact helpers. Save:

```python
torch.save(
    {
        "model_type": "gatv2",
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "node_feature_names": NODE_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "global_feature_names": list(FEATURE_NAMES),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_schema_version": 2,
        "best_validation": best,
        "training_manifest_sha256": train_hash,
        "validation_manifest_sha256": validation_hash,
    },
    checkpoint_path,
)
```

Add a GAT predictor implementation to `predictors.py`. It should create all
action graphs for one parent, batch them, apply `sigmoid`, and return one value
per input action in exactly the same order. Assert output length and action IDs.

### 4.4 GAT tests

Add `tests/test_graph_model.py`:

- one parent with two actions produces two different action graphs;
- fragment/core masks are nonempty and do not overlap incorrectly;
- selected attachment edge is marked in both directions;
- batched forward output shape is `(batch_size,)`;
- backward pass with `binomial_nll` produces finite gradients;
- checkpoint round-trip preserves predictions;
- CPU inference works without GenMol or CUDA.

## 5. Can every molecule have more actions?

No. Raising `max_eligible_actions` only stops truncating molecules that already
have more eligible pendants. It cannot create additional actions for a molecule
with two or three eligible fragments.

Under the current definition, an action must be:

- a pendant fragment directly attached to the Bemis-Murcko core;
- attached through one non-ring single bond;
- disconnected by one simple cut without another core contact;
- uniquely aligned to a SAFE fragment; and
- successfully round-tripped.

Many valid molecules naturally have fewer than four such pendants. The current
configuration accepts two, three, or four. Molecules with more than four are
sorted deterministically and truncated to four in
`enumerate_eligible_actions` near lines 382-386 of
`src/rollout_policy/eligibility.py`.

There are three distinct policy choices:

1. **Raise only the maximum**, for example `min=2, max=8`. Parents still have a
   variable number of actions; molecules with 5-8 actions stop being truncated.
2. **Require exactly four**, using `min=4, max=4`. Molecules with fewer than four
   are rejected, while molecules with more are truncated to four. This changes
   the parent distribution toward highly substituted molecules.
3. **Broaden eligibility** to internal side-chain cuts, ring/fused edits, or
   multi-cut fragments. This creates new actions but changes the scientific task,
   SAFE mapping assumptions, and protected-core semantics. Treat it as a
   separate method, not a cap change.

Do not require four or raise the cap until the untruncated distribution has been
measured.

### 5.1 Separate enumeration, alignment, and count policy

The current `enumerate_eligible_actions` truncates before callers can inspect
the pre-truncation count, and `align_actions_to_safe_spans` silently skips
ambiguous alignments. Refactor without changing behavior:

**File:** `src/rollout_policy/eligibility.py`

```python
@dataclass(frozen=True)
class EligibilityDiagnostics:
    n_rdkit: int
    n_safe_aligned: int
    n_verified: int
    actions: tuple[EligibleAction, ...]
    rejection_reasons: tuple[str, ...]


def enumerate_aligned_actions(
    parent: Chem.Mol,
    *,
    require_safe_alignment: bool = True,
) -> EligibilityDiagnostics:
    """Enumerate and verify without applying min/max count policy."""
    rdkit_actions = enumerate_rdkit_pendant_actions(parent)
    # Production implementation should make align_actions_to_safe_spans return
    # per-action reasons such as ambiguous_element_multiset, missing_cut_piece,
    # safe_decode_failed, and roundtrip_failed rather than silently continuing.
    aligned, reasons = align_actions_to_safe_spans_diagnostic(parent, rdkit_actions)
    verified: list[EligibleAction] = []
    for action in aligned:
        try:
            verify_action_roundtrip(parent, action)
            verified.append(action)
        except Exception as exc:
            reasons.append(f"roundtrip_failed:{type(exc).__name__}")
    return EligibilityDiagnostics(
        n_rdkit=len(rdkit_actions),
        n_safe_aligned=len(aligned),
        n_verified=len(verified),
        actions=tuple(sorted(verified, key=lambda a: a.canonical_order_key)),
        rejection_reasons=tuple(reasons),
    )


def apply_action_count_policy(
    actions: tuple[EligibleAction, ...],
    *,
    min_actions: int,
    max_actions: int | None,
) -> list[EligibleAction]:
    if min_actions < 1:
        raise ValueError("min_actions must be positive")
    if max_actions is not None and max_actions < min_actions:
        raise ValueError("max_actions must be >= min_actions")
    retained = list(actions if max_actions is None else actions[:max_actions])
    return retained if len(retained) >= min_actions else []
```

Then make `enumerate_eligible_actions` a compatibility wrapper around these two
functions. Preserve the current all-or-nothing round-trip behavior for the
production manifest unless a change is explicitly preregistered; diagnostics
may report per-action failures without automatically changing inclusion.

### 5.2 Add an action-count audit script

Create `scripts/audit_action_counts.py`. It should scan standardized ChEMBL
parents without stopping after the current eligible-pool target and write only
aggregate statistics plus reproducible example IDs. It must run with real SAFE
alignment for final decisions; `--offline-no-safe` is development-only.

Core loop:

```python
counts = Counter()
reason_counts = Counter()
per_core = Counter()
records_for_split_simulation = []

for chembl_id, source_smiles in iter_chembl_smiles(raw_path):
    parent = standardize_parent(chembl_id, source_smiles, **standardization_filters)
    if parent is None:
        counts["standardization_rejected"] += 1
        continue
    mol = Chem.MolFromSmiles(parent.smiles)
    diag = enumerate_aligned_actions(mol, require_safe_alignment=True)
    counts[f"n_rdkit:{diag.n_rdkit}"] += 1
    counts[f"n_aligned:{diag.n_safe_aligned}"] += 1
    counts[f"n_verified:{diag.n_verified}"] += 1
    for reason in diag.rejection_reasons:
        reason_counts[reason] += 1
    if diag.n_verified >= 2:
        per_core[parent.core_smiles] += 1
        records_for_split_simulation.append(
            {"parent_id": chembl_id, "core_id": parent.core_smiles}
        )
```

For each proposed cap `K in {4, 6, 8, unlimited}`, report:

- parents with 0, 1, 2, 3, 4, 5, ... verified actions;
- fraction with fewer than four, exactly four, and more than four;
- mean, median, p90, p95, and maximum retained actions;
- number and fraction of actions discarded by the cap;
- number of eligible parents and distinct Murcko cores;
- largest core-group sizes and whether exact 30k/5k/5k packing succeeds;
- expected train/validation label completions:
  `8 * sum(retained_action_count)`;
- distribution shifts in heavy atoms, QED, core size, and source identifiers when
  comparing `min=2` with `min=4`;
- SAFE alignment and round-trip rejection rates.

Write `artifacts_v2/action_count_audit.json` and a concise text summary. Include
the source hash, config hash, software versions, scan count, and whether the scan
was full or capped.

### 5.3 Decision rule for the action cap

Use a rule fixed before model training, for example:

```text
Keep min_actions = 2.
Choose the smallest cap in {4, 6, 8} that retains at least 95% of all verified
actions among otherwise eligible parents, provided the resulting label workload
fits the budget. Otherwise use 8 and report the truncation rate.
```

This is only an example; preregister the actual threshold after seeing the
manifest audit but before generating labels. Do not select the cap according to
validation or test policy performance.

If a constant action count is needed for batching, pad and mask variable-length
action sets. Do not discard 2- or 3-action molecules merely to simplify tensor
shapes. PyTorch Geometric already supports variable-size graph batches, and the
tabular models operate on one action row at a time.

### 5.4 Tests for action-count behavior

Extend `tests/test_core_check.py` or add `tests/test_eligibility_counts.py`:

```python
def test_raising_cap_does_not_invent_actions(parent_with_three_actions):
    diag = enumerate_aligned_actions(parent_with_three_actions, require_safe_alignment=False)
    assert len(apply_action_count_policy(diag.actions, min_actions=2, max_actions=8)) == 3


def test_cap_truncates_in_canonical_order(parent_with_many_actions):
    diag = enumerate_aligned_actions(parent_with_many_actions, require_safe_alignment=False)
    kept = apply_action_count_policy(diag.actions, min_actions=2, max_actions=4)
    assert kept == sorted(diag.actions, key=lambda a: a.canonical_order_key)[:4]


def test_exact_four_rejects_three_action_parent(parent_with_three_actions):
    diag = enumerate_aligned_actions(parent_with_three_actions, require_safe_alignment=False)
    assert apply_action_count_policy(diag.actions, min_actions=4, max_actions=4) == []
```

Also test diagnostic reason counts for ambiguous SAFE matching and round-trip
failure. The current tests cover deterministic ordering on ibuprofen but not
pre-truncation counts or the greater-than-four path.

## 6. Shared refactors and artifact contracts

Implement these small refactors before adding four independent trainers:

1. `src/rollout_policy/labels.py`: `completion_success` aggregation and label
   schema validation.
2. `src/rollout_policy/action_table.py`: manifest/label join, action
   reconstruction, richer feature computation, and parent grouping.
3. `src/rollout_policy/metrics.py`: validation top-1 selection metrics and
   calibration metrics.
4. `src/rollout_policy/predictors.py`: common MLP, LightGBM, and GAT loading and
   action scoring.

Every trained artifact should record:

- model and feature/graph/label schema versions;
- ordered feature names;
- train and validation manifest hashes;
- label-file hashes;
- experiment-config hash;
- random seeds and package versions;
- exact model hyperparameters and trainable parameter count;
- selected epoch/iteration and validation metrics.

Fail loading on a hash/schema mismatch unless an explicit migration tool is
used. Never silently recompute actions at evaluation time: evaluation must use
the actions frozen in the manifest.

## 7. Implementation order and acceptance criteria

1. Create the V2 config and output namespaces.
2. Refactor eligibility and run the full action-count/core-frequency audit.
3. Freeze `min_eligible_actions`, `max_eligible_actions`, pool size, and the
   30k/5k/5k manifest seed; prepare and hash the fresh manifests.
4. Factor `completion_success`; add endpoint-aligned action labels while keeping
   old reward fields.
5. Implement richer features and schema validation.
6. Refactor shared action-table, metrics, and predictor loading code.
7. Train and compare the shallow MLP, LightGBM regressor, and LightGBM ranker.
8. Implement and train the compact GATv2.
9. Select exactly one primary learned policy using the frozen validation rule.
10. Freeze the chosen checkpoint and run the new test once.

Minimum acceptance checks before label generation:

```bash
pytest
python scripts/audit_action_counts.py --config configs/experiment_v2.yaml
python scripts/prepare_manifest.py --config configs/experiment_v2.yaml
```

Before training:

- all three splits have exact requested sizes;
- Murcko core sets are pairwise disjoint;
- action counts agree with the frozen cap policy;
- every action round-trips and has valid persisted attachment indices;
- manifest and raw-source hashes are stored;
- aggregation reproduces `parent_success([row])` exactly for every row;
- no training script imports or opens the V2 test manifest.

Before the one-time test:

- candidate model list and validation selection rule are frozen;
- selected checkpoint and dependency lockfile are archived;
- test seed schedule, test output directory, and metrics code are frozen;
- random and largest-fragment baselines use the same eligible action sets and
  completion budget as the learned policy.

The old `results/test` and old test manifest remain read-only throughout.
