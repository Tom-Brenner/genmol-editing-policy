# GAT implementation and pre-training audit handoff

This handoff describes the work required **before the first run of
`scripts/train_gat.py`**. The existing train and validation generations are
valuable and must not be regenerated:

- train: 584,944 completion records, 73,118 action labels;
- validation: 98,504 completion records, 12,313 action labels;
- total: **683,448 completion records**.

The generated molecules and their raw score components are already stored in
`results_v2/labels/{train,validation}.jsonl`. All feature changes below are
computed from the frozen manifests, and the exact-core audit only replays RDKit
checks over those stored outputs. Neither task needs GenMol, its checkpoint, a
GPU, or new molecular generation.

The current configured split is 30,000 train / 5,000 validation / 5,000 test.
The observed validation action count, 12,313, confirms a 5,000-parent
validation set (about 2.46 actions/parent), not a 10,000-parent set.

## Required outcome

Before training:

1. recompute and validate the Bemis-Murcko core identity and unique protected
   core for every parent in all three manifests;
2. stream all 683,448 existing train/validation records and independently
   recompute exact-core preservation, comparing it with the stored flag;
3. expand the global action feature vector to the complete descriptor set below;
4. replace top-1-success early stopping with validation binomial NLL;
5. make validation tie-breaking identical to `select_learned`;
6. replace simultaneous materialization of all action graphs with a lazy or
   sharded dataset;
7. implement checkpoint-compatible GAT inference before opening the test set.

If the audits find a correctness problem, stop and write a report. Do not
silently discard records, rewrite source JSONL, or regenerate candidates.

---

## 1. Full manifest and exact-core audit

### What is and is not currently checked

`scripts/prepare_manifest.py` groups records by the stored canonical
Bemis-Murcko `core_id` and asserts pairwise core-disjoint train, validation, and
test sets. `generate_action_rollouts` also extracts a mapped protected core and
checks every generated candidate before storing `exact_core`.

However:

- `scripts/generate_labels.py` trusts the prepared split and does not recompute
  split disjointness;
- `scripts/audit_records.py` randomly samples completion records and merely
  copies the stored `exact_core` flag;
- parent standardization requires a nonempty Murcko scaffold, but
  `extract_protected_core` is stricter and rejects a parent if its scaffold has
  more than one substructure match;
- a 50-record random sample is not a full parent/scaffold audit.

The new audit should therefore have two distinct phases.

### Phase A: audit all 40,000 manifest parents

Create `scripts/audit_exact_cores.py`. Read all three manifests but do not read
the test generation outputs and do not generate anything.

For every parent:

1. parse `parent["smiles"]`;
2. recompute the unmapped canonical Murcko core with
   `murcko_core_smiles(parent_mol)`;
3. create `parent_mapped = assign_atom_maps(parent_mol)`, exactly as
   `generate_action_rollouts` does;
4. call `extract_protected_core(parent_mapped)`;
5. require a non-`None` protected core, proving unique extraction;
6. compare the recomputed unmapped core ID with `parent["core_id"]`;
7. validate every persisted action's atom indices and attachment bond;
8. collect the recomputed core IDs per split and check pairwise disjointness;
9. check that parent IDs do not occur in more than one split.

Do **not** directly compare `ProtectedCore.core_smiles` with `core_id` unless atom
maps have first been removed: the protected core is extracted from a mapped
parent, whereas the manifest ID is an unmapped canonical SMILES. Recompute the
manifest comparison from the unmapped parent molecule.

Stub:

```python
from collections import Counter
from rdkit import Chem

from rollout_policy.chembl import (
    assign_atom_maps,
    murcko_core_smiles,
)
from rollout_policy.core_check import extract_protected_core
from rollout_policy.dry_run import action_from_dict
from rollout_policy.graph_features import validate_action_indices
from rollout_policy.io import iter_jsonl, resolve_path


def audit_manifest_split(path, split: str):
    core_ids: set[str] = set()
    parent_ids: set[str] = set()
    failures: list[dict] = []
    counts = Counter()

    for row in iter_jsonl(path):
        counts["parents"] += 1
        parent_id = str(row["parent_id"])
        if parent_id in parent_ids:
            failures.append(
                {"split": split, "parent_id": parent_id, "reason": "duplicate_parent"}
            )
            continue
        parent_ids.add(parent_id)

        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            failures.append(
                {"split": split, "parent_id": parent_id, "reason": "invalid_parent"}
            )
            continue

        recomputed_core_id = murcko_core_smiles(mol)
        if recomputed_core_id != row["core_id"]:
            failures.append(
                {
                    "split": split,
                    "parent_id": parent_id,
                    "reason": "core_id_mismatch",
                    "stored": row["core_id"],
                    "recomputed": recomputed_core_id,
                }
            )
        if recomputed_core_id is not None:
            core_ids.add(recomputed_core_id)

        parent_mapped = assign_atom_maps(mol)
        protected_core = extract_protected_core(parent_mapped)
        if protected_core is None:
            failures.append(
                {
                    "split": split,
                    "parent_id": parent_id,
                    "reason": "protected_core_not_unique",
                }
            )
            continue
        counts["unique_protected_core"] += 1

        for action_row in row["actions"]:
            try:
                validate_action_indices(mol, action_from_dict(action_row))
                counts["valid_actions"] += 1
            except Exception as exc:
                failures.append(
                    {
                        "split": split,
                        "parent_id": parent_id,
                        "action_id": action_row.get("action_id"),
                        "reason": f"invalid_action:{type(exc).__name__}",
                    }
                )

    return {
        "split": split,
        "core_ids": core_ids,
        "parent_ids": parent_ids,
        "counts": counts,
        "failures": failures,
    }
```

After reading the splits:

```python
assert train["core_ids"].isdisjoint(validation["core_ids"])
assert train["core_ids"].isdisjoint(test["core_ids"])
assert validation["core_ids"].isdisjoint(test["core_ids"])
assert train["parent_ids"].isdisjoint(validation["parent_ids"])
assert train["parent_ids"].isdisjoint(test["parent_ids"])
assert validation["parent_ids"].isdisjoint(test["parent_ids"])
```

Do not rely only on `assert`, which can be disabled with optimized Python.
Convert overlap into an explicit nonzero exit and include overlapping IDs in the
report.

### Phase B: replay exact-core checking for all 683,448 records

Use `iter_jsonl`, not `load_jsonl`, so the audit streams records instead of
loading both completion files into RAM.

Build a parent cache from the train and validation manifests. For each parent,
cache:

- canonical parent SMILES;
- `parent_mapped = assign_atom_maps(parent_mol)`;
- `protected_core = extract_protected_core(parent_mapped)`;
- recomputed unmapped `core_id`.

Then stream:

```text
results_v2/labels/train.jsonl
results_v2/labels/validation.jsonl
```

For every record:

1. require that `parent_id` occurs in the matching manifest split;
2. require that record `parent_smiles` and `core_id` agree with the manifest;
3. if the record is invalid or has no output SMILES, recomputed exact-core is
   `False`;
4. otherwise sanitize the stored `output_smiles`;
5. call `check_exact_core(parent_mapped, candidate, protected_core)`;
6. compare the result with `bool(record["exact_core"])`;
7. count mismatches by split, parent, action, core, and stored/recomputed value;
8. verify record-key uniqueness and the expected total record counts;
9. retain only a bounded sample of mismatch records in the human-readable
   report.

Stub:

```python
from dataclasses import dataclass

from rollout_policy.core_check import check_exact_core
from rollout_policy.io import iter_jsonl
from rollout_policy.scoring import sanitize_smiles


@dataclass(frozen=True)
class ParentAuditContext:
    split: str
    smiles: str
    core_id: str
    parent_mapped: Chem.Mol
    protected_core: object


def recompute_record_exact_core(
    record: dict,
    context: ParentAuditContext,
) -> bool:
    if not record.get("valid") or not record.get("output_smiles"):
        return False
    candidate = sanitize_smiles(record["output_smiles"])
    if candidate is None:
        return False
    return bool(
        check_exact_core(
            context.parent_mapped,
            candidate,
            context.protected_core,
        )
    )


def audit_record_file(
    path,
    *,
    split: str,
    parents: dict[str, ParentAuditContext],
    expected_records: int,
    mismatch_sample_limit: int = 200,
) -> dict:
    counts = Counter()
    mismatch_samples: list[dict] = []
    seen_keys: set[tuple] = set()

    for index, record in enumerate(iter_jsonl(path), start=1):
        counts["records"] += 1
        parent_id = str(record["parent_id"])
        context = parents.get(parent_id)
        if context is None:
            counts["unknown_parent"] += 1
            continue

        if record.get("split") != split:
            counts["wrong_split"] += 1
        if record.get("parent_smiles") != context.smiles:
            counts["parent_smiles_mismatch"] += 1
        if record.get("core_id") != context.core_id:
            counts["record_core_id_mismatch"] += 1

        key = (
            record.get("phase"),
            parent_id,
            record.get("action_id"),
            int(record["candidate_index"]),
            int(record["requested_seed"]),
        )
        if key in seen_keys:
            counts["duplicate_record_key"] += 1
        else:
            seen_keys.add(key)

        recomputed = recompute_record_exact_core(record, context)
        stored = bool(record.get("exact_core"))
        counts["recomputed_exact_core_true"] += int(recomputed)
        counts["stored_exact_core_true"] += int(stored)

        if recomputed != stored:
            counts["exact_core_mismatch"] += 1
            if len(mismatch_samples) < mismatch_sample_limit:
                mismatch_samples.append(
                    {
                        "split": split,
                        "parent_id": parent_id,
                        "core_id": context.core_id,
                        "action_id": record.get("action_id"),
                        "candidate_index": record.get("candidate_index"),
                        "requested_seed": record.get("requested_seed"),
                        "stored_exact_core": stored,
                        "recomputed_exact_core": recomputed,
                        "output_smiles": record.get("output_smiles"),
                    }
                )

        if index % 10_000 == 0:
            print(
                f"{split}: audited={index}/{expected_records} "
                f"mismatches={counts['exact_core_mismatch']}",
                flush=True,
            )

    if counts["records"] != expected_records:
        raise RuntimeError(
            f"{split}: found {counts['records']} records, expected {expected_records}"
        )
    return {"counts": dict(counts), "mismatch_samples": mismatch_samples}
```

Invoke with:

```python
train_report = audit_record_file(
    results_dir / "labels" / "train.jsonl",
    split="train",
    parents=train_parents,
    expected_records=584_944,
)
validation_report = audit_record_file(
    results_dir / "labels" / "validation.jsonl",
    split="validation",
    parents=validation_parents,
    expected_records=98_504,
)
assert (
    train_report["counts"]["records"]
    + validation_report["counts"]["records"]
    == 683_448
)
```

The finished script should write:

```text
artifacts_v2/manifest_core_audit.json
artifacts_v2/exact_core_replay_audit.json
artifacts_v2/exact_core_mismatch_samples.jsonl
```

Include SHA-256 hashes of all input manifests and completion JSONL files,
software versions, start/end timestamps, elapsed seconds, expected/observed
counts, and the exact command. Never modify the input JSONL files.

### Why this audit is much faster than generation

The eight-hour generation run invoked the GenMol neural sampler and then
performed RDKit sanitization, core checking, similarity, QED, and size scoring.
This audit invokes only RDKit parsing and exact-core matching. Parent core
objects are cached and reused across all of their actions and completions.

It should therefore be much faster than eight hours. A reasonable expectation
is tens of minutes to roughly two hours on one CPU core, depending on SMILES
parsing speed. Parallel workers can reduce wall time, but first benchmark a
10,000-record prefix:

```text
estimated_full_seconds =
    benchmark_seconds * 683448 / benchmark_record_count
```

Provide `--max-records` for timing only. The production audit must run without
that option and observe exactly 683,448 records. No GPU is needed.

If multiprocessing is added, partition work by parent rather than arbitrary
lines so each worker can reuse its parent/core cache. Each worker should return
counts and bounded mismatch samples; only the parent process writes the final
atomic report.

### Audit acceptance criteria

Training must not start unless:

- exactly 30,000 / 5,000 / 5,000 manifest parents are observed;
- all 40,000 parents have a uniquely extractable protected core;
- all stored core IDs match recomputed unmapped Murcko IDs;
- train, validation, and test recomputed core sets are pairwise disjoint;
- all persisted attachment indices and bonds validate;
- exactly 584,944 train and 98,504 validation records are replayed;
- no duplicate generation keys are found;
- recomputed exact-core values match stored values for every record.

If exact-core values disagree because the checker itself has changed, preserve
the original generations and produce a reviewed derived label file from the
stored outputs. Do not call GenMol merely to rescore existing molecules. If the
manifest itself contains non-unique cores or cross-split leakage, stop and
decide separately whether a replacement manifest is required.

---

## 2. Add the complete action feature vector

### Current limitation

`src/rollout_policy/datasets.py` currently calls the original 13-element
`action_features` signature and declares `FEATURE_SCHEMA_VERSION = 1`.
The graph already includes atom/bond chemistry, selected-fragment masks, two
attachment-node indicators, and a selected-cut edge indicator, but the global
action vector still lacks the recommended fragment and attachment descriptors.

No rollout regeneration is required. Every feature below is derived from parent
SMILES and persisted action metadata in the manifests.

### Complete global feature groups

Retain the existing parent/core values:

- parent MolWt, LogP, HBD, HBA, heavy-atom count;
- protected-core heavy-atom count.

Add fragment chemistry:

- fragment MolWt;
- fragment LogP;
- fragment TPSA;
- fragment HBD and HBA;
- heteroatom count;
- total formal charge;
- rotatable-bond count;
- ring count;
- aromatic-atom count;
- heavy-atom count;
- fragment/parent heavy-atom fraction;
- fragment/parent MolWt fraction;
- fragment/parent TPSA fraction;
- ring-fragment and terminal-fragment flags.

Add separate core-side and fragment-side attachment-atom descriptors:

- one-hot element for C, N, O, F, P, S, Cl, Br, I, and other;
- degree;
- total valence;
- formal charge;
- aromatic flag;
- ring flag;
- total hydrogen count;
- one-hot SP, SP2, SP3, and other hybridization.

Add local attachment environments:

- 128-bit radius-2 centered Morgan vector for the core attachment atom;
- 128-bit radius-2 centered Morgan vector for the fragment attachment atom.

Retain the selected cut-bond type for compatibility, but note that current
eligibility only allows non-ring single cut bonds; double/aromatic cut-bond
columns will normally have zero variance.

### Refactor `features.py`

Change the API to receive `EligibleAction` so it can use persisted parent atom
indices:

```python
FEATURE_SCHEMA_VERSION = 2
ATTACHMENT_FP_BITS = 128


def action_features(
    parent: Chem.Mol,
    core: Chem.Mol,
    action: EligibleAction,
) -> np.ndarray:
    validate_action_indices(parent, action)
    fragment = induced_fragment(parent, action.fragment_atom_indices)
    core_atom = parent.GetAtomWithIdx(action.core_attachment_atom)
    fragment_atom = parent.GetAtomWithIdx(action.fragment_attachment_atom)

    names: list[str] = []
    values: list[float] = []
    add_parent_core_features(names, values, parent, core)
    add_fragment_features(names, values, parent, fragment, action)
    add_attachment_atom_features(names, values, core_atom, "core_attachment")
    add_attachment_atom_features(
        names, values, fragment_atom, "fragment_attachment"
    )
    add_centered_morgan(
        names, values, parent, action.core_attachment_atom, "core_attachment"
    )
    add_centered_morgan(
        names, values, parent, action.fragment_attachment_atom,
        "fragment_attachment",
    )
    add_cut_bond_features(names, values, parent, action)

    if tuple(names) != FEATURE_NAMES:
        raise RuntimeError("feature name/order drift")
    return np.asarray(values, dtype=np.float32)
```

Build the induced fragment from the parent atom set, not
`action.fragment_smiles`, because that stored fragment contains a dummy
attachment atom:

```python
def induced_fragment(parent: Chem.Mol, atom_indices: tuple[int, ...]) -> Chem.Mol:
    smiles = Chem.MolFragmentToSmiles(
        parent,
        atomsToUse=list(atom_indices),
        canonical=True,
        isomericSmiles=False,
    )
    fragment = Chem.MolFromSmiles(smiles)
    if fragment is None:
        raise ValueError("failed to construct selected fragment")
    return fragment
```

Centered Morgan bits:

```python
def centered_morgan_bits(parent: Chem.Mol, atom_idx: int) -> list[float]:
    fp = AllChem.GetMorganFingerprintAsBitVect(
        parent,
        radius=2,
        nBits=ATTACHMENT_FP_BITS,
        fromAtoms=[int(atom_idx)],
        useChirality=False,
    )
    return [float(fp.GetBit(i)) for i in range(ATTACHMENT_FP_BITS)]
```

Define `FEATURE_NAMES` once from ordered tuples. Do not derive feature order
from a set or an unordered mapping. Update or remove
`features_from_action_record`, which currently uses the old function signature.

### Update all callers

At minimum update:

- `src/rollout_policy/datasets.py`;
- `scripts/train_value.py`;
- `scripts/evaluate_test.py`;
- `scripts/confirm_seeds.py`;
- `scripts/dry_run_local.py`;
- the future GAT predictor used by test evaluation.

For the GAT dataset:

```python
action = action_from_dict(action_row)
global_features = action_features(
    mol,
    core_mol_from_parent(mol),
    action,
)
```

Import `FEATURE_SCHEMA_VERSION` from `features.py`; do not define a second
version constant in `datasets.py`.

### Scaling mixed continuous and binary features

The current scaler standardizes every global column. This is undesirable for
rare Morgan bits because dividing by a very small standard deviation can
produce large values.

Export a parallel `FEATURE_KINDS` tuple:

```python
FEATURE_KINDS = (
    # one item per FEATURE_NAMES entry
    "continuous",
    "binary",
    # ...
)
```

Fit mean/scale on training actions only:

- continuous/count columns: standardize with train mean/std;
- binary one-hot and Morgan-bit columns: leave as 0/1;
- zero-variance columns: leave unchanged and report them.

Save feature names, kinds, means, scales, and schema version in the checkpoint.
At load time require exact equality with runtime schema.

With roughly 300-325 global features, the GAT remains modest. The wider input
increases only the first prediction-head layer; expected total size remains
approximately 60,000-65,000 parameters, still appropriate for 73,118 training
actions.

### Feature tests

Add `tests/test_features.py`:

- exact name/value length agreement;
- deterministic output for the same parent/action;
- different actions on one parent produce different attachment features;
- induced fragment has no dummy atom;
- attachment indices and bonds are validated;
- centered Morgan vectors are each exactly 128 bits;
- train-only scaling leaves binary fields in `{0, 1}`;
- schema mismatch prevents checkpoint loading.

---

## 3. Replace top-1-success early stopping

### Why top-1 observed success is a poor early-stopping signal

`validation_top1_success_rate` first selects one action per parent and then uses
that action's observed `n_successes / 8`. It therefore:

- discards the outcomes for every unselected validation action;
- changes discontinuously when two predicted actions exchange rank;
- is noisy because each action has only eight rollout trials;
- repeatedly exposes a discrete model-selection statistic over many epochs.

Keep it as a policy-facing validation report, but do not use it as the primary
checkpoint criterion.

### Recommended criterion: validation binomial NLL

Early-stop on the same proper scoring rule used for training:

```text
primary checkpoint metric = minimum validation binomial NLL
secondary report metric = validation top-1 selected-action success
```

Binomial NLL uses all 98,504 validation completion outcomes through their
12,313 action-level success/trial counts. It is smooth, statistically efficient,
and directly evaluates predicted per-completion success probabilities.
Maximizing that probability also maximizes any-of-16 success because
`1 - (1 - p)^16` is monotonic in `p`.

Modify `scripts/train_gat.py`:

```python
best = {
    "val_binomial_nll": float("inf"),
    "validation_top1_success_rate": None,
    "epoch": -1,
}
min_delta = float(gat_cfg.get("early_stopping_min_delta", 1e-5))

improved = held["binomial_nll"] < best["val_binomial_nll"] - min_delta
if improved:
    best = {
        "val_binomial_nll": held["binomial_nll"],
        "validation_top1_success_rate": top1,
        "epoch": epoch,
    }
    stale = 0
    save_checkpoint()
else:
    stale += 1
```

Recommended starting configuration:

```yaml
models:
  gatv2:
    early_stopping_metric: "val_binomial_nll"
    early_stopping_min_delta: 0.00001
    patience: 10
    max_epochs: 100
```

Report top-1 success, Brier score, calibration bins, and largest-fragment
agreement at the selected NLL checkpoint. Do not use test results to alter the
criterion.

Accumulate train loss weighted by trial count, as validation already does,
rather than averaging batch means equally:

```python
epoch_nll_sum += float(loss.item()) * float(batch.trials.sum().item())
epoch_trials += float(batch.trials.sum().item())
train_nll = epoch_nll_sum / max(epoch_trials, 1.0)
```

---

## 4. Make validation tie-breaking match production

Production `select_learned` uses:

```python
paired.sort(
    key=lambda item: (
        -item[1],
        item[0].canonical_order_key,
        item[0].action_id,
    )
)
```

Current validation instead takes:

```python
max(items, key=lambda item: (item[0], item[1]))
```

That picks the lexicographically largest action ID on equal logits and does not
use `canonical_order_key`. It disagrees with production.

### Preferred fix

Give every graph a canonical action rank derived from the manifest:

```python
ordered_actions = sorted(
    (action_from_dict(row) for row in parent["actions"]),
    key=lambda action: (action.canonical_order_key, action.action_id),
)
rank_by_action_id = {
    action.action_id: rank for rank, action in enumerate(ordered_actions)
}
graph.canonical_action_rank = torch.tensor(
    [rank_by_action_id[action.action_id]],
    dtype=torch.long,
)
```

Assert that action IDs are unique per parent. Collect ranks during validation
and select using:

```python
chosen = min(
    items,
    key=lambda item: (
        -item.logit,
        item.canonical_action_rank,
        item.action_id,
    ),
)
```

This exactly reproduces score descending, canonical key ascending, and action
ID ascending, provided the rank was built from that same key.

Better still, add a shared `select_learned_index` helper to
`src/rollout_policy/policies.py` and use it both in production and validation:

```python
def select_learned_index(actions, predicted_values) -> int:
    indices = range(len(actions))
    return min(
        indices,
        key=lambda index: (
            -float(predicted_values[index]),
            actions[index].canonical_order_key,
            actions[index].action_id,
        ),
    )
```

`select_learned` can return `actions[select_learned_index(...)]`. Validation
should group the original action objects and scores and call the same helper
where practical, eliminating duplicate policy semantics.

Add tests with deliberately equal predictions and actions whose lexical action
ID order differs from canonical order. Require validation and production to
select the same action.

---

## 5. Avoid materializing every action graph in RAM

### Current behavior

`load_action_graphs` builds and stores:

```text
73,118 training Data objects
12,313 validation Data objects
85,431 total PyG Data objects
```

Each action duplicates its parent atom matrix, edge index, edge attributes, and
multiple masks. Raw tensor bytes are not enormous, but tens of thousands of
small tensors and Python objects have substantial overhead. The current code
also creates a dense matrix of every global feature to fit the scaler.

The 43k-parameter model is not the RAM problem; eager graph materialization is.

### Option A: lazy graph dataset

Replace `load_action_graphs` with a `torch.utils.data.Dataset` that stores only:

- compact parent records keyed by parent ID;
- action dictionaries;
- action-level `(n_successes, n_trials)`;
- canonical action rank.

Construct the PyG `Data` object in `__getitem__`:

```python
class LazyActionGraphDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, label_path, transform=None):
        self.parents = {
            row["parent_id"]: row for row in iter_jsonl(manifest_path)
        }
        labels = list(iter_jsonl(label_path))
        self.examples = build_compact_examples(self.parents, labels)
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        parent = self.parents[example.parent_id]
        action = action_from_dict(example.action_row)
        mol = cached_parent_mol(parent["smiles"])
        features = action_features(mol, core_mol_from_parent(mol), action)
        if self.transform is not None:
            features = self.transform(features)
        graph = action_to_graph(parent, action, features)
        graph.successes = torch.tensor([example.successes], dtype=torch.float32)
        graph.trials = torch.tensor([example.trials], dtype=torch.float32)
        graph.canonical_action_rank = torch.tensor(
            [example.canonical_action_rank], dtype=torch.long
        )
        return graph
```

This sharply reduces persistent RAM use, but rebuilding RDKit molecules and
graphs every epoch may become CPU-bound. Use a small per-worker LRU cache of
parsed parent molecules, set DataLoader workers deliberately, and benchmark.
Each worker has its own cache; avoid an unbounded cache that recreates the
original memory problem.

### Option B: sharded on-disk graph cache — recommended

Because the same 85,431 action graphs are reused for many epochs, a better
balance is:

1. build each action graph once from the manifest;
2. write shards of perhaps 1,000-5,000 graphs under
   `artifacts_v2/graph_cache/schema_v2/`;
3. save an index mapping global example index to shard and offset;
4. load only the needed shard(s), retaining a one- or two-shard LRU cache;
5. shuffle example indices, not all graph objects;
6. hash manifests, label files, and feature schemas into cache metadata;
7. refuse stale caches after any schema or manifest change.

Example layout:

```text
artifacts_v2/graph_cache/schema_v2/
  metadata.json
  train-00000.pt
  train-00001.pt
  ...
  validation-00000.pt
  validation-00001.pt
```

Do not create 85,431 individual files. Shards avoid inode overhead and make
sequential validation efficient.

The cache builder is feature extraction, not molecular generation. It must not
import `FrozenGenMol`, and rebuilding it after a feature change does not require
rerunning `generate_labels.py`.

### Option C: share a parent graph between actions

The average parent has about 2.44 actions. Store atom and ordinary bond
chemistry once per parent, then represent each action only by:

- fragment mask;
- core/fragment attachment-node indices;
- selected-cut edge index;
- global action feature vector;
- binomial counts.

A custom collator can combine the parent base graph with each action inside the
batch. This can reduce duplicated graph storage by roughly the mean action
count, but it is more complex than sharding. Implement only after measuring
whether the sharded cache is insufficient.

### Streaming feature-scaler fitting

Do not require all graph objects merely to fit global normalization. Fit the
scaler in a streaming pass over training action feature vectors:

```python
scaler = StreamingMixedFeatureScaler(FEATURE_KINDS)
for feature_batch in iter_training_feature_batches(...):
    scaler.partial_fit(feature_batch)
scaler.finalize()
```

Alternatively use Welford mean/variance for continuous columns. Binary columns
remain unscaled. Save the resulting transform in graph-cache metadata and the
checkpoint.

### DataLoader recommendations

Start with:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
    pin_memory=torch.cuda.is_available(),
)
```

Tune workers against available host RAM and CPU. Multiple workers prefetch
batches, so excessive workers can increase memory. For validation, use
`shuffle=False` and preserve parent/action identifiers.

---

## 6. Implement GAT checkpoint inference before training

`scripts/evaluate_test.py` currently loads only `ValueMLP`. Do not train a GAT
checkpoint that cannot be used by the frozen evaluation path.

Create `src/rollout_policy/predictors.py` with a GAT predictor that:

1. loads model dimensions and all schema metadata from `gatv2.pt`;
2. requires exact node, edge, graph, global-feature, and label schema matches;
3. restores the mixed-feature scaler;
4. creates one graph per eligible action in input order;
5. batches those graphs;
6. applies `sigmoid` to model logits;
7. returns one probability per action in the original order;
8. lets `select_learned` perform the canonical production tie-break.

Record checkpoint hash, model type, feature schema, and predictions in
`selections.json`. Use the same predictor from `evaluate_test.py` and
`confirm_seeds.py`.

Add a checkpoint round-trip test before the expensive training run:

```python
scores_before = predictor_from_live_model.predict_actions(parent, actions)
save_checkpoint(...)
scores_after = load_predictor(checkpoint).predict_actions(parent, actions)
np.testing.assert_allclose(scores_before, scores_after, rtol=1e-6, atol=1e-7)
```

---

## 7. Suggested implementation order

1. Add and test `scripts/audit_exact_cores.py`.
2. Run Phase A over all 40,000 manifest parents.
3. Benchmark Phase B on 10,000 records, estimate runtime, then replay all
   683,448 records.
4. Freeze the audit reports and input hashes.
5. Refactor `features.py` to schema version 2 with the complete feature set.
6. Add feature and mixed-scaler tests.
7. Add canonical action rank and shared production tie-breaking.
8. Replace eager graph lists with a sharded cache or lazy dataset.
9. Change early stopping to validation binomial NLL.
10. Implement and test GAT checkpoint inference.
11. Build the graph cache from existing manifests and labels.
12. Run CPU smoke tests and one short GPU epoch.
13. Only then run full `scripts/train_gat.py`.

The one-epoch smoke report should include:

- exact parameter count;
- peak host RAM and GPU VRAM;
- graph-cache build time and size;
- examples/second and estimated full training time;
- train and validation binomial NLL;
- top-1 success as a secondary metric;
- confirmation that no test manifest or test result was read during training.

## 8. Final pre-training checklist

- [ ] No command invoked GenMol or regenerated completion records.
- [ ] All 40,000 manifest parents passed unique protected-core extraction.
- [ ] Recomputed core IDs are pairwise disjoint across splits.
- [ ] All 683,448 completion records were streamed and counted.
- [ ] Recomputed exact-core flags match stored flags.
- [ ] Global features use schema version 2 and the complete ordered feature list.
- [ ] Binary fingerprints remain unscaled.
- [ ] Validation checkpointing minimizes binomial NLL.
- [ ] Validation and production use identical canonical tie-breaking.
- [ ] Graph examples are lazy or sharded rather than all resident simultaneously.
- [ ] GAT checkpoint loading and prediction pass a round-trip test.
- [ ] The V2 test set remains unopened for model selection.
