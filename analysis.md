# Feature preference and dominance analysis

`scripts/analyze_feature_preferences.py` measures which eligible pendants the
three learned policies prefer and how strongly the two tabular policies rely on
groups of input descriptors.

The script is analysis-only. It does not call GenMol, regenerate completions,
modify checkpoints, or write into the locked Phase 3 and Phase 4 directories.
Its first operation is a strict preflight covering Python dependencies, every
required artifact, non-empty files, and JSON/JSONL parseability. No model is
loaded and no output is written unless that check passes.

## Environment

Use the existing `genmol-rollout` environment if it is still available. The
project's `models` extra contains the only additions beyond the base project
environment: LightGBM and PyTorch Geometric.

```bash
cd ~/genmol-editing-policy
conda activate genmol-rollout
python -m pip install -e '.[models]'
```

No separate `shap` installation is needed. LightGBM computes exact TreeSHAP
contributions through its native `pred_contrib` interface. On the original GPU
VM, confirm the environment before downloading or recomputing anything:

```bash
python - <<'PY'
import lightgbm, rdkit, torch, torch_geometric
print("lightgbm", lightgbm.__version__)
print("rdkit", rdkit.__version__)
print("torch", torch.__version__)
print("torch-geometric", torch_geometric.__version__)
print("cuda", torch.cuda.is_available())
PY
```

CPU execution is supported. CUDA substantially shortens GATv2 rescoring:

```bash
python scripts/analyze_feature_preferences.py --device cuda
```

## Required remote artifacts

The complete analysis requires:

- `manifests_v2/validation.jsonl`
- `manifests_v2/test.jsonl`
- `manifests_v2/seeds.json`
- `results_v2/labels/validation_action_rhat.jsonl`
- `artifacts_v2/gatv2.pt`
- `artifacts_v2/shallow_mlp.pt`
- `artifacts_v2/lightgbm_regressor.txt`
- `artifacts_v2/lightgbm_regressor.json`
- the three `artifacts_v2/*_validation_report.json` files
- `results_v2/test/{metrics,selections}.json`
- `results_v2/confirmation/metrics.json`
- each tabular ablation's test and confirmation metrics and selections JSON

Raw completion JSONL files, the graph cache, the Phase 0 probe, the training
manifest, and training completion records are not needed. They are required
only for a separate analysis of completion-level failure mechanisms or exact
training reproduction.

The setup documentation contains two V2 bucket names. The backup commands name
`gs://glen-train-data`, whereas an older pull example names
`gs://genmol-rollout-data`. Locate the populated bucket before syncing:

```bash
source ~/vits/switch-gcloud.sh glen
gcloud storage ls gs://glen-train-data/manifests_v2/
```

For a compact directory-level sync from the current V2 backup:

```bash
cd ~/genmol-editing-policy
B=gs://glen-train-data

gcloud storage rsync "$B/manifests_v2" manifests_v2 \
  --recursive --exclude='train\.jsonl|hashes\.json'

mkdir -p results_v2/labels
gcloud storage cp \
  "$B/results_v2/labels/validation_action_rhat.jsonl" \
  results_v2/labels/

gcloud storage rsync "$B/artifacts_v2" artifacts_v2 \
  --recursive \
  --exclude='graph_cache/.*|eligible_parents\.jsonl|.*audit.*\.jsonl'

for section in test confirmation ablation; do
  gcloud storage rsync \
    "$B/results_v2/$section" "results_v2/$section" \
    --recursive --exclude='.*\.jsonl'
done
```

The artifact sync deliberately excludes the graph cache but may download a few
small audit summaries. Exact `gcloud storage cp` commands can reduce that
further, but are more fragile if report names change.

## Preflight and execution

Run the preflight by itself first:

```bash
python scripts/analyze_feature_preferences.py --preflight-only
```

The command reports all missing dependencies and files together. Once it
passes, run:

```bash
python scripts/analyze_feature_preferences.py \
  --device cuda \
  --permutation-repeats 20
```

For a quick implementation check, use a subset. Archive verification is
intentionally skipped for subsets:

```bash
python scripts/analyze_feature_preferences.py \
  --max-parents 100 \
  --permutation-repeats 2
```

Full outputs are written to `results_v2/feature_analysis/`:

- `preflight.json`: dependency and input audit.
- `model_summary.json`: validation ranking metrics and agreement with the
  archived test selections. GATv2 may report a few soft near-tie flips under
  CUDA; tabular models must match exactly.
- `grouped_permutation.csv`: validation top-1 success loss, oracle-regret
  increase, and selection-change rate after group permutation.
- `feature_preferences.csv`: within-parent selected-minus-alternative
  descriptor contrasts on validation and test.
- `model_agreement.csv`: fraction of parents for which model pairs select the
  same action.
- `lightgbm_feature_importance.csv`: split and gain importance.
- `lightgbm_tree_shap.csv`: mean absolute and signed TreeSHAP contributions on
  all validation actions.

## Statistical interpretation

Grouped permutation is the primary feature-reliance analysis. Fragment and
attachment groups are permuted among eligible actions of the same parent, so
the test asks whether that information helps rank sites rather than predict
which parents are generally easy. Parent-level context is constant across a
parent's actions and is therefore permuted between parents as a block.

The reported loss is

```text
original validation top-1 observed success
    - permuted validation top-1 observed success.
```

A positive value indicates model reliance on the group. It is not a causal
effect, and correlated groups divide or duplicate importance. The validation
target is based on eight stochastic completions per action, so small
differences should not be overinterpreted.

`feature_preferences.csv` describes what the models select, not why. For each
parent and descriptor it subtracts the mean of the unselected actions from the
selected action. Morgan-bit rows are especially difficult to interpret because
multiple environments can hash to one bit. Any chemical interpretation of an
important bit should regenerate RDKit `bitInfoMap` assignments and report all
colliding environments.

The held-out test has completions only for actions selected by the evaluated
policies. It therefore cannot supply unbiased counterfactual outcomes for every
eligible test action. Feature dominance is evaluated against validation
all-action labels; test manifests are used only for preference descriptions and
verification that rescoring reproduces the archived selections. A small number
of GATv2 selection flips (default: up to 1% of parents, or score margin ≤
`--near-tie-atol`) are recorded as soft mismatches and do not abort the run;
larger disagreement rates, or any tabular mismatch, still fail.
