# GCP setup (GenMol remasking policy experiment)

Project: `glen-503819` (number `398729092111`)  
gcloud config: `glen` (`source ~/vits/switch-gcloud.sh glen`)  
Conda env: **`genmol-rollout`** (Python 3.10). If this env already exists on the GPU VM, do not recreate it — add only the V2 packages in §6.  
VM: any GPU box that already has that env (locked run used an L4). A new `g2-standard-8` / 100 GB VM is §3, and only if you do not already have one.  
Bucket: `gs://genmol-rollout-data` — optional, for result backups  
Repo: `git@github.com:Tom-Brenner/genmol-editing-policy.git`

Budget (from `plan.md`): ~$200 ceiling · alerts at $150 / $175 · stop GPU before ~$185 accrued · ~180 GPU-hour soft cap. Confirm **regional** L4 on-demand price before create; dollar cap wins.

Code and configs ship via **git** (commit on the laptop, `git push`, `git pull` on the VM). Do **not** scp experiment code onto the VM.

Large files are **not** in git and are **not** rsynced from the laptop. If the locked run already left them on the VM, keep those copies. Download only a path that is missing:

| Artifact | Size | Obtained on VM via |
| --- | --- | --- |
| Experiment code / configs | — | `git pull` in the existing clone (see §6a) |
| Ibuprofen smoke fixture | 1 line | committed: `data/fixtures/CHEMBL521.tsv` |
| GenMol V1 `model.ckpt` | ~1.4 GB | already on the VM, else `python scripts/download_genmol_ckpt.py` |
| ChEMBL 37 chemreps | ~279 MB gz | already on the VM, else `python scripts/download_chembl.py` |

Configs: locked `configs/experiment.yaml`; V2 `configs/experiment_v2.yaml`. Sampling stays `configs/genmol_sampling.yaml` (`bf16: true` → L4 uses bf16).

---

## Status (2026-09-21)

The locked 1,400-parent protocol is finished. Experiment V2 (`plan.md`, `configs/experiment_v2.yaml`) is the run in progress, in the same `genmol-rollout` env.

The V2 audit and the 30,000 / 5,000 / 5,000 manifest are already on the GPU VM. `manifests_v2/test.jsonl` is frozen (`chmod a-w`).

**Next step on the GPU VM:** `git pull`, then `scripts/audit_exact_cores.py` on the labels already written. Do not rerun `prepare_manifest.py` and do not regenerate completions. After that audit passes, `scripts/train_gat.py` trains the attention model from a sharded graph cache. Do not create a second conda env. Do not re-download chemreps or the GenMol checkpoint.

Two manifests, two directories:

| Tree | Where it sits | Split | What it is |
| --- | --- | --- | --- |
| `manifests/` | In git (`manifests/train.jsonl`, `validation.jsonl`, `test.jsonl`, `seeds.json`, `hashes.json`). Laptop and VM both have it after `git pull`. | 1,000 / 200 / 200, seed `20260919` | Locked protocol. `manifests/test.jsonl` is the held-out set behind `results/test`. Labels and the value-model checkpoint from that run are under `results/` and `artifacts/`. |
| `manifests_v2/` | On the GPU VM at `~/genmol-editing-policy/manifests_v2/` (`train.jsonl`, `validation.jsonl`, `test.jsonl`, `seeds.json`, `hashes.json`). Produced there by the audit and `prepare_manifest.py`; not in git. | 30,000 / 5,000 / 5,000, seed `20261001` | Experiment V2. `manifests_v2/test.jsonl` is mode `a-w`. Audit outputs sit beside it: `artifacts_v2/eligible_parents.jsonl`, `artifacts_v2/action_count_audit.json`. Parent table: `data/processed/parents_v2.parquet`. Later labels, checkpoints, and the V2 test go to `results_v2/` and `artifacts_v2/`. |

| Item | Notes |
| --- | --- |
| Private GitHub repo | `Tom-Brenner/genmol-editing-policy` (code + pinned GenMol source; no ckpt/chemreps). |
| `genmol-rollout` env | Python 3.10, GenMol requirements, editable GenMol + this repo. Reuse it. |
| ChEMBL 37 chemreps | `data/raw/chembl_37_chemreps.txt.gz` on the VM (~279 MB). The V2 manifest was drawn from this file. Do not download another copy. |
| GenMol V1 checkpoint | `vendor/genmol/checkpoints/model.ckpt`. Leave it in place. |
| V2 still to run | `git pull`, then `scripts/audit_exact_cores.py` on the finished labels. If it exits 0, `scripts/train_gat.py`. Do not rerun `prepare_manifest.py`. |

---

## 0 — Activate glen credentials (local)

```bash
cd ~/vits   # switch script lives here
source ./switch-gcloud.sh glen
gcloud config get-value project   # glen-503819
gcloud config get-value account
```

---

## 1 — Install & authenticate gcloud (local, one-time)

Skip if `gcloud` already works with the `glen` configuration.

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates gnupg curl

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
  | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list

sudo apt-get update && sudo apt-get install -y google-cloud-cli

gcloud auth login
gcloud config configurations activate glen   # or: source ~/vits/switch-gcloud.sh glen
gcloud config set project glen-503819
gcloud auth application-default login
gcloud auth application-default set-quota-project glen-503819
```

Set billing alerts on the project ($150 / $175) in Cloud Console before leaving the GPU on overnight.

---

## 2 — Bucket (local, one-time, optional)

Useful for pulling `results/` and `artifacts/` off the VM without scp. Same region as the VM.

```bash
source ~/vits/switch-gcloud.sh glen

export REGION=us-central1   # pick with L4 quota + price; must match VM region
gcloud storage buckets create gs://genmol-rollout-data \
  --location=$REGION \
  --project=glen-503819

# Optional: stage checkpoint from laptop instead of HF on VM
# gcloud storage cp \
#   ~/genmol-editing-policy/vendor/genmol/checkpoints/model.ckpt \
#   gs://genmol-rollout-data/checkpoints/model.ckpt
```

---

## 3 — Create VM (local)

Confirm zone has `g2-standard-8` / L4 and check on-demand $/h before running.

```bash
source ~/vits/switch-gcloud.sh glen

export VM=maps-decoder-train
export ZONE=us-west4-a   # change if needed; must be in $REGION above

gcloud compute instances create $VM \
  --zone=$ZONE \
  --machine-type=g2-standard-8 \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE \
  --metadata=install-nvidia-driver=True \
  --scopes=cloud-platform
```

Plan default is **on-demand** (preemption interrupts multi-day rollouts). Spot is cheaper but risky:

```bash
# Optional cheaper Spot (can STOP with ~30s notice):
#   --provisioning-model=SPOT --instance-termination-action=STOP
```

Stop / start (disk persists; GPU billing stops when TERMINATED):

```bash
gcloud compute instances stop  $VM --zone=$ZONE
gcloud compute instances start $VM --zone=$ZONE
```

---

## 4 — Grant VM access to bucket (local, one-time)

Only needed if using `gs://genmol-rollout-data`.

```bash
source ~/vits/switch-gcloud.sh glen

# Project number (should be 398729092111 for glen-503819):
gcloud projects describe glen-503819 --format="value(projectNumber)"

gcloud storage buckets add-iam-policy-binding gs://genmol-rollout-data \
  --member="serviceAccount:398729092111-compute@developer.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

---

## 5 — SSH into VM

```bash
source ~/vits/switch-gcloud.sh glen
export VM=maps-decoder-train ZONE=us-west4-a

gcloud compute instances start $VM --zone=$ZONE   # if TERMINATED
gcloud compute ssh $VM --zone=$ZONE
```

First SSH generates `~/.ssh/google_compute_engine` automatically.

---

## 6 — Environment

Prefer the block below. The from-scratch block is only for a machine that does not already have `genmol-rollout`.

### Incremental (existing `genmol-rollout` env)

Pull the V2 commit, then add pytest, LightGBM, and PyTorch Geometric. Do not rerun `conda create`, do not reinstall `vendor/genmol/env/requirements.txt`, and do not let pip replace the CUDA torch already in the env.

```bash
cd ~/genmol-editing-policy
conda activate genmol-rollout
git pull

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Note the torch version. After the next install it must still be a CUDA build.

python -m pip install -e '.[dev,models]'

python -c "import torch; assert torch.cuda.is_available(), 'pip replaced CUDA torch; reinstall the CUDA wheel'"
python -c "import lightgbm, torch_geometric; print('lightgbm', lightgbm.__version__, 'pyg', torch_geometric.__version__)"
```

`[dev]` is pytest. `[models]` is `lightgbm>=4,<5` and `torch-geometric>=2.5,<3`. GenMol, RDKit, and the checkpoint are already installed.

If `pip install -e '.[dev,models]'` swaps in a CPU torch, put the CUDA wheel back without touching the rest of the env. Match the version printed above (the GenMol pin is `torch==2.6.0`):

```bash
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; assert torch.cuda.is_available()"
```

Chemreps and the checkpoint should already be on disk. Confirm, and download only a missing file:

```bash
test -f data/raw/chembl_37_chemreps.txt.gz && echo "chemreps present"
test -f vendor/genmol/checkpoints/model.ckpt && echo "checkpoint present"
# only if a line above did not print:
# python scripts/download_chembl.py
# python scripts/download_genmol_ckpt.py
```

### From scratch (no `genmol-rollout` env yet)

Image is Ubuntu 22.04 + CUDA 12.9 + driver 580, but system Python is **3.12**. GenMol needs **Python 3.10**. Skip this subsection when `conda activate genmol-rollout` already works.

```bash
# Drivers / GPU
nvidia-smi   # wait until driver is ready after first boot

# System deps (RDKit / SAFE)
sudo apt-get update
sudo apt-get install -y git curl wget build-essential \
  libsm6 libxext6 libxrender-dev

# Micromamba or Miniconda if absent, then:
# if `conda` is not on PATH in this SSH session:
#source /opt/conda/etc/profile.d/conda.sh
conda create -y -n genmol-rollout python=3.10
conda activate genmol-rollout

# GitHub deploy key (private repo)
ssh-keygen -t ed25519 -C "gcp-vm-genmol" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
# GitHub → Tom-Brenner/genmol-editing-policy → Settings → Deploy keys → allow read
ssh -T git@github.com

git clone git@github.com:Tom-Brenner/genmol-editing-policy.git ~/genmol-editing-policy
cd ~/genmol-editing-policy

python -m pip install --upgrade pip
# safe/wandb still import pkg_resources; setuptools 82+ removed it
python -m pip install 'setuptools>=65,<82'
python -m pip install -r vendor/genmol/env/requirements.txt
# Prefer a CUDA wheel matching the image (cu124/cu128/cu129); if pip pulls CPU torch, reinstall:
# python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e vendor/genmol
python -m pip install -e '.[dev,models]'

# GenMol SAFE import fix if needed (see vendor/genmol/README.md troubleshooting):
#   comment stock safe/__init__.py lines; append:
#   from .converter import SAFEConverter, decode, encode

python -c "import torch; print(torch.cuda.is_available(), torch.__version__, torch.cuda.get_device_name(0))"
python -c "from rollout_policy.precision import resolve_precision; print(resolve_precision(prefer_bf16=True))"
# Expect: bf16 on L4
```

### Downloads (on VM, from-scratch only)

Skip when §6 incremental already found the chemreps file and the checkpoint.

Accept the [NVIDIA Open Model License](https://huggingface.co/nvidia/NV-GenMol-89M-v1) on Hugging Face first (browser / token if gated).

The one-molecule smoke test does **not** call ChEMBL REST. Ibuprofen is committed at `data/fixtures/CHEMBL521.tsv`. Bulk parents come from the chemreps **FTP** dump (`download_chembl.py`), which is unrelated to the molecule REST API.

Production ChEMBL REST (`https://www.ebi.ac.uk/chembl/api/data/...`) often returns HTTP 500 / timeouts. If you still want to refresh a fixture, `download_chembl_molecule.py` tries production, then staging (`wwwdev`), then the committed TSV, then the local chemreps dump. `--offline` skips REST.

```bash
cd ~/genmol-editing-policy
mkdir -p vendor/genmol/checkpoints data/raw artifacts

python scripts/download_genmol_ckpt.py
python scripts/download_chembl.py

# Smoke test (uses git fixture; no REST)
python scripts/test_one_molecule.py --bf16 true --parent-tsv data/fixtures/CHEMBL521.tsv

# Optional: refresh fixture without REST
# python scripts/download_chembl_molecule.py --chembl-id CHEMBL521 --offline

# Fill provenance
cp artifacts/provenance.template.json artifacts/provenance.json
# edit: region, machine type, nvidia-smi, CUDA, torch, python, pip freeze path
python -m pip freeze | sort > artifacts/pip-freeze.txt
```

Optional pull of a staged checkpoint from GCS:

```bash
gcloud storage cp \
  gs://genmol-rollout-data/checkpoints/model.ckpt \
  ~/genmol-editing-policy/vendor/genmol/checkpoints/model.ckpt
```

---

## 6b — Tests (on the GPU VM)

Run these after §6, in `genmol-rollout`, from `~/genmol-editing-policy`. No second environment. `tests/test_gpu_models.py` skips itself without CUDA, so run it here rather than on a laptop.

```bash
conda activate genmol-rollout
cd ~/genmol-editing-policy

python -m pytest \
  tests/test_labels.py \
  tests/test_eligibility_counts.py \
  tests/test_core_check.py \
  tests/test_graph_model.py \
  tests/test_gpu_models.py \
  -q
```

| File | What it checks | Needs |
| --- | --- | --- |
| `tests/test_labels.py` | `completion_success` matches `parent_success`; schema-v2 action counts; binomial NLL ranks a correct logit above a wrong one | CPU |
| `tests/test_eligibility_counts.py` | Raising the action cap does not invent pendants; cap 4 truncates in canonical order; min=4 drops a 3-action parent; ambiguous SAFE and round-trip failures are counted | CPU, no chemreps |
| `tests/test_core_check.py` | Locked QED-threshold and heavy-atom budget tests | CPU |
| `tests/test_graph_model.py` | Two actions yield different graphs; core/fragment masks; selected cut marked both ways; CPU GATv2 backward; checkpoint reload | CPU + `torch-geometric` |
| `tests/test_gpu_models.py` | Binomial backward, a short shallow-MLP step, and a GATv2 step on CUDA | NVIDIA GPU + `torch-geometric` |

`test_graph_model.py` and the GATv2 part of `test_gpu_models.py` skip until `[models]` is installed. Do not treat a skip of `test_gpu_models.py` as a pass: `nvidia-smi` and `torch.cuda.is_available()` must be true on this machine.

These three commands already finished on the VM. They wrote `manifests_v2/` and froze the held-out file. Do not run them again.

```bash
python scripts/audit_action_counts.py --config configs/experiment_v2.yaml
python scripts/prepare_manifest.py --config configs/experiment_v2.yaml
chmod a-w manifests_v2/test.jsonl
```

Ctrl+C stops either scan, and each scan also stops after 120,000 chemreps rows. The audit stops earlier, at 40,000 parents with at least two verified pendants, and writes `artifacts_v2/eligible_parents.jsonl` and `artifacts_v2/action_count_audit.json`. `--offline-no-safe` is not a basis for the action cap. Label generation is §7 Phase 2.

---

## 6a — Updating code (laptop git → VM pull)

Edit, commit, and push from the **laptop**. On the VM only `git pull`. The deploy key is read-only.

`git add -A` is not a required step and it does not re-upload files GitHub already has. `git push` sends commits the remote is missing. Stage only the paths you changed. Skip the laptop block entirely when `git status` on the laptop is clean: the VM just needs `git pull`.

```bash
# laptop, only when there are new edits
cd ~/genmol-editing-policy
git add path/to/changed-file
git commit -m "..."
git push

# VM
cd ~/genmol-editing-policy
conda activate genmol-rollout
git status          # discard any leftover scp/local edits before pulling
git pull
```

If `git pull` refuses because of leftover local copies (e.g. from an earlier scp), discard **tracked** edits and **untracked** files separately — `git restore` cannot touch untracked paths:

```bash
git restore scripts/download_chembl_molecule.py src/rollout_policy/chembl.py
rm -f tests/test_chembl_lookup.py
git pull
git log -1 --oneline   # expect the laptop tip, not 94bb6c0
```

---

## 7 — Experiment phases with tmux (on VM)

Phases below are Experiment V2: 30,000 train / 5,000 validation / 5,000 test, config `configs/experiment_v2.yaml`. Every command passes that config, so reads come from `manifests_v2/` and writes go to `results_v2/` and `artifacts_v2/`. Omitting `--config` falls back to `configs/experiment.yaml` and the locked `manifests/` split.

`manifests/` (1,000 / 200 / 200, seed `20260919`) stays the locked protocol. Its test result is `results/test`. Leave that tree in place.

Always `stop` the VM when idle. Run long jobs in tmux.

```bash
tmux new -s genmol
conda activate genmol-rollout   # or: micromamba activate genmol-rollout
cd ~/genmol-editing-policy
```

### Phase 0 — throughput gate (already passed)

The frozen sampler already cleared the gate on this checkpoint (validity at least 25%, at most 20 s/completion). Repeat only if the VM image, `vendor/genmol/checkpoints/model.ckpt`, or `configs/genmol_sampling.yaml` changes. A repeat must exclude parents already in `manifests_v2/` and must not write into `results/`:

```bash
python scripts/probe_genmol.py --config configs/experiment_v2.yaml --bf16 true \
  --output results_v2/probes.jsonl 2>&1 | tee ~/probe.log
python scripts/audit_records.py results_v2/probes.jsonl --sample 50 \
  --output artifacts_v2/audit_sample.jsonl
```

### Phase 1 — manifest (already done)

`prepare_manifest.py --config configs/experiment_v2.yaml` has written the core-disjoint 30,000 / 5,000 / 5,000 split under `manifests_v2/`, and `chmod a-w manifests_v2/test.jsonl` has frozen the held-out file. Do not rebuild it. Do not open `manifests_v2/test.jsonl` until the candidate set and the validation rule in `plan.md` are frozen.

### Phase 2 — labels, then the V2 models

The GPU hours go into rollouts. Eight completions per train or validation action. The same files then train every V2 candidate. Outputs: `results_v2/labels/{split}.jsonl` and `results_v2/labels/{split}_action_rhat.jsonl`. Train and validation only.

```bash
python scripts/generate_labels.py --config configs/experiment_v2.yaml \
  --split validation --rollouts 8 2>&1 | tee ~/labels_val.log
python scripts/generate_labels.py --config configs/experiment_v2.yaml \
  --split train --rollouts 8 2>&1 | tee ~/labels_train.log
```

`scripts/train_value.py` fits the locked two-layer MLP with weighted Huber loss on the original 13 descriptors. That is the model already reported for the 1,400-parent protocol. Leave it on `configs/experiment.yaml`. Running it on `manifests_v2/` does not produce a V2 candidate.

After the labels exist, fit the four models in `experiment-v2-handoff.md` and `plan.md`, all on the schema-v2 binomial labels and the richer fragment/attachment features. Selection uses validation top-1 observed success rate. None of them open `manifests_v2/test.jsonl`.

| Candidate | What it is | Trainer in the handoff |
| --- | --- | --- |
| `gatv2` | Action-conditioned GATv2. Each example is one parent/action graph: atom features, both directed bonds, a selected-cut flag on the attachment edge, core and fragment masks. Three layers, four heads, 64 channels, then a binomial logit. This is the attention model. | `scripts/train_gat.py` |
| `shallow_mlp` | New `TabularMLP`: one 64-unit hidden layer, dropout 0.15, weight decay `1e-4`, binomial NLL, on the V2 feature vector. A deeper net on the old 13 descriptors is not a substitute; the handoff’s point is that depth cannot invent the missing atom types and functional groups. | `scripts/train_tabular_models.py` |
| `lightgbm_regressor` | Boosted tree. Pointwise regression on `n_successes / n_trials`, sample weight `n_trials`. LightGBM is the tree library pinned in the handoff (same role as XGBoost). | `scripts/train_tabular_models.py` |
| `lightgbm_ranker` | LightGBM LambdaRank over actions of one parent. Scores are ranks, not probabilities. | `scripts/train_tabular_models.py` |

Label generation for train and validation is finished. Leave those JSONL files in place. The next CPU job replays scaffold identity on all 40,000 manifest parents and exact-core flags on the stored completions. It does not call GenMol.

```bash
python scripts/audit_exact_cores.py --config configs/experiment_v2.yaml --max-records 10000
# timing only; then the full audit, which must see 683448 records:
python scripts/audit_exact_cores.py --config configs/experiment_v2.yaml --workers 4 \
  2>&1 | tee ~/audit_exact_cores.log
```

Train only if that command exits 0. `scripts/train_gat.py` then builds `artifacts_v2/graph_cache/schema_v2/` (one shard in memory at a time) and checkpoints on validation binomial NLL. Top-1 success is reported beside that checkpoint.

```bash
python scripts/train_gat.py --config configs/experiment_v2.yaml \
  2>&1 | tee ~/train_gat.log
```

`src/rollout_policy/predictors.py` loads `artifacts_v2/best.pt` for the GATv2 test and confirmation. The tabular trainer does not open `manifests_v2/test.jsonl`.

### Phase 3 — held-out test (5,000 parents, 16 completions)

Phase 3 is done. It opened `manifests_v2/test.jsonl` once and scored the frozen GATv2 checkpoint (`artifacts_v2/best.pt`) plus random and largest-fragment, 16 completions each. Do not run it again: a second pass rewrites `results_v2/test/selections.json` and `results_v2/test/metrics.json`. The command that produced the locked files was:

```bash
python scripts/evaluate_test.py --config configs/experiment_v2.yaml --rollouts 16 \
  2>&1 | tee ~/eval_test.log
chmod -R a-w results_v2/test
```

Metrics then land in `results_v2/test` and `artifacts_v2/test_metrics.json`.

### Phase 4 — seed confirmation

Repeat the GATv2 policy and largest-fragment with 16 new samples and no further fitting. `confirm_seeds.py` reuses `results_v2/test/selections.json` and loads `artifacts_v2/best.pt`. It writes `results_v2/confirmation/`, not the Phase 3 files:

```bash
python scripts/confirm_seeds.py --config configs/experiment_v2.yaml \
  --policies learned largest_fragment --rollouts 16 \
  2>&1 | tee ~/confirm.log
```

### Tabular ablation — MLP and LightGBM

Random and largest-fragment do not depend on the learned checkpoint. Their Phase 3 completions stay in `results_v2/test/random.jsonl` and `results_v2/test/largest_fragment.jsonl`. Phase 4 largest-fragment stays in `results_v2/confirmation/largest_fragment.jsonl`. `scripts/evaluate_ablation.py` reads those files and does not write them. A new completion is sampled only when the tabular model picks a cut that is not already stored for the same phase, parent, action, candidate index, and seed (`test_base_seed` or `confirmation_base_seed`, 16 rollouts).

Train first. This does not open the test manifest. Checkpoints land in `artifacts_v2/shallow_mlp.pt` and `artifacts_v2/lightgbm_regressor.txt` with `lightgbm_regressor.json`.

```bash
python scripts/train_tabular_models.py --config configs/experiment_v2.yaml \
  2>&1 | tee ~/train_tabular.log
```

After Phase 3 has all 5,000 parents in both baseline files, score the test schedule:

```bash
python scripts/evaluate_ablation.py --config configs/experiment_v2.yaml \
  --model shallow_mlp --schedule test --rollouts 16 \
  2>&1 | tee ~/ablation_mlp_test.log
python scripts/evaluate_ablation.py --config configs/experiment_v2.yaml \
  --model lightgbm_regressor --schedule test --rollouts 16 \
  2>&1 | tee ~/ablation_lgb_test.log
```

After Phase 4 has all 5,000 parents in `results_v2/confirmation/largest_fragment.jsonl`, repeat with the confirmation seed:

```bash
python scripts/evaluate_ablation.py --config configs/experiment_v2.yaml \
  --model shallow_mlp --schedule confirmation --rollouts 16 \
  2>&1 | tee ~/ablation_mlp_confirm.log
python scripts/evaluate_ablation.py --config configs/experiment_v2.yaml \
  --model lightgbm_regressor --schedule confirmation --rollouts 16 \
  2>&1 | tee ~/ablation_lgb_confirm.log
```

Metrics are `results_v2/ablation/<model>/{test,confirmation}_metrics.json` and `artifacts_v2/ablation_<model>_<schedule>_metrics.json`. The GATv2 Phase 3 report stays the primary test. These runs are the feature ablation against the same molecules and seeds.

- **Detach:** `Ctrl-b d`
- **Reattach:** `tmux attach -t genmol`

Periodic backup to GCS (resume-safe; cheap):

```bash
gcloud storage rsync ~/genmol-editing-policy/results \
  gs://genmol-rollout-data/results --recursive
gcloud storage rsync ~/genmol-editing-policy/artifacts \
  gs://genmol-rollout-data/artifacts --recursive
gcloud storage rsync ~/genmol-editing-policy/manifests \
  gs://genmol-rollout-data/manifests --recursive
gcloud storage rsync ~/genmol-editing-policy/results_v2 \
  gs://glen-train-data/results_v2 --recursive
gcloud storage rsync ~/genmol-editing-policy/artifacts_v2 \
  gs://glen-train-data/artifacts_v2 --recursive
gcloud storage rsync ~/genmol-editing-policy/manifests_v2 \
  gs://glen-train-data/manifests_v2 --recursive
```

---

## 8 — VM lifecycle

```bash
source ~/vits/switch-gcloud.sh glen
export VM=maps-decoder-train ZONE=us-central1-a

gcloud compute instances stop   $VM --zone=$ZONE
gcloud compute instances start  $VM --zone=$ZONE
gcloud compute ssh $VM --zone=$ZONE
gcloud compute instances delete $VM --zone=$ZONE   # done for good
gcloud compute instances list --sort-by=zone
```

After restart: `gcloud compute ssh` → `tmux attach -t genmol`. Label scripts resume missing `(phase, parent, action, candidate, seed)` keys.

---

## 9 — Pull results locally

```bash
source ~/vits/switch-gcloud.sh glen
mkdir -p ~/genmol-editing-policy-gcp/{results,artifacts,manifests}

gcloud storage rsync \
  gs://genmol-rollout-data/results \
  ~/genmol-editing-policy-gcp/results --recursive

gcloud storage rsync \
  gs://genmol-rollout-data/artifacts \
  ~/genmol-editing-policy-gcp/artifacts --recursive

gcloud storage rsync \
  gs://genmol-rollout-data/manifests \
  ~/genmol-editing-policy-gcp/manifests --recursive

gcloud storage rsync \
  gs://genmol-rollout-data/results_v2 \
  ~/genmol-editing-policy-gcp/results_v2 --recursive

gcloud storage rsync \
  gs://genmol-rollout-data/artifacts_v2 \
  ~/genmol-editing-policy-gcp/artifacts_v2 --recursive

gcloud storage rsync \
  gs://genmol-rollout-data/manifests_v2 \
  ~/genmol-editing-policy-gcp/manifests_v2 --recursive
```

Or scp:

```bash
gcloud compute scp --recurse \
  maps-decoder-train:~/genmol-editing-policy/results \
  ~/genmol-editing-policy-gcp/results \
  --zone=$ZONE
```

---

## Cost / stop checklist

- [ ] Billing alerts $150 / $175
- [ ] Prefer **stop** over leave-idle GPU
- [ ] Hard stop before ~$185 accrued
- [ ] JSONL / checkpoints written after each batch (scripts append + fsync)
- [ ] Do not train GenMol; do not use A100
- [ ] Freeze `configs/genmol_sampling.yaml` before Phase 3
