# GCP setup (GenMol remasking policy experiment)

Project: `glen-503819` (number `398729092111`)  
gcloud config: `glen` (`source ~/vits/switch-gcloud.sh glen`)  
VM: **`genmol-rollout`** — **not created yet** (target: `g2-standard-8` / L4, **100 GB** boot disk)  
Bucket: `gs://genmol-rollout-data` — **not created yet** (optional; for result backups)  
Repo: `git@github.com:Tom-Brenner/genmol-editing-policy.git`

Budget (from `plan.md`): ~$200 ceiling · alerts at $150 / $175 · stop GPU before ~$185 accrued · ~180 GPU-hour soft cap. Confirm **regional** L4 on-demand price before create; dollar cap wins.

Code and configs ship via **git** (commit on the laptop, `git push`, `git pull` on the VM). Do **not** scp experiment code onto the VM.

Large files are **not** in git and are **not** rsynced from the laptop. Download on the VM:

| Artifact | Size | Obtained on VM via |
| --- | --- | --- |
| Experiment code / configs | — | `git clone` then `git pull` (see §6a) |
| Ibuprofen smoke fixture | 1 line | committed: `data/fixtures/CHEMBL521.tsv` |
| GenMol V1 `model.ckpt` | ~1.4 GB | `python scripts/download_genmol_ckpt.py` |
| ChEMBL 37 chemreps | ~279 MB gz | `python scripts/download_chembl.py` (FTP; independent of ChEMBL REST) |

Configs: `configs/experiment.yaml`, `configs/genmol_sampling.yaml` (`bf16: true` → L4 uses bf16).

---

## Status (2026-09-21)

### Done

| Item | Notes |
| --- | --- |
| Private GitHub repo | `Tom-Brenner/genmol-editing-policy` (code + pinned GenMol source; no ckpt/chemreps). |
| Local smoke | `--bf16` fp16 fallback on 1080 Ti; one-molecule SAFE remask passed. |
| Disk sizing | **100 GB** boot disk (150 GB optional headroom). |
| This doc | Modeled on `~/Dropbox/DiffSinger/GCP_SETUP.md`. |

### Not done — pick up here

**Next step: §1 credentials (if needed) → §2 bucket (optional) → §3 create VM → §5 SSH → §6 VM setup.**

Do **not** upload local `vendor/genmol/checkpoints/model.ckpt` unless you prefer GCS staging over Hugging Face on the VM.

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

export VM=genmol-rollout
export ZONE=us-central1-a   # change if needed; must be in $REGION above

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
export VM=genmol-rollout ZONE=us-central1-a

gcloud compute instances start $VM --zone=$ZONE   # if TERMINATED
gcloud compute ssh $VM --zone=$ZONE
```

First SSH generates `~/.ssh/google_compute_engine` automatically.

---

## 6 — VM setup (on VM, one-time)

Image is Ubuntu 22.04 + CUDA 12.9 + driver 580, but system Python is **3.12**. GenMol needs **Python 3.10**.

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
python -m pip install -e ".[dev]"

# GenMol SAFE import fix if needed (see vendor/genmol/README.md troubleshooting):
#   comment stock safe/__init__.py lines; append:
#   from .converter import SAFEConverter, decode, encode

python -c "import torch; print(torch.cuda.is_available(), torch.__version__, torch.cuda.get_device_name(0))"
python -c "from rollout_policy.precision import resolve_precision; print(resolve_precision(prefer_bf16=True))"
# Expect: bf16 on L4
```

### Downloads (on VM)

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

## 6a — Updating code (laptop git → VM pull)

Edit, commit, and push from the **laptop**. On the VM only `git pull`. The deploy key is read-only.

```bash
# laptop
cd ~/genmol-editing-policy
git add -A
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

Always `stop` the VM when idle. Run long jobs in tmux.

```bash
tmux new -s genmol
conda activate genmol-rollout   # or: micromamba activate genmol-rollout
cd ~/genmol-editing-policy
```

### Phase 0 — throughput gate (max ~8 GPU-h)

```bash
python scripts/probe_genmol.py --config configs/experiment.yaml --bf16 true \
  2>&1 | tee ~/probe.log
python scripts/audit_records.py results/probes.jsonl --sample 50
# Manual audit: zero core/mapping errors; validity ≥25%; ≤20 s/completion
```

### Phase 1 — manifest

The frozen 1000/200/200 split is committed under `manifests/` (git pull). Rebuild only if you intend to change the seed or filters:

```bash
python scripts/prepare_manifest.py --config configs/experiment.yaml --seed 20260919
chmod a-w manifests/test.jsonl
```

### Phase 1–2 — labels + value model

```bash
python scripts/generate_labels.py --split validation --rollouts 8 2>&1 | tee ~/labels_val.log
python scripts/generate_labels.py --split train --rollouts 8       2>&1 | tee ~/labels_train.log
python scripts/train_value.py --config configs/experiment.yaml     2>&1 | tee ~/train_value.log
```

### Phase 3 — locked test

```bash
python scripts/evaluate_test.py --config configs/experiment.yaml --rollouts 16 \
  2>&1 | tee ~/eval_test.log
chmod -R a-w results/test
```

### Phase 4 — seed confirmation

```bash
python scripts/confirm_seeds.py --policies learned largest_fragment --rollouts 16 \
  2>&1 | tee ~/confirm.log
```

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
```

---

## 8 — VM lifecycle

```bash
source ~/vits/switch-gcloud.sh glen
export VM=genmol-rollout ZONE=us-central1-a

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
```

Or scp:

```bash
gcloud compute scp --recurse \
  genmol-rollout:~/genmol-editing-policy/results \
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
