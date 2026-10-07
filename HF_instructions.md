# Hugging Face ZeroGPU Space — remaining steps

`app.py` and `lightgbm_predict.py` are the Space application. They are torch-free and expect the frozen LightGBM checkpoint next to the app (or under `artifacts_v2/` when run from this repo).

Nothing below has been executed here (no extra pip installs, no Space create/push).

## 1. Files the Space needs

At the **root of the Space repo** (or this folder if you point the Space at it):

| File | Source |
|------|--------|
| `app.py` | this repo |
| `lightgbm_predict.py` | this repo |
| `artifacts/lightgbm_regressor.txt` | copy from `artifacts_v2/lightgbm_regressor.txt` |
| `artifacts/lightgbm_regressor.json` | copy from `artifacts_v2/lightgbm_regressor.json` |
| `requirements.txt` | create (see below) |
| `README.md` | Space card with YAML front matter (see below) |

`lightgbm_predict.default_model_paths()` looks for:

1. `./artifacts/lightgbm_regressor.txt` (+ `.json`)
2. `./artifacts_v2/lightgbm_regressor.txt` (+ `.json`) — convenient when testing inside this git repo

## 2. `requirements.txt` for the Space

Create with at least:

```text
gradio
spaces
lightgbm>=4,<5
rdkit
numpy
```

Do **not** add `torch` or `torch-geometric` for this LightGBM-only Space.

## 3. Space `README.md` front matter

```yaml
---
title: GenMol edit-site policy (LightGBM)
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
hardware: zerogpu
---
```

Adjust `sdk_version` to a current Gradio release if needed. Hardware **must** be ZeroGPU for the free personal-account Gradio hosting path (CPU Basic Gradio creation is gated behind PRO).

## 4. Create / configure the Space on HF

1. Account: verified email, age ≥ 30 days (free ZeroGPU hosting: up to 2 Spaces).
2. New Space → Gradio → hardware **ZeroGPU**.
3. Upload or git-push the files from §1–3.
4. Confirm the build installs deps and `app.py` launches.
5. Smoke-test a few SMILES in the UI (including one with no eligible pendants).

Optional CLI sketch:

```bash
huggingface-cli login
# create Space in the website UI as ZeroGPU, then:
git clone https://huggingface.co/spaces/<user>/<space-name>
cd <space-name>
cp /path/to/genmol-editing-policy/app.py .
cp /path/to/genmol-editing-policy/lightgbm_predict.py .
mkdir -p artifacts
cp /path/to/genmol-editing-policy/artifacts_v2/lightgbm_regressor.txt artifacts/
cp /path/to/genmol-editing-policy/artifacts_v2/lightgbm_regressor.json artifacts/
# add requirements.txt + README.md, then:
git add app.py lightgbm_predict.py artifacts requirements.txt README.md
git commit -m "Add LightGBM cut-site Space"
git push
```

## 5. Behaviour notes

- Protected core = Bemis–Murcko scaffold (derived, not user input).
- Actions = RDKit pendant cuts only (`require_safe_alignment=False`). No GenMol remask.
- Eligibility policy matches the experiment defaults: keep 2–4 actions after canonical ordering; otherwise return an error.
- `@spaces.GPU` wraps the predict function so the Space fits ZeroGPU hosting; the booster itself runs on CPU.

## 6. Optional local check (elsewhere)

When you have an env with the Space deps installed:

```bash
python -c "from lightgbm_predict import predict_cut_site, format_prediction as f; r=predict_cut_site('CCO'); print(f(r))"
python app.py
```

Do not rely on the current project `venv` until `lightgbm` / `gradio` / `spaces` are installed there on purpose.
