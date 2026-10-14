# Editing-site selection for frozen GenMol remasking

This repository evaluates policies that select one eligible peripheral fragment
to remask with frozen [GenMol](https://github.com/NVIDIA/GenMol) V1
(Lee et al., ICML 2025). The selected edit must preserve the Bemis–Murcko core
and satisfy similarity, size, validity, and QED-improvement constraints. The
learned LightGBM policy is compared with random and largest-fragment baselines
on a core-disjoint ChEMBL holdout.

- [Preprint](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15010340/v2)
- [Interactive Hugging Face demo](https://huggingface.co/spaces/one-more-beauty/genmol-edit-site-policy)

The demo scores eligible cut sites; it does not generate molecules or make
claims about potency, safety, or synthesis.

## Reproduce the code path

Python 3.10+ is required. Create an environment with RDKit available, then
install the pinned GenMol source and this project:

```bash
git clone https://github.com/Tom-Brenner/genmol-editing-policy.git
cd genmol-editing-policy

python -m pip install --upgrade pip
python -m pip install -r vendor/genmol/env/requirements.txt
python -m pip install -e vendor/genmol
python -m pip install -e ".[dev,models]"
python -m pytest -q
```

The lightweight local smoke path requires neither the GenMol checkpoint nor the
ChEMBL bulk download:

```bash
python scripts/dry_run_local.py --bf16 false
```

## Full experiment

The experiment uses ChEMBL 37 and the gated GenMol V1 checkpoint. Download
them locally; neither is committed to this repository:

```bash
python scripts/download_genmol_ckpt.py
python scripts/download_chembl.py
```

Accept the [NVIDIA Open Model License](https://huggingface.co/nvidia/NV-GenMol-89M-v1)
before downloading the checkpoint. ChEMBL is distributed under
[CC BY-SA 3.0](https://www.ebi.ac.uk/chembl/).

Run the V2 workflow in order. Do not inspect or tune on the test split before
the model-selection step is frozen.

```bash
python scripts/audit_action_counts.py --config configs/experiment_v2.yaml
python scripts/prepare_manifest.py --config configs/experiment_v2.yaml
python scripts/generate_labels.py --config configs/experiment_v2.yaml --split validation
python scripts/generate_labels.py --config configs/experiment_v2.yaml --split train
python scripts/train_tabular_models.py --config configs/experiment_v2.yaml
python scripts/train_gat.py --config configs/experiment_v2.yaml
python scripts/evaluate_test.py --config configs/experiment_v2.yaml
python scripts/confirm_seeds.py --config configs/experiment_v2.yaml
```

`configs/experiment_v2.yaml` and `configs/genmol_sampling.yaml` define the
frozen split, reward, model, and sampling settings. The scripts write generated
data to `manifests_v2/`, `results_v2/`, and `artifacts_v2/`.

## Repository contents and generated data

The repository contains source code, configurations, tests, tiny fixtures,
vendor source, the 2.4 MB LightGBM model used by the demo, and compact
provenance/metric summaries. It deliberately excludes ChEMBL source data,
manifests, rollout JSONL, graph caches, PyTorch checkpoints, and bulk
intermediate outputs. Those files are reproducible generated data and are
ignored by Git to keep clones practical.

The retained `manifests_v2/{hashes,seeds}.json`, compact metric reports, and
feature-analysis summaries identify the evaluated run. Model and result files
that are not explicitly included should be regenerated using the workflow
above.

## Layout

```text
configs/             frozen experiment and sampling configuration
src/rollout_policy/  policy, scoring, feature, and model code
scripts/             data preparation, training, evaluation, and audit entry points
tests/               unit and integration tests
vendor/genmol/       pinned GenMol source; checkpoints are excluded
artifacts_v2/        compact model and run summaries only
```

## References

GenMol is the frozen remasking generator used throughout this study. Vendored
source lives under `vendor/genmol/`; weights are the NVIDIA GenMol V1
checkpoint on Hugging Face.

```bibtex
@article{lee2025genmol,
  title     = {GenMol: A Drug Discovery Generalist with Discrete Diffusion},
  author    = {Lee, Seul and Kreis, Karsten and Veccham, Srimukh Prasad and Liu, Meng and Reidenbach, Danny and Peng, Yuxing and Paliwal, Saee and Nie, Weili and Vahdat, Arash},
  journal   = {International Conference on Machine Learning},
  year      = {2025}
}
```

- GenMol paper: Lee et al., ICML 2025 (citation above)
- GenMol code: [NVIDIA/GenMol](https://github.com/NVIDIA/GenMol)
- GenMol V1 weights: [nvidia/NV-GenMol-89M-v1](https://huggingface.co/nvidia/NV-GenMol-89M-v1)
- This work (preprint): [ChemRxiv](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15010340/v2)
- Interactive demo: [Hugging Face Space](https://huggingface.co/spaces/one-more-beauty/genmol-edit-site-policy)
