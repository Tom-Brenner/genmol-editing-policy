"""Gradio entrypoint for a LightGBM-only ZeroGPU Space.

ZeroGPU is required for free-tier Gradio hosting even though this model runs on CPU.
"""

from __future__ import annotations

import json

import gradio as gr

from lightgbm_predict import format_prediction, predict_cut_site

try:
    import spaces
except ImportError:  # local runs without the HF spaces package
    spaces = None


def _predict(smiles: str) -> tuple[str, str]:
    result = predict_cut_site(smiles)
    return format_prediction(result), json.dumps(result, indent=2)


if spaces is not None:
    _predict = spaces.GPU(_predict)


EXAMPLE_SMILES = [
    "COc1ccc(CC(=O)N2CCN(c3ccccc3)CC2)cc1",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
]


demo = gr.Interface(
    fn=_predict,
    inputs=gr.Textbox(
        label="Parent SMILES",
        lines=2,
        placeholder="Paste a drug-like parent SMILES",
    ),
    outputs=[
        gr.Textbox(label="Cut site", lines=16),
        gr.Code(label="JSON", language="json"),
    ],
    title="GenMol edit-site policy (LightGBM)",
    description=(
        "Scores eligible Murcko-pendant cut sites with the frozen LightGBM "
        "regressor and returns the learned top-1 action. RDKit enumeration only "
        "(no SAFE/GenMol generation)."
    ),
    examples=EXAMPLE_SMILES,
    allow_flagging="never",
)


if __name__ == "__main__":
    demo.launch()
