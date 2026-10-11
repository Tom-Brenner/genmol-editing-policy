"""Gradio entrypoint for a LightGBM-only ZeroGPU Space; runs on CPU :).
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
    title="Editing-site selection for constrained QED improvement through frozen GenMol remasking",
    description=(
        "Scores eligible Murcko-pendant cut sites with a trained LightGBM "
        "regressor and returns the learned top-1 action."
        " Selection shown to maximize probability of QED increase through GenMol remasking, see preprint TBA"
    ),
    examples=EXAMPLE_SMILES,
    allow_flagging="never",
)


if __name__ == "__main__":
    demo.launch()
