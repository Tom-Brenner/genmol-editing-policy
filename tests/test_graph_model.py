"""Action-conditioned graphs and GATv2. CPU only; no GenMol and no CUDA.

The CUDA training step lives in tests/test_gpu_models.py.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from rdkit import Chem

pytest.importorskip("torch_geometric")

from rollout_policy.eligibility import enumerate_aligned_actions
from rollout_policy.graph_features import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    action_to_graph,
    validate_action_indices,
)
from rollout_policy.model import binomial_nll
from rollout_policy.model_gat import ActionGATv2
from torch_geometric.data import Batch

MESITYLENE = "Cc1cc(C)cc(C)c1"


def _graphs():
    parent = Chem.MolFromSmiles(MESITYLENE)
    diag = enumerate_aligned_actions(parent, require_safe_alignment=False)
    assert len(diag.actions) >= 2
    record = {"parent_id": "mesitylene", "smiles": MESITYLENE}
    global_features = np.zeros(4, dtype=np.float32)
    graphs = [
        action_to_graph(record, action, global_features) for action in diag.actions[:2]
    ]
    return parent, diag.actions[:2], graphs


def test_two_actions_produce_different_graphs():
    _parent, _actions, graphs = _graphs()
    assert graphs[0].action_id != graphs[1].action_id
    assert not torch.equal(graphs[0].fragment_mask, graphs[1].fragment_mask)


def test_masks_are_nonempty_and_disjoint():
    _parent, _actions, graphs = _graphs()
    for graph in graphs:
        assert bool(graph.fragment_mask.any())
        assert bool(graph.core_mask.any())
        assert not bool(torch.any(graph.fragment_mask & graph.core_mask))
        assert int(graph.core_attachment_mask.sum()) == 1
        assert int(graph.fragment_attachment_mask.sum()) == 1


def test_selected_attachment_edge_is_marked_both_ways():
    _parent, _actions, graphs = _graphs()
    selected_col = EDGE_FEATURE_NAMES.index("selected_cut")
    for graph in graphs:
        assert int(graph.edge_attr[:, selected_col].sum()) == 2
        assert graph.x.size(1) == len(NODE_FEATURE_NAMES)


def test_invalid_atom_indices_fail():
    parent, actions, _graphs_unused = _graphs()
    bad = replace(actions[0], fragment_atom_indices=(10_000,))
    with pytest.raises(ValueError):
        validate_action_indices(parent, bad)


def test_batched_forward_and_backward_on_cpu():
    _parent, _actions, graphs = _graphs()
    batch = Batch.from_data_list(graphs)
    model = ActionGATv2(
        node_dim=batch.x.size(1),
        edge_dim=batch.edge_attr.size(1),
        global_dim=4,
    )
    model.train()
    logits = model(batch)
    assert logits.shape == (batch.num_graphs,)
    loss = binomial_nll(
        logits,
        torch.tensor([2.0, 6.0]),
        torch.tensor([8.0, 8.0]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
        for param in model.parameters()
    )


def test_checkpoint_roundtrip_preserves_predictions(tmp_path):
    _parent, _actions, graphs = _graphs()
    graph = graphs[0]
    model = ActionGATv2(
        node_dim=graph.x.size(1),
        edge_dim=graph.edge_attr.size(1),
        global_dim=4,
    )
    model.eval()
    with torch.no_grad():
        before = model(graph)
    path = tmp_path / "gat.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "model_config": model.model_config()},
        path,
    )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    restored = ActionGATv2(**blob["model_config"])
    restored.load_state_dict(blob["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        after = restored(graph)
    assert torch.allclose(before, after)


def test_cpu_inference_does_not_require_cuda():
    assert not action_to_graph.__module__.startswith("genmol")
    _parent, _actions, graphs = _graphs()
    model = ActionGATv2(
        node_dim=graphs[0].x.size(1),
        edge_dim=graphs[0].edge_attr.size(1),
        global_dim=4,
    )
    model.eval()
    with torch.no_grad():
        logits = model(graphs[0])
    # One parent/action graph is a batch of size 1, same layout as a DataLoader batch.
    assert logits.shape == (1,)
    assert torch.isfinite(logits)
    assert logits.device.type == "cpu"
