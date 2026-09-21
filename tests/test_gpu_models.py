"""CUDA training checks for the V2 models.

Run on a machine with a GPU:

    python -m pytest tests/test_gpu_models.py -q

The module is skipped when CUDA is not available. The GATv2 step also needs
``torch-geometric`` (``pip install -e '.[dev,models]'``).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA is required; run tests/test_gpu_models.py on a GPU machine",
        allow_module_level=True,
    )

from rollout_policy.model import TabularMLP, binomial_nll

pytestmark = pytest.mark.gpu


def test_binomial_nll_backward_on_cuda():
    device = torch.device("cuda")
    logits = torch.tensor([0.2, -0.4], device=device, requires_grad=True)
    loss = binomial_nll(
        logits,
        torch.tensor([6.0, 1.0], device=device),
        torch.tensor([8.0, 8.0], device=device),
    )
    loss.backward()
    assert loss.device.type == "cuda"
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_tabular_mlp_train_step_on_cuda():
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    features = torch.randn(16, 8, generator=generator, device="cpu").to(device)
    successes = torch.randint(0, 9, (16,), generator=generator, device="cpu").to(
        device=device, dtype=torch.float32
    )
    trials = torch.full((16,), 8.0, device=device)
    model = TabularMLP(n_features=8, hidden_sizes=(16,), dropout=0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    model.train()
    before = None
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = binomial_nll(logits, successes, trials)
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
        if before is None:
            before = float(loss.detach())
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(features))
    assert probabilities.shape == (16,)
    assert torch.all(probabilities > 0) and torch.all(probabilities < 1)
    assert probabilities.device.type == "cuda"
    assert before is not None and before > 0


def test_gat_train_step_on_cuda():
    pytest.importorskip("torch_geometric")
    from rdkit import Chem
    from torch_geometric.data import Batch

    from rollout_policy.eligibility import enumerate_aligned_actions
    from rollout_policy.graph_features import action_to_graph
    from rollout_policy.model_gat import ActionGATv2

    smiles = "Cc1cc(C)cc(C)c1"
    parent = Chem.MolFromSmiles(smiles)
    actions = enumerate_aligned_actions(parent, require_safe_alignment=False).actions[:2]
    record = {"parent_id": "mesitylene", "smiles": smiles}
    graphs = [
        action_to_graph(record, action, np.zeros(4, dtype=np.float32)) for action in actions
    ]
    batch = Batch.from_data_list(graphs).to("cuda")
    model = ActionGATv2(
        node_dim=batch.x.size(1),
        edge_dim=batch.edge_attr.size(1),
        global_dim=4,
        hidden_dim=32,
        heads=4,
        n_layers=2,
        dropout=0.0,
    ).to("cuda")
    model.train()
    logits = model(batch)
    loss = binomial_nll(
        logits,
        torch.tensor([1.0, 7.0], device="cuda"),
        torch.tensor([8.0, 8.0], device="cuda"),
    )
    loss.backward()
    assert logits.shape == (2,)
    assert logits.device.type == "cuda"
    assert torch.isfinite(loss)
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all()
        for param in model.parameters()
    )
    n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    assert n_params > 0
