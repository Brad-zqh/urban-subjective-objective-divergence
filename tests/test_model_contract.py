"""CPU-compatible smoke tests for the exact V211 architecture contract."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mm_gtgnnwr.model_contract import (
    V207TransparentMMGTGNNWR,
    legal_relation_mean,
    legal_relation_mean_from_edges,
)


def make_case():
    """Create a small deterministic multimodal contract case."""
    torch.manual_seed(211)
    names = ("google_image", "google_text")
    model = V207TransparentMMGTGNNWR(
        modality_names=names,
        numeric_dim=3,
        relation_count=2,
        quality_dim=2,
        space_time_dim=3,
        embedding_dim=8,
        latent_dim=4,
        context_dim=6,
    )
    rows = 4
    case = {
        "numeric_x": torch.randn(rows, 3),
        "embeddings": {name: torch.randn(rows, 8) for name in names},
        "qualities": {name: torch.randn(rows, 2) for name in names},
        "availability": {name: torch.ones(rows, dtype=torch.bool) for name in names},
        "adjacency": torch.eye(rows).repeat(2, 1, 1),
        "legal_senders": torch.ones(2, rows, dtype=torch.bool),
        "space_time": torch.randn(rows, 3),
    }
    return model, case


def test_prediction_is_exact_contribution_sum() -> None:
    model, case = make_case()
    output = model(**case)
    assert torch.equal(output["prediction"], output["contributions"].sum(dim=1))


def test_missing_channel_is_nan_safe_and_gated_to_zero() -> None:
    model, case = make_case()
    case["availability"]["google_text"][0] = False
    case["embeddings"]["google_text"][0] = torch.nan
    output = model(**case)
    assert output["gates"]["google_text"][0].item() == 0.0
    assert torch.isfinite(output["prediction"]).all()


def test_zero_weight_sparse_padding_matches_dense_contract() -> None:
    context = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    edges = torch.tensor(
        [
            [[0, 0, 1, 2], [0, 1, 2, 3]],
            [[1, 3, 0, 0], [4, 0, 0, 0]],
        ],
        dtype=torch.long,
    )
    legal = torch.ones(2, 5, dtype=torch.bool)
    weights = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    sparse = legal_relation_mean_from_edges(
        context,
        edges,
        legal,
        relation_count=2,
        edge_weight=weights,
    )
    dense = torch.zeros(2, 5, 5)
    for relation in range(2):
        for edge in range(edges.shape[2]):
            dense[relation, edges[relation, 0, edge], edges[relation, 1, edge]] = weights[
                relation, edge
            ]
    assert torch.equal(sparse, legal_relation_mean(context, dense, legal))
