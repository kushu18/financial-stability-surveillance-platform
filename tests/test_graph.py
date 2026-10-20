"""tests/test_graph.py — 20 tests for GraphBuilder, graph structure, node mapping."""
import pytest
import numpy as np
import pandas as pd

from src.graph.builder import (
    GraphBuilder, GNN_FEATURES, ALL_BANK_ORDER,
    CORE_BANK_ORDER, PERIPHERY_BANK_ORDER,
)


# ── Section 1: Constants ──────────────────────────────────────────────────────

def test_gnn_features_count():
    """Exactly 14 GNN features defined."""
    assert len(GNN_FEATURES) == 14


def test_gnn_features_unique():
    """All GNN feature names are unique."""
    assert len(GNN_FEATURES) == len(set(GNN_FEATURES))


def test_all_bank_order_count():
    """ALL_BANK_ORDER has exactly 33 banks."""
    assert len(ALL_BANK_ORDER) == 33


def test_core_first_in_order():
    """First 5 entries in ALL_BANK_ORDER are core banks."""
    assert ALL_BANK_ORDER[:5] == CORE_BANK_ORDER


def test_periphery_count():
    """28 periphery banks defined."""
    assert len(PERIPHERY_BANK_ORDER) == 28


def test_no_overlap_core_periphery():
    """No bank appears in both core and periphery lists."""
    assert len(set(CORE_BANK_ORDER) & set(PERIPHERY_BANK_ORDER)) == 0


def test_yes_bank_in_periphery():
    """Yes Bank is a periphery bank (not a D-SIB)."""
    assert "Yes Bank" in PERIPHERY_BANK_ORDER
    assert "Yes Bank" not in CORE_BANK_ORDER


# ── Section 2: PyG graph structure ───────────────────────────────────────────

@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not installed"),
    reason="PyTorch not installed"
)
def test_graph_node_count(sample_graph):
    """Graph has exactly 33 nodes."""
    if sample_graph is None:
        pytest.skip("PyTorch not available")
    assert sample_graph.num_nodes == 33
    assert sample_graph.x.shape[0] == 33


def test_graph_feature_dim(sample_graph):
    """Node feature matrix has shape [33, 14]."""
    if sample_graph is None:
        pytest.skip("PyTorch not available")
    assert sample_graph.x.shape == (33, 14)


def test_graph_edge_index_shape(sample_graph):
    """edge_index has shape [2, N_edges]."""
    if sample_graph is None:
        pytest.skip("PyTorch not available")
    assert sample_graph.edge_index.shape[0] == 2


def test_graph_edge_attr_shape(sample_graph):
    """edge_attr has shape [N_edges, 2]."""
    if sample_graph is None:
        pytest.skip("PyTorch not available")
    n_edges = sample_graph.edge_index.shape[1]
    assert sample_graph.edge_attr.shape == (n_edges, 2)


def test_graph_no_nan_features(sample_graph):
    """No NaN values in node feature matrix."""
    if sample_graph is None:
        pytest.skip("PyTorch not available")
    import torch
    assert not torch.isnan(sample_graph.x).any()


def test_graph_core_core_edges(sample_graph):
    """Core banks (indices 0-4) have bidirectional edges among themselves."""
    if sample_graph is None:
        pytest.skip("PyTorch not available")
    edge_index = sample_graph.edge_index.numpy()
    # Check at least one core-core edge exists
    core_src = edge_index[0][edge_index[0] < 5]
    core_dst = edge_index[1][edge_index[0] < 5]
    core_to_core = (core_dst < 5).sum()
    assert core_to_core >= 10, \
        f"Expected >= 10 core-core directed edges, got {core_to_core}"


def test_graph_periphery_no_direct_edges(sample_graph):
    """Periphery banks (indices 5-32) have no direct edges to each other."""
    if sample_graph is None:
        pytest.skip("PyTorch not available")
    edge_index = sample_graph.edge_index.numpy()
    peri_src = edge_index[0][edge_index[0] >= 5]
    peri_dst = edge_index[1][edge_index[0] >= 5]
    peri_to_peri = (peri_dst >= 5).sum()
    assert peri_to_peri == 0, \
        f"Expected 0 periphery-to-periphery edges, got {peri_to_peri}"


# ── Section 3: Node mapping ───────────────────────────────────────────────────

def test_node_mapping_df_structure(node_mapping_df):
    """node_mapping.csv has bank_name and matrix_index columns."""
    assert "bank_name"     in node_mapping_df.columns
    assert "matrix_index"  in node_mapping_df.columns


def test_node_mapping_count(node_mapping_df):
    """node_mapping has exactly 33 rows."""
    assert len(node_mapping_df) == 33


def test_node_mapping_indices_unique(node_mapping_df):
    """All matrix indices are unique."""
    assert node_mapping_df["matrix_index"].nunique() == 33


def test_node_mapping_indices_range(node_mapping_df):
    """Matrix indices span 0 to 32."""
    assert node_mapping_df["matrix_index"].min() == 0
    assert node_mapping_df["matrix_index"].max() == 32


def test_yes_bank_in_mapping(node_mapping_df):
    """Yes Bank is present in the node mapping."""
    assert "Yes Bank" in node_mapping_df["bank_name"].values


def test_sbi_index_zero(node_mapping_df):
    """SBI has matrix index 0 (first core bank)."""
    sbi_row = node_mapping_df[node_mapping_df["bank_name"] == "State Bank of India (SBI)"]
    assert len(sbi_row) == 1
    assert int(sbi_row["matrix_index"].values[0]) == 0
