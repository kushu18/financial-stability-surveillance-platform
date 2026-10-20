"""tests/test_gnn.py — 10 tests for FinancialGCN model, trainer, explainer."""
import pytest
import numpy as np


def test_model_output_shape(sample_graph):
    """FinancialGCN output has shape [33, 2]."""
    if sample_graph is None:
        pytest.skip("PyTorch not available")
    try:
        import torch
        from src.gnn.model import FinancialGCN
        model     = FinancialGCN()
        model.eval()
        with torch.no_grad():
            out = model(sample_graph.x, sample_graph.edge_index)
        assert out.shape == (33, 2), f"Expected (33,2), got {out.shape}"
    except ImportError:
        pytest.skip("PyTorch not available")


def test_model_output_is_log_softmax(sample_graph):
    """Output values are log-probabilities (all <= 0)."""
    if sample_graph is None:
        pytest.skip("PyTorch not available")
    try:
        import torch
        from src.gnn.model import FinancialGCN
        model = FinancialGCN()
        model.eval()
        with torch.no_grad():
            out = model(sample_graph.x, sample_graph.edge_index)
        assert (out <= 0).all(), "log_softmax values should all be <= 0"
    except ImportError:
        pytest.skip("PyTorch not available")


def test_model_probabilities_sum_to_one(sample_graph):
    """exp(log_probs) sums to ~1 per node."""
    if sample_graph is None:
        pytest.skip("PyTorch not available")
    try:
        import torch
        from src.gnn.model import FinancialGCN
        model = FinancialGCN()
        model.eval()
        with torch.no_grad():
            log_probs = model(sample_graph.x, sample_graph.edge_index)
            probs     = torch.exp(log_probs)
        row_sums = probs.sum(dim=1).numpy()
        np.testing.assert_allclose(row_sums, np.ones(33), atol=1e-5)
    except ImportError:
        pytest.skip("PyTorch not available")


def test_model_parameter_count():
    """FinancialGCN has the expected parameter count."""
    try:
        from src.gnn.model import FinancialGCN, count_parameters
        model  = FinancialGCN()
        n_params = count_parameters(model)
        # 14→64→32→2 GCN: rough lower bound
        assert n_params > 1000, f"Too few parameters: {n_params}"
        assert n_params < 100_000, f"Unexpectedly many parameters: {n_params}"
    except ImportError:
        pytest.skip("PyTorch not available")


def test_model_dropout_disabled_in_eval(sample_graph):
    """Model produces same output in two eval() passes (dropout disabled)."""
    if sample_graph is None:
        pytest.skip("PyTorch not available")
    try:
        import torch
        from src.gnn.model import FinancialGCN
        model = FinancialGCN()
        model.eval()
        with torch.no_grad():
            out1 = model(sample_graph.x, sample_graph.edge_index)
            out2 = model(sample_graph.x, sample_graph.edge_index)
        torch.testing.assert_close(out1, out2)
    except ImportError:
        pytest.skip("PyTorch not available")


def test_model_get_embeddings_shape(sample_graph):
    """get_embeddings() returns [33, 32] tensor."""
    if sample_graph is None:
        pytest.skip("PyTorch not available")
    try:
        import torch
        from src.gnn.model import FinancialGCN
        model = FinancialGCN()
        model.eval()
        with torch.no_grad():
            emb = model.get_embeddings(sample_graph.x, sample_graph.edge_index)
        assert emb.shape == (33, 32)
    except ImportError:
        pytest.skip("PyTorch not available")


def test_inference_engine_mock(sample_snapshots):
    """MockGNNInferenceEngine returns dict with 33 entries."""
    from src.gnn.inference import MockGNNInferenceEngine
    engine = MockGNNInferenceEngine()
    probs  = engine.predict(sample_snapshots)
    assert len(probs) == 33
    assert all(0.0 <= v <= 1.0 for v in probs.values())


def test_inference_engine_yes_bank_high(sample_snapshots):
    """MockGNNInferenceEngine gives Yes Bank high vulnerability."""
    from src.gnn.inference import MockGNNInferenceEngine
    engine    = MockGNNInferenceEngine()
    probs     = engine.predict(sample_snapshots)
    yes_prob  = probs.get("Yes Bank", 0.0)
    assert yes_prob >= 0.90, f"Yes Bank probability should be high, got {yes_prob:.4f}"


def test_risk_level_thresholds():
    """get_risk_level() returns correct labels at boundaries."""
    from src.gnn.inference import MockGNNInferenceEngine
    engine = MockGNNInferenceEngine()
    assert engine.get_risk_level(0.85) == "CRITICAL"
    assert engine.get_risk_level(0.65) == "HIGH"
    assert engine.get_risk_level(0.45) == "MODERATE"
    assert engine.get_risk_level(0.25) == "LOW"
    assert engine.get_risk_level(0.10) == "SAFE"


def test_explainer_returns_result(sample_snapshots):
    """GNNBankExplainer.explain_bank returns ExplainerResult."""
    from src.gnn.inference import MockGNNInferenceEngine
    from src.gnn.explainer import GNNBankExplainer, ExplainerResult
    engine   = MockGNNInferenceEngine()
    explainer = GNNBankExplainer(engine)
    result   = explainer.explain_bank("Yes Bank", sample_snapshots)
    assert isinstance(result, ExplainerResult)
    assert result.bank_name == "Yes Bank"
    assert result.risk_level in ("CRITICAL", "HIGH", "MODERATE", "LOW", "SAFE")
    assert result.vulnerability_score >= 0.90
