"""
Financial Contagion Monitor — src/gnn/inference.py
M.S. Ramaiah Institute of Technology | AIP67 | 2026

GNNInferenceEngine — runs live GNN inference on every DayResult at dashboard runtime.

This is the bridge between the trained model and the live dashboard.
It loads gnn_model.pt once at startup, then provides a predict() method
that maps 33 BankSnapshots → {bank_name: vulnerability_probability}.

Critical: The GNN feature order MUST match the order used during training
(defined in GNN_FEATURES in builder.py). Any mismatch silently corrupts predictions.

Used by:
  ReplayEngine.seek()       — historical mode predictions
  LiveDataLoader.get_current() — live mode predictions
  DayResult.top_risks construction
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.gnn.model import FinancialGCN, load_model

# Must match GNN_FEATURES in builder.py EXACTLY — same order, same names
FEATURE_ORDER = [
    "tier1_capital_ratio",
    "total_capital_ratio",
    "leverage_ratio",
    "lcr",
    "nsfr",
    "npl_ratio",
    "cds_spread_bps",
    "equity_volatility_30d",
    "interbank_borrowing_ratio",
    "interbank_exposure_ratio",
    "network_degree",
    "eigenvector_centrality",
    "gsib_score",
    "capital_buffer",
]


class GNNInferenceEngine:
    """
    Loads the trained FinancialGCN once at dashboard startup and provides
    fast per-bank vulnerability predictions from BankSnapshot lists.

    Usage:
        gnn = GNNInferenceEngine()
        vuln_probs = gnn.predict(snapshots)
        # vuln_probs = {"State Bank of India (SBI)": 0.12, "Yes Bank": 0.94, ...}
    """

    def __init__(
        self,
        model_path: str = "data/raw/gnn_model.pt",
        graph_path: str = "data/raw/graph.pt",
        node_mapping_path: str = "data/raw/node_mapping.csv",
    ):
        """
        Loads model and graph at initialization.
        Raises FileNotFoundError if any required file is missing.

        Args:
            model_path:        Path to trained GNN weights
            graph_path:        Path to PyG graph (for edge_index)
            node_mapping_path: Path to bank_name → matrix_index CSV
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for GNN inference. "
                "Install: pip install torch torch_geometric"
            )

        # Load model
        self.model = load_model(model_path)
        self.model.eval()

        # Load graph (we only need edge_index — structure doesn't change)
        if not os.path.exists(graph_path):
            raise FileNotFoundError(
                f"{graph_path} not found. Run python main.py first."
            )
        self._graph = torch.load(graph_path, map_location="cpu")

        # Load node mapping
        if not os.path.exists(node_mapping_path):
            raise FileNotFoundError(
                f"{node_mapping_path} not found. Run python main.py first."
            )
        self._node_mapping = pd.read_csv(node_mapping_path)
        # Cache as dict for fast lookup: bank_name -> matrix_index
        self._name_to_idx: Dict[str, int] = dict(
            zip(
                self._node_mapping["bank_name"],
                self._node_mapping["matrix_index"].astype(int)
            )
        )
        self._idx_to_name: Dict[int, str] = {v: k for k, v in self._name_to_idx.items()}

        # Cache feature normalization stats from training graph
        # (used to normalize live snapshot features consistently with training)
        self._feature_min: Optional[np.ndarray] = None
        self._feature_max: Optional[np.ndarray] = None
        self._compute_normalization_stats()

        print(f"[OK] GNNInferenceEngine initialized: "
              f"{len(self._name_to_idx)} banks, "
              f"{len(FEATURE_ORDER)} features")

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, snapshots: "List") -> Dict[str, float]:
        """
        Runs GNN inference on the current day's 33 BankSnapshots.

        Args:
            snapshots: List of BankSnapshot namedtuples (from loader.py)
                       Should have exactly 33 items, one per bank.

        Returns:
            Dict mapping bank institution_id → vulnerability probability [0, 1]
            Example: {"State Bank of India (SBI)": 0.08, "Yes Bank": 0.94, ...}

        Notes:
            - Banks missing from snapshots default to 0.5 (unknown risk)
            - Prediction is deterministic (model.eval(), no dropout)
            - Uses the SAME feature normalization as training
        """
        if not snapshots:
            return {name: 0.5 for name in self._name_to_idx}

        # Build feature tensor from current snapshots
        x = self._snapshots_to_tensor(snapshots)

        # Run inference
        with torch.no_grad():
            log_probs = self.model(x, self._graph.edge_index)
            probs = torch.exp(log_probs)  # [33, 2]
            vuln_probs = probs[:, 1].numpy()  # P(vulnerable) per node

        # Map matrix indices back to bank names
        result = {}
        for idx in range(33):
            bank_name = self._idx_to_name.get(idx, f"Unknown_{idx}")
            result[bank_name] = float(np.clip(vuln_probs[idx], 0.0, 1.0))

        return result

    def predict_single(self, bank_name: str, snapshots: "List") -> float:
        """
        Convenience method: returns vulnerability probability for a single bank.

        Args:
            bank_name: Full institution_id (e.g. "Yes Bank")
            snapshots: Full list of 33 BankSnapshots

        Returns:
            Vulnerability probability [0, 1]
        """
        all_probs = self.predict(snapshots)
        return all_probs.get(bank_name, 0.5)

    def get_risk_level(self, vuln_prob: float) -> str:
        """
        Converts a vulnerability probability to a categorical risk level.
        Used by GNNBankExplainer and dashboard displays.

        Thresholds:
            >= 0.80 → CRITICAL
            >= 0.60 → HIGH
            >= 0.40 → MODERATE
            >= 0.20 → LOW
            <  0.20 → SAFE
        """
        if vuln_prob >= 0.80:
            return "CRITICAL"
        elif vuln_prob >= 0.60:
            return "HIGH"
        elif vuln_prob >= 0.40:
            return "MODERATE"
        elif vuln_prob >= 0.20:
            return "LOW"
        else:
            return "SAFE"

    # ── Private: Feature Tensor Construction ──────────────────────────────────

    def _snapshots_to_tensor(self, snapshots: "List") -> "torch.Tensor":
        """
        Maps 33 BankSnapshots to a [33, 14] float32 feature tensor.
        Features are normalized using the same min-max stats as training.

        Args:
            snapshots: List of BankSnapshot namedtuples

        Returns:
            x: torch.Tensor of shape [33, 14], dtype=float32
        """
        # Build snapshot lookup: institution_id → snapshot
        snap_dict = {s.institution_id: s for s in snapshots}

        x = np.zeros((33, 14), dtype=np.float32)

        for bank_name, matrix_idx in self._name_to_idx.items():
            snap = snap_dict.get(bank_name)
            if snap is None:
                # Bank missing from snapshots — use zero vector
                # (will produce near-0.5 prediction — treated as unknown)
                continue

            for feat_idx, feat_name in enumerate(FEATURE_ORDER):
                val = getattr(snap, feat_name, 0.0)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = 0.0
                x[matrix_idx, feat_idx] = float(val)

        # Normalize using training stats (min-max per feature column)
        if self._feature_min is not None and self._feature_max is not None:
            for col_idx in range(14):
                col_min = self._feature_min[col_idx]
                col_max = self._feature_max[col_idx]
                if col_max - col_min > 1e-8:
                    x[:, col_idx] = (x[:, col_idx] - col_min) / (col_max - col_min)
                    x[:, col_idx] = np.clip(x[:, col_idx], 0.0, 1.0)
                else:
                    x[:, col_idx] = 0.0

        return torch.tensor(x, dtype=torch.float32)

    def _compute_normalization_stats(self) -> None:
        """
        Extracts per-feature min/max from the training graph's feature matrix.
        These stats are used to normalize live snapshot features consistently.
        """
        try:
            x_train = self._graph.x.numpy()  # [33, 14]
            self._feature_min = x_train.min(axis=0)   # [14]
            self._feature_max = x_train.max(axis=0)   # [14]
        except Exception:
            self._feature_min = None
            self._feature_max = None

    # ── Public: Gradient-based Feature Importance ─────────────────────────────

    def compute_feature_importance(
        self,
        bank_name: str,
        snapshots: "List"
    ) -> List[tuple]:
        """
        Computes gradient-based feature importance for a specific bank.
        Used by GNNBankExplainer.

        Method: Compute gradient of P(vulnerable) w.r.t. each input feature,
        then take absolute value as importance score.

        Args:
            bank_name: Full institution_id
            snapshots: Full list of 33 BankSnapshots

        Returns:
            List of (feature_name, importance_score) sorted by importance descending.
            Example: [("npl_ratio", 0.42), ("cds_spread_bps", 0.28), ...]
        """
        matrix_idx = self._name_to_idx.get(bank_name)
        if matrix_idx is None:
            return [(f, 0.0) for f in FEATURE_ORDER]

        x = self._snapshots_to_tensor(snapshots)
        x.requires_grad_(True)

        # Forward pass
        self.model.train()  # enable gradients through dropout (for explanation)
        log_probs = self.model(x, self._graph.edge_index)
        probs = torch.exp(log_probs)

        # Target: P(vulnerable) for this specific bank
        target = probs[matrix_idx, 1]
        target.backward()

        # Importance = |gradient| of bank's row
        importance = x.grad[matrix_idx].abs().detach().numpy()

        self.model.eval()  # restore eval mode

        # Pair with feature names and sort
        result = list(zip(FEATURE_ORDER, importance.tolist()))
        result.sort(key=lambda t: t[1], reverse=True)

        return result


class MockGNNInferenceEngine:
    """
    Drop-in replacement for GNNInferenceEngine when model files don't exist yet.
    Returns plausible predictions based on vulnerability_label_injected from snapshots.
    Used during development before main.py has been run.
    """

    def __init__(self):
        print("[INFO] Using MockGNNInferenceEngine — run main.py to train the real GNN")
        self._idx_to_name = {}

    def predict(self, snapshots: "List") -> Dict[str, float]:
        """Returns vulnerability_label_injected directly as the 'prediction'."""
        return {
            s.institution_id: float(s.vulnerability_label_injected)
            for s in snapshots
        }

    def predict_single(self, bank_name: str, snapshots: "List") -> float:
        all_probs = self.predict(snapshots)
        return all_probs.get(bank_name, 0.5)

    def get_risk_level(self, vuln_prob: float) -> str:
        if vuln_prob >= 0.80: return "CRITICAL"
        elif vuln_prob >= 0.60: return "HIGH"
        elif vuln_prob >= 0.40: return "MODERATE"
        elif vuln_prob >= 0.20: return "LOW"
        else: return "SAFE"

    def compute_feature_importance(self, bank_name: str, snapshots: "List") -> List[tuple]:
        return [(f, 0.0) for f in FEATURE_ORDER]


def get_inference_engine(
    model_path: str = "data/raw/gnn_model.pt",
    graph_path: str = "data/raw/graph.pt",
    node_mapping_path: str = "data/raw/node_mapping.csv",
) -> "GNNInferenceEngine":
    """
    Factory function — returns real engine if model files exist,
    mock engine otherwise. Used in app.py startup guard.
    """
    all_exist = all([
        os.path.exists(model_path),
        os.path.exists(graph_path),
        os.path.exists(node_mapping_path),
    ])
    if all_exist and TORCH_AVAILABLE:
        return GNNInferenceEngine(model_path, graph_path, node_mapping_path)
    else:
        return MockGNNInferenceEngine()
