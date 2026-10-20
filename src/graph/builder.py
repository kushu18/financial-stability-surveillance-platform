"""
Financial Contagion Monitor — src/graph/builder.py
M.S. Ramaiah Institute of Technology | AIP67 | 2026

GraphBuilder — constructs the PyTorch Geometric Data object from the merged
dataframe. Computes average feature values per bank across all dates, then
builds the interbank edge index based on asset size and exposure.

Outputs:
  data/raw/graph.pt             — PyG Data object [33, 14]
  data/raw/node_mapping.csv     — bank_name -> matrix_index
  data/raw/node_positions.json  — {bank_name: [x, y, z]} for 3D dashboard
  data/raw/feature_names.txt    — 14 GNN feature names (one per line)

Edge construction rules:
  Core-to-Core:       All 10 bidirectional edges (fully connected 5-node subgraph)
  Core-to-Periphery:  1–3 edges per periphery bank based on total_assets_usd_bn
  Periphery-to-Periphery: NONE (prevents spaghetti, reflects regulatory reality)
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False


# ── 14 GNN features (must match feature_names.txt and GNNInferenceEngine) ────
GNN_FEATURES = [
    "tier1_capital_ratio",      # 0  Capital adequacy — primary signal
    "total_capital_ratio",      # 1  Regulatory capital buffer
    "leverage_ratio",           # 2  Systemic leverage risk
    "lcr",                      # 3  Short-term liquidity resilience
    "nsfr",                     # 4  Long-term funding stability
    "npl_ratio",                # 5  Loan book quality — key distress predictor
    "cds_spread_bps",           # 6  Market perception of default risk
    "equity_volatility_30d",    # 7  Market-implied uncertainty
    "interbank_borrowing_ratio",# 8  Contagion transmission (borrowing side)
    "interbank_exposure_ratio", # 9  Contagion transmission (lending side)
    "network_degree",           # 10 Connectivity in interbank network
    "eigenvector_centrality",   # 11 Systemic importance by network position
    "gsib_score",               # 12 Global systemic importance classification
    "capital_buffer",           # 13 Cushion above regulatory floor
]

# 5 Core D-SIBs — ordered by systemic importance (order determines matrix index)
CORE_BANK_ORDER = [
    "State Bank of India (SBI)",
    "HDFC Bank",
    "ICICI Bank",
    "Axis Bank",
    "Kotak Mahindra Bank",
]

# 28 Periphery banks — ordered alphabetically for deterministic indexing
PERIPHERY_BANK_ORDER = [
    "AU Small Finance Bank",
    "Bandhan Bank",
    "Bank of Baroda",
    "Bank of India",
    "Bank of Maharashtra",
    "Canara Bank",
    "Central Bank of India",
    "City Union Bank",
    "CSB Bank",
    "Equitas Small Finance Bank",
    "Federal Bank",
    "IDBI Bank",
    "IDFC FIRST Bank",
    "Indian Bank",
    "Indian Overseas Bank (IOB)",
    "IndusInd Bank",
    "J&K Bank",
    "Karnataka Bank",
    "Karur Vysya Bank",
    "Punjab & Sind Bank",
    "Punjab National Bank (PNB)",
    "RBL Bank",
    "South Indian Bank",
    "Tamilnad Mercantile Bank",
    "UCO Bank",
    "Ujjivan Small Finance Bank",
    "Union Bank of India",
    "Yes Bank",
]

# Final ordered list: indices 0–4 = Core, 5–32 = Periphery
ALL_BANK_ORDER = CORE_BANK_ORDER + PERIPHERY_BANK_ORDER


class GraphBuilder:
    """
    Builds the interbank network graph for GNN training and dashboard visualization.

    Usage (in main.py):
        builder = GraphBuilder(df)
        graph = builder.build_graph_from_real_data()
        torch.save(graph, "data/raw/graph.pt")
        builder.save_node_mapping("data/raw/node_mapping.csv")
        pos_3d = builder.build_3d_positions()
    """

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: The full 1,205,490-row merged dataframe from RealDataLoader.
                Must have all 28+ columns including GNN_FEATURES.
        """
        self.df = df
        self._bank_avgs: Optional[pd.DataFrame] = None  # per-bank average features
        self._node_mapping: Optional[Dict[str, int]] = None  # bank_name -> matrix index
        self._edge_list: Optional[List[Tuple[int, int]]] = None
        self._edge_attrs: Optional[List[Tuple[float, float]]] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def build_graph_from_real_data(self) -> "Data":
        """
        Main entry point. Builds and returns a PyG Data object.

        Returns:
            torch_geometric.data.Data with:
              x          — [33, 14] float32 node feature matrix
              edge_index — [2, N_edges] long edge index
              edge_attr  — [N_edges, 2] float32 edge attributes
              num_nodes  — 33
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and torch_geometric are required. "
                "Install with: pip install torch torch_geometric"
            )

        print("  Computing per-bank average features...")
        self._bank_avgs = self._compute_bank_averages()

        print("  Building node mapping...")
        self._node_mapping = self._build_node_mapping()

        print("  Building edge index...")
        self._edge_list, self._edge_attrs = self._build_edges()

        print("  Constructing feature matrix...")
        x = self._build_feature_matrix()

        print("  Building PyG Data object...")
        graph = self._build_pyg_data(x)

        n_edges = graph.edge_index.shape[1]
        print(f"  Graph: {graph.num_nodes} nodes, {n_edges} edges, "
              f"{graph.x.shape[1]} features/node")

        # Validate
        assert graph.x.shape == (33, 14), \
            f"Expected x shape (33, 14), got {graph.x.shape}"
        assert graph.edge_index.shape[0] == 2, \
            "edge_index must have shape [2, N_edges]"
        assert not torch.isnan(graph.x).any(), \
            "NaN values found in node feature matrix — check source data"

        return graph

    def save_node_mapping(self, path: str) -> None:
        """
        Saves node_mapping.csv with columns [bank_name, matrix_index].
        Used by GNNInferenceEngine, GNNBankExplainer, and ContagionSimulator.

        Format:
            bank_name,matrix_index
            State Bank of India (SBI),0
            HDFC Bank,1
            ...
            Yes Bank,32
        """
        if self._node_mapping is None:
            raise RuntimeError("Call build_graph_from_real_data() first.")

        rows = [
            {"bank_name": name, "matrix_index": idx}
            for name, idx in sorted(self._node_mapping.items(), key=lambda x: x[1])
        ]
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  node_mapping.csv saved: {len(rows)} banks")

    def build_3d_positions(self) -> Dict[str, List[float]]:
        """
        Computes 3D spring layout positions for all 33 bank nodes.
        - X, Y: from NetworkX spring layout (seed=42 for reproducibility)
        - Z: from eigenvector centrality (more central banks float higher)

        Per-tick drift is applied in callbacks.py, NOT here.
        These are the base positions loaded once at dashboard startup.

        Returns:
            {bank_name: [x, y, z]} — all floats, suitable for JSON serialization
        """
        if not NX_AVAILABLE:
            print("  [WARNING] networkx not installed — using fallback 3D positions")
            return self._fallback_3d_positions()

        if self._edge_list is None:
            raise RuntimeError("Call build_graph_from_real_data() first.")

        # Build networkx graph from our edge list
        G = nx.Graph()
        G.add_nodes_from(ALL_BANK_ORDER)

        mapping_inv = {v: k for k, v in self._node_mapping.items()}
        for (src_idx, dst_idx) in self._edge_list:
            src_name = mapping_inv.get(src_idx, f"Bank_{src_idx}")
            dst_name = mapping_inv.get(dst_idx, f"Bank_{dst_idx}")
            G.add_edge(src_name, dst_name)

        # 2D spring layout — k controls spacing (higher = more spread out)
        pos_2d = nx.spring_layout(G, seed=42, k=2.5, iterations=100)

        # Eigenvector centrality for Z axis
        try:
            centrality = nx.eigenvector_centrality(G, max_iter=500, tol=1e-6)
        except nx.PowerIterationFailedConvergence:
            centrality = nx.degree_centrality(G)  # fallback

        pos_3d = {}
        for bank_name in ALL_BANK_ORDER:
            if bank_name in pos_2d:
                x, y = pos_2d[bank_name]
                z = centrality.get(bank_name, 0.5) * 2.0  # [0, 2] range
            else:
                # Fallback for missing bank
                x, y, z = np.random.uniform(-1, 1, 3).tolist()
            pos_3d[bank_name] = [float(x), float(y), float(z)]

        print(f"  3D positions computed for {len(pos_3d)} banks")
        return pos_3d

    def get_node_mapping(self) -> Dict[str, int]:
        """Returns {bank_name: matrix_index} dict."""
        if self._node_mapping is None:
            raise RuntimeError("Call build_graph_from_real_data() first.")
        return self._node_mapping.copy()

    def get_edge_list(self) -> List[Tuple[int, int]]:
        """Returns list of (src_idx, dst_idx) tuples."""
        if self._edge_list is None:
            raise RuntimeError("Call build_graph_from_real_data() first.")
        return self._edge_list.copy()

    # ── Private: Feature Matrix ───────────────────────────────────────────────

    def _compute_bank_averages(self) -> pd.DataFrame:
        """
        Computes the mean value of each GNN feature per bank across all dates.
        Missing features are filled with the global mean for that feature.
        Returns a DataFrame indexed by institution_id.
        """
        # Only keep columns we need
        available_features = [f for f in GNN_FEATURES if f in self.df.columns]
        missing_features = [f for f in GNN_FEATURES if f not in self.df.columns]

        if missing_features:
            print(f"  [WARNING] Missing GNN features in source data: {missing_features}")
            print("  These will be filled with 0.0 — check EXPECTED_COLUMNS in loader.py")

        cols = ["institution_id"] + available_features
        bank_avgs = self.df[cols].groupby("institution_id").mean()

        # Fill missing feature columns with 0.0
        for feat in missing_features:
            bank_avgs[feat] = 0.0

        # Ensure all GNN features are present and in correct order
        bank_avgs = bank_avgs[GNN_FEATURES]

        # Fill any NaN with column mean
        for feat in GNN_FEATURES:
            col_mean = bank_avgs[feat].mean()
            if pd.isna(col_mean):
                col_mean = 0.0
            bank_avgs[feat] = bank_avgs[feat].fillna(col_mean)

        return bank_avgs

    def _build_node_mapping(self) -> Dict[str, int]:
        """
        Creates bank_name -> matrix_index mapping.
        Core banks: indices 0–4 (in CORE_BANK_ORDER)
        Periphery banks: indices 5–32 (in PERIPHERY_BANK_ORDER)

        If a bank from ALL_BANK_ORDER is missing from the data,
        it gets a zero-filled feature vector (logged as warning).
        """
        mapping = {}
        for idx, bank_name in enumerate(ALL_BANK_ORDER):
            mapping[bank_name] = idx

        # Check for banks in data but not in our expected list
        data_banks = set(self.df["institution_id"].unique())
        expected_banks = set(ALL_BANK_ORDER)

        unexpected = data_banks - expected_banks
        if unexpected:
            print(f"  [WARNING] Banks in data not in ALL_BANK_ORDER: {unexpected}")
            print("  These banks will be EXCLUDED from the graph.")

        missing_from_data = expected_banks - data_banks
        if missing_from_data:
            print(f"  [WARNING] Expected banks missing from data: {missing_from_data}")
            print("  These banks will have ZERO feature vectors in the graph.")

        return mapping

    def _build_feature_matrix(self) -> "torch.Tensor":
        """
        Builds the [33, 14] node feature matrix X.
        Row i = feature vector for bank at matrix index i.
        Features are normalized to [0, 1] range using min-max scaling.
        """
        x = np.zeros((33, 14), dtype=np.float32)

        for bank_name, idx in self._node_mapping.items():
            if bank_name in self._bank_avgs.index:
                features = self._bank_avgs.loc[bank_name, GNN_FEATURES].values
                x[idx] = features.astype(np.float32)
            else:
                # Zero vector for missing banks — already initialized
                pass

        # Min-max normalize each feature column to [0, 1]
        # This prevents features with large ranges (e.g. cds_spread_bps: 0-2000)
        # from dominating features with small ranges (e.g. lcr: 0.9-2.5)
        for col_idx in range(14):
            col = x[:, col_idx]
            col_min, col_max = col.min(), col.max()
            if col_max - col_min > 1e-8:
                x[:, col_idx] = (col - col_min) / (col_max - col_min)
            else:
                x[:, col_idx] = 0.0  # constant column — zero it out

        return torch.tensor(x, dtype=torch.float32)

    # ── Private: Edge Construction ────────────────────────────────────────────

    def _build_edges(self) -> Tuple[List[Tuple[int, int]], List[Tuple[float, float]]]:
        """
        Constructs the interbank edge list according to architecture rules:

        Rule 1: Core-to-Core
          All 5 core banks are fully connected (bidirectional).
          10 unique pairs × 2 directions = 20 directed edges.

        Rule 2: Core-to-Periphery
          Each periphery bank connects to 1–3 core banks based on total_assets_usd_bn:
            assets > $50Bn  → 3 core banks
            assets $10–50Bn → 2 core banks
            assets < $10Bn  → 1 core bank (SBI as largest)
          Connections are bidirectional.

        Rule 3: Periphery-to-Periphery
          NO direct connections (reflects regulatory reality, prevents spaghetti).

        Edge attributes: [interbank_exposure_ratio_src, interbank_borrowing_ratio_src]
        """
        edges = []       # list of (src_idx, dst_idx)
        edge_attrs = []  # list of (exposure_ratio, borrowing_ratio)

        # Get per-bank asset sizes and ratios for edge weight computation
        asset_col = "total_assets_usd_bn"
        exp_col = "interbank_exposure_ratio"
        borrow_col = "interbank_borrowing_ratio"

        def get_bank_val(bank_name: str, col: str, default: float = 0.1) -> float:
            if bank_name in self._bank_avgs.index:
                if col in self._bank_avgs.columns:
                    val = self._bank_avgs.loc[bank_name, col]
                    return float(val) if not pd.isna(val) else default
            # Try from original df
            rows = self.df[self.df["institution_id"] == bank_name]
            if len(rows) > 0 and col in self.df.columns:
                return float(rows[col].mean())
            return default

        core_indices = [self._node_mapping[b] for b in CORE_BANK_ORDER]
        periphery_indices = [self._node_mapping[b] for b in PERIPHERY_BANK_ORDER]

        # ── Rule 1: Core ↔ Core (bidirectional, all pairs) ───────────────────
        for i in range(len(CORE_BANK_ORDER)):
            for j in range(i + 1, len(CORE_BANK_ORDER)):
                src_name = CORE_BANK_ORDER[i]
                dst_name = CORE_BANK_ORDER[j]
                src_idx = self._node_mapping[src_name]
                dst_idx = self._node_mapping[dst_name]

                # Edge weight = average of both banks' exposure ratios
                src_exp = get_bank_val(src_name, exp_col, 0.15)
                src_borrow = get_bank_val(src_name, borrow_col, 0.12)
                dst_exp = get_bank_val(dst_name, exp_col, 0.15)
                dst_borrow = get_bank_val(dst_name, borrow_col, 0.12)

                avg_exp = (src_exp + dst_exp) / 2
                avg_borrow = (src_borrow + dst_borrow) / 2

                # Add both directions
                edges.append((src_idx, dst_idx))
                edge_attrs.append((avg_exp, avg_borrow))
                edges.append((dst_idx, src_idx))
                edge_attrs.append((avg_exp, avg_borrow))

        # ── Rule 2: Periphery ↔ Core (based on asset size) ───────────────────
        for peri_name in PERIPHERY_BANK_ORDER:
            peri_idx = self._node_mapping[peri_name]
            peri_assets = get_bank_val(peri_name, asset_col, 5.0)

            # Determine how many core banks to connect to
            if peri_assets > 50.0:
                n_connections = 3
            elif peri_assets > 10.0:
                n_connections = 2
            else:
                n_connections = 1

            # Compute exposure weight of each core bank to this periphery bank
            # Connect to the n_connections core banks with highest exposure
            core_scores = []
            for core_name in CORE_BANK_ORDER:
                core_exp = get_bank_val(core_name, exp_col, 0.15)
                core_borrow = get_bank_val(core_name, borrow_col, 0.12)
                core_assets = get_bank_val(core_name, asset_col, 100.0)
                # Score: larger core banks + more exposure = higher connectivity
                score = (core_assets / 1000.0) + core_exp
                core_scores.append((core_name, score, core_exp, core_borrow))

            # Sort by score descending, take top n_connections
            core_scores.sort(key=lambda x: x[1], reverse=True)
            selected_cores = core_scores[:n_connections]

            peri_exp = get_bank_val(peri_name, exp_col, 0.08)
            peri_borrow = get_bank_val(peri_name, borrow_col, 0.06)

            for core_name, _, core_exp, core_borrow in selected_cores:
                core_idx = self._node_mapping[core_name]

                # Periphery → Core edge (periphery borrows from/exposes to core)
                edges.append((peri_idx, core_idx))
                edge_attrs.append((peri_exp, peri_borrow))

                # Core → Periphery edge (core exposed to periphery)
                # Slightly lower weight — core is less exposed to periphery than vice versa
                edges.append((core_idx, peri_idx))
                edge_attrs.append((core_exp * 0.7, core_borrow * 0.7))

        print(f"  Core-core edges: {len(CORE_BANK_ORDER) * (len(CORE_BANK_ORDER)-1)} "
              f"(bidirectional)")
        print(f"  Core-periphery edges: {len(edges) - len(CORE_BANK_ORDER) * (len(CORE_BANK_ORDER)-1)} "
              f"(bidirectional)")
        print(f"  Total directed edges: {len(edges)}")

        return edges, edge_attrs

    def _build_pyg_data(self, x: "torch.Tensor") -> "Data":
        """
        Assembles the final PyG Data object from feature matrix and edge list.
        """
        if not self._edge_list:
            raise RuntimeError("Edge list is empty. build_graph_from_real_data() failed.")

        # Convert edge list to tensor
        src = [e[0] for e in self._edge_list]
        dst = [e[1] for e in self._edge_list]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        # Convert edge attributes to tensor
        # col 0: interbank_exposure_ratio of source bank (lending exposure)
        # col 1: interbank_borrowing_ratio of source bank (borrowing channel)
        edge_attr = torch.tensor(self._edge_attrs, dtype=torch.float32)

        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=33
        )

        return graph

    # ── Private: Fallback ─────────────────────────────────────────────────────

    def _fallback_3d_positions(self) -> Dict[str, List[float]]:
        """
        Fallback 3D positions when networkx is not available.
        Core banks placed in a tight central cluster at high Z.
        Periphery banks arranged in a ring at low Z.
        """
        pos_3d = {}
        # Core banks: center cluster
        core_positions = [
            [0.0, 0.0, 2.0],   # SBI — center top
            [0.5, 0.3, 1.8],   # HDFC
            [-0.5, 0.3, 1.8],  # ICICI
            [0.5, -0.3, 1.6],  # Axis
            [-0.5, -0.3, 1.6], # Kotak
        ]
        for i, name in enumerate(CORE_BANK_ORDER):
            pos_3d[name] = core_positions[i]

        # Periphery banks: ring around core
        n_peri = len(PERIPHERY_BANK_ORDER)
        for i, name in enumerate(PERIPHERY_BANK_ORDER):
            angle = 2 * np.pi * i / n_peri
            r = 2.5 + np.random.uniform(-0.3, 0.3)
            x = float(r * np.cos(angle))
            y = float(r * np.sin(angle))
            z = float(np.random.uniform(0.3, 1.0))
            pos_3d[name] = [x, y, z]

        return pos_3d


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: Load saved graph artifacts
# ══════════════════════════════════════════════════════════════════════════════

def load_graph(graph_path: str = "data/raw/graph.pt") -> "Data":
    """Loads the saved PyG graph. Used by trainer and inference engine."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required to load graph.")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(
            f"{graph_path} not found. Run python main.py first."
        )
    return torch.load(graph_path, map_location="cpu")


def load_node_mapping(path: str = "data/raw/node_mapping.csv") -> Dict[str, int]:
    """Returns {bank_name: matrix_index} from saved CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run python main.py first.")
    df = pd.read_csv(path)
    return dict(zip(df["bank_name"], df["matrix_index"].astype(int)))


def load_node_positions(path: str = "data/raw/node_positions.json") -> Dict[str, List[float]]:
    """Returns {bank_name: [x, y, z]} from saved JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run python main.py first.")
    with open(path, "r") as f:
        return json.load(f)
