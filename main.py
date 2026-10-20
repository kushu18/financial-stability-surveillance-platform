"""
Financial Contagion Monitor — main.py
M.S. Ramaiah Institute of Technology | AIP67 | 2026

Master Orchestrator — runs the full data pipeline ONCE before starting the dashboard.
Produces 6 output files that all downstream modules depend on:
  1. data/raw/banks_daily_data.csv   — 1,205,490 rows, 35 columns
  2. data/raw/graph.pt               — PyTorch Geometric graph [33 x 14]
  3. data/raw/gnn_model.pt           — trained GNN weights (best AUC checkpoint)
  4. data/raw/sri_series.csv         — pre-computed daily SRI 2015–2024
  5. data/security/users.db          — all 35 user accounts (bcrypt-hashed)
  6. data/cache.db                   — yfinance live data cache (empty at init)

Usage:
  conda activate contagion
  python main.py
  # Then: python dashboard/app.py

DO NOT re-run main.py unless you want to retrain the GNN or regenerate all data.
"""

import os
import sys
import time

# ── Step 0: Runtime directory creation (must happen before any src imports) ───
RUNTIME_DIRS = [
    "data/raw",
    "data/security",
    "reports/exports",
    "logs",
]
for d in RUNTIME_DIRS:
    os.makedirs(d, exist_ok=True)
print("[OK] Runtime directories verified")

# ── Step 0b: Verify source CSV files exist before loading any module ──────────
CSV_FILES = [
    "data/raw/data_part1_800k.csv",
    "data/raw/data_part2_405k.csv",
]
missing_csv = [f for f in CSV_FILES if not os.path.exists(f)]
if missing_csv:
    print("\n[ERROR] CSV source files missing. Place them in data/raw/ and re-run.")
    for f in missing_csv:
        print(f"  MISSING: {f}")
    print("\nExpected files:")
    print("  data/raw/data_part1_800k.csv  (800,000 rows)")
    print("  data/raw/data_part2_405k.csv  (405,490 rows)")
    sys.exit(1)
print("[OK] Both CSV source files found")

# ── Now safe to import src modules ────────────────────────────────────────────
import numpy as np
import pandas as pd

print("\n" + "=" * 70)
print("  FINANCIAL CONTAGION MONITOR — DATA PIPELINE")
print("  M.S. Ramaiah Institute of Technology | AIP67 | 2026")
print("=" * 70 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load and merge CSV data
# ─────────────────────────────────────────────────────────────────────────────
print("STEP 1: Loading and merging CSV data...")
t0 = time.time()

from src.data.loader import RealDataLoader

loader = RealDataLoader("data/raw/")
df = loader.load_from_parts()

elapsed = time.time() - t0
print(f"[OK] banks_daily_data.csv: {len(df):,} rows, {len(df.columns)} columns ({elapsed:.1f}s)")

# Validate row count
if len(df) != 1_205_490:
    print(f"[WARNING] Expected 1,205,490 rows but got {len(df):,}. Check CSV files.")

# Validate all 33 banks present
n_banks = df["institution_id"].nunique()
if n_banks != 33:
    print(f"[ERROR] Expected 33 unique banks, found {n_banks}. Check institution_id column.")
    sys.exit(1)
print(f"[OK] All {n_banks} banks present in dataset")

# Validate date range
date_min = df["date"].min()
date_max = df["date"].max()
print(f"[OK] Date range: {date_min} to {date_max}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Validate COVID spike (non-negotiable — demo depends on this)
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 2: Validating crisis injection...")

covid_rows = df[df["date"] == "2020-03-15"]
if len(covid_rows) == 0:
    print("[ERROR] No rows found for 2020-03-15. Crisis injection may have failed.")
    sys.exit(1)

covid_sri = float(covid_rows["vulnerability_label_injected"].mean())
print(f"[OK] COVID-19 peak (2020-03-15) avg vulnerability: {covid_sri:.4f}")

if covid_sri < 0.85:
    print(f"[ERROR] COVID spike too low ({covid_sri:.4f}). Must be > 0.85.")
    print("        Check _inject_crisis_markers() COVID forced override in loader.py.")
    sys.exit(1)
print("[OK] COVID spike validated — demo-ready")

# Validate Yes Bank moratorium period
yes_rows = df[(df["date"] >= "2020-03-05") & (df["date"] <= "2020-03-13") &
              (df["institution_id"].str.contains("Yes Bank"))]
if len(yes_rows) > 0:
    yes_max = yes_rows["vulnerability_label_injected"].max()
    print(f"[OK] Yes Bank moratorium period max vulnerability: {yes_max:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Pre-compute SRI series for all dates (avoids 1.2M row scan at runtime)
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 3: Pre-computing SRI series for all trading days...")
t0 = time.time()

sri_rows = []
all_dates = sorted(df["date"].unique())
for date in all_dates:
    day_df = df[df["date"] == date]
    weights = day_df["total_assets_usd_bn"].values
    vulns = day_df["vulnerability_label_injected"].values
    total_weight = weights.sum()
    if total_weight > 0:
        sri = float(np.average(vulns, weights=weights))
    else:
        sri = float(vulns.mean())
    sri_rows.append({"date": date, "sri": round(sri, 6)})

sri_df = pd.DataFrame(sri_rows)
sri_df.to_csv("data/raw/sri_series.csv", index=False)

elapsed = time.time() - t0
print(f"[OK] sri_series.csv: {len(sri_df):,} rows ({elapsed:.1f}s)")

# Validate COVID peak in SRI series
covid_sri_val = sri_df[sri_df["date"] == "2020-03-15"]["sri"].values
if len(covid_sri_val) > 0:
    print(f"[OK] COVID peak SRI in series: {covid_sri_val[0]:.4f}")
    assert covid_sri_val[0] > 0.85, f"COVID SRI in series too low: {covid_sri_val[0]}"

# Find and report SRI peaks for all 9 crises
print("\n  Crisis SRI Peaks (top 5 dates by SRI):")
top5 = sri_df.nlargest(5, "sri")
for _, row in top5.iterrows():
    print(f"    {row['date']}: SRI = {row['sri']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Build graph
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 4: Building interbank graph...")
t0 = time.time()

try:
    import torch
    from src.graph.builder import GraphBuilder

    builder = GraphBuilder(df)
    graph = builder.build_graph_from_real_data()

    # Save graph
    torch.save(graph, "data/raw/graph.pt")
    print(f"[OK] graph.pt saved: {graph.x.shape[0]} nodes, "
          f"{graph.edge_index.shape[1]} edges, "
          f"{graph.x.shape[1]} features per node")

    # Save node mapping
    builder.save_node_mapping("data/raw/node_mapping.csv")
    print("[OK] node_mapping.csv saved")

    # Save 3D positions
    pos_3d = builder.build_3d_positions()
    import json
    with open("data/raw/node_positions.json", "w") as f:
        json.dump(pos_3d, f, indent=2)
    print(f"[OK] node_positions.json saved: {len(pos_3d)} bank positions")

    # Save feature names (14 GNN features, one per line)
    feature_names = [
        "tier1_capital_ratio", "total_capital_ratio", "leverage_ratio",
        "lcr", "nsfr", "npl_ratio", "cds_spread_bps",
        "equity_volatility_30d", "interbank_borrowing_ratio",
        "interbank_exposure_ratio", "network_degree",
        "eigenvector_centrality", "gsib_score", "capital_buffer"
    ]
    with open("data/raw/feature_names.txt", "w") as f:
        f.write("\n".join(feature_names) + "\n")
    print("[OK] feature_names.txt saved")

    elapsed = time.time() - t0
    print(f"[OK] Graph pipeline complete ({elapsed:.1f}s)")

except ImportError as e:
    print(f"[WARNING] PyTorch not installed — skipping graph build: {e}")
    print("         Install PyTorch and torch_geometric, then re-run main.py")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Train GNN
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 5: Training GNN (FinancialGCN)...")
t0 = time.time()

try:
    import torch
    from src.gnn.trainer import GNNTrainer

    if not os.path.exists("data/raw/graph.pt"):
        print("[WARNING] graph.pt missing — skipping GNN training. Run Step 4 first.")
    else:
        trainer = GNNTrainer(
            graph_path="data/raw/graph.pt",
            labels_df=df,
            node_mapping_path="data/raw/node_mapping.csv"
        )
        best_auc = trainer.train(max_epochs=200, lr=0.01)
        elapsed = time.time() - t0
        print(f"[OK] GNN trained: Mean CV AUC = {best_auc:.4f} ({elapsed:.1f}s)")

        if best_auc < 0.90:
            print(f"[WARNING] Mean AUC {best_auc:.4f} is below 0.90.")
            print("         This may affect vulnerability predictions.")
            print("         Consider: more epochs, different lr, or check label quality.")
        elif best_auc >= 0.95:
            print(f"[OK] AUC target 0.95 met — model saved to data/raw/gnn_model.pt")
        else:
            print(f"[OK] AUC {best_auc:.4f} — acceptable. Target is 0.95.")

except ImportError as e:
    print(f"[WARNING] PyTorch/PyG not available — skipping GNN training: {e}")
    print("         Install PyTorch and torch_geometric, then re-run main.py")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5.5: Monte Carlo Stress Test — generate mc_results.csv + metrics JSON
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 5.5: Running Monte Carlo Stress Test (1000 runs)...")
t0 = time.time()

try:
    from src.simulation.montecarlo import MonteCarloStressTester
    from src.data.loader import BankSnapshot

    # Build COVID-peak snapshots for the stress test
    covid_df = df[df["date"] == "2020-03-15"]
    if len(covid_df) == 0:
        # Fallback: use nearest available date
        covid_df = df[df["date"] >= "2020-03-14"].head(330)

    # Aggregate to 33 bank snapshots
    mc_snapshots = []
    for bank_name in covid_df["institution_id"].unique():
        bank_rows = covid_df[covid_df["institution_id"] == bank_name].iloc[0]
        def sf(key, default=0.0):
            try:
                v = bank_rows.get(key, default)
                return default if pd.isna(v) else float(v)
            except: return default
        def si(key, default=0):
            try:
                v = bank_rows.get(key, default)
                return default if pd.isna(v) else int(v)
            except: return default

        mc_snapshots.append(BankSnapshot(
            institution_id=str(bank_rows.get("institution_id", "")),
            bank_short=str(bank_rows.get("bank_short", "")),
            bank_type=str(bank_rows.get("bank_type", "periphery")),
            date="2020-03-15",
            tier1_capital_ratio=sf("tier1_capital_ratio"),
            total_capital_ratio=sf("total_capital_ratio"),
            leverage_ratio=sf("leverage_ratio"),
            capital_buffer=sf("capital_buffer"),
            rwa_density=sf("rwa_density"),
            lcr=sf("lcr"), nsfr=sf("nsfr"),
            short_term_funding_ratio=sf("short_term_funding_ratio"),
            npl_ratio=sf("npl_ratio"),
            loan_loss_provision_ratio=sf("loan_loss_provision_ratio"),
            cds_spread_bps=sf("cds_spread_bps"),
            equity_volatility_30d=sf("equity_volatility_30d"),
            beta_to_financials=sf("beta_to_financials"),
            interbank_borrowing_ratio=sf("interbank_borrowing_ratio"),
            interbank_exposure_ratio=sf("interbank_exposure_ratio"),
            network_degree=sf("network_degree"),
            eigenvector_centrality=sf("eigenvector_centrality"),
            gsib_score=sf("gsib_score"),
            total_assets_usd_bn=sf("total_assets_usd_bn", 1.0),
            roe=sf("roe"), cost_to_income=sf("cost_to_income"),
            concentration_hhi=sf("concentration_hhi"),
            vulnerability_label_injected=sf("vulnerability_label_injected"),
            distress_event_injected=si("distress_event_injected"),
            cds_spread_bps_boosted=sf("cds_spread_bps_boosted"),
            equity_volatility_boosted=sf("equity_volatility_boosted"),
        ))

    if mc_snapshots:
        mc = MonteCarloStressTester(n_runs=1000)
        mc_result = mc.run_sync(
            snapshots=mc_snapshots,
            shock_bank="Yes Bank",
            magnitude=0.85,
            shock_type="Liquidity Crisis",
        )
        mc.save_outputs(mc_result)
        elapsed = time.time() - t0
        print(f"[OK] Monte Carlo complete: SRI CI = [{mc_result.sri_ci_low:.3f}, "
              f"{mc_result.sri_ci_high:.3f}] ({elapsed:.1f}s)")
    else:
        print("[WARNING] No COVID-peak snapshots found — skipping MC stress test.")

except Exception as e:
    print(f"[WARNING] Monte Carlo stress test failed: {e}")
    import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Initialize security — create users.db with all 35 accounts
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 6: Initializing security database...")
t0 = time.time()

from src.security.auth import AuthManager

auth = AuthManager("data/security/users.db")
auth.initialize_all_users()

elapsed = time.time() - t0
print(f"[OK] users.db initialized ({elapsed:.1f}s)")

# Verify user count
import sqlite3
conn = sqlite3.connect("data/security/users.db")
user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
role_counts = conn.execute(
    "SELECT role, COUNT(*) FROM users GROUP BY role"
).fetchall()
conn.close()

print(f"[OK] Total users: {user_count}")
for role, count in role_counts:
    print(f"     {role}: {count}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Initialize live data cache
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 7: Initializing live data cache...")

from src.data.cache import LiveCache

cache = LiveCache("data/cache.db")
cache.initialize()
print("[OK] cache.db initialized (SQLite WAL mode active)")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL: Summary and next steps
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PIPELINE COMPLETE — VALIDATION SUMMARY")
print("=" * 70)

checks = [
    ("banks_daily_data.csv", os.path.exists("data/raw/banks_daily_data.csv")),
    ("sri_series.csv", os.path.exists("data/raw/sri_series.csv")),
    ("graph.pt", os.path.exists("data/raw/graph.pt")),
    ("node_mapping.csv", os.path.exists("data/raw/node_mapping.csv")),
    ("node_positions.json", os.path.exists("data/raw/node_positions.json")),
    ("feature_names.txt", os.path.exists("data/raw/feature_names.txt")),
    ("gnn_model.pt", os.path.exists("data/raw/gnn_model.pt")),
    ("gnn_labels.csv", os.path.exists("data/raw/gnn_labels.csv")),
    ("mc_results.csv", os.path.exists("data/raw/mc_results.csv")),
    ("systemic_risk_metrics.json", os.path.exists("data/raw/systemic_risk_metrics.json")),
    ("users.db", os.path.exists("data/security/users.db")),
    ("cache.db", os.path.exists("data/cache.db")),
]

all_ok = True
for name, exists in checks:
    status = "[OK]" if exists else "[MISSING]"
    if not exists:
        all_ok = False
    print(f"  {status:10s} {name}")

print()
if all_ok:
    print("  All required files present.")
    print("\n[READY] Start the dashboard:")
    print("        python dashboard/app.py")
    print("        Open browser: http://localhost:8050")
    print("        Multi-device: http://<server-ip>:8050\n")
else:
    print("  Some files are missing (likely PyTorch not installed).")
    print("  Install PyTorch + torch_geometric and re-run main.py.\n")
