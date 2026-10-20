"""
Financial Contagion Monitor — dashboard/app.py
M.S. Ramaiah Institute of Technology | AIP67 | 2026

Dash application entry point.
Initializes all modules, registers Flask API routes, mounts layout,
registers callbacks, and starts the server.

Run with:
    python dashboard/app.py
    # Open: http://localhost:8050
    # Multi-device demo: http://<server-ip>:8050

IMPORTANT: Run python main.py FIRST before starting this file.
           app.py will exit cleanly with instructions if required files are missing.
"""

import json
import os
import sys
from datetime import datetime

# ── Ensure project root is on sys.path and CWD ──────────────────────────────
# When run as 'python dashboard/app.py', the CWD is the project root but
# 'dashboard/' isn't recognized as a package unless root is in sys.path.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

# ── Startup Guard — check all required files before importing heavy modules ───
REQUIRED_FILES = [
    "data/raw/banks_daily_data.csv",
    "data/raw/graph.pt",
    "data/raw/gnn_model.pt",
    "data/raw/node_mapping.csv",
    "data/raw/node_positions.json",
    "data/raw/sri_series.csv",
    "data/raw/feature_names.txt",
]

missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing:
    print("\n[ERROR] Required files missing. Run python main.py first.")
    for f in missing:
        print(f"  MISSING: {f}")
    sys.exit(1)

# ── Global Playback State (server-side, shared across all sessions) ───────────
# Only RBI (SUPERADMIN) can write; all other clients read and sync.
import threading
GLOBAL_PLAYBACK_STATE = {
    "date": "2020-03-15",
    "playing": False,
    "speed": 2,
    "updated_by": "",
}
GLOBAL_PLAYBACK_LOCK = threading.Lock()

# ── Imports ───────────────────────────────────────────────────────────────────
import dash
import dash_bootstrap_components as dbc
from flask import jsonify, request

from src.security.auth import AuthManager
from src.data.loader import RealDataLoader
from src.data.cache import LiveCache
from src.data.live_loader import LiveDataLoader
from src.graph.builder import load_node_positions
from src.gnn.inference import get_inference_engine
from src.simulation.contagion import ContagionSimulator
from src.simulation.montecarlo import MonteCarloStressTester
from src.replay.engine import ReplayEngine
from src.ews.detector import EarlyWarningDetector
from src.alerts.manager import AlertManager
from src.reports.llm_client import OllamaClient
from src.reports.generator import ReportGenerator
from src.gnn.explainer import GNNBankExplainer
from src.reports.pdf_export import PDFExporter

import plotly.io as pio
pio.templates.default = "plotly_dark"

# ── App Initialization ────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Financial Contagion Monitor",
    update_title=None,
    url_base_pathname="/dashboard/",
)
server = app.server   # Flask server — used for API routes

from flask import send_from_directory

@server.route("/")
def serve_landing():
    return send_from_directory("../Website/rbi-command-portal", "index.html")

@server.route("/<path:filename>")
def serve_static(filename):
    # Only serve if it exists in the portal directory, else 404
    portal_dir = os.path.join(PROJECT_ROOT, "Website", "rbi-command-portal")
    if os.path.exists(os.path.join(portal_dir, filename)):
        return send_from_directory(portal_dir, filename)
    return "Not Found", 404


# ── Module Initialization ─────────────────────────────────────────────────────
print("[app.py] Initializing modules...")

# Security
auth = AuthManager("data/security/users.db")

# Data
loader = RealDataLoader("data/raw/")
cache  = LiveCache("data/cache.db")

# GNN
gnn_engine = get_inference_engine(
    model_path="data/raw/gnn_model.pt",
    graph_path="data/raw/graph.pt",
    node_mapping_path="data/raw/node_mapping.csv",
)

# EWS
ews = EarlyWarningDetector()

# Replay engine
mc             = MonteCarloStressTester(n_runs=1000)
replay_engine  = ReplayEngine(loader, ews=ews, gnn_engine=gnn_engine, mc=mc)
replay_engine.load()
print("[app.py] ReplayEngine loaded")

# Live loader
live_loader = LiveDataLoader(cache, loader, ews=ews, gnn_engine=gnn_engine)

# Simulation
contagion  = ContagionSimulator()

# Explainer
explainer = GNNBankExplainer(gnn_engine)

# Alerts
alert_manager = AlertManager("data/security/users.db", ews)

# Reports
ollama    = OllamaClient()
generator = ReportGenerator(ollama, contagion, mc, explainer, ews)
pdf_exp   = PDFExporter()

# Node positions for 3D network
NODE_POSITIONS = load_node_positions("data/raw/node_positions.json")

# Ollama availability check (non-fatal)
if ollama.is_available():
    print(f"[app.py] Ollama available — model: {ollama.get_model_name()}")
else:
    print("[app.py] WARNING: Ollama not running. LLM reports will use template fallback.")
    print("         Start Ollama: ollama serve")

# ── CORS Configuration ────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:8050,http://127.0.0.1:8050"
).split(",")

@server.after_request
def apply_cors(response):
    origin = request.headers.get("Origin", "")
    if any(origin.startswith(o.strip()) for o in ALLOWED_ORIGINS):
        response.headers["Access-Control-Allow-Origin"]  = origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# ── Flask API Routes ──────────────────────────────────────────────────────────

@server.route("/api/login", methods=["POST"])
def api_login():
    """Login endpoint — rate limited, returns JWT on success."""
    data = request.get_json() or {}
    result = auth.login(
        data.get("username", ""),
        data.get("password", ""),
        request.remote_addr,
    )
    if result["success"]:
        return jsonify({"token": result["token"]}), 200
    return jsonify({"error": result["error"]}), 401


@server.route("/api/alerts", methods=["GET"])
def api_get_alerts():
    """Returns unacknowledged alerts for the authenticated user's bank."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    payload = auth.verify_token(token)
    if not payload:
        return jsonify([]), 401
    since  = request.args.get("since", "2000-01-01T00:00:00")
    alerts = alert_manager.get_alerts(payload["bank_name"], since)
    return jsonify([{
        "alert_id":       a.alert_id,
        "target_bank":    a.target_bank,
        "severity":       a.severity,
        "title":          a.title,
        "message":        a.message,
        "source":         a.source,
        "triggered_at":   a.triggered_at,
        "acknowledged":   a.acknowledged,
    } for a in alerts]), 200


@server.route("/api/all-alerts", methods=["GET"])
def api_all_alerts():
    """Returns all alerts — SUPERADMIN only."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    payload = auth.verify_token(token)
    if not payload or payload.get("role") != "SUPERADMIN":
        return jsonify({"error": "Forbidden"}), 403
    alerts = alert_manager.get_all_alerts()
    return jsonify([{
        "alert_id":     a.alert_id,
        "target_bank":  a.target_bank,
        "severity":     a.severity,
        "title":        a.title,
        "message":      a.message,
        "triggered_at": a.triggered_at,
        "acknowledged": a.acknowledged,
    } for a in alerts]), 200


@server.route("/api/acknowledge-alert", methods=["POST"])
def api_acknowledge():
    """Marks an alert as acknowledged (own bank only)."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    payload = auth.verify_token(token)
    if not payload:
        return jsonify({}), 401
    data = request.get_json() or {}
    ok = alert_manager.acknowledge(
        int(data.get("alert_id", 0)),
        payload["bank_name"],
    )
    return jsonify({"ok": ok}), 200


@server.route("/api/force-alert", methods=["POST"])
def api_force_alert():
    """Manually fires an alert — SUPERADMIN only."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    payload = auth.verify_token(token)
    if not payload or payload.get("role") != "SUPERADMIN":
        return jsonify({"error": "Forbidden"}), 403
    data = request.get_json() or {}
    aid = alert_manager.force_alert(
        target_bank=data.get("target_bank", ""),
        message=data.get("message", "RBI Manual Alert"),
        severity=data.get("severity", "CRITICAL"),
    )
    return jsonify({"ok": True, "alert_id": aid}), 200


@server.route("/api/audit-log", methods=["GET"])
def api_audit_log():
    """Returns audit log integrity status — SUPERADMIN only."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    payload = auth.verify_token(token)
    if not payload or payload.get("role") != "SUPERADMIN":
        return jsonify({"error": "Forbidden"}), 403
    intact = auth.verify_audit_chain()
    return jsonify({"audit_chain_intact": intact}), 200


@server.route("/api/global-date", methods=["GET"])
def api_get_global_date():
    """Returns the current global playback date set by RBI."""
    with GLOBAL_PLAYBACK_LOCK:
        return jsonify(GLOBAL_PLAYBACK_STATE), 200


@server.route("/api/global-date", methods=["POST"])
def api_set_global_date():
    """Sets the global playback date — SUPERADMIN only."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    payload = auth.verify_token(token)
    if not payload or payload.get("role") != "SUPERADMIN":
        return jsonify({"error": "Forbidden"}), 403
    data = request.get_json() or {}
    with GLOBAL_PLAYBACK_LOCK:
        if "date" in data:
            GLOBAL_PLAYBACK_STATE["date"] = data["date"]
        if "playing" in data:
            GLOBAL_PLAYBACK_STATE["playing"] = data["playing"]
        if "speed" in data:
            GLOBAL_PLAYBACK_STATE["speed"] = data["speed"]
        GLOBAL_PLAYBACK_STATE["updated_by"] = payload.get("bank_short", "RBI")
    return jsonify({"ok": True}), 200


# ── Bank Data Ingestion API (Deploy-Ready) ────────────────────────────────────
# In a real-world deployment, each of the 33 banks would push their internal
# financial metrics (NPA ratio, capital adequacy, liquidity ratios, etc.) to
# this endpoint on a scheduled basis (e.g., every 24 hours).
#
# This endpoint:
#   1. Validates JWT authentication (only authenticated bank users can submit)
#   2. Validates the payload schema against the 14 GNN input features
#   3. Logs the submission for audit trail
#   4. Stores the data for the next GNN inference cycle
#
# Currently operates in "dry-run" mode — accepts and validates payloads but
# does not overwrite the historical CSV data. In production, this would feed
# directly into the live GNN inference pipeline.
# ──────────────────────────────────────────────────────────────────────────────

# In-memory store for ingested bank data (production would use a database)
_INGESTED_BANK_DATA = {}

# The 14 GNN features that banks must report
_REQUIRED_FEATURES = [
    "tier1_capital_ratio", "total_capital_ratio", "leverage_ratio",
    "lcr", "nsfr", "npl_ratio", "cds_spread_bps",
    "equity_volatility_30d", "interbank_borrowing_ratio",
    "interbank_exposure_ratio", "network_degree",
    "eigenvector_centrality", "gsib_score", "capital_buffer",
]


@server.route("/api/v1/ingest_bank_data", methods=["POST"])
def api_ingest_bank_data():
    """
    Bank Data Ingestion Endpoint — accepts financial metrics from individual banks.

    Expected JSON payload:
    {
        "institution_id": "Yes Bank",
        "reporting_date": "2026-05-04",
        "metrics": {
            "tier1_capital_ratio": 0.085,
            "total_capital_ratio": 0.112,
            "leverage_ratio": 0.045,
            "lcr": 1.15,
            "nsfr": 1.08,
            "npl_ratio": 0.062,
            "cds_spread_bps": 285.0,
            "equity_volatility_30d": 0.38,
            "interbank_borrowing_ratio": 0.14,
            "interbank_exposure_ratio": 0.09,
            "network_degree": 3.0,
            "eigenvector_centrality": 0.15,
            "gsib_score": 0.02,
            "capital_buffer": 0.025
        }
    }

    Returns:
        201: Data accepted and stored
        400: Invalid payload (missing fields, bad values)
        401: Authentication failed
        403: Insufficient permissions
    """
    # ── Authentication ────────────────────────────────────────────────────
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    payload = auth.verify_token(token)
    if not payload:
        return jsonify({"error": "Authentication required. Provide a valid JWT."}), 401

    # All authenticated roles can submit data for their own bank
    # SUPERADMIN can submit for any bank
    submitter_bank = payload.get("bank_name", "")
    submitter_role = payload.get("role", "")

    # ── Parse payload ─────────────────────────────────────────────────────
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON payload required."}), 400

    institution_id = data.get("institution_id", "")
    reporting_date = data.get("reporting_date", "")
    metrics = data.get("metrics", {})

    if not institution_id:
        return jsonify({"error": "Missing required field: institution_id"}), 400
    if not reporting_date:
        return jsonify({"error": "Missing required field: reporting_date"}), 400
    if not metrics or not isinstance(metrics, dict):
        return jsonify({"error": "Missing or invalid 'metrics' object."}), 400

    # ── Authorization: banks can only submit their own data ───────────────
    if submitter_role != "SUPERADMIN" and institution_id != submitter_bank:
        return jsonify({
            "error": f"Forbidden: {submitter_bank} cannot submit data for {institution_id}. "
                     f"Only SUPERADMIN can submit on behalf of other banks."
        }), 403

    # ── Validate metrics schema ───────────────────────────────────────────
    missing_features = [f for f in _REQUIRED_FEATURES if f not in metrics]
    if missing_features:
        return jsonify({
            "error": f"Missing required metrics: {', '.join(missing_features)}",
            "required_features": _REQUIRED_FEATURES,
        }), 400

    # Validate all values are numeric
    invalid_values = {}
    for feat in _REQUIRED_FEATURES:
        val = metrics[feat]
        if not isinstance(val, (int, float)):
            invalid_values[feat] = f"Expected numeric, got {type(val).__name__}"
    if invalid_values:
        return jsonify({
            "error": "Non-numeric values in metrics.",
            "invalid_fields": invalid_values,
        }), 400

    # ── Store ingested data ───────────────────────────────────────────────
    _INGESTED_BANK_DATA[institution_id] = {
        "institution_id": institution_id,
        "reporting_date": reporting_date,
        "metrics": {f: float(metrics[f]) for f in _REQUIRED_FEATURES},
        "submitted_by": payload.get("username", "unknown"),
        "submitted_at": datetime.now().isoformat(),
    }

    print(f"[INGEST] Bank data received: {institution_id} for {reporting_date} "
          f"(submitted by {payload.get('username', 'unknown')})")

    return jsonify({
        "status": "accepted",
        "institution_id": institution_id,
        "reporting_date": reporting_date,
        "features_received": len(_REQUIRED_FEATURES),
        "message": f"Data for {institution_id} ingested successfully. "
                   f"Will be used in next GNN inference cycle.",
    }), 201


@server.route("/api/v1/ingested_data", methods=["GET"])
def api_get_ingested_data():
    """Returns all currently ingested bank data — SUPERADMIN only."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    payload = auth.verify_token(token)
    if not payload or payload.get("role") != "SUPERADMIN":
        return jsonify({"error": "Forbidden — SUPERADMIN only."}), 403

    return jsonify({
        "count": len(_INGESTED_BANK_DATA),
        "banks": list(_INGESTED_BANK_DATA.keys()),
        "data": _INGESTED_BANK_DATA,
    }), 200


# ── Layout + Callbacks ────────────────────────────────────────────────────────
# Import after app is created to avoid circular imports
from dashboard.layout import build_layout
from dashboard.callbacks import register_callbacks

app.layout = build_layout()

register_callbacks(
    app=app,
    auth=auth,
    replay_engine=replay_engine,
    live_loader=live_loader,
    alert_manager=alert_manager,
    generator=generator,
    pdf_exp=pdf_exp,
    contagion=contagion,
    mc=mc,
    explainer=explainer,
    ews=ews,
    node_positions=NODE_POSITIONS,
)

print("[app.py] Dashboard ready.")

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.getenv("DASH_PORT", "8050"))
    host  = os.getenv("DASH_HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "False").lower() == "true"

    print(f"\n{'='*60}")
    print(f"  Financial Contagion Monitor")
    print(f"  http://localhost:{port}")
    print(f"  Multi-device: http://<server-ip>:{port}")
    print(f"{'='*60}\n")

    app.run(debug=debug, host=host, port=port)
