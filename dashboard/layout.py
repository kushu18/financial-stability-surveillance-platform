"""
Financial Contagion Monitor — dashboard/layout.py
M.S. Ramaiah Institute of Technology | AIP67 | 2026

Complete UI layout — all 12 sections, all component IDs.
Returns a single html.Div from build_layout() — mounted by app.py.

Component ID reference (used by callbacks.py):
  Auth:        login-page, username-input, password-input, login-btn, login-error
  Stores:      session-store, current-day-result, data-mode-store, last-alert-ts, sim-state-store
  Intervals:   live-update-interval, alert-poll-interval, play-interval, sim-step-interval
  Header:      header-bank-name, data-mode-toggle, market-status-badge, logout-btn
  Splash:      splash-screen, splash-status, splash-banks-list, splash-timer
  Alerts:      alert-overlay, alert-modal, alert-title, alert-message, alert-source, ack-btn
  KPI:         kpi-date, kpi-sri, kpi-collapse-pct, kpi-stressed, kpi-day-count
  Banners:     crisis-banner, crisis-banner-text, ews-banner
  Network:     network-graph, clear-selection-btn, compare-mode-btn, second-degree-toggle
  Sidebar:     date-picker-single, step-back-btn, step-forward-btn, play-btn,
               speed-slider, shock-bank-dropdown, shock-magnitude-slider,
               shock-type-radio, shock-steps-slider, run-shock-btn,
               reset-shock-btn, animate-steps-check
  Simulation:  sim-banner, cascade-table, rbi-flag-banner, injection-estimate
  Report:      generate-report-btn, report-output, report-spinner, export-report-btn, pdf-download
  SRI Chart:   sri-chart
  Stock Chart: stock-bank-dropdown, stock-chart
  Tables:      top-risk-table, all-banks-table
  Stats Panel: bank-stats-panel
  Compare:     compare-panel, date-a-picker, date-b-picker, compare-left, compare-right
  Contradiction: contradiction-panel, live-output-panel, historical-benchmark-panel
  Status:      status-bar
"""

import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# All 33 bank names for dropdowns
ALL_BANK_NAMES = [
    "State Bank of India (SBI)", "HDFC Bank", "ICICI Bank",
    "Axis Bank", "Kotak Mahindra Bank", "AU Small Finance Bank",
    "Bandhan Bank", "Bank of Baroda", "Bank of India",
    "Bank of Maharashtra", "Canara Bank", "Central Bank of India",
    "City Union Bank", "CSB Bank", "Equitas Small Finance Bank",
    "Federal Bank", "IDBI Bank", "IDFC FIRST Bank",
    "Indian Bank", "Indian Overseas Bank (IOB)", "IndusInd Bank",
    "J&K Bank", "Karnataka Bank", "Karur Vysya Bank",
    "Punjab & Sind Bank", "Punjab National Bank (PNB)", "RBL Bank",
    "South Indian Bank", "Tamilnad Mercantile Bank", "UCO Bank",
    "Ujjivan Small Finance Bank", "Union Bank of India", "Yes Bank",
]

SHOCK_TYPES = [
    "Liquidity Crisis", "Capital Shortfall", "NPL Spike", "Market Panic"
]


def _dark_empty_fig(height=380, is_3d=True):
    """Returns an empty dark-background figure for initial graph state."""
    fig = go.Figure()
    if is_3d:
        fig.update_layout(
            paper_bgcolor="#06080d",
            scene=dict(bgcolor="#06080d",
                       xaxis=dict(showgrid=False, showticklabels=False, backgroundcolor="#06080d", showspikes=False),
                       yaxis=dict(showgrid=False, showticklabels=False, backgroundcolor="#06080d", showspikes=False),
                       zaxis=dict(showgrid=False, showticklabels=False, backgroundcolor="#06080d", showspikes=False)),
            margin=dict(l=0, r=0, t=0, b=0),
            height=height,
        )
    else:
        fig.update_layout(
            paper_bgcolor="#06080d", plot_bgcolor="#0b1018",
            margin=dict(l=40, r=20, t=10, b=30),
            height=height,
            xaxis=dict(showgrid=False, color="#b5bdc8", tickfont=dict(size=9)),
            yaxis=dict(showgrid=True, gridcolor="#21262D", color="#b5bdc8", tickfont=dict(size=9)),
        )
    return fig


def build_layout() -> html.Div:
    """Returns the complete app layout."""
    return html.Div(
        id="app-root",
        style={"backgroundColor": "#06080d", "minHeight": "100vh",
               "fontFamily": '"Segoe UI Variable", "Segoe UI", Tahoma, sans-serif'},
        children=[
            # ── Client-side Stores ──────────────────────────────────────────
            dcc.Store(id="session-store",      storage_type="session", data={}),
            dcc.Store(id="current-day-result", storage_type="memory",  data={}),
            dcc.Store(id="data-mode-store",    storage_type="session", data="historical"),
            dcc.Store(id="last-alert-ts",      storage_type="memory",  data="2000-01-01T00:00:00"),
            dcc.Store(id="sim-state-store",    storage_type="memory",  data={"step": 0, "result": None}),
            dcc.Store(id="selected-bank-store", storage_type="memory",  data=""),
            dcc.Store(id="selected-live-bank-store", storage_type="memory", data=""),
            dcc.Store(id="siren-state-store",  storage_type="memory",  data={"active": False, "affected_banks": [], "origin": "", "crisis_name": ""}),
            dcc.Store(id="siren-fired-store",  storage_type="memory",  data={"crisis": ""}),
            dcc.Store(id="live-gnn-result",    storage_type="memory",  data={}),

            # ── Intervals ───────────────────────────────────────────────────
            dcc.Interval(id="live-update-interval",  interval=300_000, disabled=True,  n_intervals=0),
            dcc.Interval(id="alert-poll-interval",   interval=3_000,   disabled=False, n_intervals=0),
            dcc.Interval(id="play-interval",         interval=1_000,   disabled=True,  n_intervals=0),
            dcc.Interval(id="sim-step-interval",     interval=1_200,   disabled=True,  n_intervals=0),
            dcc.Interval(id="splash-timer",          interval=3_000,   disabled=True,  n_intervals=0, max_intervals=1),
            dcc.Interval(id="siren-dismiss-interval",interval=30_000,  disabled=True,  n_intervals=0, max_intervals=1),
            dcc.Interval(id="live-gnn-interval",     interval=60_000,  disabled=True,  n_intervals=0),
            dcc.Interval(id="global-sync-interval",  interval=1_000,   disabled=False, n_intervals=0),
            dcc.Interval(id="live-mc-poll-interval", interval=3_000,   disabled=True,  n_intervals=0),

            # ── PDF Download ────────────────────────────────────────────────
            dcc.Download(id="pdf-download"),
            dcc.Download(id="live-pdf-download"),

            # ── Login Page ──────────────────────────────────────────────────
            _build_login_page(),

            # ── Main Content (hidden until login) ───────────────────────────
            html.Div(
                id="main-content",
                style={"display": "none"},
                children=[
                    _build_alert_overlay(),
                    _build_siren_overlay(),
                    _build_splash_screen(),
                    _build_header(),
                    _build_crisis_banners(),
                    _build_kpi_row(),
                    html.Div(
                        style={"display": "flex", "gap": "12px",
                               "padding": "0 16px 16px 16px"},
                        children=[
                            _build_sidebar(),
                            _build_main_panel(),
                        ]
                    ),
                    _build_status_bar(),
                ]
            ),
        ]
    )


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

def _build_login_page() -> html.Div:
    return html.Div(
        id="login-page",
        style={
            "display": "flex", "alignItems": "center", "justifyContent": "center",
            "minHeight": "100vh", "backgroundColor": "#06080d",
        },
        children=[
            html.Div(
                style={
                    "width": "400px", "padding": "40px",
                    "backgroundColor": "#0b1018",
                    "border": "1px solid rgba(181, 150, 77, 0.22)",
                    "borderTop": "3px solid #c79b45",
                },
                children=[
                    html.H2("FCM", style={"color": "#c79b45", "margin": "0 0 4px 0",
                                              "fontFamily": "inherit"}),
                    html.P("Financial Contagion Monitor",
                           style={"color": "#b5bdc8", "margin": "0 0 8px 0", "fontSize": "12px"}),
                    html.P("M.S. Ramaiah Institute of Technology | AIP67",
                           style={"color": "#b5bdc8", "margin": "0 0 32px 0", "fontSize": "11px"}),

                    html.Label("Username", style={"color": "#b5bdc8", "fontSize": "12px"}),
                    dcc.Input(
                        id="username-input",
                        type="text", placeholder="Enter username",
                        style=_input_style(),
                        debounce=False,
                        n_submit=0,
                    ),
                    html.Br(),

                    html.Label("Password", style={"color": "#b5bdc8", "fontSize": "12px",
                                                   "marginTop": "16px", "display": "block"}),
                    dcc.Input(
                        id="password-input",
                        type="password", placeholder="Enter password",
                        style=_input_style(),
                        debounce=False,
                        n_submit=0,
                    ),
                    html.Br(),

                    html.Button(
                        "LOGIN", id="login-btn",
                        n_clicks=0,
                        style={
                            "width": "100%", "padding": "12px",
                            "backgroundColor": "#c79b45",
                            "color": "#e9ecef", "border": "none",
                            "cursor": "pointer", "fontFamily": "inherit",
                            "fontSize": "14px", "fontWeight": "bold",
                            "marginTop": "24px", "letterSpacing": "2px",
                        }
                    ),

                    html.Div(
                        id="login-error",
                        style={"color": "#d36d5b", "fontSize": "12px",
                               "marginTop": "12px", "minHeight": "20px"},
                    ),

                    html.Hr(style={"borderColor": "rgba(181, 150, 77, 0.22)", "margin": "24px 0 16px 0"}),
                    html.P("Demo credentials: demo_user / Demo@ReadOnly#2026!",
                           style={"color": "#b5bdc8", "fontSize": "10px", "margin": 0}),
                ]
            )
        ]
    )


# ══════════════════════════════════════════════════════════════════════════════
# ALERT OVERLAY
# ══════════════════════════════════════════════════════════════════════════════

def _build_alert_overlay() -> html.Div:
    return html.Div(
        id="alert-overlay",
        style={"display": "none"},  # shown by callback on alert
        children=[
            html.Div(
                id="alert-modal",
                style={
                    "position": "fixed", "top": "50%", "left": "50%",
                    "transform": "translate(-50%, -50%)",
                    "backgroundColor": "#0b1018",
                    "border": "2px solid #d36d5b",
                    "padding": "32px", "maxWidth": "500px", "width": "90%",
                    "textAlign": "center", "zIndex": "10000",
                    "fontFamily": "inherit",
                },
                children=[
                    html.Div("⚠", style={"fontSize": "48px", "marginBottom": "16px"}),
                    html.H3(id="alert-title",
                            style={"color": "#d36d5b", "margin": "0 0 12px 0",
                                   "fontFamily": "inherit"}),
                    html.P(id="alert-message",
                           style={"color": "#e9ecef", "fontSize": "13px",
                                  "margin": "0 0 8px 0"}),
                    html.P(id="alert-source",
                           style={"color": "#b5bdc8", "fontSize": "11px",
                                  "margin": "0 0 24px 0"}),
                    html.Button(
                        "ACKNOWLEDGE", id="ack-btn",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#d36d5b", "color": "#e9ecef",
                            "border": "none", "padding": "10px 32px",
                            "cursor": "pointer", "fontFamily": "inherit",
                            "fontWeight": "bold", "letterSpacing": "2px",
                        }
                    ),
                ]
            )
        ]
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIREN OVERLAY — Full-screen crisis warning (role-aware)
# ══════════════════════════════════════════════════════════════════════════════

def _build_siren_overlay() -> html.Div:
    """Full-screen red warning overlay with siren audio. Only shown to affected banks + RBI."""
    return html.Div(
        id="siren-overlay",
        style={"display": "none"},
        children=[
            html.Div(
                style={
                    "position": "fixed", "top": 0, "left": 0,
                    "width": "100vw", "height": "100vh",
                    "backgroundColor": "rgba(200,0,0,0.95)",
                    "zIndex": "99999",
                    "display": "flex", "alignItems": "center",
                    "justifyContent": "center", "flexDirection": "column",
                    "textAlign": "center",
                    "animation": "siren-pulse 1s ease-in-out infinite alternate",
                },
                children=[
                    html.Div("⚠", style={"fontSize": "120px", "marginBottom": "10px",
                                          "animation": "siren-shake 0.3s ease-in-out infinite"}),
                    html.H1("CRISIS ALERT", style={
                        "color": "#e9ecef", "fontSize": "56px", "fontWeight": "bold",
                        "letterSpacing": "12px", "margin": "0 0 10px 0",
                        "fontFamily": "inherit",
                        "textShadow": "0 0 30px rgba(255,255,255,0.5)",
                    }),
                    html.Div(id="siren-crisis-name", style={
                        "color": "#d0a24f", "fontSize": "28px", "fontWeight": "bold",
                        "letterSpacing": "4px", "marginBottom": "20px",
                    }),
                    html.Div(id="siren-origin-bank", style={
                        "color": "#e9ecef", "fontSize": "18px", "marginBottom": "10px",
                    }),
                    html.Div(id="siren-affected-list", style={
                        "color": "rgba(255,255,255,0.8)", "fontSize": "14px",
                        "maxWidth": "600px", "lineHeight": "1.8",
                        "marginBottom": "30px",
                    }),
                    html.Div("CONTAGION CHAIN DETECTED — IMMEDIATE ACTION REQUIRED",
                             style={"color": "#d0a24f", "fontSize": "14px",
                                    "letterSpacing": "3px", "marginBottom": "30px",
                                    "animation": "siren-blink 0.8s ease-in-out infinite"}),
                    html.Button(
                        "ACKNOWLEDGE & DISMISS", id="siren-dismiss-btn",
                        n_clicks=0,
                        style={
                            "backgroundColor": "transparent",
                            "color": "#e9ecef", "border": "3px solid #e9ecef",
                            "padding": "14px 40px", "cursor": "pointer",
                            "fontFamily": "inherit", "fontSize": "16px",
                            "fontWeight": "bold", "letterSpacing": "3px",
                        }
                    ),
                    html.Div("Auto-dismiss in 30 seconds",
                             style={"color": "rgba(255,255,255,0.5)", "fontSize": "11px",
                                    "marginTop": "12px"}),
                    # Hidden audio element for siren — uses Web Audio oscillator via clientside
                    html.Div(id="siren-audio-trigger", style={"display": "none"}),
                ]
            ),
            # CSS animations injected via style tag
            html.Div(
                dangerously_allow_html=True,
                children=[],
                style={"display": "none"},
            ) if False else None,
        ]
    )


# ══════════════════════════════════════════════════════════════════════════════
# SPLASH SCREEN
# ══════════════════════════════════════════════════════════════════════════════

def _build_splash_screen() -> html.Div:
    return html.Div(
        id="splash-screen",
        style={
            "display": "none",
            "position": "fixed", "top": 0, "left": 0,
            "width": "100vw", "height": "100vh",
            "backgroundColor": "#06080d", "zIndex": "9000",
            "alignItems": "center", "justifyContent": "center",
            "flexDirection": "column",
        },
        children=[
            html.H1("FINANCIAL CONTAGION MONITOR",
                    style={"color": "#c79b45", "fontFamily": "inherit",
                           "letterSpacing": "4px", "fontSize": "24px"}),
            html.Div(id="splash-status",
                     style={"color": "#e9ecef", "fontSize": "32px",
                            "fontWeight": "bold", "margin": "20px 0",
                            "fontFamily": "inherit"}),
            html.Div(id="splash-banks-list",
                     style={"color": "#b5bdc8", "fontSize": "12px",
                            "maxWidth": "600px", "textAlign": "center"}),
        ]
    )


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

def _build_header() -> html.Div:
    return html.Div(
        style={
            "display": "flex", "alignItems": "center",
            "justifyContent": "space-between",
            "padding": "10px 16px",
            "backgroundColor": "#0b1018",
            "borderBottom": "2px solid #c79b45",
        },
        children=[
            # Left: project name + bank name
            html.Div([
                html.Span("FCM  ",
                          style={"color": "#c79b45", "fontWeight": "bold",
                                 "fontSize": "14px"}),
                html.Span("▸ ", style={"color": "#b5bdc8"}),
                html.Span(id="header-bank-name",
                          style={"color": "#e9ecef", "fontSize": "14px"}),
            ]),

            # Center: mode toggle
            html.Div([
                dcc.RadioItems(
                    id="data-mode-toggle",
                    options=[
                        {"label": "HISTORICAL REPLAY", "value": "historical"},
                        {"label": "LIVE MARKET", "value": "live"},
                    ],
                    value="historical",
                    inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={
                        "marginRight": "16px", "fontSize": "12px",
                        "cursor": "pointer", "color": "#b5bdc8",
                    },
                )
            ]),

            # Right: API status + market status + logout
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "10px"},
                children=[
                    html.Span(
                        id="api-status-badge",
                        children=["● NSE API: CONNECTED"],
                        style={"fontSize": "10px", "color": "#7da57b",
                               "padding": "3px 8px",
                               "border": "1px solid #7da57b",
                               "letterSpacing": "1px"},
                    ),
                    html.Span(id="market-status-badge",
                              style={"fontSize": "11px", "color": "#b5bdc8",
                                     "padding": "4px 10px",
                                     "border": "1px solid rgba(181, 150, 77, 0.22)"}),
                    html.Button(
                        "EXPORT STATE", id="export-state-btn",
                        n_clicks=0,
                        style={
                            "backgroundColor": "transparent",
                            "color": "#c79b45", "border": "1px solid #c79b45",
                            "padding": "4px 12px", "cursor": "pointer",
                            "fontFamily": "inherit", "fontSize": "11px",
                        }
                    ),
                    dcc.Download(id="export-download"),
                    html.Button(
                        "LOGOUT", id="logout-btn",
                        n_clicks=0,
                        style={
                            "backgroundColor": "transparent",
                            "color": "#b5bdc8", "border": "1px solid rgba(181, 150, 77, 0.22)",
                            "padding": "4px 12px", "cursor": "pointer",
                            "fontFamily": "inherit", "fontSize": "11px",
                        }
                    ),
                ]
            ),
        ]
    )


# ══════════════════════════════════════════════════════════════════════════════
# CRISIS / EWS BANNERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_crisis_banners() -> html.Div:
    return html.Div([
        html.Div(
            id="crisis-banner",
            style={"display": "none", "backgroundColor": "#d36d5b",
                   "color": "#e9ecef", "padding": "8px 16px",
                   "textAlign": "center", "fontWeight": "bold",
                   "fontSize": "13px", "letterSpacing": "1px"},
            children=[html.Span(id="crisis-banner-text")],
        ),
        html.Div(
            id="ews-banner",
            style={"display": "none", "backgroundColor": "#d0a24f",
                   "color": "#0b1018", "padding": "6px 16px",
                   "textAlign": "center", "fontSize": "12px",
                   "fontWeight": "bold"},
        ),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════

def _build_kpi_row() -> html.Div:
    def kpi_card(label, value_id, default="—"):
        return html.Div(
            style={
                "flex": "1", "minWidth": "140px",
                "backgroundColor": "#0b1018",
                "border": "1px solid rgba(181, 150, 77, 0.22)",
                "borderTop": "3px solid #c79b45",
                "padding": "14px 16px",
            },
            children=[
                html.Div(default, id=value_id,
                         style={"fontSize": "22px", "fontWeight": "bold",
                                "color": "#c79b45", "fontFamily": "inherit"}),
                html.Div(label, style={"fontSize": "10px", "color": "#b5bdc8",
                                       "textTransform": "uppercase",
                                       "letterSpacing": "1px", "marginTop": "4px"}),
            ]
        )

    return html.Div(
        style={"display": "flex", "gap": "8px", "padding": "12px 16px"},
        children=[
            kpi_card("Date",            "kpi-date",        "—"),
            kpi_card("SRI",             "kpi-sri",         "0.000"),
            kpi_card("Collapse Risk",   "kpi-collapse-pct","0.0%"),
            kpi_card("Banks Stressed",  "kpi-stressed",    "0/33"),
            kpi_card("EWS Level",       "kpi-day-count",   "NORMAL"),
        ]
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def _build_sidebar() -> html.Div:
    return html.Div(
        style={
            "width": "280px", "flexShrink": "0",
            "backgroundColor": "#0b1018",
            "border": "1px solid rgba(181, 150, 77, 0.22)",
            "padding": "16px", "overflowY": "auto",
            "maxHeight": "calc(100vh - 200px)",
        },
        children=[
            # ── Replay Controls ──
            html.Div("REPLAY CONTROLS",
                     style=_sidebar_section_label()),

            # RBI control badge — shown only to non-RBI users
            html.Div(
                id="rbi-control-badge",
                style={"display": "none"},
                children=[
                    html.Div(
                        "🔒 RBI CONTROLLED",
                        style={"backgroundColor": "rgba(199, 155, 69, 0.15)",
                               "border": "1px solid #c79b45",
                               "color": "#c79b45", "fontSize": "10px",
                               "letterSpacing": "2px", "textAlign": "center",
                               "padding": "6px 8px", "marginBottom": "10px",
                               "fontWeight": "bold"},
                    ),
                    html.Div(
                        "Date synced from RBI dashboard",
                        style={"color": "#b5bdc8", "fontSize": "9px",
                               "textAlign": "center", "marginBottom": "10px"},
                    ),
                ]
            ),

            html.Div(
                style={"display": "flex", "gap": "8px", "marginBottom": "12px"},
                children=[
                    html.Button("◀", id="step-back-btn", n_clicks=0,
                                style=_ctrl_btn_style()),
                    html.Button("▶ PLAY", id="play-btn", n_clicks=0,
                                style={**_ctrl_btn_style(), "flex": "1"}),
                    html.Button("▶", id="step-forward-btn", n_clicks=0,
                                style=_ctrl_btn_style()),
                ]
            ),

            html.Label("Speed", style=_label_style()),
            dcc.Slider(
                id="speed-slider", className="gold-slider", min=1, max=4, step=1, value=2,
                marks={1: "Slow", 2: "Norm", 3: "Fast", 4: "Turbo"},
                tooltip={"placement": "bottom", "always_visible": False},
            ),

            html.Br(),
            html.Label("Jump to Date", style=_label_style()),
            html.Div([
                dcc.DatePickerSingle(
                    id="date-picker-single",
                    className="dark-calendar",
                    min_date_allowed="2015-01-01",
                    max_date_allowed="2024-12-31",
                    initial_visible_month="2020-03-01",
                    date="2020-03-15",
                    display_format="DD MMM YYYY",
                    style={"width": "100%"},
                )
            ], style={"position": "relative", "zIndex": 9999, "marginRight": "15px"}),

            html.Hr(style={"borderColor": "rgba(181, 150, 77, 0.22)", "margin": "20px 0"}),

            # ── Shock Simulation ──
            html.Div("SHOCK SIMULATION", style=_sidebar_section_label()),
            html.Div(
                id="sim-banner",
                style={"display": "none",
                       "backgroundColor": "#d0a24f", "color": "#0b1018",
                       "padding": "6px", "fontSize": "11px",
                       "textAlign": "center", "fontWeight": "bold",
                       "marginBottom": "8px"},
                children="SIMULATION MODE — NOT LIVE DATA",
            ),

            html.Label("Pre-Built Scenarios", style=_label_style()),
            dcc.Dropdown(
                id="scenario-dropdown",
                options=[
                    {"label": "Select Scenario...", "value": "none"},
                    {"label": "Yes Bank Crisis (Mar 2020)", "value": "yes_bank_2020"},
                    {"label": "IL&FS Contagion (Sep 2018)", "value": "ilfs_2018"},
                    {"label": "COVID-19 Global Shock (Mar 2020)", "value": "covid_2020"},
                    {"label": "SBI Systemic Failure (Hypothetical)", "value": "sbi_hypothetical"}
                ],
                value="none",
                clearable=False,
                style={"backgroundColor": "#06080d", "color": "#e9ecef",
                       "border": "1px solid #c79b45", "fontSize": "12px",
                       "marginBottom": "12px"},
            ),

            html.Label("Shock Origin Bank", style=_label_style()),
            dcc.Dropdown(
                id="shock-bank-dropdown",
                options=[{"label": b, "value": b} for b in ALL_BANK_NAMES],
                value="Yes Bank",
                clearable=False,
                style={"backgroundColor": "#0b1018", "color": "#e9ecef",
                       "border": "1px solid rgba(181, 150, 77, 0.22)", "fontSize": "12px"},
            ),

            html.Br(),
            html.Label("Shock Magnitude", style=_label_style()),
            dcc.Slider(
                id="shock-magnitude-slider", className="gold-slider",
                min=10, max=100, step=5, value=75,
                marks={10: "10%", 50: "50%", 100: "100%"},
                tooltip={"placement": "bottom", "always_visible": True,
                         "template": "{value}%"},
            ),

            html.Br(),
            html.Label("Shock Type", style=_label_style()),
            dcc.RadioItems(
                id="shock-type-radio",
                options=[{"label": t, "value": t} for t in SHOCK_TYPES],
                value="Liquidity Crisis",
                labelStyle={"display": "block", "fontSize": "12px",
                            "color": "#e9ecef", "marginBottom": "4px",
                            "cursor": "pointer"},
            ),

            html.Br(),
            html.Label("Cascade Steps", style=_label_style()),
            dcc.Slider(
                id="shock-steps-slider", className="gold-slider",
                min=1, max=5, step=1, value=3,
                marks={1: "1", 2: "2", 3: "3", 4: "4", 5: "5"},
            ),

            html.Br(),
            dcc.Checklist(
                id="animate-steps-check",
                options=[{"label": " Animate steps", "value": "animate"}],
                value=["animate"],
                style={"fontSize": "12px", "color": "#e9ecef"},
            ),

            html.Div(
                style={"display": "flex", "gap": "8px", "marginTop": "16px"},
                children=[
                    html.Button("RUN SIMULATION", id="run-shock-btn",
                                n_clicks=0,
                                style={**_ctrl_btn_style(), "flex": "1",
                                       "backgroundColor": "#d36d5b",
                                       "color": "#e9ecef"}),
                    html.Button("RESET", id="reset-shock-btn",
                                n_clicks=0,
                                style=_ctrl_btn_style()),
                ]
            ),

            # RBI flag
            html.Div(
                id="rbi-flag-banner",
                style={"display": "none",
                       "backgroundColor": "#5c0000", "color": "#d36d5b",
                       "padding": "8px", "fontSize": "11px",
                       "fontWeight": "bold", "marginTop": "12px",
                       "textAlign": "center", "border": "1px solid #d36d5b"},
            ),
            html.Div(
                id="injection-estimate",
                style={"color": "#d0a24f", "fontSize": "11px",
                       "marginTop": "4px", "textAlign": "center"},
            ),

            html.Hr(style={"borderColor": "rgba(181, 150, 77, 0.22)", "margin": "20px 0"}),

            # ── Report ──
            html.Div("CRISIS REPORT", style=_sidebar_section_label()),
            html.Button(
                "GENERATE REPORT", id="generate-report-btn",
                n_clicks=0,
                style={**_ctrl_btn_style(), "width": "100%",
                       "backgroundColor": "#c79b45", "color": "#e9ecef",
                       "marginBottom": "8px"},
            ),
            html.Button(
                "EXPORT PDF", id="export-report-btn",
                n_clicks=0,
                style={**_ctrl_btn_style(), "width": "100%"},
            ),
        ]
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════════════════════

def _build_main_panel() -> html.Div:
    return html.Div(
        style={"flex": "1", "minWidth": "0", "display": "flex",
               "flexDirection": "column", "gap": "12px"},
        children=[
            # Row 1: 3D Network + Bank Stats
            html.Div(
                style={"display": "flex", "gap": "12px"},
                children=[
                    # 3D Network
                    html.Div(
                        style={"flex": "1", "backgroundColor": "#0b1018",
                               "border": "1px solid rgba(181, 150, 77, 0.22)",
                               "minHeight": "420px", "position": "relative"},
                        children=[
                            html.Div(
                                style={"display": "flex", "justifyContent": "space-between",
                                       "alignItems": "center",
                                       "padding": "8px 12px",
                                       "borderBottom": "1px solid rgba(181, 150, 77, 0.22)"},
                                children=[
                                    html.Span("INTERBANK NETWORK",
                                              style={"color": "#b5bdc8", "fontSize": "11px",
                                                     "letterSpacing": "2px"}),
                                    html.Div([
                                        dcc.Checklist(
                                            id="second-degree-toggle",
                                            options=[{"label": " 2nd-degree chain",
                                                      "value": "show2"}],
                                            value=[],
                                            style={"display": "inline",
                                                   "fontSize": "11px",
                                                   "color": "#b5bdc8"},
                                        ),
                                        html.Button(
                                            "CLEAR SELECTION",
                                            id="clear-selection-btn",
                                            n_clicks=0,
                                            style={**_ctrl_btn_style(),
                                                   "marginLeft": "8px",
                                                   "fontSize": "10px",
                                                   "padding": "3px 8px"},
                                        ),
                                        html.Button(
                                            "COMPARE",
                                            id="compare-mode-btn",
                                            n_clicks=0,
                                            style={**_ctrl_btn_style(),
                                                   "marginLeft": "4px",
                                                   "fontSize": "10px",
                                                   "padding": "3px 8px"},
                                        ),
                                        html.Button(
                                            "GAT ATTENTION",
                                            id="gat-attention-btn",
                                            n_clicks=0,
                                            style={**_ctrl_btn_style(),
                                                   "marginLeft": "4px",
                                                   "fontSize": "10px",
                                                   "padding": "3px 8px",
                                                   "color": "#d0a24f",
                                                   "borderColor": "#d0a24f"},
                                        ),
                                    ]),
                                ]
                            ),
                            dcc.Graph(
                                id="network-graph",
                                figure=_dark_empty_fig(380, is_3d=True),
                                style={"height": "380px"},
                                config={"displayModeBar": False},
                            ),
                        ]
                    ),

                    # Bank Stats Panel
                    html.Div(
                        id="bank-stats-panel",
                        style={
                            "width": "220px", "flexShrink": "0",
                            "backgroundColor": "#0b1018",
                            "border": "1px solid rgba(181, 150, 77, 0.22)",
                            "padding": "12px", "overflowY": "auto",
                            "maxHeight": "420px",
                        },
                        children=[
                            html.Div("CLICK A NODE", style={
                                "color": "#b5bdc8", "fontSize": "11px",
                                "textAlign": "center", "marginTop": "40px",
                            })
                        ]
                    ),
                ]
            ),

            # Row 2: SRI Chart
            html.Div(
                style={"backgroundColor": "#0b1018",
                       "border": "1px solid rgba(181, 150, 77, 0.22)"},
                children=[
                    html.Div("SYSTEMIC RISK INDEX — 2015 to 2024",
                             style={"color": "#b5bdc8", "fontSize": "11px",
                                    "letterSpacing": "2px",
                                    "padding": "8px 12px",
                                    "borderBottom": "1px solid rgba(181, 150, 77, 0.22)"}),
                    dcc.Graph(
                        id="sri-chart",
                        figure=_dark_empty_fig(200, is_3d=False),
                        style={"height": "200px"},
                        config={"displayModeBar": False},
                    ),
                    html.Div([
                        html.Button("Toggle 10-Year Risk Heatmap", id="toggle-heatmap-btn", n_clicks=0,
                                    style={"flex": "1", "backgroundColor": "transparent",
                                           "color": "#c79b45", "border": "none",
                                           "borderRight": "1px solid rgba(181, 150, 77, 0.22)",
                                           "padding": "6px", "fontSize": "10px",
                                           "cursor": "pointer", "textAlign": "center"}),
                        html.Button("Toggle 30-Day Forecast", id="toggle-forecast-btn", n_clicks=0,
                                    style={"flex": "1", "backgroundColor": "transparent",
                                           "color": "#d0a24f", "border": "none",
                                           "padding": "6px", "fontSize": "10px",
                                           "cursor": "pointer", "textAlign": "center"}),
                    ], style={"display": "flex", "borderTop": "1px solid rgba(181, 150, 77, 0.22)", "width": "100%"}),
                    html.Div(
                        id="heatmap-container",
                        style={"display": "none"},
                        children=[
                            dcc.Graph(
                                id="sri-heatmap",
                                figure=_dark_empty_fig(160, is_3d=False),
                                style={"height": "160px"},
                                config={"displayModeBar": False},
                            )
                        ]
                    )
                ]
            ),

            # Row 2.5: Live Stock Market Chart
            html.Div(
                style={"backgroundColor": "#0b1018",
                       "border": "1px solid rgba(181, 150, 77, 0.22)"},
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between",
                               "alignItems": "center",
                               "padding": "8px 12px",
                               "borderBottom": "1px solid rgba(181, 150, 77, 0.22)"},
                        children=[
                            html.Span("LIVE STOCK MARKET",
                                      style={"color": "#b5bdc8", "fontSize": "11px",
                                             "letterSpacing": "2px"}),
                            dcc.Dropdown(
                                id="stock-bank-dropdown",
                                options=[{"label": b, "value": b} for b in ALL_BANK_NAMES],
                                value="HDFC Bank",
                                clearable=False,
                                style={"width": "250px", "backgroundColor": "#0b1018",
                                       "color": "#e9ecef", "border": "none",
                                       "fontSize": "11px"},
                            ),
                        ]
                    ),
                    dcc.Graph(
                        id="stock-chart",
                        figure=_dark_empty_fig(250, is_3d=False),
                        style={"height": "250px"},
                        config={"displayModeBar": False},
                    ),
                ]
            ),

            # Row 3: Cascade table (shown after simulation)
            html.Div(
                id="cascade-table-container",
                style={"display": "none", "gap": "12px", "flexDirection": "row"},
                children=[
                    dcc.Interval(id="historical-mc-poll-interval", interval=2_000, disabled=False, n_intervals=0),
                    html.Div(
                        style={"backgroundColor": "#0d1117",
                               "border": "1px solid rgba(181, 150, 77, 0.22)", "flex": "1"},
                        children=[
                            html.Div("CONTAGION CASCADE",
                                     style={"color": "#d36d5b", "fontSize": "11px",
                                            "letterSpacing": "2px",
                                            "padding": "8px 12px",
                                            "borderBottom": "1px solid #5c0000"}),
                            dash_table.DataTable(
                                id="cascade-table",
                                columns=[
                                    {"name": "Step",        "id": "Step"},
                                    {"name": "Bank",        "id": "Bank"},
                                    {"name": "Channel",     "id": "Channel"},
                                    {"name": "Exposure",    "id": "Exposure"},
                                    {"name": "Vulnerability","id": "Vulnerability"},
                                    {"name": "Status",      "id": "Status"},
                                ],
                                data=[],
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "backgroundColor": "#0b1018",
                                    "color": "#d0a24f",
                                    "border": "1px solid rgba(181, 150, 77, 0.15)",
                                    "fontSize": "12px",
                                    "fontFamily": "'JetBrains Mono', monospace",
                                    "padding": "6px 10px",
                                },
                                style_header={
                                    "backgroundColor": "#5c0000",
                                    "color": "#e9ecef",
                                    "fontWeight": "bold",
                                    "border": "1px solid rgba(181, 150, 77, 0.22)",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "1px",
                                    "fontSize": "10px",
                                },
                                style_data_conditional=[
                                    {"if": {"filter_query": '{Status} contains "DISTRESS"'},
                                     "color": "#d36d5b", "fontWeight": "bold"},
                                    {"if": {"filter_query": '{Status} contains "STRESSED"'},
                                     "color": "#d0a24f"},
                                    {"if": {"filter_query": '{Status} contains "ELEVATED"'},
                                     "color": "#c79b45"},
                                    {"if": {"row_index": "odd"},
                                     "backgroundColor": "#0b1018"},
                                    {"if": {"row_index": "even"},
                                     "backgroundColor": "#111821"},
                                ],
                            ),
                        ]
                    ),
                    # Right: Monte Carlo Historical Predictions
                    html.Div(
                        style={"width": "340px", "flexShrink": "0",
                               "backgroundColor": "#06080d",
                               "border": "1px solid rgba(181, 150, 77, 0.22)",
                               "display": "flex", "flexDirection": "column"},
                        children=[
                            html.Div("MONTE CARLO PREDICTIONS (SIMULATION)",
                                     style={"color": "#b08ee0", "fontSize": "10px",
                                            "letterSpacing": "2px",
                                            "padding": "8px 12px",
                                            "borderBottom": "1px solid rgba(181, 150, 77, 0.22)"}),
                            html.Div(
                                id="historical-mc-predictions",
                                style={"padding": "12px", "color": "#b5bdc8",
                                       "fontSize": "12px", "lineHeight": "1.8",
                                       "flex": "1"},
                                children=[
                                    html.Div("Run a simulation to see Monte Carlo predictions...",
                                             style={"color": "#b5bdc8",
                                                    "fontSize": "11px",
                                                    "textAlign": "center",
                                                    "marginTop": "30px"}),
                                ]
                            ),
                        ]
                    ),
                    # Store to hold the last generated report text for chat context
                    dcc.Store(id="report-text-store", data=""),
                ]
            ),

            # Row 4: LLM Analysis Report
            html.Div(
                style={"backgroundColor": "#0b1018", "border": "1px solid rgba(181, 150, 77, 0.22)",
                       "padding": "16px", "marginTop": "4px"},
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between",
                               "alignItems": "center", "marginBottom": "12px"},
                        children=[
                            html.Div(
                                style={"display": "flex", "alignItems": "center", "gap": "12px"},
                                children=[
                                    html.Span("LLM CRISIS ANALYSIS REPORT",
                                              style={"color": "#b5bdc8", "fontSize": "11px",
                                                     "letterSpacing": "2px"}),
                                    dcc.Loading(
                                        id="report-spinner",
                                        type="circle",
                                        color="#c79b45",
                                        children=html.Div(style={"width": "20px", "height": "20px"}),
                                    ),
                                ]
                            ),
                        ]
                    ),
                    html.Div(
                        id="report-output",
                        style={"color": "#e9ecef", "fontSize": "12px",
                               "lineHeight": "1.6", "whiteSpace": "pre-wrap",
                               "maxHeight": "300px", "overflowY": "auto"},
                        children="Click GENERATE REPORT in the sidebar to produce an AI crisis analysis.",
                    ),
                    # ── Interactive LLM Chat (Historical) ───────────────
                    html.Div(
                        id="report-chat-section",
                        style={"marginTop": "12px", "borderTop": "1px solid rgba(181, 150, 77, 0.22)",
                               "paddingTop": "12px", "display": "none"},
                        children=[
                            html.Div(
                                style={"display": "flex", "alignItems": "center",
                                       "gap": "8px", "marginBottom": "8px"},
                                children=[
                                    html.Span("ASK FOLLOW-UP",
                                              style={"color": "#c79b45", "fontSize": "10px",
                                                     "letterSpacing": "2px", "flexShrink": "0"}),
                                    html.Span("Ask about this analysis",
                                              style={"color": "#b5bdc8", "fontSize": "10px",
                                                     "fontStyle": "italic"}),
                                ]
                            ),
                            html.Div(
                                style={"display": "flex", "gap": "8px"},
                                children=[
                                    dcc.Input(
                                        id="report-chat-input",
                                        type="text",
                                        placeholder="e.g. Which banks are most at risk?",
                                        debounce=True,
                                        style={"flex": "1", "backgroundColor": "#06080d",
                                               "color": "#e9ecef", "border": "1px solid rgba(181, 150, 77, 0.22)",
                                               "padding": "8px 12px", "fontSize": "12px",
                                               "fontFamily": "inherit", "outline": "none"},
                                    ),
                                    html.Button(
                                        "ASK", id="report-chat-btn", n_clicks=0,
                                        style={"backgroundColor": "#c79b45", "color": "#0b1018",
                                               "border": "none", "padding": "8px 16px",
                                               "cursor": "pointer", "fontFamily": "inherit",
                                               "fontSize": "11px", "fontWeight": "bold",
                                               "letterSpacing": "1px"},
                                    ),
                                ]
                            ),
                            dcc.Loading(
                                id="report-chat-spinner",
                                type="circle",
                                color="#c79b45",
                                children=[
                                    html.Div(
                                        id="report-chat-output",
                                        style={"marginTop": "8px", "color": "#e9ecef",
                                               "fontSize": "12px", "lineHeight": "1.6",
                                               "whiteSpace": "pre-wrap",
                                               "maxHeight": "200px", "overflowY": "auto",
                                               "backgroundColor": "#06080d",
                                               "padding": "10px",
                                               "border": "1px solid rgba(181, 150, 77, 0.12)",
                                               "display": "none"},
                                    ),
                                ]
                            ),
                        ]
                    ),
                ]
            ),

            # Row 5: Crisis Backtesting Module
            html.Div(
                style={"backgroundColor": "#0b1018", "border": "1px solid rgba(181, 150, 77, 0.22)",
                       "padding": "16px", "marginTop": "16px"},
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "marginBottom": "12px"},
                        children=[
                            html.Span("CRISIS BACKTESTING (GAT MODEL ACCURACY)",
                                      style={"color": "#b5bdc8", "fontSize": "11px",
                                             "letterSpacing": "2px"}),
                            html.Div(
                                style={"display": "flex", "alignItems": "center", "gap": "8px"},
                                children=[
                                    dcc.Dropdown(
                                        id="backtest-crisis-dropdown",
                                        options=[
                                            {"label": "China Yuan Devaluation", "value": "china"},
                                            {"label": "Demonetization", "value": "demo"},
                                            {"label": "IL&FS Crisis", "value": "ilfs"},
                                            {"label": "Yes Bank Stress", "value": "yes_stress"},
                                            {"label": "Yes Bank Moratorium", "value": "yes_moratorium"},
                                            {"label": "COVID-19 PANDEMIC", "value": "covid"},
                                            {"label": "Second COVID Wave", "value": "covid2"},
                                            {"label": "RBI Rate Hike Stress", "value": "rbi_hike"},
                                            {"label": "SVB Global Contagion", "value": "svb"},
                                        ],
                                        value="covid",
                                        clearable=False,
                                        style={"width": "200px", "backgroundColor": "#0b1018",
                                               "color": "#e9ecef", "border": "1px solid rgba(181, 150, 77, 0.22)",
                                               "fontSize": "10px"}
                                    ),
                                    html.Button(
                                        "RUN BACKTEST", id="run-backtest-btn", n_clicks=0,
                                        style={"backgroundColor": "transparent", "color": "#c79b45",
                                               "border": "1px solid #c79b45", "padding": "4px 12px",
                                               "cursor": "pointer", "fontFamily": "inherit",
                                               "fontSize": "10px"}
                                    ),
                                ]
                            ),
                        ]
                    ),
                    dcc.Loading(
                        id="backtest-spinner",
                        type="circle",
                        color="#c79b45",
                        children=html.Div(
                            id="backtest-output",
                            style={"color": "#e9ecef", "fontSize": "12px",
                                   "lineHeight": "1.6", "whiteSpace": "pre-wrap",
                                   "maxHeight": "300px", "overflowY": "auto"},
                            children="Click to run historical GNN validation against COVID-19 real data.",
                        )
                    ),
                ]
            ),

            # Row 5: Top Risk + All Banks Tables
            html.Div(
                style={"display": "flex", "gap": "12px"},
                children=[
                    _build_table_panel(
                        "TOP AT-RISK BANKS",
                        "top-risk-table",
                        [
                            {"name": "Bank",          "id": "bank"},
                            {"name": "Vulnerability", "id": "vulnerability"},
                            {"name": "NPL",           "id": "npl"},
                            {"name": "CDS (bps)",     "id": "cds"},
                            {"name": "Status",        "id": "status"},
                        ],
                        flex="1",
                    ),
                    _build_table_panel(
                        "ALL 33 BANKS",
                        "all-banks-table",
                        [
                            {"name": "Bank",          "id": "bank"},
                            {"name": "Type",          "id": "type"},
                            {"name": "Vulnerability", "id": "vulnerability"},
                            {"name": "Tier 1",        "id": "tier1"},
                            {"name": "NPL",           "id": "npl"},
                            {"name": "Basel III",     "id": "basel_iii"},
                            {"name": "Status",        "id": "status"},
                        ],
                        flex="2",
                    ),
                ]
            ),

            # Row 6: Compare Panel (hidden by default)
            html.Div(
                id="compare-panel",
                style={"display": "none"},
                children=[_build_compare_panel()]
            ),

            # Row 7: Contradiction Panel (live mode only)
            html.Div(
                id="contradiction-panel",
                style={"display": "none"},
                children=[_build_contradiction_panel()]
            ),

            # ═══════════════════════════════════════════════════════════════
            # Row 8: LIVE GNN MONITORING SECTION
            # ═══════════════════════════════════════════════════════════════
            html.Div(
                id="live-gnn-panel",
                style={"display": "none", "backgroundColor": "#0b1018",
                       "border": "1px solid rgba(181, 150, 77, 0.22)",
                       "padding": "0", "marginTop": "4px"},
                children=[
                    # Section header
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between",
                               "alignItems": "center",
                               "padding": "10px 16px",
                               "backgroundColor": "#c79b45",
                               "borderBottom": "1px solid rgba(181, 150, 77, 0.22)"},
                        children=[
                            html.Div([
                                html.Span("LIVE GNN MONITORING",
                                          style={"color": "#e9ecef", "fontSize": "13px",
                                                 "fontWeight": "bold",
                                                 "letterSpacing": "3px"}),
                                html.Span("  ●  REAL-TIME MARKET DATA → GNN INFERENCE",
                                          style={"color": "rgba(255,255,255,0.6)",
                                                 "fontSize": "10px",
                                                 "letterSpacing": "1px"}),
                            ]),
                            html.Div([
                                html.Span(id="live-gnn-status",
                                          children="STANDBY",
                                          style={"fontSize": "10px", "color": "#d0a24f",
                                                 "padding": "3px 10px",
                                                 "border": "1px solid #d0a24f",
                                                 "letterSpacing": "2px"}),
                                html.Button(
                                    "▶ START LIVE GNN", id="start-live-gnn-btn",
                                    n_clicks=0,
                                    style={
                                        "marginLeft": "10px",
                                        "backgroundColor": "#7da57b",
                                        "color": "#e9ecef", "border": "none",
                                        "padding": "5px 14px", "cursor": "pointer",
                                        "fontFamily": "inherit", "fontSize": "11px",
                                        "fontWeight": "bold", "letterSpacing": "1px",
                                    }
                                ),
                            ]),
                        ]
                    ),
                    # Live GNN content — KPIs + Network side by side
                    html.Div(
                        style={"display": "flex", "gap": "12px", "padding": "12px"},
                        children=[
                            # Live Network Graph
                            html.Div(
                                style={"flex": "1", "backgroundColor": "#06080d",
                                       "border": "1px solid rgba(181, 150, 77, 0.22)",
                                       "minHeight": "350px"},
                                children=[
                                    html.Div("LIVE INTERBANK NETWORK — GNN OUTPUT",
                                             style={"color": "#b5bdc8", "fontSize": "10px",
                                                    "letterSpacing": "2px",
                                                    "padding": "8px 12px",
                                                    "borderBottom": "1px solid rgba(181, 150, 77, 0.22)"}),
                                    dcc.Graph(
                                        id="live-network-graph",
                                        figure=_dark_empty_fig(330, is_3d=True),
                                        style={"height": "330px"},
                                        config={"displayModeBar": False},
                                    ),
                                ]
                            ),
                            # Live KPI panel
                            html.Div(
                                id="live-kpi-panel",
                                style={"width": "280px", "flexShrink": "0",
                                       "display": "flex", "flexDirection": "column",
                                       "gap": "8px"},
                                children=[
                                    _build_live_kpi("LIVE SRI",        "live-kpi-sri",       "—", "#c79b45"),
                                    _build_live_kpi("LIVE EWS",        "live-kpi-ews",       "—", "#7da57b"),
                                    _build_live_kpi("BANKS AT RISK",   "live-kpi-at-risk",   "0/33", "#d0a24f"),
                                    _build_live_kpi("DATA SOURCE",     "live-kpi-source",    "yfinance NSE", "#b5bdc8"),
                                    _build_live_kpi("LAST UPDATE",     "live-kpi-timestamp", "—", "#b5bdc8"),
                                    _build_live_kpi("GNN CONFIDENCE",  "live-kpi-confidence","—", "#b08ee0"),
                                    html.Div(id="live-connected-banks-panel", style={"marginTop": "8px"})
                                ]
                            ),
                        ]
                    ),
                    # ── Live Risky Banks Table + MC Predictions ──────────────
                    html.Div(
                        style={"display": "flex", "gap": "12px", "padding": "0 12px 12px 12px"},
                        children=[
                            # Left: Live Risky Banks Table
                            html.Div(
                                style={"flex": "1", "backgroundColor": "#06080d",
                                       "border": "1px solid rgba(181, 150, 77, 0.22)"},
                                children=[
                                    html.Div("LIVE AT-RISK BANKS — REAL-TIME VULNERABILITY",
                                             style={"color": "#d36d5b", "fontSize": "10px",
                                                    "letterSpacing": "2px",
                                                    "padding": "8px 12px",
                                                    "borderBottom": "1px solid rgba(181, 150, 77, 0.22)"}),
                                    dash_table.DataTable(
                                        id="live-risky-banks-table",
                                        columns=[
                                            {"name": "#",               "id": "rank"},
                                            {"name": "Bank",            "id": "bank"},
                                            {"name": "Vulnerability",   "id": "vulnerability"},
                                            {"name": "Stock Change",    "id": "stock_change"},
                                            {"name": "Risk Level",      "id": "risk_level"},
                                        ],
                                        data=[],
                                        style_table={"overflowX": "auto", "maxHeight": "280px", "overflowY": "auto"},
                                        style_cell={
                                            "backgroundColor": "#0b1018",
                                            "color": "#d0a24f",
                                            "border": "1px solid rgba(181, 150, 77, 0.15)",
                                            "fontSize": "11px",
                                            "fontFamily": "inherit",
                                            "padding": "5px 10px",
                                        },
                                        style_header={
                                            "backgroundColor": "#5c0000",
                                            "color": "#e9ecef",
                                            "fontWeight": "bold",
                                            "border": "1px solid rgba(181, 150, 77, 0.22)",
                                            "fontSize": "10px",
                                            "letterSpacing": "1px",
                                        },
                                        style_data_conditional=[
                                            {"if": {"filter_query": '{risk_level} = "CRITICAL"'},
                                             "color": "#d36d5b", "fontWeight": "bold"},
                                            {"if": {"filter_query": '{risk_level} = "HIGH"'},
                                             "color": "#d0a24f"},
                                            {"if": {"filter_query": '{risk_level} = "MODERATE"'},
                                             "color": "#c79b45"},
                                            {"if": {"filter_query": '{risk_level} = "LOW"'},
                                             "color": "#7da57b"},
                                            {"if": {"row_index": "odd"},
                                             "backgroundColor": "#0b1018"},
                                            {"if": {"row_index": "even"},
                                             "backgroundColor": "#111821"},
                                        ],
                                        page_size=10,
                                        sort_action="native",
                                    ),
                                ]
                            ),
                            # Right: Monte Carlo Live Predictions
                            html.Div(
                                style={"width": "340px", "flexShrink": "0",
                                       "backgroundColor": "#06080d",
                                       "border": "1px solid rgba(181, 150, 77, 0.22)",
                                       "display": "flex", "flexDirection": "column"},
                                children=[
                                    html.Div("MONTE CARLO LIVE PREDICTIONS",
                                             style={"color": "#b08ee0", "fontSize": "10px",
                                                    "letterSpacing": "2px",
                                                    "padding": "8px 12px",
                                                    "borderBottom": "1px solid rgba(181, 150, 77, 0.22)"}),
                                    html.Div(
                                        id="live-mc-predictions",
                                        style={"padding": "12px", "color": "#b5bdc8",
                                               "fontSize": "12px", "lineHeight": "1.8",
                                               "flex": "1"},
                                        children=[
                                            html.Div("Waiting for Live GNN data...",
                                                     style={"color": "#b5bdc8",
                                                            "fontSize": "11px",
                                                            "textAlign": "center",
                                                            "marginTop": "30px"}),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                    # ── Live LLM Report ──
                    html.Div(
                        style={"padding": "0 12px 12px 12px"},
                        children=[
                            html.Div(
                                style={"backgroundColor": "#06080d", "border": "1px solid rgba(181, 150, 77, 0.22)", "padding": "16px", "display": "flex", "flexDirection": "column"},
                                children=[
                                    html.Div(
                                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "12px"},
                                        children=[
                                            html.Div(
                                                style={"display": "flex", "alignItems": "center", "gap": "12px"},
                                                children=[
                                                    html.Span("LIVE MARKET ANALYSIS REPORT", style={"color": "#b5bdc8", "fontSize": "11px", "letterSpacing": "2px"}),
                                                    dcc.Loading(
                                                        id="live-report-spinner",
                                                        type="circle",
                                                        color="#c79b45",
                                                        children=html.Div(style={"width": "20px", "height": "20px"}),
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                style={"display": "flex", "gap": "8px"},
                                                children=[
                                                    html.Button("GENERATE REPORT", id="generate-live-report-btn", n_clicks=0, style={"backgroundColor": "#c79b45", "color": "#e9ecef", "border": "1px solid rgba(181, 150, 77, 0.22)", "padding": "6px 12px", "cursor": "pointer", "fontFamily": "inherit", "fontSize": "12px"}),
                                                    html.Button("EXPORT PDF", id="export-live-report-btn", n_clicks=0, style={"backgroundColor": "#0b1018", "color": "#e9ecef", "border": "1px solid rgba(181, 150, 77, 0.22)", "padding": "6px 12px", "cursor": "pointer", "fontFamily": "inherit", "fontSize": "12px"}),
                                                ]
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        id="live-report-output",
                                        style={"color": "#e9ecef", "fontSize": "12px", "lineHeight": "1.6", "whiteSpace": "pre-wrap", "maxHeight": "200px", "overflowY": "auto"},
                                        children="Click GENERATE REPORT for a real-time AI analysis of the current market conditions.",
                                    ),
                                    # ── Interactive LLM Chat (Live) ───────────────
                                    html.Div(
                                        id="live-chat-section",
                                        style={"marginTop": "12px", "borderTop": "1px solid rgba(181, 150, 77, 0.22)",
                                               "paddingTop": "12px", "display": "none"},
                                        children=[
                                            html.Div(
                                                style={"display": "flex", "alignItems": "center",
                                                       "gap": "8px", "marginBottom": "8px"},
                                                children=[
                                                    html.Span("ASK FOLLOW-UP",
                                                              style={"color": "#c79b45", "fontSize": "10px",
                                                                     "letterSpacing": "2px", "flexShrink": "0"}),
                                                    html.Span("Ask about this live analysis",
                                                              style={"color": "#b5bdc8", "fontSize": "10px",
                                                                     "fontStyle": "italic"}),
                                                ]
                                            ),
                                            html.Div(
                                                style={"display": "flex", "gap": "8px"},
                                                children=[
                                                    dcc.Input(
                                                        id="live-chat-input",
                                                        type="text",
                                                        placeholder="e.g. What if SBI also gets stressed?",
                                                        debounce=True,
                                                        style={"flex": "1", "backgroundColor": "#06080d",
                                                               "color": "#e9ecef", "border": "1px solid rgba(181, 150, 77, 0.22)",
                                                               "padding": "8px 12px", "fontSize": "12px",
                                                               "fontFamily": "inherit", "outline": "none"},
                                                    ),
                                                    html.Button(
                                                        "ASK", id="live-chat-btn", n_clicks=0,
                                                        style={"backgroundColor": "#c79b45", "color": "#0b1018",
                                                               "border": "none", "padding": "8px 16px",
                                                               "cursor": "pointer", "fontFamily": "inherit",
                                                               "fontSize": "11px", "fontWeight": "bold",
                                                               "letterSpacing": "1px"},
                                                    ),
                                                ]
                                            ),
                                            dcc.Loading(
                                                id="live-chat-spinner",
                                                type="circle",
                                                color="#c79b45",
                                                children=[
                                                    html.Div(
                                                        id="live-chat-output",
                                                        style={"marginTop": "8px", "color": "#e9ecef",
                                                               "fontSize": "12px", "lineHeight": "1.6",
                                                               "whiteSpace": "pre-wrap",
                                                               "maxHeight": "200px", "overflowY": "auto",
                                                               "backgroundColor": "#06080d",
                                                               "padding": "10px",
                                                               "border": "1px solid rgba(181, 150, 77, 0.12)",
                                                               "display": "none"},
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                    dcc.Store(id="live-report-text-store", data=""),
                                ]
                            )
                        ]
                    ),
                ]
            ),
        ]
    )


# ══════════════════════════════════════════════════════════════════════════════
# COMPARE PANEL
# ══════════════════════════════════════════════════════════════════════════════

def _build_compare_panel() -> html.Div:
    return html.Div(
        style={"backgroundColor": "#0b1018", "border": "1px solid rgba(181, 150, 77, 0.22)",
               "padding": "16px"},
        children=[
            html.Div("CRISIS COMPARISON MODE",
                     style={"color": "#c79b45", "fontSize": "11px",
                            "letterSpacing": "2px", "marginBottom": "12px"}),
            html.Div(
                style={"display": "flex", "gap": "16px",
                       "marginBottom": "12px", "alignItems": "center"},
                children=[
                    html.Div([
                        html.Label("Date A", style=_label_style()),
                        dcc.DatePickerSingle(
                            id="date-a-picker",
                            className="dark-calendar",
                            min_date_allowed="2015-01-01",
                            max_date_allowed="2024-12-31",
                            date="2020-03-14",
                            display_format="DD MMM YYYY",
                        ),
                    ]),
                    html.Span("vs", style={"color": "#b5bdc8", "fontSize": "18px",
                                           "marginTop": "16px"}),
                    html.Div([
                        html.Label("Date B", style=_label_style()),
                        dcc.DatePickerSingle(
                            id="date-b-picker",
                            className="dark-calendar",
                            min_date_allowed="2015-01-01",
                            max_date_allowed="2024-12-31",
                            date="2020-03-15",
                            display_format="DD MMM YYYY",
                        ),
                    ]),
                    html.P("Best demo: 2020-03-14 vs 2020-03-15 — most dramatic contrast.",
                           style={"color": "#b5bdc8", "fontSize": "11px",
                                  "marginTop": "16px"}),
                ]
            ),
            html.Div(
                style={"display": "flex", "gap": "12px"},
                children=[
                    html.Div(id="compare-left",  style={"flex": "1"}),
                    html.Div(id="compare-right", style={"flex": "1"}),
                ]
            ),
        ]
    )


# ══════════════════════════════════════════════════════════════════════════════
# CONTRADICTION PANEL
# ══════════════════════════════════════════════════════════════════════════════

def _build_contradiction_panel() -> html.Div:
    return html.Div(
        style={"backgroundColor": "#0b1018", "border": "1px solid #c79b45",
               "padding": "16px"},
        children=[
            html.Div("LIVE vs HISTORICAL BENCHMARK",
                     style={"color": "#c79b45", "fontSize": "11px",
                            "letterSpacing": "2px", "marginBottom": "8px"}),
            html.P(
                "Historical crisis data used for model calibration. "
                "Live stock signals update the real-time risk assessment.",
                style={"color": "#b5bdc8", "fontSize": "11px", "margin": "0 0 12px 0"}
            ),
            html.Div(
                style={"display": "flex", "gap": "12px"},
                children=[
                    html.Div(id="live-output-panel",       style={"flex": "1"}),
                    html.Div(id="historical-benchmark-panel", style={"flex": "1"}),
                ]
            ),
        ]
    )


# ══════════════════════════════════════════════════════════════════════════════
# STATUS BAR
# ══════════════════════════════════════════════════════════════════════════════

def _build_status_bar() -> html.Div:
    return html.Div(
        id="status-bar",
        style={
            "backgroundColor": "#0b1018",
            "borderTop": "1px solid rgba(181, 150, 77, 0.22)",
            "padding": "4px 16px",
            "fontSize": "11px", "color": "#b5bdc8",
            "display": "flex", "justifyContent": "space-between",
        },
        children=[
            html.Span("Ready"),
            html.Span("M.S. Ramaiah Institute of Technology | AIP67 | 2026"),
        ]
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_table_panel(title, table_id, columns, flex="1") -> html.Div:
    return html.Div(
        style={"flex": flex, "backgroundColor": "#0d1117",
               "border": "1px solid rgba(181, 150, 77, 0.22)"},
        children=[
            html.Div(title, style={"color": "#7fc3d5", "fontSize": "11px",
                                   "letterSpacing": "2px", "padding": "8px 12px",
                                   "borderBottom": "1px solid rgba(181, 150, 77, 0.22)"}),
            dash_table.DataTable(
                id=table_id,
                columns=columns,
                data=[],
                style_table={"overflowX": "auto"},
                style_cell={
                    "backgroundColor": "#0b1018",
                    "color": "#d0a24f",
                    "border": "1px solid rgba(181, 150, 77, 0.15)",
                    "fontSize": "11px",
                    "fontFamily": "'JetBrains Mono', monospace",
                    "padding": "6px 10px",
                    "maxWidth": "200px",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                },
                style_header={
                    "backgroundColor": "#5c0000",
                    "color": "#e9ecef",
                    "fontWeight": "bold",
                    "border": "1px solid rgba(181, 150, 77, 0.22)",
                    "fontSize": "10px",
                    "letterSpacing": "1px",
                    "textTransform": "uppercase",
                },
                style_data_conditional=[
                    {"if": {"filter_query": '{status} contains "DISTRESS"'},
                     "color": "#d36d5b", "fontWeight": "bold"},
                    {"if": {"filter_query": '{status} contains "STRESSED"'},
                     "color": "#d0a24f"},
                    {"if": {"filter_query": '{status} contains "ELEVATED"'},
                     "color": "#c79b45"},
                    {"if": {"row_index": "odd"},
                     "backgroundColor": "#0b1018"},
                    {"if": {"row_index": "even"},
                     "backgroundColor": "#111821"},
                ],
                page_size=10,
                sort_action="native",
            ),
        ]
    )


def _input_style() -> dict:
    return {
        "width": "100%", "padding": "10px",
        "backgroundColor": "#0b1018",
        "color": "#e9ecef", "border": "1px solid rgba(181, 150, 77, 0.22)",
        "fontFamily": "inherit", "fontSize": "13px",
        "marginTop": "6px", "marginBottom": "4px",
        "boxSizing": "border-box",
    }


def _ctrl_btn_style() -> dict:
    return {
        "backgroundColor": "#0b1018",
        "color": "#e9ecef",
        "border": "1px solid rgba(181, 150, 77, 0.22)",
        "padding": "6px 12px",
        "cursor": "pointer",
        "fontFamily": "inherit",
        "fontSize": "12px",
    }


def _label_style() -> dict:
    return {
        "color": "#b5bdc8", "fontSize": "11px",
        "textTransform": "uppercase", "letterSpacing": "1px",
        "display": "block", "marginBottom": "6px",
    }


def _sidebar_section_label() -> dict:
    return {
        "color": "#c79b45", "fontSize": "10px",
        "letterSpacing": "2px", "textTransform": "uppercase",
        "marginBottom": "12px", "borderBottom": "1px solid rgba(181, 150, 77, 0.22)",
        "paddingBottom": "6px",
    }


def _build_live_kpi(label: str, value_id: str, default: str = "—",
                    color: str = "#c79b45") -> html.Div:
    """Builds a single KPI card for the live GNN monitoring panel."""
    return html.Div(
        style={
            "backgroundColor": "#0b1018",
            "border": "1px solid rgba(181, 150, 77, 0.22)",
            "borderLeft": f"3px solid {color}",
            "padding": "10px 12px",
        },
        children=[
            html.Div(label, style={"fontSize": "9px", "color": "#b5bdc8",
                                    "letterSpacing": "2px",
                                    "textTransform": "uppercase",
                                    "marginBottom": "4px"}),
            html.Div(default, id=value_id,
                     style={"fontSize": "16px", "fontWeight": "bold",
                            "color": color, "fontFamily": "inherit"}),
        ]
    )

