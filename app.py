"""
app.py — Carbon-Aware Adaptive AI Inference System
===================================================
Main Streamlit dashboard entry point.

Run with:
    streamlit run app.py

Architecture overview
---------------------
User Input
    ↓
inference_pipeline.run_pipeline()
    ├── Stage 1: rule_engine      (near-zero energy)
    ├── Stage 2: DistilBERT       (small model)
    └── Stage 3: BERT             (large model)
    ↓
energy_tracker.estimate_energy() — CO₂ / kWh accounting
    ↓
Streamlit dashboard — charts, metrics, log table
"""

import streamlit as st
import pandas as pd
import time

from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, EXAMPLE_TEXTS,
    ENERGY_LARGE_MODEL,
)
from inference_pipeline import run_pipeline
from dashboard_utils import (
    energy_bar_chart,
    stage_distribution_pie,
    carbon_timeline,
    green_score_gauge,
    build_log_df,
    STAGE_COLOURS,
)
from ollama_dashboard import render_dashboard
from comparative_dashboard import render_comparative_dashboard

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark startup-dashboard aesthetic ─────────────────────────────
st.markdown(
    """
    <style>
    /* ── Base ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Sora:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
        background-color: #0f172a;
        color: #f1f5f9;
    }

    /* ── Hide Streamlit chrome ────────────────────────── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Cards ────────────────────────────────────────── */
    .card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .card-accent { border-left: 4px solid #22c55e; }

    /* ── Hero header ──────────────────────────────────── */
    .hero {
        background: linear-gradient(135deg, #064e3b 0%, #0f172a 60%);
        border: 1px solid #065f46;
        border-radius: 18px;
        padding: 2rem 2.4rem 1.6rem;
        margin-bottom: 2rem;
    }
    .hero h1 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #4ade80;
        margin: 0 0 .4rem;
    }
    .hero p { color: #94a3b8; font-size: .95rem; margin: 0; }

    /* ── Stage badge ──────────────────────────────────── */
    .badge {
        display: inline-block;
        border-radius: 999px;
        padding: .25rem .85rem;
        font-size: .8rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: .04em;
    }
    .badge-green  { background: #14532d; color: #4ade80; border: 1px solid #22c55e; }
    .badge-blue   { background: #1e3a5f; color: #93c5fd; border: 1px solid #3b82f6; }
    .badge-orange { background: #431407; color: #fdba74; border: 1px solid #f97316; }

    /* ── Metric overrides ─────────────────────────────── */
    [data-testid="stMetric"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: .9rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: .82rem; }
    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.3rem;
    }
    [data-testid="stMetricDelta"] { font-size: .78rem; }

    /* ── Table ────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        background: #1e293b !important;
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Buttons ──────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #065f46, #047857);
        color: #ecfdf5;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        padding: .55rem 1.4rem;
        transition: opacity .2s;
    }
    .stButton > button:hover { opacity: .85; }

    /* ── Text input ───────────────────────────────────── */
    .stTextArea textarea {
        background: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        font-family: 'Sora', sans-serif !important;
    }

    /* ── Section titles ───────────────────────────────── */
    .section-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        font-weight: 600;
        color: #4ade80;
        letter-spacing: .06em;
        text-transform: uppercase;
        margin-bottom: 1rem;
        padding-bottom: .4rem;
        border-bottom: 1px solid #1e3a2b;
    }

    /* ── Progress bar ─────────────────────────────────── */
    .stProgress > div > div { background: #22c55e !important; border-radius: 9px; }
    .stProgress { border-radius: 9px; }

    /* ── Selectbox / example picker ───────────────────── */
    .stSelectbox > div > div {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #f1f5f9 !important;
        border-radius: 10px !important;
    }

    /* ── Divider ──────────────────────────────────────── */
    hr { border-color: #1e293b; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state initialisation ──────────────────────────────────────────────
if "inference_log"    not in st.session_state:
    st.session_state["inference_log"]    = []
if "total_saved_kwh"  not in st.session_state:
    st.session_state["total_saved_kwh"]  = 0.0
if "last_result"      not in st.session_state:
    st.session_state["last_result"]      = None

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    app_mode = st.radio(
        "Select Dashboard", 
        ["Sentiment Tracker", "Ollama Compare", "Comparative Analysis"],
        help="Switch between the sentiment tracker, Ollama model comparison, or the head-to-head comparative analysis."
    )
    st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════
if app_mode == "Sentiment Tracker":
    # ── HEADER ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero">
            <h1>🌿 Carbon-Aware AI Inference System</h1>
            <p>
                A Green AI demonstration that uses a <strong>three-stage adaptive pipeline</strong>
                to minimise energy consumption and carbon emissions during NLP inference.
                Simple inputs are resolved with near-zero energy; complex ones escalate to
                heavier models only when needed.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 1 — INPUT
    # ═══════════════════════════════════════════════════════════════════════════════
    st.markdown('<p class="section-title">📝 Input</p>', unsafe_allow_html=True)

    col_input, col_sidebar = st.columns([3, 1])

    with col_input:
        # Example sentence picker
        example_choice = st.selectbox(
            "💡 Load an example sentence",
            options=["— choose an example —"] + EXAMPLE_TEXTS,
            key="example_picker",
        )

        default_text = (
            "" if example_choice == "— choose an example —" else example_choice
        )
        user_text = st.text_area(
            "Enter text for sentiment analysis",
            value=default_text,
            height=110,
            placeholder="Type a sentence and click Run Inference…",
            label_visibility="collapsed",
        )

    with col_sidebar:
        st.markdown("<br>", unsafe_allow_html=True)
        run_clicked = st.button("⚡ Run Inference", use_container_width=True)
        clear_log   = st.button("🗑 Clear Log",     use_container_width=True)
        if clear_log:
            st.session_state["inference_log"]   = []
            st.session_state["total_saved_kwh"] = 0.0
            st.session_state["last_result"]     = None
            st.rerun()

    # ── Run pipeline ──────────────────────────────────────────────────────────────
    if run_clicked:
        if not user_text.strip():
            st.warning("⚠️ Please enter some text first.")
        else:
            with st.spinner("Running adaptive pipeline…"):
                result = run_pipeline(user_text.strip())

            result["text"] = user_text.strip()
            st.session_state["last_result"]       = result
            st.session_state["total_saved_kwh"]  += result["saved_kwh"]
            st.session_state["inference_log"].append(result)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 2 — PREDICTION RESULT
    # ═══════════════════════════════════════════════════════════════════════════════
    st.markdown('<p class="section-title">🎯 Prediction Result</p>', unsafe_allow_html=True)

    result = st.session_state.get("last_result")

    if result is None:
        st.info("No inference run yet. Enter text above and click **Run Inference**.")
    else:
        stage = result["stage"]
        label = result["label"]
        conf  = result["confidence"]

        # Stage badge
        badge_class = {
            "Rule Engine": "badge-green",
            "DistilBERT":  "badge-blue",
            "BERT":        "badge-orange",
        }.get(stage, "badge-green")

        emoji_label = (
            "🟢 POSITIVE" if label == "POSITIVE" 
            else "🔴 NEGATIVE" if label == "NEGATIVE"
            else "🟡 NEUTRAL"
        )
        stage_icon  = {"Rule Engine": "🌿", "DistilBERT": "💙", "BERT": "🟠"}.get(stage, "")

        # Three result cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"""
                <div class="card card-accent">
                    <div style="font-size:.78rem;color:#94a3b8;text-transform:uppercase;
                                letter-spacing:.06em;margin-bottom:.5rem;">Prediction</div>
                    <div style="font-size:1.7rem;font-weight:700;
                                color:{'#4ade80' if label=='POSITIVE' else '#f87171' if label=='NEGATIVE' else '#eab308'};">
                        {emoji_label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            bar_width = int(conf * 100)
            bar_col   = "#4ade80" if conf >= 0.8 else "#eab308"
            st.markdown(
                f"""
                <div class="card">
                    <div style="font-size:.78rem;color:#94a3b8;text-transform:uppercase;
                                letter-spacing:.06em;margin-bottom:.6rem;">Confidence</div>
                    <div style="font-size:1.7rem;font-weight:700;
                                font-family:'JetBrains Mono',monospace;color:#f1f5f9;">
                        {conf:.1%}
                    </div>
                    <div style="background:#334155;border-radius:6px;height:6px;margin-top:.6rem;">
                        <div style="background:{bar_col};width:{bar_width}%;height:6px;
                                    border-radius:6px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="card">
                    <div style="font-size:.78rem;color:#94a3b8;text-transform:uppercase;
                                letter-spacing:.06em;margin-bottom:.6rem;">Inference Stage</div>
                    <div style="font-size:1.1rem;font-weight:600;margin-bottom:.5rem;">
                        {stage_icon} {stage}
                    </div>
                    <span class="badge {badge_class}">{stage.upper()}</span>
                    <div style="font-size:.78rem;color:#64748b;margin-top:.55rem;">
                        ⏱ {result.get('latency_ms', 0):.1f} ms
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Rule engine keyword hint
        if stage == "Rule Engine" and "matched" in result:
            st.success(f"🔑 Matched keywords: `{', '.join(result['matched'])}`")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 3 — ENERGY METRICS
    # ═══════════════════════════════════════════════════════════════════════════════
    st.markdown('<p class="section-title">⚡ Energy & Carbon Metrics</p>', unsafe_allow_html=True)

    if result:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("⚡ Energy Used",    f"{result['energy_kwh']:.6f} kWh")
        m2.metric("🌍 CO₂ Emissions",  f"{result['co2_kg']:.6f} kg")
        m3.metric("💚 Energy Saved",   f"{result['saved_pct']:.1f}%",
                  delta=f"vs always running BERT")
        m4.metric("🏭 Total Saved Today",
                  f"{st.session_state['total_saved_kwh'] * 1e6:.2f} µWh",
                  delta=f"{len(st.session_state['inference_log'])} inferences")

        st.markdown("<br>", unsafe_allow_html=True)

        # Green Score + Carbon Meter side by side
        gs_col, cm_col = st.columns([1, 1])

        with gs_col:
            st.plotly_chart(
                green_score_gauge(result["green_score"]),
                use_container_width=True,
            )

        with cm_col:
            st.markdown(
                '<div class="card" style="margin-top:1rem;">'
                '<div style="font-size:.85rem;color:#94a3b8;margin-bottom:.8rem;">'
                '🌱 Carbon Efficiency Meter</div>',
                unsafe_allow_html=True,
            )
            efficiency = result["saved_pct"] / 100
            colour = (
                "normal" if efficiency >= 0.6
                else "off"
            )
            st.progress(max(min(efficiency, 1.0), 0.0))
            saved_label = f"Saved {result['saved_pct']:.1f}% compared to always running BERT"
            st.caption(saved_label)

            # Stage energy comparison table
            st.markdown("<br>", unsafe_allow_html=True)
            stage_df = pd.DataFrame({
                "Stage":      ["Rule Engine", "DistilBERT", "BERT"],
                "Energy (µWh)": [0.001, 0.120, 0.320],
                "CO₂ (µg)":  [
                    round(0.000001 * 475_000, 4),
                    round(0.000120 * 475_000, 4),
                    round(0.000320 * 475_000, 4),
                ],
            })
            st.dataframe(stage_df, hide_index=True, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("Run an inference to see energy metrics.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 4 — VISUALISATION DASHBOARD
    # ═══════════════════════════════════════════════════════════════════════════════
    st.markdown('<p class="section-title">📊 Visualisation Dashboard</p>', unsafe_allow_html=True)

    log        = st.session_state["inference_log"]
    log_df     = build_log_df(log)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.plotly_chart(
            energy_bar_chart(result["stage"] if result else None),
            use_container_width=True,
        )

    with chart_col2:
        st.plotly_chart(stage_distribution_pie(log_df), use_container_width=True)

    st.plotly_chart(carbon_timeline(log_df), use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 5 — INFERENCE LOG TABLE
    # ═══════════════════════════════════════════════════════════════════════════════
    st.markdown('<p class="section-title">📋 Inference Log</p>', unsafe_allow_html=True)

    if log_df.empty:
        st.info("No inferences logged yet.")
    else:
        # Colour-code Stage column using pandas styler
        def colour_stage(val: str) -> str:
            colours = {
                "Rule Engine": "background-color:#14532d;color:#4ade80",
                "DistilBERT":  "background-color:#1e3a5f;color:#93c5fd",
                "BERT":        "background-color:#431407;color:#fdba74",
            }
            return colours.get(val, "")

        styled = log_df.style.applymap(colour_stage, subset=["Stage"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center;margin-top:3rem;padding:1rem;
                    color:#334155;font-size:.78rem;font-family:'JetBrains Mono',monospace;">
            Carbon-Aware AI Inference System · Green AI Demo ·
            Built with 🌿 Streamlit + HuggingFace Transformers + Plotly
        </div>
        """,
        unsafe_allow_html=True,
    )

elif app_mode == "Ollama Compare":
    # ── OLLAMA COMPARE DASHBOARD ──────────────────────────────────────────────
    render_dashboard()

else:
    # ── COMPARATIVE ANALYSIS DASHBOARD ────────────────────────────────────────
    render_comparative_dashboard()
