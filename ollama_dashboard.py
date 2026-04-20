"""
ollama_dashboard.py — UI Module for Multi-Model Comparison.
Renders the 3-panel dashboard and handles state for concurrent execution.
Now includes sentiment analysis mode with carbon emissions tracking.
"""

import streamlit as st
import concurrent.futures
import time
import plotly.graph_objects as go
from ollama_service import (
    is_ollama_running, 
    start_ollama, 
    get_installed_models, 
    run_ollama_inference,
    run_ollama_sentiment,
)

# ── Colour constants (reuse main dashboard palette) ───────────────────────────
BG_COLOUR   = "#0f172a"
CARD_COLOUR = "#1e293b"
TEXT_COLOUR  = "#f1f5f9"
GRID_COLOUR  = "#334155"
LABEL_COLOURS = {
    "POSITIVE": "#4ade80",
    "NEGATIVE": "#f87171",
    "NEUTRAL":  "#eab308",
}


def _emissions_bar_chart(responses: list[dict]) -> go.Figure:
    """Bar chart comparing energy/CO₂ across the completed Ollama panels."""
    valid = [r for r in responses if r and r.get("status") == "success" and "energy_kwh" in r]
    if not valid:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor=CARD_COLOUR, plot_bgcolor=CARD_COLOUR,
            font=dict(color=TEXT_COLOUR),
            title=dict(text="No emissions data yet", font=dict(color=TEXT_COLOUR)),
            height=280,
        )
        return fig

    models   = [r["model"].split(":")[0] for r in valid]
    energy   = [r["energy_kwh"] * 1e6 for r in valid]  # µWh
    co2      = [r["co2_kg"] * 1e6 for r in valid]      # µg
    colours  = ["#22c55e", "#3b82f6", "#f97316", "#a855f7", "#ec4899"]

    fig = go.Figure()
    for i, (m, e, c) in enumerate(zip(models, energy, co2)):
        fig.add_trace(go.Bar(
            name=m,
            x=[m], y=[e],
            marker=dict(color=colours[i % len(colours)]),
            text=[f"{e:.2f} µWh"],
            textposition="outside",
            hovertemplate=f"<b>{m}</b><br>Energy: {e:.2f} µWh<br>CO₂: {c:.2f} µg<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="⚡ Energy Consumed Per Model (µWh)", font=dict(color=TEXT_COLOUR, size=14)),
        paper_bgcolor=CARD_COLOUR, plot_bgcolor=CARD_COLOUR,
        font=dict(color=TEXT_COLOUR, family="monospace"),
        yaxis=dict(title="Energy (µWh)", gridcolor=GRID_COLOUR),
        xaxis=dict(gridcolor=GRID_COLOUR),
        showlegend=False, height=300,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def render_ollama_status():
    """Render the Ollama status badge and auto-start logic."""
    if "ollama_started" not in st.session_state:
        st.session_state.ollama_started = False

    is_running = is_ollama_running()
    
    if not is_running and not st.session_state.ollama_started:
        st.session_state.ollama_started = True # Mark as tried immediately
        with st.status("🚀 Ollama not running. Attempting auto-start...", expanded=False):
            success = start_ollama()
            if success:
                st.success("Ollama started successfully!")
            else:
                st.error("Failed to start Ollama automatically. Please ensure it is installed and running.")

    # Status indicator
    is_running = is_ollama_running() # Check again
    if is_running:
        st.markdown('<span class="badge badge-green">● OLLAMA RUNNING</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-orange">○ OLLAMA DISCONNECTED</span>', unsafe_allow_html=True)

def render_dashboard():
    """Main entry point for the Comparison Dashboard."""
    
    # Custom CSS for the Ollama panels (reusing existing theme patterns)
    st.markdown("""
        <style>
        .panel-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            height: 500px;
            display: flex;
            flex-direction: column;
        }
        .panel-header {
            font-size: 0.85rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.8rem;
            display: flex;
            justify-content: space-between;
        }
        .panel-output {
            background: #0f172a;
            border-radius: 8px;
            padding: 1rem;
            flex-grow: 1;
            overflow-y: auto;
            font-size: 0.9rem;
            line-height: 1.5;
            color: #f1f5f9;
            white-space: pre-wrap;
            border: 1px solid #1e293b;
        }
        .latency-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: #64748b;
            margin-top: 0.5rem;
        }
        .sentiment-result {
            background: #0f172a;
            border-radius: 10px;
            padding: 1rem;
            border: 1px solid #334155;
        }
        .emission-row {
            display: flex;
            justify-content: space-between;
            padding: 0.3rem 0;
            font-size: 0.8rem;
            color: #94a3b8;
            font-family: 'JetBrains Mono', monospace;
        }
        .emission-row .val {
            color: #f1f5f9;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">🤖 Multi-Model Compare</p>', unsafe_allow_html=True)
    
    # Init session state for panels
    if "ollama_responses" not in st.session_state:
        st.session_state.ollama_responses = [None, None, None]
    if "ollama_loading" not in st.session_state:
        st.session_state.ollama_loading = False
    if "ollama_mode" not in st.session_state:
        st.session_state.ollama_mode = None  # "general" or "sentiment"

    # ── Top Section: Prompt ───────────────────────────────────────────────────
    col_prompt, col_status = st.columns([3, 1])
    
    with col_prompt:
        user_prompt = st.text_area(
            "Enter your prompt",
            placeholder="Ask something to compare models, or enter text for sentiment analysis...",
            height=120,
            label_visibility="collapsed",
            key="ollama_prompt"
        )
    
    with col_status:
        st.markdown("<br>", unsafe_allow_html=True)
        render_ollama_status()
        
        models = get_installed_models()
        if st.button("🔄 Refresh Models", use_container_width=True):
            st.rerun()
        
        # Two action buttons
        run_btn = st.button(
            "🚀 Run Models", use_container_width=True,
            disabled=st.session_state.ollama_loading
        )
        sentiment_btn = st.button(
            "🎯 Sentiment Analysis", use_container_width=True,
            type="primary",
            disabled=st.session_state.ollama_loading,
            help="Classify sentiment as POSITIVE / NEGATIVE / NEUTRAL and measure carbon emissions"
        )
        if st.button("🗑 Clear All", use_container_width=True):
            st.session_state.ollama_responses = [None, None, None]
            st.session_state.ollama_mode = None
            st.rerun()

    # ── Main Section: Panels ──────────────────────────────────────────────────
    cols = st.columns(3)
    
    # Model selection persist
    if "panel_models" not in st.session_state:
        st.session_state.panel_models = [
            models[0] if len(models) > 0 else "n/a",
            models[1] if len(models) > 1 else (models[0] if len(models) > 0 else "n/a"),
            models[2] if len(models) > 2 else (models[0] if len(models) > 0 else "n/a"),
        ]

    for i in range(3):
        with cols[i]:
            st.session_state.panel_models[i] = st.selectbox(
                f"Model {i+1}",
                options=models if models else ["No models found"],
                index=min(i, len(models)-1) if models else 0,
                key=f"model_select_{i}"
            )
            
            # Panel Container
            container = st.container()
            with container:
                resp = st.session_state.ollama_responses[i]
                
                # Header
                st.markdown(f"""
                    <div style="font-size:0.75rem; color:#94a3b8; font-weight:600; margin-bottom:5px;">
                        PANEL {i+1}
                    </div>
                """, unsafe_allow_html=True)
                
                # Output area
                if st.session_state.ollama_loading:
                    st.info("Thinking...")
                elif resp:
                    if resp["status"] == "success":
                        # Check if this is a sentiment response
                        if st.session_state.ollama_mode == "sentiment" and "label" in resp:
                            label = resp["label"]
                            conf  = resp.get("confidence", 0)
                            lbl_col = LABEL_COLOURS.get(label, "#94a3b8")
                            emoji = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "🟡"}.get(label, "⚪")
                            
                            st.markdown(f"""
                            <div class="sentiment-result">
                                <div style="font-size:1.4rem; font-weight:700; color:{lbl_col}; margin-bottom:0.5rem;">
                                    {emoji} {label}
                                </div>
                                <div style="background:#334155;border-radius:6px;height:6px;margin:0.5rem 0;">
                                    <div style="background:{lbl_col};width:{int(conf*100)}%;height:6px;border-radius:6px;"></div>
                                </div>
                                <div style="font-size:0.85rem;color:#94a3b8;">Confidence: <b style="color:#f1f5f9">{conf:.1%}</b></div>
                                <hr style="border-color:#334155;margin:0.6rem 0;">
                                <div class="emission-row">
                                    <span>⚡ Energy</span>
                                    <span class="val">{resp.get('energy_kwh', 0)*1e6:.2f} µWh</span>
                                </div>
                                <div class="emission-row">
                                    <span>🌍 CO₂</span>
                                    <span class="val">{resp.get('co2_kg', 0)*1e6:.2f} µg</span>
                                </div>
                                <div class="emission-row">
                                    <span>⏱ Latency</span>
                                    <span class="val">{resp.get('latency', 0)}s</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show raw response in an expander
                            with st.expander("📝 Raw model response"):
                                st.text(resp.get("raw_response", ""))
                        else:
                            # General inference (original behaviour)
                            st.markdown(f'<div class="panel-output">{resp["response"]}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="latency-badge">⏱ {resp["latency"]}s · {resp["model"]}</div>', unsafe_allow_html=True)
                    else:
                        st.error(f"Error: {resp['message']}")
                else:
                    st.markdown('<div class="panel-output" style="color:#334155;">Idle... waiting for prompt.</div>', unsafe_allow_html=True)

    # ── Carbon Emissions Chart (sentiment mode only) ──────────────────────────
    if st.session_state.ollama_mode == "sentiment" and any(
        r and r.get("status") == "success" and "energy_kwh" in r
        for r in st.session_state.ollama_responses
    ):
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<p class="section-title">⚡ Carbon Emissions Comparison</p>', unsafe_allow_html=True)
        st.plotly_chart(
            _emissions_bar_chart(st.session_state.ollama_responses),
            use_container_width=True,
        )

    # ── Execution Logic ───────────────────────────────────────────────────────
    if run_btn:
        if not user_prompt.strip():
            st.warning("⚠️ Please enter a prompt.")
        elif not models:
            st.error("❌ No models selected or installed.")
        else:
            st.session_state.ollama_loading = True
            st.session_state.ollama_mode = "general"
            st.rerun()

    if sentiment_btn:
        if not user_prompt.strip():
            st.warning("⚠️ Please enter text for sentiment analysis.")
        elif not models:
            st.error("❌ No models selected or installed.")
        else:
            st.session_state.ollama_loading = True
            st.session_state.ollama_mode = "sentiment"
            st.rerun()

    # Triggered rerun handler
    if st.session_state.ollama_loading:
        selected_models = st.session_state.panel_models
        mode = st.session_state.ollama_mode
        
        # Parallel Execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            if mode == "sentiment":
                future_to_panel = {
                    executor.submit(run_ollama_sentiment, selected_models[i], user_prompt): i 
                    for i in range(3)
                }
            else:
                future_to_panel = {
                    executor.submit(run_ollama_inference, selected_models[i], user_prompt): i 
                    for i in range(3)
                }
            
            for future in concurrent.futures.as_completed(future_to_panel):
                panel_idx = future_to_panel[future]
                try:
                    data = future.result()
                    st.session_state.ollama_responses[panel_idx] = data
                except Exception as exc:
                    st.session_state.ollama_responses[panel_idx] = {
                        "status": "error", 
                        "message": str(exc),
                        "model": selected_models[panel_idx]
                    }
        
        st.session_state.ollama_loading = False
        st.rerun()
