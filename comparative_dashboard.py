"""
comparative_dashboard.py — Head-to-head comparison of the 3-Step Adaptive
Pipeline vs Ollama models for sentiment analysis with carbon emissions.
"""

import streamlit as st
import concurrent.futures
import plotly.graph_objects as go
import pandas as pd

from inference_pipeline import run_pipeline
from config import ENERGY_LARGE_MODEL
from ollama_service import (
    is_ollama_running,
    get_installed_models,
    run_ollama_sentiment,
)

# ── Colour palette ─────────────────────────────────────────────────────────────
BG_COLOUR    = "#0f172a"
CARD_COLOUR  = "#1e293b"
TEXT_COLOUR   = "#f1f5f9"
GRID_COLOUR   = "#334155"
ACCENT_GREEN  = "#22c55e"
ACCENT_BLUE   = "#3b82f6"
ACCENT_ORANGE = "#f97316"
ACCENT_PURPLE = "#a855f7"
ACCENT_PINK   = "#ec4899"

LABEL_COLOURS = {
    "POSITIVE": "#4ade80",
    "NEGATIVE": "#f87171",
    "NEUTRAL":  "#eab308",
}
LABEL_EMOJI = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "🟡"}

MODEL_COLOURS = [ACCENT_GREEN, ACCENT_BLUE, ACCENT_ORANGE, ACCENT_PURPLE, ACCENT_PINK]


def _unique_model_name(model_full: str, seen: dict) -> str:
    """Return a unique display name for the model, adding a suffix if needed."""
    base = model_full  # use full name including tag (e.g. gemma3:4b)
    if base in seen:
        seen[base] += 1
        return f"{base} #{seen[base]}"
    else:
        seen[base] = 1
        return base


def _compute_green_score(energy_kwh: float) -> int:
    """Compute a 0-100 green score: 100 = zero energy, 0 = BERT-level energy."""
    if ENERGY_LARGE_MODEL <= 0:
        return 0
    score = round(100 * (1 - energy_kwh / ENERGY_LARGE_MODEL))
    return max(0, min(100, score))


def _base_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=TEXT_COLOUR, size=14)),
        paper_bgcolor=CARD_COLOUR,
        plot_bgcolor=CARD_COLOUR,
        font=dict(color=TEXT_COLOUR, family="monospace"),
        margin=dict(l=40, r=20, t=50, b=40),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_chart_data(pipeline_result, ollama_results):
    """Build parallel lists of (name, energy_µWh, co2_µg, latency_ms, colour)
    for all successful models. Uses full model names to avoid duplicates."""
    names, energies, co2s, latencies, colours = [], [], [], [], []
    seen = {}

    if pipeline_result:
        stage = pipeline_result.get("stage", "Pipeline")
        names.append(f"3-Step ({stage})")
        energies.append(pipeline_result["energy_kwh"] * 1e6)
        co2s.append(pipeline_result["co2_kg"] * 1e6)
        latencies.append(pipeline_result.get("latency_ms", 0))
        colours.append(ACCENT_GREEN)

    for i, r in enumerate(ollama_results):
        if r and r.get("status") == "success":
            name = _unique_model_name(r["model"], seen)
            names.append(name)
            energies.append(r.get("energy_kwh", 0) * 1e6)
            co2s.append(r.get("co2_kg", 0) * 1e6)
            latencies.append(r.get("latency", 0) * 1000)
            colours.append(MODEL_COLOURS[(i + 1) % len(MODEL_COLOURS)])

    return names, energies, co2s, latencies, colours


def _energy_comparison_bar(pipeline_result, ollama_results):
    names, energies, _, _, colours = _collect_chart_data(pipeline_result, ollama_results)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=energies,
        marker=dict(color=colours, line=dict(width=0)),
        text=[f"{e:.3f}" for e in energies],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Energy: %{y:.4f} µWh<extra></extra>",
    ))
    layout = _base_layout("⚡ Energy Consumed (µWh)")
    layout.update(
        yaxis=dict(title="Energy (µWh)", gridcolor=GRID_COLOUR),
        xaxis=dict(gridcolor=GRID_COLOUR),
        showlegend=False, height=320,
    )
    fig.update_layout(**layout)
    return fig


def _co2_comparison_bar(pipeline_result, ollama_results):
    names, _, co2s, _, colours = _collect_chart_data(pipeline_result, ollama_results)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=co2s,
        marker=dict(color=colours, line=dict(width=0)),
        text=[f"{c:.3f}" for c in co2s],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>CO₂: %{y:.4f} µg<extra></extra>",
    ))
    layout = _base_layout("🌍 CO₂ Emissions (µg)")
    layout.update(
        yaxis=dict(title="CO₂ (µg)", gridcolor=GRID_COLOUR),
        xaxis=dict(gridcolor=GRID_COLOUR),
        showlegend=False, height=320,
    )
    fig.update_layout(**layout)
    return fig


def _latency_comparison_bar(pipeline_result, ollama_results):
    names, _, _, latencies, colours = _collect_chart_data(pipeline_result, ollama_results)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=latencies,
        marker=dict(color=colours, line=dict(width=0)),
        text=[f"{l:.0f} ms" for l in latencies],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Latency: %{y:.1f} ms<extra></extra>",
    ))
    layout = _base_layout("⏱ Inference Latency (ms)")
    layout.update(
        yaxis=dict(title="Latency (ms)", gridcolor=GRID_COLOUR),
        xaxis=dict(gridcolor=GRID_COLOUR),
        showlegend=False, height=320,
    )
    fig.update_layout(**layout)
    return fig


def _energy_pie(pipeline_result, ollama_results):
    names, energies, _, _, colours = _collect_chart_data(pipeline_result, ollama_results)
    fig = go.Figure(go.Pie(
        labels=names, values=energies,
        marker=dict(colors=colours, line=dict(color=BG_COLOUR, width=2)),
        hole=0.45,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Energy: %{value:.4f} µWh<extra></extra>",
    ))
    layout = _base_layout("🥧 Energy Share")
    layout.update(height=340, showlegend=True, legend=dict(orientation="h", y=-0.15))
    fig.update_layout(**layout)
    return fig


def _savings_waterfall(pipeline_result, ollama_results):
    if not pipeline_result:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=CARD_COLOUR, plot_bgcolor=CARD_COLOUR,
                          font=dict(color=TEXT_COLOUR), height=320)
        return fig

    pipe_energy = pipeline_result["energy_kwh"] * 1e6
    names, savings, colours = [], [], []
    seen = {}

    for r in ollama_results:
        if r and r.get("status") == "success":
            name = _unique_model_name(r["model"], seen)
            ollama_energy = r.get("energy_kwh", 0) * 1e6
            saved = ollama_energy - pipe_energy
            names.append(name)
            savings.append(saved)
            colours.append(ACCENT_GREEN if saved > 0 else "#f87171")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=savings,
        marker=dict(color=colours),
        text=[f"{s:.3f} µWh\nsaved" if s > 0 else f"{abs(s):.3f} µWh\nextra" for s in savings],
        textposition="outside",
    ))
    layout = _base_layout("💚 Energy Saved by 3-Step vs Each Ollama Model (µWh)")
    layout.update(
        yaxis=dict(title="Energy Saved (µWh)", gridcolor=GRID_COLOUR),
        showlegend=False, height=320,
    )
    fig.update_layout(**layout)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT CARD RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

def _render_result_card(title: str, label: str, confidence: float,
                        energy_kwh: float, co2_kg: float,
                        latency_display: str, green_score: int,
                        subtitle: str = "", accent: str = "#22c55e"):
    lbl_col = LABEL_COLOURS.get(label, "#94a3b8")
    emoji   = LABEL_EMOJI.get(label, "⚪")
    gs_col  = "#22c55e" if green_score >= 80 else "#eab308" if green_score >= 50 else "#f87171"
    st.markdown(f"""
    <div class="card" style="border-top:3px solid {accent};">
        <div style="font-size:.75rem;color:#94a3b8;text-transform:uppercase;
                    letter-spacing:.06em;margin-bottom:.3rem;">{title}</div>
        {'<div style="font-size:.7rem;color:#64748b;margin-bottom:.5rem;">' + subtitle + '</div>' if subtitle else ''}
        <div style="font-size:1.5rem;font-weight:700;color:{lbl_col};margin-bottom:.4rem;">
            {emoji} {label}
        </div>
        <div style="background:#334155;border-radius:6px;height:5px;margin-bottom:.5rem;">
            <div style="background:{lbl_col};width:{int(confidence*100)}%;height:5px;border-radius:6px;"></div>
        </div>
        <div style="font-size:.82rem;color:#94a3b8;margin-bottom:.6rem;">
            Confidence: <b style="color:#f1f5f9">{confidence:.1%}</b>
        </div>
        <hr style="border-color:#334155;margin:.4rem 0;">
        <div style="display:flex;justify-content:space-between;font-size:.78rem;color:#94a3b8;margin-top:.3rem;">
            <span>⚡ {energy_kwh*1e6:.3f} µWh</span>
            <span>🌍 {co2_kg*1e6:.3f} µg</span>
            <span>⏱ {latency_display}</span>
        </div>
        <div style="display:flex;justify-content:center;margin-top:.5rem;">
            <span style="font-size:.75rem;color:{gs_col};font-family:'JetBrains Mono',monospace;
                         font-weight:600;">🌿 Green Score: {green_score}/100</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def render_comparative_dashboard():
    """Render the full Comparative Analysis dashboard."""

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>📊 Comparative Carbon Analysis</h1>
        <p>
            Run the <strong>same text</strong> through the 3-Step Adaptive Pipeline
            <em>and</em> your selected Ollama models, then compare sentiment results
            and carbon emissions side-by-side.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Session state ─────────────────────────────────────────────────────────
    if "comp_pipeline_result" not in st.session_state:
        st.session_state.comp_pipeline_result = None
    if "comp_ollama_results" not in st.session_state:
        st.session_state.comp_ollama_results = []
    if "comp_loading" not in st.session_state:
        st.session_state.comp_loading = False

    # ── Input section ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">📝 Input</p>', unsafe_allow_html=True)

    col_input, col_controls = st.columns([3, 1])

    with col_input:
        comp_text = st.text_area(
            "Text for comparative analysis",
            placeholder="Enter text to analyse sentiment and compare carbon emissions across models...",
            height=110,
            label_visibility="collapsed",
            key="comp_text_input",
        )

        # Ollama model selector
        models = get_installed_models()
        if models:
            selected_ollama = st.multiselect(
                "🤖 Select Ollama models to compare",
                options=models,
                default=models[:min(2, len(models))],
                max_selections=5,
                key="comp_model_select",
                help="Pick up to 5 Ollama models for side-by-side comparison"
            )
        else:
            selected_ollama = []
            st.warning("No Ollama models found. Ensure Ollama is running with models pulled.")

    with col_controls:
        st.markdown("<br>", unsafe_allow_html=True)

        # Ollama status
        is_running = is_ollama_running()
        if is_running:
            st.markdown('<span class="badge badge-green">● OLLAMA RUNNING</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-orange">○ OLLAMA DISCONNECTED</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run_comp = st.button(
            "⚡ Run Comparative Analysis",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.comp_loading,
        )
        if st.button("🗑 Clear Results", use_container_width=True):
            st.session_state.comp_pipeline_result = None
            st.session_state.comp_ollama_results = []
            st.rerun()

    # ── Trigger execution ─────────────────────────────────────────────────────
    if run_comp:
        if not comp_text.strip():
            st.warning("⚠️ Please enter some text first.")
        elif not selected_ollama and not models:
            st.error("❌ No Ollama models available.")
        else:
            st.session_state.comp_loading = True
            st.session_state._comp_text = comp_text.strip()
            st.session_state._comp_models = selected_ollama
            st.rerun()

    # ── Background execution on rerun ─────────────────────────────────────────
    if st.session_state.comp_loading:
        text = st.session_state.get("_comp_text", "")
        sel_models = st.session_state.get("_comp_models", [])

        with st.spinner("Running 3-Step Pipeline + Ollama models in parallel…"):
            # Run 3-step pipeline (fast, runs in main thread)
            pipeline_result = run_pipeline(text)
            pipeline_result["text"] = text
            st.session_state.comp_pipeline_result = pipeline_result

            # Run Ollama models in parallel
            ollama_results = [None] * len(sel_models)
            if sel_models:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(sel_models)) as executor:
                    future_map = {
                        executor.submit(run_ollama_sentiment, m, text): i
                        for i, m in enumerate(sel_models)
                    }
                    for future in concurrent.futures.as_completed(future_map):
                        idx = future_map[future]
                        try:
                            ollama_results[idx] = future.result()
                        except Exception as exc:
                            ollama_results[idx] = {
                                "status": "error",
                                "message": str(exc),
                                "model": sel_models[idx],
                            }

            st.session_state.comp_ollama_results = ollama_results

        st.session_state.comp_loading = False
        st.rerun()

    # ── Display results ───────────────────────────────────────────────────────
    pipe_r  = st.session_state.comp_pipeline_result
    ollama_rs = st.session_state.comp_ollama_results

    if pipe_r is None and not ollama_rs:
        st.info("Enter text above and click **Run Comparative Analysis** to begin.")
        return

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">🎯 Sentiment Results</p>', unsafe_allow_html=True)

    # Build columns: 1 for pipeline + 1 per Ollama model
    n_ollama = len([r for r in ollama_rs if r])
    total_cols = 1 + n_ollama
    result_cols = st.columns(total_cols)

    # 3-Step pipeline card
    with result_cols[0]:
        if pipe_r:
            _render_result_card(
                title="3-Step Adaptive Pipeline",
                label=pipe_r["label"],
                confidence=pipe_r["confidence"],
                energy_kwh=pipe_r["energy_kwh"],
                co2_kg=pipe_r["co2_kg"],
                latency_display=f"{pipe_r.get('latency_ms', 0):.0f} ms",
                green_score=pipe_r.get("green_score", 0),
                subtitle=f"Stage used: {pipe_r['stage']}",
                accent=ACCENT_GREEN,
            )
            if pipe_r.get("stage") == "Rule Engine" and "matched" in pipe_r:
                st.caption(f"🔑 Keywords: {', '.join(pipe_r['matched'])}")

    # Ollama model cards
    col_idx = 1
    for r in ollama_rs:
        if r is None:
            continue
        with result_cols[col_idx]:
            if r.get("status") == "success":
                ollama_gs = _compute_green_score(r.get("energy_kwh", 0))
                _render_result_card(
                    title=f"Ollama: {r['model'].split(':')[0]}",
                    label=r["label"],
                    confidence=r["confidence"],
                    energy_kwh=r.get("energy_kwh", 0),
                    co2_kg=r.get("co2_kg", 0),
                    latency_display=f"{r.get('latency', 0)*1000:.0f} ms",
                    green_score=ollama_gs,
                    subtitle=r["model"],
                    accent=MODEL_COLOURS[col_idx % len(MODEL_COLOURS)],
                )
                with st.expander("📝 Raw response"):
                    st.text(r.get("raw_response", ""))
            else:
                st.error(f"❌ {r.get('model', '?')}: {r.get('message', 'Unknown error')}")
        col_idx += 1

    # ═══════════════════════════════════════════════════════════════════════════
    # VISUALISATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">📊 Carbon Emissions Analysis</p>', unsafe_allow_html=True)

    chart_c1, chart_c2 = st.columns(2)
    with chart_c1:
        st.plotly_chart(_energy_comparison_bar(pipe_r, ollama_rs), use_container_width=True)
    with chart_c2:
        st.plotly_chart(_co2_comparison_bar(pipe_r, ollama_rs), use_container_width=True)

    chart_c3, chart_c4 = st.columns(2)
    with chart_c3:
        st.plotly_chart(_latency_comparison_bar(pipe_r, ollama_rs), use_container_width=True)
    with chart_c4:
        st.plotly_chart(_energy_pie(pipe_r, ollama_rs), use_container_width=True)

    # Savings chart
    st.plotly_chart(_savings_waterfall(pipe_r, ollama_rs), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">📋 Full Comparison Table</p>', unsafe_allow_html=True)

    rows = []
    if pipe_r:
        rows.append({
            "Model": f"3-Step ({pipe_r['stage']})",
            "Sentiment": pipe_r["label"],
            "Confidence": f"{pipe_r['confidence']:.1%}",
            "Energy (µWh)": f"{pipe_r['energy_kwh']*1e6:.4f}",
            "CO₂ (µg)": f"{pipe_r['co2_kg']*1e6:.4f}",
            "Latency (ms)": f"{pipe_r.get('latency_ms', 0):.1f}",
            "Green Score": f"{pipe_r.get('green_score', 0)} / 100",
        })
    for r in ollama_rs:
        if r and r.get("status") == "success":
            gs = _compute_green_score(r.get("energy_kwh", 0))
            rows.append({
                "Model": r["model"],
                "Sentiment": r["label"],
                "Confidence": f"{r['confidence']:.1%}",
                "Energy (µWh)": f"{r.get('energy_kwh', 0)*1e6:.4f}",
                "CO₂ (µg)": f"{r.get('co2_kg', 0)*1e6:.4f}",
                "Latency (ms)": f"{r.get('latency', 0)*1000:.1f}",
                "Green Score": f"{gs} / 100",
            })

    if rows:
        df = pd.DataFrame(rows)

        def _colour_sentiment(val: str) -> str:
            colours = {
                "POSITIVE": "background-color:#14532d;color:#4ade80",
                "NEGATIVE": "background-color:#431407;color:#f87171",
                "NEUTRAL":  "background-color:#422006;color:#eab308",
            }
            return colours.get(val, "")

        styled = df.style.applymap(_colour_sentiment, subset=["Sentiment"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("No results to display.")
