"""
inference_pipeline.py — Three-stage Carbon-Aware Inference Pipeline.

Pipeline logic
--------------
Stage 1  Rule Engine    (near-zero energy)
         ↓ if ambiguous
Stage 2  DistilBERT     (small model, ~38% of BERT energy)
         ↓ if confidence < 0.80
Stage 3  BERT           (large model, full quality)

Optimisations
-------------
- Stage 2 runs *without* CodeCarbon overhead — the tracker is only
  invoked if Stage 2 is accepted or when Stage 3 fires.
- Avoids unnecessary function-wrapping where possible.

The function ``run_pipeline`` is the single public entry point.
It returns a rich result dict consumed by the Streamlit dashboard.
"""

from __future__ import annotations
import time
import logging

from config import (
    SMALL_MODEL_NAME,
    LARGE_MODEL_NAME,
    SMALL_MODEL_THRESHOLD,
)
from rule_engine  import detect_sentiment
from model_loader import run_inference
from energy_tracker import estimate_energy, measure_with_tracker

logger = logging.getLogger(__name__)


def run_pipeline(text: str) -> dict:
    """
    Execute the adaptive inference pipeline on *text*.

    Parameters
    ----------
    text : str
        Raw input sentence from the user.

    Returns
    -------
    dict
        label        – "POSITIVE" | "NEGATIVE" | "NEUTRAL"
        confidence   – float 0-1
        stage        – "Rule Engine" | "DistilBERT" | "BERT"
        latency_ms   – wall-clock inference time in milliseconds
        energy_kwh   – estimated energy consumed
        co2_kg       – estimated CO₂ emissions
        green_score  – 0-100 environmental efficiency
        saved_pct    – % energy saved vs always running BERT
        saved_kwh    – absolute kWh saved
        matched      – (Rule Engine only) matched keywords
    """
    t0 = time.perf_counter()

    # ── Stage 1: Rule Engine ──────────────────────────────────────────────────
    rule_result = detect_sentiment(text)
    if rule_result is not None:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        energy = estimate_energy("Rule Engine")
        logger.info("Stage 1 fired: %s", rule_result["label"])
        return {
            **rule_result,
            "latency_ms": latency_ms,
            **energy,
        }

    # ── Stage 2: DistilBERT ───────────────────────────────────────────────────
    # Run WITHOUT CodeCarbon wrapper to reduce overhead.
    # If accepted, we use empirical estimates (or re-measure if needed).
    small_raw = run_inference(SMALL_MODEL_NAME, text)
    small_result = {
        "label":      small_raw["label"],
        "confidence": small_raw["confidence"],
        "stage":      "DistilBERT",
    }

    if small_result["confidence"] >= SMALL_MODEL_THRESHOLD:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        energy = estimate_energy("DistilBERT")
        logger.info(
            "Stage 2 accepted: %s @ %.3f", small_result["label"], small_result["confidence"]
        )
        return {
            **small_result,
            "latency_ms": latency_ms,
            **energy,
        }

    # ── Stage 3: BERT ─────────────────────────────────────────────────────────
    def _run_large():
        return run_inference(LARGE_MODEL_NAME, text)

    large_raw, tracker_data = measure_with_tracker(_run_large)
    large_result = {
        "label":      large_raw["label"],
        "confidence": large_raw["confidence"],
        "stage":      "BERT",
    }

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    energy = estimate_energy("BERT")

    # If CodeCarbon measured real energy for Stage 3, prefer it
    if tracker_data.get("source") == "CodeCarbon":
        energy["energy_kwh"] = tracker_data["energy_kwh"]
        energy["co2_kg"]     = tracker_data["co2_kg"]

    logger.info("Stage 3 used: %s @ %.3f", large_result["label"], large_result["confidence"])
    return {
        **large_result,
        "latency_ms": latency_ms,
        **energy,
    }
