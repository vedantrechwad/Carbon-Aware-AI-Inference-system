"""
model_loader.py — Lazy, cached loading of transformer models.

Models are loaded *once* per process and reused across every inference call.
This avoids the multi-second startup penalty on every request and prevents
unnecessary memory churn.

Caching strategy
----------------
We use a module-level dictionary ``_cache`` keyed by model name.
Streamlit's @st.cache_resource would work too, but keeping model loading
*outside* Streamlit makes this module testable independently.
"""

from __future__ import annotations
import logging
from typing import Any

from transformers import pipeline

logger = logging.getLogger(__name__)

# Module-level singleton cache: { model_name: HuggingFace pipeline }
_cache: dict[str, Any] = {}


def load_model(model_name: str) -> Any:
    """
    Return a HuggingFace text-classification pipeline for *model_name*.

    The pipeline is loaded with ``device=-1`` (CPU) so it works on any
    machine without a GPU.  ``truncation=True`` prevents tokeniser errors
    on long inputs.

    Parameters
    ----------
    model_name : str
        HuggingFace model hub identifier.

    Returns
    -------
    transformers.Pipeline
        Ready-to-call sentiment-analysis pipeline.
    """
    if model_name not in _cache:
        logger.info("Loading model: %s", model_name)
        _cache[model_name] = pipeline(
            task="text-classification",
            model=model_name,
            device=-1,          # CPU inference — no GPU required
            truncation=True,
            max_length=512,
        )
        logger.info("Model loaded: %s", model_name)
    return _cache[model_name]


def preload_models() -> None:
    """
    Eagerly load all pipeline models into the cache.

    Call once at application startup (e.g. from ``app.py``) to avoid
    cold-start latency on the first user request.
    """
    from config import SMALL_MODEL_NAME, LARGE_MODEL_NAME
    for name in (SMALL_MODEL_NAME, LARGE_MODEL_NAME):
        load_model(name)
    logger.info("All models pre-loaded into cache.")


def run_inference(model_name: str, text: str) -> dict:
    """
    Run sentiment inference and return a normalised result dict.

    Parameters
    ----------
    model_name : str
        Model to use.
    text : str
        Input sentence.

    Returns
    -------
    dict with keys ``label`` (str) and ``confidence`` (float 0-1).
    """
    clf = load_model(model_name)
    # top_k=1: only compute the best prediction — skip sorting all classes
    raw = clf(text, top_k=1)[0]
    
    # Handle different label formats from different models
    label = raw["label"].upper()
    
    # Map common label variations to standard format
    label_mapping = {
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "NEUTRAL", 
        "LABEL_2": "POSITIVE",
        "NEG": "NEGATIVE",
        "NEU": "NEUTRAL",
        "POS": "POSITIVE",
        # nlptown/bert-base-multilingual-uncased-sentiment uses star ratings
        "1 STAR": "NEGATIVE",
        "2 STARS": "NEGATIVE",
        "3 STARS": "NEUTRAL",
        "4 STARS": "POSITIVE",
        "5 STARS": "POSITIVE",
    }
    
    label = label_mapping.get(label, label)
    
    return {
        "label":      label,
        "confidence": round(float(raw["score"]), 4),
    }
