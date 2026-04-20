"""
ollama_service.py — Backend service for Ollama model management and inference.
Handles process lifecycle, model discovery, and API communication.
"""

import subprocess
import platform
import time
import json
import re
import requests
import os
import logging

from energy_tracker import measure_with_tracker

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_API_TAGS = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_API_GENERATE = f"{OLLAMA_BASE_URL}/api/generate"

logger = logging.getLogger(__name__)

# ── Sentiment analysis prompt template ─────────────────────────────────────────
_SENTIMENT_SYSTEM = (
    "You are a precise sentiment classifier. Analyse the user's text carefully "
    "and classify the sentiment as exactly one of: POSITIVE, NEGATIVE, or NEUTRAL.\n"
    "You must respond with ONLY a valid JSON object — no markdown, no explanation.\n"
    "The confidence value must be a precise decimal reflecting how certain you are "
    "(e.g. 0.7234, 0.912, 0.5541 — do NOT round to multiples of 0.05).\n"
    'Example: {"label": "POSITIVE", "confidence": 0.8723}\n'
)

def is_ollama_running() -> bool:
    """Check if the Ollama server is responsive."""
    try:
        response = requests.get(OLLAMA_API_TAGS, timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False

def get_ollama_path():
    """Find the ollama binary path, prioritizing known locations on Windows."""
    if platform.system() == "Windows":
        # Check standard installation path
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        standard_path = os.path.join(local_app_data, "Programs", "Ollama", "ollama.exe")
        if os.path.exists(standard_path):
            return standard_path
    return "ollama" # Fallback to PATH

def start_ollama():
    """Start the Ollama server in the background based on the OS."""
    if is_ollama_running():
        return True

    ollama_bin = get_ollama_path()
    try:
        if platform.system() == "Windows":
            # Quoting path for spaces safety and using shell=True for better execution
            cmd = f'"{ollama_bin}" serve'
            subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            # Start ollama serve on Mac/Linux
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setpgrp # ensures it keeps running independently
            )
        
        # Poll until ready
        max_retries = 10
        for _ in range(max_retries):
            time.sleep(2)
            if is_ollama_running():
                return True
        return False
    except Exception as e:
        logger.error(f"Failed to start Ollama: {e}")
        return False

def get_installed_models() -> list[str]:
    """Fetch all installed Ollama models."""
    try:
        response = requests.get(OLLAMA_API_TAGS, timeout=5)
        if response.status_code == 200:
            data = response.json()
            # models is a list of dicts with 'name' key
            return [m['name'] for m in data.get('models', [])]
        return []
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return []

def run_ollama_inference(model: str, prompt: str) -> dict:
    """Run a non-streaming inference on a specific model."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    start_time = time.time()
    try:
        response = requests.post(OLLAMA_API_GENERATE, json=payload, timeout=60)
        latency = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            return {
                "status": "success",
                "response": result.get("response", ""),
                "latency": round(latency, 2),
                "model": model
            }
        else:
            return {
                "status": "error",
                "message": f"API Error: {response.status_code}",
                "model": model
            }
    except requests.Timeout:
        return {
            "status": "error",
            "message": "Request timed out",
            "model": model
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "model": model
        }


def _parse_sentiment_response(raw_text: str) -> dict:
    """
    Extract {"label": ..., "confidence": ...} from Ollama's free-text response.
    Handles models that wrap JSON in markdown code fences or add extra text.
    """
    text = raw_text.strip()

    # Try to find JSON in the response
    # Pattern 1: raw JSON
    # Pattern 2: ```json ... ```
    # Pattern 3: ``` ... ```
    json_patterns = [
        r'\{[^{}]*"label"[^{}]*\}',     # bare JSON object with "label"
        r'```json\s*(\{.*?\})\s*```',     # markdown json fence
        r'```\s*(\{.*?\})\s*```',         # generic markdown fence
    ]

    for pattern in json_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(0) if match.lastindex is None else match.group(1)
            try:
                parsed = json.loads(json_str)
                label = parsed.get("label", "NEUTRAL").upper().strip()
                confidence = float(parsed.get("confidence", 0.5))
                # Normalise label
                if label not in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
                    if "POS" in label:
                        label = "POSITIVE"
                    elif "NEG" in label:
                        label = "NEGATIVE"
                    else:
                        label = "NEUTRAL"
                return {"label": label, "confidence": round(min(max(confidence, 0.0), 1.0), 4)}
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

    # Fallback: keyword scan on the raw response
    upper = text.upper()
    if "POSITIVE" in upper:
        return {"label": "POSITIVE", "confidence": 0.7}
    elif "NEGATIVE" in upper:
        return {"label": "NEGATIVE", "confidence": 0.7}
    else:
        return {"label": "NEUTRAL", "confidence": 0.5}


def run_ollama_sentiment(model: str, text: str) -> dict:
    """
    Run sentiment analysis on *text* using an Ollama model.

    Wraps the call in CodeCarbon (if available) to measure real energy
    consumption, making the result directly comparable with the 3-step
    pipeline's carbon metrics.

    Returns
    -------
    dict with keys:
        status       – "success" | "error"
        label        – "POSITIVE" | "NEGATIVE" | "NEUTRAL"
        confidence   – float 0-1
        latency      – seconds
        model        – model name
        energy_kwh   – measured or estimated energy
        co2_kg       – measured or estimated CO₂
        raw_response – the raw text from the model
    """
    prompt = f"{_SENTIMENT_SYSTEM}\n\nText: {text}"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    def _call():
        return requests.post(OLLAMA_API_GENERATE, json=payload, timeout=120)

    start_time = time.time()
    try:
        response, tracker_data = measure_with_tracker(_call)
        latency = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            raw_text = result.get("response", "")
            parsed = _parse_sentiment_response(raw_text)

            energy_kwh = tracker_data.get("energy_kwh", 0.0)
            co2_kg = tracker_data.get("co2_kg", 0.0)

            # Fallback estimate if CodeCarbon not available
            if tracker_data.get("source") == "empirical":
                # Estimate: ~65W TDP average CPU × inference seconds
                energy_kwh = (65 * latency) / (1000 * 3600)  # watts × seconds → kWh
                co2_kg = energy_kwh * 0.475  # IEA world average

            return {
                "status": "success",
                "label": parsed["label"],
                "confidence": parsed["confidence"],
                "latency": round(latency, 2),
                "model": model,
                "energy_kwh": energy_kwh,
                "co2_kg": co2_kg,
                "raw_response": raw_text,
            }
        else:
            return {
                "status": "error",
                "message": f"API Error: {response.status_code}",
                "model": model,
            }
    except requests.Timeout:
        return {"status": "error", "message": "Request timed out (120s)", "model": model}
    except Exception as e:
        return {"status": "error", "message": str(e), "model": model}
