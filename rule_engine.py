"""
rule_engine.py — Stage 1: Ultra-low-energy rule-based sentiment classifier.

This module uses handcrafted keyword lists to detect *obvious* sentiment
without loading any AI model.  When it fires it returns confidence = 1.0
so the pipeline can skip all downstream stages.

Optimisations
-------------
- ``frozenset`` for immutable, hash-optimised lookups.
- Punctuation is stripped before tokenising so ``"amazing!"`` matches.
- Negation detection: ``"not great"`` is suppressed to avoid false positives.

Energy cost: negligible (pure Python set operations).
"""

from __future__ import annotations
import re

# ── Compiled regex: strip anything that isn't a letter, digit, or space ────────
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# ── Negation window: if any of these appear right before a keyword, suppress ──
_NEGATORS: frozenset[str] = frozenset({
    "not", "no", "never", "neither", "nor", "hardly", "barely",
    "scarcely", "don't", "dont", "doesn't", "doesnt", "didn't",
    "didnt", "wasn't", "wasnt", "isn't", "isnt", "aren't", "arent",
    "won't", "wont", "wouldn't", "wouldnt", "shouldn't", "shouldnt",
    "couldn't", "couldnt", "cannot", "can't", "cant",
})

# ── Keyword lexicons (frozenset for immutability + fast lookup) ────────────────
POSITIVE_KEYWORDS: frozenset[str] = frozenset({
    "great", "excellent", "amazing", "fantastic", "wonderful", "outstanding",
    "superb", "brilliant", "awesome", "love", "loved", "perfect", "best",
    "incredible", "exceptional", "delightful", "impressive", "magnificent",
    "splendid", "terrific", "phenomenal", "remarkable", "beautiful", "happy",
    "joyful", "pleased", "satisfied", "thrilled", "excited", "glad",
})

NEGATIVE_KEYWORDS: frozenset[str] = frozenset({
    "bad", "terrible", "worst", "horrible", "awful", "dreadful", "disgusting",
    "hate", "hated", "pathetic", "useless", "poor", "disappointing",
    "disappointed", "waste", "rubbish", "trash", "garbage", "atrocious",
    "abysmal", "disastrous", "inferior", "defective", "broken", "failed",
    "failure", "frustrated", "angry", "furious", "miserable", "unacceptable",
})

NEUTRAL_KEYWORDS: frozenset[str] = frozenset({
    "okay", "ok", "fine", "average", "normal", "standard", "typical", "usual",
    "ordinary", "regular", "acceptable", "adequate", "sufficient", "decent",
    "fair", "moderate", "reasonable", "neutral", "balanced", "mixed", "so-so",
    "alright", "nothing", "whatever", "meh", "bland", "plain", "basic",
})


def _strip_negated(tokens: list[str], hits: frozenset[str]) -> set[str]:
    """Remove keywords that are immediately preceded by a negator."""
    clean: set[str] = set()
    for i, tok in enumerate(tokens):
        if tok in hits:
            # Check the preceding token for a negator
            if i > 0 and tokens[i - 1] in _NEGATORS:
                continue          # negated — suppress
            clean.add(tok)
    return clean


def detect_sentiment(text: str) -> dict | None:
    """
    Scan *text* for obvious positive / negative / neutral keywords.

    Returns
    -------
    dict  with keys ``label``, ``confidence``, ``stage`` if a keyword fires.
    None  if the text is ambiguous — pipeline should proceed to Stage 2.
    """
    # Normalise: lowercase, strip punctuation, split
    cleaned = _PUNCT_RE.sub("", text.lower())
    token_list = cleaned.split()           # ordered list (for negation check)
    tokens     = frozenset(token_list)      # fast membership tests

    # Raw keyword intersections
    raw_pos = tokens & POSITIVE_KEYWORDS
    raw_neg = tokens & NEGATIVE_KEYWORDS
    raw_neu = tokens & NEUTRAL_KEYWORDS

    # Suppress negated hits
    pos_hits = _strip_negated(token_list, raw_pos) if raw_pos else set()
    neg_hits = _strip_negated(token_list, raw_neg) if raw_neg else set()
    neu_hits = _strip_negated(token_list, raw_neu) if raw_neu else set()

    # Multiple sentiment types detected → ambiguous; let the model decide
    hit_count = sum([bool(pos_hits), bool(neg_hits), bool(neu_hits)])
    if hit_count > 1:
        return None

    if pos_hits:
        return {
            "label":      "POSITIVE",
            "confidence": 1.0,
            "stage":      "Rule Engine",
            "matched":    sorted(pos_hits),
        }

    if neg_hits:
        return {
            "label":      "NEGATIVE",
            "confidence": 1.0,
            "stage":      "Rule Engine",
            "matched":    sorted(neg_hits),
        }

    if neu_hits:
        return {
            "label":      "NEUTRAL",
            "confidence": 1.0,
            "stage":      "Rule Engine",
            "matched":    sorted(neu_hits),
        }

    return None  # No clear signal — escalate to AI
