"""
Phase 9 — Explainability Layer
================================

For every file decision produced by Phase 7 (importance scoring) and Phase 8
(action assignment), this phase generates:

  1. SHAP-based feature attribution   — which of the 9 signals drove the score
  2. Plain-language explanation        — LLM-generated sentence (Groq or local)
  3. Confidence indicator              — model certainty for the assigned label
  4. Counterfactual tip                — "do X to move this file to a better category"
  5. Audit record                      — full JSON stored in the DB for the UI

Pipeline
─────────
  1. Load file_scores  (ph7)  +  decision labels
  2. Reconstruct feature matrix → compute SHAP values
  3. Build ranked signal attribution per file
  4. Call LLM for natural-language explanation  (Groq if key available, else
     rule-based template fallback — no silent failures)
  5. Persist to  file_explanations  table
  6. Write  phase9_report.json

Reads from  : file_scores  (ph7)
Writes to   : file_explanations  (PostgreSQL)
              phase9_report.json

Install:
    pip install shap requests psycopg2-binary pandas numpy scikit-learn joblib python-dotenv
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

DB_URL         = os.getenv("DB_URL")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"
REPORT_PATH    = os.getenv("PH9_REPORT_PATH", "phase9_report.json")
MODEL_PATH     = os.getenv("MODEL_PATH", os.getenv("PH7_MODEL_PATH", "phase7_model.joblib"))
BATCH_SIZE     = int(os.getenv("PH9_BATCH_SIZE", 50))      # LLM calls per flush
SHAP_MAX_FILES = int(os.getenv("PH9_SHAP_MAX", 50_000))    # cap for memory
PH9_DELETE_LLM_CONF_THRESHOLD = float(os.getenv("PH9_DELETE_LLM_CONF_THRESHOLD", 0.55))
PH9_LLM_SAMPLE_PERCENT = int(os.getenv("PH9_LLM_SAMPLE_PERCENT", 100))
SIGNAL_COLS = [
    "s_content_richness",
    "s_recency",
    "s_type_importance",
    "s_uniqueness",
    "s_extraction_quality",
    "s_content_depth",
    "s_cluster_density",
    "s_llm_quality",
    "s_semantic_proximity",
]

SIGNAL_LABELS = {
    "s_content_richness":   "Content richness",
    "s_recency":            "Recency",
    "s_type_importance":    "File type importance",
    "s_uniqueness":         "Uniqueness (not a duplicate)",
    "s_extraction_quality": "Extraction quality",
    "s_content_depth":      "Content depth (chunk count)",
    "s_cluster_density":    "Semantic cluster density",
    "s_llm_quality":        "AI content quality assessment",
    "s_semantic_proximity": "Semantic proximity to top files",
}

logging.basicConfig(
    filename="phase9_errors.log",
    level=logging.WARNING,
    format="%(asctime)s — %(levelname)s — %(message)s",
)

# ── stdout safety ─────────────────────────────────────────────────────────────

def _safe_print(*args, sep=" ", end="\n", file=None, flush=False):
    if file is None:
        file = sys.stdout
    text = sep.join(str(a) for a in args) + end
    try:
        file.write(text)
    except Exception:
        sys.__stdout__.buffer.write(text.encode("utf-8", errors="backslashreplace"))
    if flush:
        try:
            file.flush()
        except Exception:
            pass

print = _safe_print

# ── DB ────────────────────────────────────────────────────────────────────────

def _conn():
    if not DB_URL:
        raise RuntimeError(
            "DB_URL not set. Please set DB_URL in your .env or environment before running Phase 9."
        )
    return psycopg2.connect(DB_URL)


def _create_tables(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS file_explanations (
                id                  SERIAL PRIMARY KEY,
                file_id             INTEGER UNIQUE REFERENCES files(id) ON DELETE CASCADE,
                path                TEXT,
                name                TEXT,
                label               TEXT,
                importance_score    REAL,

                -- Top-3 contributing signals (name + direction + magnitude)
                top_signal_1        TEXT,
                top_signal_1_value  REAL,
                top_signal_1_shap   REAL,

                top_signal_2        TEXT,
                top_signal_2_value  REAL,
                top_signal_2_shap   REAL,

                top_signal_3        TEXT,
                top_signal_3_value  REAL,
                top_signal_3_shap   REAL,

                -- Full SHAP JSON  {"signal": shap_value, ...}
                shap_json           JSONB,

                -- LLM-generated plain-language explanation
                explanation_text    TEXT,

                -- Counterfactual: what would change the decision
                counterfactual_tip  TEXT,

                -- Model confidence (max class probability)
                confidence          REAL,

                -- Generation method: "llm" | "template"
                explanation_method  TEXT,

                explained_at        TIMESTAMP DEFAULT NOW()
            );
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_fe_label ON file_explanations(label);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_fe_score ON file_explanations(importance_score DESC);"
        )
    conn.commit()

# ── Step 1 — Load data ────────────────────────────────────────────────────────

def _load_scores(conn) -> pd.DataFrame:
    print("  [1/5] Loading file_scores from ph7 …")
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT fs.file_id, fs.path, fs.name, fs.ext,
                   fs.label, fs.importance_score,
                   {', '.join(f'fs.{s}' for s in SIGNAL_COLS)}
            FROM file_scores fs
            LEFT JOIN file_explanations fe ON fs.file_id = fe.file_id
            WHERE fe.file_id IS NULL           -- only un-explained files
            ORDER BY fs.importance_score DESC
        """)
        rows = cur.fetchall()

    df = pd.DataFrame(rows)
    print(f"      → {len(df):,} files to explain")
    return df

# ── Step 2 — SHAP attributions ────────────────────────────────────────────────

def _normalized_proxy(X: np.ndarray) -> np.ndarray:
    """Min-max normalize raw signal values to [0,1] as an attribution proxy
    when real SHAP values aren't available for a row."""
    mins  = X.min(axis=0, keepdims=True)
    maxs  = X.max(axis=0, keepdims=True)
    denom = np.where(maxs - mins > 0, maxs - mins, 1.0)
    return (X - mins) / denom


def _compute_shap(df: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values using the saved RF model from ph7.
    Returns a (n_files, n_signals) array of mean absolute SHAP values
    across all output classes.

    Falls back to raw feature values normalised to [0,1] if the model
    or shap library is unavailable, or for any individual file beyond
    SHAP_MAX_FILES (rather than fabricating attribution by copying
    another file's SHAP row).
    """
    X = df[SIGNAL_COLS].fillna(0.0).values.astype(np.float32)

    try:
        import shap
        import joblib

        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"RF model not found at {MODEL_PATH}")

        rf = joblib.load(MODEL_PATH)

        # cap for memory
        X_sample = X[:SHAP_MAX_FILES]

        print(f"      Computing SHAP for {len(X_sample):,} files (TreeExplainer) …")
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_sample)

        # FIX (CRITICAL): shap.TreeExplainer.shap_values() for a multi-class
        # model returns different shapes depending on the installed shap
        # version:
        #   - older shap (<0.45ish): list[n_classes] of (n_samples, n_features)
        #   - current shap (0.45+, e.g. 0.52 as of this fix): a single
        #     ndarray of shape (n_samples, n_features, n_classes)
        # The original code only handled the list case; on any reasonably
        # current `pip install shap`, the ndarray branch produced a 3D
        # array that then broke np.vstack against the 2D padding array a
        # few lines down, raising an exception on every single call and
        # silently falling back to the crude min-max proxy below - meaning
        # genuine SHAP attribution never actually ran in practice. Both
        # shapes are now normalized to (n_samples, n_features) by averaging
        # absolute contributions across the class axis, whichever axis
        # that turns out to be.
        if isinstance(shap_values, list):
            abs_shap = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # (n_samples, n_features, n_classes) -> average over last axis
            abs_shap = np.mean(np.abs(shap_values), axis=2)
        else:
            abs_shap = np.abs(shap_values)

        if abs_shap.shape != X_sample.shape:
            raise ValueError(
                f"Unexpected SHAP output shape {abs_shap.shape}, "
                f"expected {X_sample.shape}"
            )

        # FIX (HIGH): files beyond SHAP_MAX_FILES previously got the LAST
        # computed file's SHAP row tiled across all of them — i.e. every
        # overflow file was shown someone else's attribution as if it were
        # its own. That's worse than no explanation: it's a wrong one,
        # which matters a lot in an audit/governance feature. Overflow
        # files now get their own genuine (proxy) attribution computed
        # from their own feature values instead of a copy of another
        # file's real SHAP values.
        if len(X) > SHAP_MAX_FILES:
            overflow_proxy = _normalized_proxy(X[SHAP_MAX_FILES:])
            abs_shap = np.vstack([abs_shap, overflow_proxy])
            print(
                f"      ⚠  {len(X) - SHAP_MAX_FILES:,} files exceeded "
                f"SHAP_MAX_FILES={SHAP_MAX_FILES:,} — using proxy "
                f"attribution for those (not fabricated SHAP)."
            )

        print(f"      → SHAP done")
        return abs_shap

    except ImportError:
        print("      ⚠  shap not installed — using raw feature values as proxy")
    except Exception as e:
        logging.warning(f"SHAP computation failed: {e}")
        print(f"      ⚠  SHAP failed ({e}) — using raw feature values as proxy")

    # Fallback: normalise raw signals to [0,1] as attribution proxy
    return _normalized_proxy(X)


def _top_signals(shap_row: np.ndarray, feature_values: np.ndarray, n: int = 3):
    """Return top-n signals sorted by SHAP magnitude."""
    order = np.argsort(shap_row)[::-1]
    result = []
    for idx in order[:n]:
        result.append({
            "signal": SIGNAL_COLS[idx],
            "label":  SIGNAL_LABELS[SIGNAL_COLS[idx]],
            "value":  float(feature_values[idx]),
            "shap":   float(shap_row[idx]),
        })
    return result


def _weak_signals(shap_row: np.ndarray, feature_values: np.ndarray, n: int = 4):
    """Return the least influential signals sorted by SHAP magnitude."""
    order = np.argsort(shap_row)
    result = []
    for idx in order[:n]:
        result.append({
            "signal": SIGNAL_COLS[idx],
            "label":  SIGNAL_LABELS[SIGNAL_COLS[idx]],
            "value":  float(feature_values[idx]),
            "shap":   float(shap_row[idx]),
        })
    return result


def _use_llm(row: dict, confidence: float) -> bool:
    """Determine whether this row should use the LLM for the explanation."""
    import random

    label = row.get("label", "REVIEW")
    if label == "KEEP":
        return random.random() * 100 < PH9_LLM_SAMPLE_PERCENT
    if label == "DELETE_CANDIDATE":
        return confidence <= PH9_DELETE_LLM_CONF_THRESHOLD
    return False

# ── Step 3 — Model confidence ─────────────────────────────────────────────────

def _compute_confidence(df: pd.DataFrame) -> np.ndarray:
    """Max class probability from the RF model, or 0.5 if unavailable."""
    try:
        import joblib
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError()
        rf = joblib.load(MODEL_PATH)
        X  = df[SIGNAL_COLS].fillna(0.0).values.astype(np.float32)
        proba = rf.predict_proba(X)
        return proba.max(axis=1)
    except Exception as e:
        logging.warning(f"Confidence computation failed: {e}")
        return np.full(len(df), 0.5)

# ── Step 4 — Explanation generation ──────────────────────────────────────────

_SYSTEM_PROMPT = """You are an enterprise data governance AI. Your job is to explain 
file importance decisions in clear, professional language for non-technical managers.

Given a file's metadata and the top signals that drove its score, generate:
1. A 2-3 sentence explanation of WHY this file received its label and score.
2. One concrete tip for what could change the decision (counterfactual).

Return ONLY a JSON object in this exact format (no markdown, no extra text):
{
  "explanation": "2-3 sentence explanation here.",
  "tip": "One concrete actionable tip here."
}"""


def _build_llm_prompt(row: dict, top3: list, weak_signals: list) -> str:
    top_text = "\n".join(
        f"  - {s['label']}: {s['value']:.2f} (impact: {s['shap']:.3f})"
        for s in top3
    )
    weak_text = "\n".join(
        f"  - {s['label']}: {s['value']:.2f} (low impact: {s['shap']:.3f})"
        for s in weak_signals
    )
    return (
        f"File: {row.get('name', 'unknown')}\n"
        f"Type: {row.get('ext', 'unknown')}\n"
        f"Score: {row.get('importance_score', 0):.1f}/100\n"
        f"Decision: {row.get('label', 'REVIEW')}\n"
        f"Top contributing signals:\n{top_text}\n"
        f"Weakest signals:\n{weak_text}"
    )


def _call_groq(prompt: str) -> Optional[dict]:
    try:
        import requests
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       GROQ_MODEL,
                "messages":    [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                "max_tokens":  300,
                "temperature": 0.2,
            },
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip any markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        logging.warning(f"Groq call failed: {e}")
        return None


def _template_explanation(row: dict, top3: list, weak_signals: list) -> dict:
    """Rule-based fallback when LLM is unavailable."""
    label = row.get("label", "REVIEW")
    score = row.get("importance_score", 0.0)
    name  = row.get("name", "This file")
    ext   = (row.get("ext") or "").upper().lstrip(".")

    top_signal  = top3[0] if top3 else None
    top_label   = top_signal["label"] if top_signal else "multiple factors"
    top_val     = top_signal["value"] if top_signal else 0.0

    weak_labels = [s["label"] for s in weak_signals]
    weakest_text = ", ".join(weak_labels) if weak_labels else "multiple weaker factors"

    def delete_tip():
        improvements = {
            "Content richness": "add more descriptive or relevant content",
            "Recency": "refresh or update the file so it becomes more current",
            "File type importance": "use a more appropriate format or context for your audience",
            "Uniqueness (not a duplicate)": "reduce duplication and make this file uniquely valuable",
            "Extraction quality": "improve extraction accuracy or source metadata",
            "Content depth (chunk count)": "expand the document with deeper coverage or supporting details",
            "Semantic cluster density": "connect it more clearly to high-value topics",
            "AI content quality assessment": "improve clarity and correctness of the content",
            "Semantic proximity to top files": "make it more aligned with top-priority documents",
        }
        actions = [improvements.get(label, "strengthen this signal") for label in weak_labels[:3]]
        action_text = ", ".join(actions)
        return (
            f"To move out of DELETE_CANDIDATE, focus on the weakest signals: {weakest_text}. "
            f"Specifically, {action_text}."
        )

    # Score tier labels
    if score >= 80:
        tier = "very high"
    elif score >= 50:
        tier = "moderate to high"
    elif score >= 20:
        tier = "low to moderate"
    else:
        tier = "very low"

    label_reasons = {
        "KEEP":             f"scored {score:.0f}/100 indicating {tier} importance to the organization",
        "ARCHIVE":          f"scored {score:.0f}/100, suggesting it has some value but lower priority for active use",
        "REVIEW":           f"scored {score:.0f}/100, which places it in the uncertain range requiring human judgment",
        "DELETE_CANDIDATE": f"scored {score:.0f}/100, indicating {tier} value with no strong retention signals",
    }

    reason = label_reasons.get(label, f"scored {score:.0f}/100")
    signal_note = (
        f"The most influential factor was {top_label.lower()} "
        f"(value: {top_val:.2f}), "
        f"which accounts for the largest share of its score."
    ) if top_signal else ""

    if label == "DELETE_CANDIDATE":
        explanation = (
            f"{name}{' (' + ext + ')' if ext else ''} {reason}. "
            f"The weakest signals were {weakest_text}, which held this file back. "
            f"This decision is grounded in SHAP-based attribution over the nine explainability signals."
        ).strip()
        tip = delete_tip()
    elif label == "ARCHIVE":
        explanation = (
            f"{name}{' (' + ext + ')' if ext else ''} {reason}. "
            f"The highest contributing signals were {top_label.lower()} and related importance factors. "
            f"Review whether updating the content or recency could raise its value."
        ).strip()
        tip = (
            "Accessing or updating this file would increase its recency score and may elevate it to KEEP."
        )
    elif label == "REVIEW":
        explanation = (
            f"{name}{' (' + ext + ')' if ext else ''} {reason}. "
            f"It has some useful content but also weaker signals such as {weakest_text}. "
            f"Human review is recommended to confirm whether it should be kept or archived."
        ).strip()
        tip = (
            "To move this file to KEEP, ensure it has rich content, is recently modified, "
            "and is not a duplicate of another file."
        )
    else:
        explanation = (
            f"{name}{' (' + ext + ')' if ext else ''} {reason}. "
            f"The most influential factor was {top_label.lower()} (value: {top_val:.2f})."
        ).strip()
        tip = (
            "Continue keeping this file accessible and up to date to maintain its high score."
            if label == "KEEP"
            else "Review this file manually to confirm the AI decision."
        )

    return {"explanation": explanation, "tip": tip}


def _generate_explanations(
    df: pd.DataFrame,
    shap_vals: np.ndarray,
    confidence: np.ndarray,
) -> list[dict]:
    """Generate explanation records for all files in df."""
    use_llm_key = bool(GROQ_API_KEY)
    method  = "llm" if use_llm_key else "template"

    print(
        f"  [4/5] Generating explanations "
        f"({'Hybrid LLM + templates' if use_llm_key else 'rule-based templates'}) …"
    )

    feature_matrix = df[SIGNAL_COLS].fillna(0.0).values.astype(np.float32)
    rows = df.to_dict(orient="records")
    records = []
    total   = len(rows)

    for i, row in enumerate(rows):
        feature_vals = feature_matrix[i]
        shap_row     = shap_vals[i]
        top3         = _top_signals(shap_row, feature_vals, n=3)
        weak         = _weak_signals(shap_row, feature_vals, n=4)
        conf         = float(confidence[i])

        if use_llm_key and _use_llm(row, conf):
            prompt = _build_llm_prompt(row, top3, weak)
            result = _call_groq(prompt)
            if result is None:
                result = _template_explanation(row, top3, weak)
                rec_method = "template"
            else:
                rec_method = "llm"
        else:
            result     = _template_explanation(row, top3, weak)
            rec_method = "template"

        shap_dict = {SIGNAL_COLS[j]: float(shap_row[j]) for j in range(len(SIGNAL_COLS))}

        records.append({
            "file_id":           int(row["file_id"]),
            "path":              row.get("path", ""),
            "name":              row.get("name", ""),
            "label":             row.get("label", ""),
            "importance_score":  float(row.get("importance_score", 0)),
            "top3":              top3,
            "shap_json":         shap_dict,
            "explanation_text":  result.get("explanation", ""),
            "counterfactual_tip":result.get("tip", ""),
            "confidence":        conf,
            "explanation_method":rec_method,
        })

        if (i + 1) % BATCH_SIZE == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            bar = "#" * int(pct // 5) + "-" * (20 - int(pct // 5))
            print(
                f"  [{bar}] {pct:5.1f}%  {i+1}/{total}",
                end="\r", flush=True,
            )

    print()
    return records

# ── Step 5 — Persist to DB ────────────────────────────────────────────────────

def _save_explanations(conn, records: list[dict]):
    print(f"  [5/5] Writing {len(records):,} explanations to file_explanations …")
    rows = []
    for r in records:
        t = r["top3"]
        s1 = t[0] if len(t) > 0 else {}
        s2 = t[1] if len(t) > 1 else {}
        s3 = t[2] if len(t) > 2 else {}
        rows.append((
            r["file_id"],
            r["path"],
            r["name"],
            r["label"],
            r["importance_score"],
            s1.get("label", ""),
            s1.get("value", 0.0),
            s1.get("shap",  0.0),
            s2.get("label", ""),
            s2.get("value", 0.0),
            s2.get("shap",  0.0),
            s3.get("label", ""),
            s3.get("value", 0.0),
            s3.get("shap",  0.0),
            json.dumps(r["shap_json"]),
            r["explanation_text"],
            r["counterfactual_tip"],
            r["confidence"],
            r["explanation_method"],
        ))

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO file_explanations (
                file_id, path, name, label, importance_score,
                top_signal_1, top_signal_1_value, top_signal_1_shap,
                top_signal_2, top_signal_2_value, top_signal_2_shap,
                top_signal_3, top_signal_3_value, top_signal_3_shap,
                shap_json, explanation_text, counterfactual_tip,
                confidence, explanation_method
            ) VALUES %s
            ON CONFLICT (file_id) DO UPDATE SET
                label               = EXCLUDED.label,
                importance_score    = EXCLUDED.importance_score,
                top_signal_1        = EXCLUDED.top_signal_1,
                top_signal_1_value  = EXCLUDED.top_signal_1_value,
                top_signal_1_shap   = EXCLUDED.top_signal_1_shap,
                top_signal_2        = EXCLUDED.top_signal_2,
                top_signal_2_value  = EXCLUDED.top_signal_2_value,
                top_signal_2_shap   = EXCLUDED.top_signal_2_shap,
                top_signal_3        = EXCLUDED.top_signal_3,
                top_signal_3_value  = EXCLUDED.top_signal_3_value,
                top_signal_3_shap   = EXCLUDED.top_signal_3_shap,
                shap_json           = EXCLUDED.shap_json::jsonb,
                explanation_text    = EXCLUDED.explanation_text,
                counterfactual_tip  = EXCLUDED.counterfactual_tip,
                confidence          = EXCLUDED.confidence,
                explanation_method  = EXCLUDED.explanation_method,
                explained_at        = NOW()
        """, rows)
    conn.commit()
    print(f"      → Done")

# ── Public helper: explain a single file on demand ────────────────────────────

def explain_file(file_id: int) -> dict:
    """
    Retrieve the stored explanation for a single file.
    Returns a dict suitable for rendering in a UI or API response.
    Raises ValueError if no explanation exists yet.
    """
    conn = _conn()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT * FROM file_explanations WHERE file_id = %s", (file_id,)
        )
        row = cur.fetchone()
    conn.close()

    if row is None:
        raise ValueError(f"No explanation found for file_id={file_id}. Run phase 9 first.")

    return dict(row)


def explain_files_batch(file_ids: list[int]) -> list[dict]:
    """Retrieve stored explanations for a list of file IDs."""
    conn = _conn()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT * FROM file_explanations WHERE file_id = ANY(%s) ORDER BY importance_score DESC",
            (file_ids,),
        )
        rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ── Report ────────────────────────────────────────────────────────────────────

def _build_report(records: list[dict], elapsed: float) -> dict:
    if not records:
        return {"phase": 9, "elapsed_seconds": elapsed, "total": 0}

    df = pd.DataFrame(records)
    method_counts = df["explanation_method"].value_counts().to_dict()

    by_label = df.groupby("label").agg(
        count=("file_id", "count"),
        avg_confidence=("confidence", "mean"),
    ).reset_index().to_dict(orient="records")

    top_signal_freq: dict[str, int] = {}
    for rec in records:
        if rec["top3"]:
            sig = rec["top3"][0]["signal"]
            top_signal_freq[sig] = top_signal_freq.get(sig, 0) + 1

    sample_explanations = []
    for label in ["KEEP", "DELETE_CANDIDATE", "REVIEW", "ARCHIVE"]:
        subset = [r for r in records if r["label"] == label]
        if subset:
            pick = subset[0]
            sample_explanations.append({
                "label":       label,
                "name":        pick["name"],
                "score":       pick["importance_score"],
                "explanation": pick["explanation_text"],
                "tip":         pick["counterfactual_tip"],
            })

    return {
        "phase":                  9,
        "elapsed_seconds":        round(elapsed, 2),
        "total_files_explained":  len(records),
        "explanation_methods":    method_counts,
        "label_breakdown":        by_label,
        "most_common_top_signal": sorted(
            top_signal_freq.items(), key=lambda x: -x[1]
        )[:5],
        "avg_confidence":         round(float(df["confidence"].mean()), 3),
        "sample_explanations":    sample_explanations,
    }

# ── Entry point ───────────────────────────────────────────────────────────────

def run_phase9() -> dict:
    t0 = time.time()
    print("\n" + "=" * 62)
    print("  Phase 9 — Explainability Layer")
    print("=" * 62 + "\n")

    if GROQ_API_KEY:
        print(f"  LLM backend : Groq ({GROQ_MODEL})")
    else:
        print("  LLM backend : rule-based templates  (set GROQ_API_KEY for LLM)")

    if Path(MODEL_PATH).exists():
        print(f"  RF model    : {MODEL_PATH}")
    else:
        print(f"  RF model    : NOT FOUND at {MODEL_PATH} — SHAP will use proxy")

    conn = _conn()
    _create_tables(conn)

    df = _load_scores(conn)
    if df.empty:
        print("  Nothing to explain — all files already have explanations.")
        conn.close()
        elapsed = round(time.time() - t0, 2)
        summary = {"phase": 9, "elapsed_seconds": elapsed, "total_files_explained": 0}
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    print(f"\n  [2/5] Computing SHAP attributions …")
    shap_vals  = _compute_shap(df)

    print(f"  [3/5] Computing model confidence …")
    confidence = _compute_confidence(df)

    records = _generate_explanations(df, shap_vals, confidence)
    _save_explanations(conn, records)
    conn.close()

    elapsed = round(time.time() - t0, 2)
    report  = _build_report(records, elapsed)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print(f"  Files explained       : {report['total_files_explained']:,}")
    print(f"  Avg confidence        : {report.get('avg_confidence', 0):.1%}")
    print(f"  Explanation methods   : {report['explanation_methods']}")
    print(f"\n  Label breakdown:")
    for lb in report.get("label_breakdown", []):
        print(
            f"    {lb['label']:<20}  {lb['count']:>6,} files   "
            f"avg confidence {lb['avg_confidence']:.1%}"
        )
    print(f"\n  Most common top signal:")
    for sig, cnt in report.get("most_common_top_signal", []):
        bar = "█" * int(cnt / max(1, report["total_files_explained"]) * 40)
        print(f"    {SIGNAL_LABELS.get(sig, sig):<40}  {cnt:,}  {bar}")
    print(f"\n  Sample explanations:")
    for ex in report.get("sample_explanations", []):
        print(f"\n    [{ex['label']}]  {ex['name']}  (score: {ex['score']:.1f})")
        print(f"    Explanation : {ex['explanation'][:200]}…" if len(ex['explanation']) > 200 else f"    Explanation : {ex['explanation']}")
        print(f"    Tip         : {ex['tip']}")
    print(f"\n  Elapsed       : {elapsed}s")
    print(f"  Report        : {REPORT_PATH}")
    print(f"{'=' * 62}")
    print("\nPhase 9 complete.")

    return report


if __name__ == "__main__":
    run_phase9()
