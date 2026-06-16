"""
Phase 7: File Importance Scoring
=================================
Hybrid traditional + AI scoring pipeline that assigns every file an
importance_score (0–100) and a label (KEEP / ARCHIVE / REVIEW / DELETE_CANDIDATE).

Architecture mirrors Google Drive Gemini suggestions / Dropbox Dash Dash smart search.
"""

import os
import json
import time
import logging
import warnings
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
import joblib

warnings.filterwarnings("ignore")
load_dotenv()

# ─────────────────────────────────────────────
# CONFIG (all overridable via .env)
# ─────────────────────────────────────────────
DB_URL               = os.getenv("DATABASE_URL", "postgresql://localhost/docdb")
FAISS_PATH           = os.getenv("FAISS_PATH", "embeddings/chunk_embeddings.faiss")
CHUNK_INDEX_PATH     = os.getenv("CHUNK_INDEX_PATH", "embeddings/chunk_index.json")
MODEL_PATH           = os.getenv("MODEL_PATH", "phase7_rf_model.joblib")
REPORT_PATH          = os.getenv("REPORT_PATH", "phase7_report.json")
ERROR_LOG            = os.getenv("ERROR_LOG", "phase7_errors.log")

MAX_STALENESS_DAYS   = float(os.getenv("MAX_STALENESS_DAYS", "1825"))   # 5 years
DBSCAN_EPS           = float(os.getenv("DBSCAN_EPS", "0.3"))
DBSCAN_MIN_SAMPLES   = int(os.getenv("DBSCAN_MIN_SAMPLES", "3"))
LLM_MAX_WORDS        = int(os.getenv("LLM_MAX_WORDS", "400"))
LLM_BATCH_SIZE       = int(os.getenv("LLM_BATCH_SIZE", "100"))
LLM_TIMEOUT          = float(os.getenv("LLM_TIMEOUT", "15"))
KEEP_CENTROID_TOP_N  = int(os.getenv("KEEP_CENTROID_TOP_N", "200"))
GROQ_API_KEY         = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL           = os.getenv("GROQ_MODEL", "llama3-8b-8192")

# Bootstrap weights (S1–S8 only; S9 not yet available in boot phase)
BOOT_WEIGHTS = np.array([0.10, 0.12, 0.10, 0.15, 0.08, 0.10, 0.18, 0.17], dtype=np.float32)

# RF scoring weights (dot product with class midpoints)
CLASS_MIDPOINTS = np.array([10, 35, 65, 90], dtype=np.float32)

# Label thresholds (bootstrap)
LABEL_THRESHOLDS = {"KEEP": 80, "ARCHIVE": 50, "REVIEW": 20}

IDX_TO_LABEL = {0: "DELETE_CANDIDATE", 1: "REVIEW", 2: "ARCHIVE", 3: "KEEP"}
LABEL_TO_IDX = {v: k for k, v in IDX_TO_LABEL.items()}

# Extension weights
EXT_WEIGHTS = {
    "pdf": 1.0, "docx": 1.0, "doc": 1.0, "txt": 1.0,
    "xlsx": 0.95, "xls": 0.95, "csv": 0.95,
    "pptx": 0.85, "ppt": 0.85,
    "py": 0.80, "js": 0.80, "ts": 0.80, "java": 0.80, "cpp": 0.80,
    "c": 0.80, "go": 0.80, "rs": 0.80, "sql": 0.80, "sh": 0.80,
    "json": 0.75, "xml": 0.75, "yaml": 0.75, "yml": 0.75,
    "png": 0.50, "jpg": 0.50, "jpeg": 0.50, "gif": 0.50,
    "zip": 0.40, "tar": 0.40, "gz": 0.40, "7z": 0.40,
    "tmp": 0.00, "bak": 0.00, "log": 0.00,
}


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(ERROR_LOG),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("phase7")


def _safe(fn, default, label=""):
    """Execute fn(); return default + log on any exception."""
    try:
        return fn()
    except Exception as e:
        log.warning(f"[SAFE:{label}] {e}")
        return default


# ═══════════════════════════════════════════════════════════════
# STEP 1 & 2: DATA FETCH
# ═══════════════════════════════════════════════════════════════

def fetch_files(conn) -> pd.DataFrame:
    """Fetch all files + extracted_content + aggregated chunk stats."""
    log.info("► STEP 1: Fetching files from PostgreSQL…")
    sql = """
    SELECT
        f.id                        AS file_id,
        f.path,
        f.name,
        f.stem,
        f.extension                 AS ext,
        f.folder,
        f.depth,
        f.size_bytes,
        f.created_time,
        f.modified_time,
        f.access_time,
        f.hash,
        COALESCE(ec.extraction_status, 'UNKNOWN')  AS extraction_status,
        COALESCE(ec.ocr_applied, FALSE)             AS ocr_applied,
        COALESCE(ec.word_count, 0)                  AS word_count,
        COALESCE(ec.language_hint, '')              AS language_hint,
        COALESCE(ch.chunk_total, 0)                 AS chunk_total,
        COALESCE(ch.clean_chunks, 0)                AS clean_chunks
    FROM files f
    LEFT JOIN extracted_content ec ON ec.file_id = f.id
    LEFT JOIN (
        SELECT file_id,
               MAX(chunk_total)                                    AS chunk_total,
               COUNT(*) FILTER (WHERE clean_status = 'SUCCESS')   AS clean_chunks
        FROM chunks
        GROUP BY file_id
    ) ch ON ch.file_id = f.id
    ORDER BY f.id
    """
    df = pd.read_sql(sql, conn)
    log.info(f"  Loaded {len(df):,} files.")
    return df


def fetch_duplicate_paths(conn) -> set:
    """Return set of paths that are duplicates or redundant (Phase 6)."""
    log.info("► STEP 2: Fetching duplicate/redundancy paths…")
    dup_paths: set = set()

    for table, cols in [
        ("duplicate_files", ("path_1", "path_2")),
        ("file_redundancy", ("path_1", "path_2")),
    ]:
        try:
            cur = conn.cursor()
            cur.execute(f"SELECT {cols[0]}, {cols[1]} FROM {table}")
            for r in cur.fetchall():
                dup_paths.update(r)
            cur.close()
            log.info(f"  {table}: +{len(dup_paths)} paths so far.")
        except Exception as e:
            log.warning(f"  Table {table} missing or error: {e}")
            conn.rollback()

    log.info(f"  Total duplicate paths: {len(dup_paths)}")
    return dup_paths


# ═══════════════════════════════════════════════════════════════
# STEP 3: FAISS EMBEDDINGS
# ═══════════════════════════════════════════════════════════════

def load_embeddings(df: pd.DataFrame):
    """
    Returns embeddings dict {file_id: np.ndarray} by averaging chunk vectors.
    Returns empty dict if FAISS unavailable.
    """
    log.info("► STEP 3: Loading FAISS embeddings…")
    try:
        import faiss  # type: ignore

        index = faiss.read_index(FAISS_PATH)
        dim = index.d
        n_total = index.ntotal

        with open(CHUNK_INDEX_PATH) as f:
            raw_index = json.load(f)

        # Support both list-of-dicts and list-of-chunk_id-strings
        if raw_index and isinstance(raw_index[0], dict):
            chunk_meta = raw_index  # [{file_id, ...}, ...]
        else:
            # plain list of chunk_ids — build synthetic meta
            chunk_meta = [{"chunk_id": i, "file_id": None} for i in raw_index]

        if len(chunk_meta) != n_total:
            log.warning(f"  chunk_index length {len(chunk_meta)} ≠ FAISS ntotal {n_total}; truncating.")
            chunk_meta = chunk_meta[:n_total]

        # Reconstruct all vectors
        all_vecs = np.zeros((n_total, dim), dtype=np.float32)
        index.reconstruct_n(0, n_total, all_vecs)

        # Group by file_id
        from collections import defaultdict
        fid_vecs = defaultdict(list)
        for i, meta in enumerate(chunk_meta):
            fid = meta.get("file_id") if isinstance(meta, dict) else None
            if fid is not None:
                fid_vecs[int(fid)].append(all_vecs[i])

        embeddings = {fid: np.mean(vecs, axis=0) for fid, vecs in fid_vecs.items()}
        log.info(f"  Embeddings loaded for {len(embeddings):,} files. Dim={dim}.")
        return embeddings

    except ImportError:
        log.warning("  faiss not installed — S7/S9 will be 0.")
    except FileNotFoundError as e:
        log.warning(f"  FAISS file missing: {e} — S7/S9 will be 0.")
    except Exception as e:
        log.warning(f"  Embedding load error: {e}")

    return {}


# ═══════════════════════════════════════════════════════════════
# STEP 4: SIGNALS S1–S6 (vectorized)
# ═══════════════════════════════════════════════════════════════

def compute_s1_to_s6(df: pd.DataFrame, dup_paths: set) -> pd.DataFrame:
    """Compute signals S1–S6 as float32 columns on df."""
    log.info("► STEP 4: Computing S1–S6…")
    now = datetime.now(timezone.utc)

    # S1 — content richness
    df["s_content_richness"] = (
        np.log1p(df["word_count"].fillna(0).clip(lower=0)) /
        np.log1p(20_000)
    ).clip(0, 1).astype(np.float32)

    # S2 — recency
    def best_ts(row):
        candidates = []
        for col in ("modified_time", "access_time"):
            v = row.get(col)
            if pd.notna(v):
                ts = pd.Timestamp(v)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                candidates.append(ts)
        return max(candidates) if candidates else None

    df["_best_ts"] = df.apply(best_ts, axis=1)
    df["_days_since"] = df["_best_ts"].apply(
        lambda ts: (now - ts).days if ts else MAX_STALENESS_DAYS
    )
    df["s_recency"] = (
        1 - (df["_days_since"] / MAX_STALENESS_DAYS)
    ).clip(0, 1).astype(np.float32)

    # S3 — type importance
    df["s_type_importance"] = df["ext"].str.lower().map(
        lambda e: EXT_WEIGHTS.get(str(e).lstrip("."), 0.30)
    ).fillna(0.30).astype(np.float32)

    # S4 — uniqueness
    df["s_uniqueness"] = (~df["path"].isin(dup_paths)).astype(np.float32)

    # S5 — extraction quality
    status_map = {
        "SUCCESS": 1.0, "CLEAN": 1.0,
        "OCR": 0.6, "OCR_ASSISTED": 0.6,
        "PARTIAL": 0.3,
        "FAILED": 0.0, "ERROR": 0.0,
    }
    df["s_extraction_quality"] = df["extraction_status"].str.upper().map(
        lambda s: status_map.get(s, 0.5)
    ).fillna(0.5).astype(np.float32)

    # S6 — content depth (successful clean chunks)
    df["s_content_depth"] = (
        np.log1p(df["clean_chunks"].fillna(0).clip(lower=0)) /
        np.log1p(20)
    ).clip(0, 1).astype(np.float32)

    log.info("  S1–S6 computed.")
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 5: S7 — DBSCAN CLUSTER DENSITY
# ═══════════════════════════════════════════════════════════════

def compute_s7_cluster_density(df: pd.DataFrame, embeddings: dict) -> pd.DataFrame:
    """Compute S7 via DBSCAN on file embeddings."""
    log.info("► STEP 5: Computing S7 (cluster density via DBSCAN)…")

    fids = df["file_id"].tolist()
    vecs = [embeddings.get(fid) for fid in fids]
    has_emb = [v is not None for v in vecs]

    if sum(has_emb) < DBSCAN_MIN_SAMPLES:
        log.warning("  Not enough embeddings for DBSCAN — S7=0.")
        df["s_cluster_density"] = np.float32(0.0)
        return df

    emb_matrix = np.vstack([v for v in vecs if v is not None]).astype(np.float32)
    emb_normed = normalize(emb_matrix, norm="l2")

    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="cosine", n_jobs=-1)
    labels = db.fit_predict(emb_normed)

    from collections import Counter
    label_counts = Counter(labels)
    max_cluster = max((cnt for lbl, cnt in label_counts.items() if lbl != -1), default=1)

    # Map back to all files
    emb_idx = 0
    s7_vals = []
    for has in has_emb:
        if has:
            lbl = labels[emb_idx]
            emb_idx += 1
            if lbl == -1:
                s7_vals.append(0.0)
            else:
                s7_vals.append(label_counts[lbl] / max_cluster)
        else:
            s7_vals.append(0.0)

    df["s_cluster_density"] = np.array(s7_vals, dtype=np.float32)
    unique_clusters = len(set(labels) - {-1})
    noise_count = (labels == -1).sum()
    log.info(f"  DBSCAN: {unique_clusters} clusters, {noise_count} noise points.")
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 6: S8 — LLM QUALITY SCORING
# ═══════════════════════════════════════════════════════════════

def _heuristic_s8(row) -> float:
    return float(0.3 * row["s_content_richness"] +
                 0.4 * row["s_content_depth"] +
                 0.3 * row["s_extraction_quality"])


def _groq_rate_batch(texts: list[str]) -> list[float]:
    """Call Groq API for a batch of texts; return list of [0,1] scores."""
    try:
        from groq import Groq  # type: ignore
        client = Groq(api_key=GROQ_API_KEY)
        scores = []
        for text in texts:
            truncated = " ".join(text.split()[:LLM_MAX_WORDS])
            prompt = (
                "Rate the business relevance of this document excerpt on a scale "
                "from 1 (irrelevant/noise) to 5 (critical business document). "
                "Reply with ONLY the integer rating.\n\n"
                f"EXCERPT:\n{truncated}"
            )
            try:
                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    timeout=LLM_TIMEOUT,
                )
                rating_str = resp.choices[0].message.content.strip()
                rating = int("".join(c for c in rating_str if c.isdigit())[:1] or "3")
                rating = max(1, min(5, rating))
                scores.append((rating - 1) / 4)
            except Exception as e:
                log.warning(f"  Groq per-item error: {e}")
                scores.append(0.5)
        return scores
    except ImportError:
        log.warning("  groq package not installed.")
        raise
    except Exception as e:
        log.warning(f"  Groq batch error: {e}")
        raise


def compute_s8_llm_quality(df: pd.DataFrame, conn, dup_paths: set) -> pd.DataFrame:
    """Compute S8 — Groq LLM business relevance or heuristic fallback."""
    log.info("► STEP 6: Computing S8 (LLM quality)…")

    use_llm = bool(GROQ_API_KEY)
    if not use_llm:
        log.info("  No GROQ_API_KEY — using heuristic fallback for S8.")
        df["s_llm_quality"] = df.apply(_heuristic_s8, axis=1).astype(np.float32)
        return df

    # Build candidate map: file_id → first chunk text
    log.info("  Fetching representative chunk texts…")
    try:
        eligible_fids = df.loc[
            (~df["path"].isin(dup_paths)) & (df["word_count"] > 20), "file_id"
        ].tolist()

        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT ON (file_id) file_id, chunk_text
            FROM chunks
            WHERE file_id = ANY(%s) AND clean_status = 'SUCCESS'
            ORDER BY file_id, chunk_index
        """, (eligible_fids,))
        rows = cur.fetchall()
        cur.close()
        fid_to_text = {r[0]: r[1] for r in rows}
        log.info(f"  Got chunk texts for {len(fid_to_text):,} files.")
    except Exception as e:
        log.warning(f"  Could not fetch chunk texts: {e}; using heuristic.")
        conn.rollback()
        df["s_llm_quality"] = df.apply(_heuristic_s8, axis=1).astype(np.float32)
        return df

    # Batch LLM calls
    s8_map: dict = {}
    fids_to_score = [fid for fid in eligible_fids if fid in fid_to_text]
    for i in range(0, len(fids_to_score), LLM_BATCH_SIZE):
        batch_fids = fids_to_score[i:i + LLM_BATCH_SIZE]
        batch_texts = [fid_to_text[fid] for fid in batch_fids]
        pct = 100 * (i + len(batch_fids)) / max(len(fids_to_score), 1)
        log.info(f"  LLM batch {i // LLM_BATCH_SIZE + 1}: {len(batch_fids)} files [{pct:.0f}%]")
        try:
            scores = _groq_rate_batch(batch_texts)
        except Exception:
            scores = [0.5] * len(batch_fids)
        for fid, sc in zip(batch_fids, scores):
            s8_map[fid] = sc

    def get_s8(row):
        if row["file_id"] in s8_map:
            return s8_map[row["file_id"]]
        return _heuristic_s8(row)

    df["s_llm_quality"] = df.apply(get_s8, axis=1).astype(np.float32)
    log.info("  S8 complete.")
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 7: S9 — SEMANTIC PROXIMITY TO KEEP CENTROID
# ═══════════════════════════════════════════════════════════════

def compute_s9_semantic_proximity(df: pd.DataFrame, embeddings: dict) -> pd.DataFrame:
    """Compute S9 — cosine similarity to centroid of top-N KEEP bootstrap files."""
    log.info("► STEP 7: Computing S9 (semantic proximity)…")

    if not embeddings:
        log.warning("  No embeddings — S9=0.")
        df["s_semantic_proximity"] = np.float32(0.0)
        return df

    # Bootstrap: weighted sum S1–S8
    sig_cols = ["s_content_richness", "s_recency", "s_type_importance",
                "s_uniqueness", "s_extraction_quality", "s_content_depth",
                "s_cluster_density", "s_llm_quality"]
    sig_matrix = df[sig_cols].fillna(0).values.astype(np.float32)
    boot_scores = sig_matrix @ BOOT_WEIGHTS  # shape (N,)

    top_n_idx = np.argsort(boot_scores)[::-1][:KEEP_CENTROID_TOP_N]
    top_fids = df["file_id"].iloc[top_n_idx].tolist()

    keep_vecs = [embeddings[fid] for fid in top_fids if fid in embeddings]
    if not keep_vecs:
        log.warning("  No embedding vectors for top KEEP files — S9=0.")
        df["s_semantic_proximity"] = np.float32(0.0)
        return df

    centroid = np.mean(keep_vecs, axis=0).astype(np.float32)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)

    def cosine_to_centroid(fid):
        v = embeddings.get(fid)
        if v is None:
            return 0.5  # neutral
        vn = v / (np.linalg.norm(v) + 1e-9)
        cos = float(np.dot(vn, centroid_norm))
        return (cos + 1) / 2  # rescale [-1,1] → [0,1]

    df["s_semantic_proximity"] = df["file_id"].map(cosine_to_centroid).astype(np.float32)
    log.info(f"  S9 computed. Centroid built from {len(keep_vecs)} vectors.")
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 8: RF CLASSIFIER → IMPORTANCE SCORE + LABEL
# ═══════════════════════════════════════════════════════════════

ALL_SIGNAL_COLS = [
    "s_content_richness", "s_recency", "s_type_importance",
    "s_uniqueness", "s_extraction_quality", "s_content_depth",
    "s_cluster_density", "s_llm_quality", "s_semantic_proximity",
]
BOOT_WEIGHTS_9 = np.append(BOOT_WEIGHTS, 0.0).astype(np.float32)  # S9 has 0 in boot phase


def _label_to_idx(score: float) -> int:
    if score >= LABEL_THRESHOLDS["KEEP"]:
        return LABEL_TO_IDX["KEEP"]
    elif score >= LABEL_THRESHOLDS["ARCHIVE"]:
        return LABEL_TO_IDX["ARCHIVE"]
    elif score >= LABEL_THRESHOLDS["REVIEW"]:
        return LABEL_TO_IDX["REVIEW"]
    else:
        return LABEL_TO_IDX["DELETE_CANDIDATE"]


def train_or_load_rf(df: pd.DataFrame) -> RandomForestClassifier:
    if Path(MODEL_PATH).exists():
        log.info(f"  Loading saved RF model from {MODEL_PATH}…")
        return joblib.load(MODEL_PATH)

    log.info("  Training RandomForest (bootstrap labels)…")
    X = df[ALL_SIGNAL_COLS].fillna(0).values.astype(np.float32)

    # Boot score using BOOT_WEIGHTS_9
    boot_scores = X @ BOOT_WEIGHTS_9 * 100  # scale to 0–100
    y = np.array([_label_to_idx(s) for s in boot_scores])

    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X, y)
    joblib.dump(clf, MODEL_PATH)
    log.info(f"  RF trained & saved → {MODEL_PATH}")
    return clf


def score_with_rf(clf: RandomForestClassifier, df: pd.DataFrame) -> pd.DataFrame:
    """Apply RF → importance_score + label."""
    log.info("► STEP 8: Scoring with RF classifier…")
    X = df[ALL_SIGNAL_COLS].fillna(0).values.astype(np.float32)
    proba = clf.predict_proba(X).astype(np.float32)

    # Ensure columns match IDX_TO_LABEL ordering
    class_order = clf.classes_
    full_proba = np.zeros((len(df), 4), dtype=np.float32)
    for col_i, cls_idx in enumerate(class_order):
        full_proba[:, cls_idx] = proba[:, col_i]

    importance_scores = (full_proba @ CLASS_MIDPOINTS).clip(0, 100)
    label_indices = full_proba.argmax(axis=1)
    labels = [IDX_TO_LABEL[i] for i in label_indices]

    df["importance_score"] = importance_scores.astype(np.float32)
    df["label"] = labels
    log.info("  Scoring complete.")
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 9: UPSERT TO file_scores
# ═══════════════════════════════════════════════════════════════

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS file_scores (
    file_id              BIGINT PRIMARY KEY,
    path                 TEXT,
    name                 TEXT,
    ext                  TEXT,
    category             TEXT,
    s_content_richness   REAL,
    s_recency            REAL,
    s_type_importance    REAL,
    s_uniqueness         REAL,
    s_extraction_quality REAL,
    s_content_depth      REAL,
    s_cluster_density    REAL,
    s_llm_quality        REAL,
    s_semantic_proximity REAL,
    importance_score     REAL,
    label                TEXT,
    scored_at            TIMESTAMPTZ DEFAULT NOW()
);
"""

UPSERT_SQL = """
INSERT INTO file_scores (
    file_id, path, name, ext, category,
    s_content_richness, s_recency, s_type_importance,
    s_uniqueness, s_extraction_quality, s_content_depth,
    s_cluster_density, s_llm_quality, s_semantic_proximity,
    importance_score, label, scored_at
) VALUES %s
ON CONFLICT (file_id) DO UPDATE SET
    path                 = EXCLUDED.path,
    name                 = EXCLUDED.name,
    ext                  = EXCLUDED.ext,
    category             = EXCLUDED.category,
    s_content_richness   = EXCLUDED.s_content_richness,
    s_recency            = EXCLUDED.s_recency,
    s_type_importance    = EXCLUDED.s_type_importance,
    s_uniqueness         = EXCLUDED.s_uniqueness,
    s_extraction_quality = EXCLUDED.s_extraction_quality,
    s_content_depth      = EXCLUDED.s_content_depth,
    s_cluster_density    = EXCLUDED.s_cluster_density,
    s_llm_quality        = EXCLUDED.s_llm_quality,
    s_semantic_proximity = EXCLUDED.s_semantic_proximity,
    importance_score     = EXCLUDED.importance_score,
    label                = EXCLUDED.label,
    scored_at            = EXCLUDED.scored_at
"""


def upsert_scores(conn, df: pd.DataFrame) -> None:
    log.info("► STEP 9: Upserting scores to file_scores…")
    now_str = datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_SQL)

    rows = []
    for _, row in df.iterrows():
        rows.append((
            int(row["file_id"]),
            str(row.get("path", "")),
            str(row.get("name", "")),
            str(row.get("ext", "")),
            str(row.get("folder", "")),  # category ← folder
            float(row["s_content_richness"]),
            float(row["s_recency"]),
            float(row["s_type_importance"]),
            float(row["s_uniqueness"]),
            float(row["s_extraction_quality"]),
            float(row["s_content_depth"]),
            float(row["s_cluster_density"]),
            float(row["s_llm_quality"]),
            float(row["s_semantic_proximity"]),
            float(row["importance_score"]),
            str(row["label"]),
            now_str,
        ))

    execute_values(cur, UPSERT_SQL, rows, page_size=500)
    conn.commit()
    cur.close()
    log.info(f"  Upserted {len(rows):,} rows into file_scores.")


# ═══════════════════════════════════════════════════════════════
# STEP 10: JSON REPORT
# ═══════════════════════════════════════════════════════════════

def write_report(df: pd.DataFrame, clf: RandomForestClassifier, elapsed: float) -> None:
    log.info("► STEP 10: Writing JSON report…")
    scores = df["importance_score"]

    label_counts = df["label"].value_counts().to_dict()
    total = len(df)
    label_pcts = {k: round(100 * v / total, 2) for k, v in label_counts.items()}

    top10 = (
        df.nlargest(10, "importance_score")[["file_id", "path", "importance_score", "label"]]
        .to_dict(orient="records")
    )
    bottom10 = (
        df.nsmallest(10, "importance_score")[["file_id", "path", "importance_score", "label"]]
        .to_dict(orient="records")
    )

    fi = dict(zip(ALL_SIGNAL_COLS, clf.feature_importances_.tolist()))

    report = {
        "phase": 7,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "total_files": total,
        "score_distribution": {
            "min": round(float(scores.min()), 2),
            "max": round(float(scores.max()), 2),
            "mean": round(float(scores.mean()), 2),
            "median": round(float(scores.median()), 2),
            "p25": round(float(scores.quantile(0.25)), 2),
            "p75": round(float(scores.quantile(0.75)), 2),
        },
        "label_counts": label_counts,
        "label_percentages": label_pcts,
        "rf_feature_importances": fi,
        "top_10_files": top10,
        "bottom_10_files": bottom10,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"  Report written → {REPORT_PATH}")
    return report


def print_summary(df: pd.DataFrame, clf: RandomForestClassifier, elapsed: float) -> None:
    total = len(df)
    scores = df["importance_score"]
    label_counts = df["label"].value_counts()

    print("\n" + "═" * 60)
    print("  PHASE 7 COMPLETE — FILE IMPORTANCE SCORING")
    print("═" * 60)
    print(f"  Total files scored : {total:,}")
    print(f"  Elapsed time       : {elapsed:.1f}s")
    print(f"\n  Score Distribution:")
    print(f"    Min    : {scores.min():.1f}")
    print(f"    Mean   : {scores.mean():.1f}")
    print(f"    Median : {scores.median():.1f}")
    print(f"    Max    : {scores.max():.1f}")

    print(f"\n  Label Breakdown:")
    for label, count in label_counts.items():
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"    {label:<20} {count:>6,}  ({pct:5.1f}%)  {bar}")

    print(f"\n  RF Feature Importances:")
    fi = list(zip(ALL_SIGNAL_COLS, clf.feature_importances_))
    fi.sort(key=lambda x: -x[1])
    for name, importance in fi:
        bar = "▓" * int(importance * 40)
        print(f"    {name:<25} {importance:.4f}  {bar}")

    print("═" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_pipeline():
    t0 = time.time()
    log.info("═" * 60)
    log.info("PHASE 7: FILE IMPORTANCE SCORING — STARTING")
    log.info("═" * 60)

    conn = psycopg2.connect(DB_URL)

    try:
        # Steps 1–2: data fetch
        df = fetch_files(conn)
        dup_paths = fetch_duplicate_paths(conn)

        # Step 3: embeddings
        embeddings = load_embeddings(df)

        # Steps 4–6: traditional signals
        df = compute_s1_to_s6(df, dup_paths)
        df = compute_s7_cluster_density(df, embeddings)
        df = compute_s8_llm_quality(df, conn, dup_paths)

        # Step 7: semantic proximity
        df = compute_s9_semantic_proximity(df, embeddings)

        # Clamp all signal columns
        for col in ALL_SIGNAL_COLS:
            df[col] = df[col].fillna(0).clip(0, 1).astype(np.float32)

        # Step 8: RF scoring
        clf = train_or_load_rf(df)
        df = score_with_rf(clf, df)

        # Step 9: persist
        upsert_scores(conn, df)

        # Step 10: report
        elapsed = time.time() - t0
        report = write_report(df, clf, elapsed)
        print_summary(df, clf, elapsed)

    except Exception:
        log.error("FATAL: " + traceback.format_exc())
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run_pipeline()
