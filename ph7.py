# phase7.py
"""
Phase 7 — File Importance Scoring (Dropbox Dash / Gemini-style)
Nine signals with real AI reasoning (embeddings + LLM).
"""

import argparse
import json
import logging
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import requests
from dotenv import load_dotenv
from psycopg2.extras import execute_values, RealDictCursor
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
import joblib

load_dotenv()

# ========================= CONFIG =========================
DB_URL = os.getenv("DB_URL")
REPORT_PATH = os.getenv("PH7_REPORT_PATH", "phase7_report.json")
MODEL_PATH = os.getenv("PH7_MODEL_PATH", "phase7_model.joblib")
LLM_CACHE_PATH = Path("llm_cache.json")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

DBSCAN_EPS = float(os.getenv("DBSCAN_EPS", 0.3))
DBSCAN_MIN_SAMPLES = int(os.getenv("DBSCAN_MIN_SAMPLES", 3))
LLM_MAX_WORDS = int(os.getenv("LLM_MAX_WORDS", 200))
KEEP_CENTROID_N = int(os.getenv("KEEP_CENTROID_N", 200))
MAX_STALENESS_DAYS = float(os.getenv("MAX_STALENESS_DAYS", 1825))

_BOOT_WEIGHTS = {
    "s_content_richness": 0.20,
    "s_recency": 0.10,
    "s_type_importance": 0.15,
    "s_uniqueness": 0.15,
    "s_extraction_quality": 0.05,
    "s_content_depth": 0.05,
    "s_cluster_density": 0.15,
    "s_llm_quality": 0.10,
    "s_semantic_proximity": 0.05,
}

RF_PARAMS = {
    "n_estimators": int(os.getenv("RF_N_ESTIMATORS", 300)),
    "max_depth": int(os.getenv("RF_MAX_DEPTH", 15)) or None,
    "min_samples_leaf": int(os.getenv("RF_MIN_SAMPLES_LEAF", 2)),
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

CLASS_MIDPOINTS = np.array([10.0, 35.0, 65.0, 90.0])

IDX_TO_LABEL = {
    0: "DELETE_CANDIDATE",
    1: "REVIEW",
    2: "ARCHIVE",
    3: "KEEP",
}

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

# ========================= LOGGING =========================
logging.basicConfig(
    filename="phase7_errors.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)

# ========================= HELPERS =========================
def _conn():
    if not DB_URL:
        raise RuntimeError("DB_URL not set in .env")
    return psycopg2.connect(DB_URL)


def _create_tables(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS file_scores (
                id SERIAL PRIMARY KEY,
                file_id INTEGER UNIQUE REFERENCES files(id) ON DELETE CASCADE,
                path TEXT,
                name TEXT,
                ext TEXT,
                category TEXT,
                s_content_richness REAL,
                s_recency REAL,
                s_type_importance REAL,
                s_uniqueness REAL,
                s_extraction_quality REAL,
                s_content_depth REAL,
                s_cluster_density REAL,
                s_llm_quality REAL,
                s_semantic_proximity REAL,
                importance_score REAL,
                label TEXT,
                scored_at TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()

    with conn.cursor() as cur:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fs_label ON file_scores(label)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fs_score ON file_scores(importance_score DESC)")
        conn.commit()


def _fetch_all_data(conn) -> pd.DataFrame:
    print(" [1/7] Fetching data from ph1–ph3 tables …")
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT 
                f.id AS file_id,
                f.path,
                f.name,
                f.stem,
                f.extension,
                f.folder,
                f.depth,
                f.size_bytes,
                f.created_time,
                f.modified_time,
                f.access_time,
                f.hash,
                e.extraction_status,
                e.ocr_applied,
                e.word_count AS extracted_words,
                e.language_hint,
                ch.chunk_total,
                ch.clean_word_count
            FROM files f
            LEFT JOIN extracted_content e ON f.id = e.file_id
            LEFT JOIN (
                SELECT 
                    file_id, 
                    MAX(chunk_total) AS chunk_total,
                    SUM(clean_word_count) AS clean_word_count
                FROM chunks 
                WHERE clean_status = 'SUCCESS' 
                GROUP BY file_id
            ) ch ON f.id = ch.file_id
        """)
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError("No files found — run phases 1-3 first.")
    df = pd.DataFrame(rows)
    print(f" → {len(df):,} files loaded")
    return df


def _fetch_duplicate_paths(conn) -> set:
    print(" [2/7] Fetching duplicate/redundancy data …")
    dup_paths = set()
    with conn.cursor() as cur:
        for query in [
            "SELECT path_1 FROM duplicate_files UNION SELECT path_2 FROM duplicate_files",
            "SELECT path_1 FROM file_redundancy WHERE action = 'DELETE duplicate' UNION SELECT path_2 FROM file_redundancy WHERE action = 'DELETE duplicate'",
        ]:
            try:
                cur.execute(query)
                for row in cur.fetchall():
                    if row[0]:
                        dup_paths.add(row[0])
            except Exception:
                conn.rollback()
    print(f" → {len(dup_paths):,} duplicate/redundant paths")
    return dup_paths


def _fetch_embeddings(conn) -> dict:
    print(" [3/7] Loading embeddings from FAISS …")
    faiss_path = os.getenv("FAISS_PATH", "vector_store.faiss")
    index_path = os.getenv("CHUNK_INDEX_PATH", "chunk_index.json")

    if not Path(faiss_path).exists() or not Path(index_path).exists():
        print(" → FAISS files not found — S7/S9 will be 0")
        return {}

    try:
        import faiss
        index = faiss.read_index(faiss_path)
        with open(index_path, "r", encoding="utf-8") as f:
            chunk_index = json.load(f)

        n = index.ntotal
        dim = index.d
        print(f" → FAISS index: {n:,} vectors, dim={dim}")

        all_vecs = np.zeros((n, dim), dtype=np.float32)
        for i in range(n):
            all_vecs[i] = index.reconstruct(i)

        file_vecs = defaultdict(list)
        chunk_ids_to_resolve = []

        for i, entry in enumerate(chunk_index):
            if i >= n:
                break
            if isinstance(entry, dict):
                fid = entry.get("file_id") or entry.get("fileId")
                if fid is not None:
                    file_vecs[int(fid)].append(all_vecs[i])
                    continue
            # string chunk_id
            chunk_id = entry if isinstance(entry, (str, int)) else None
            if chunk_id:
                chunk_ids_to_resolve.append((i, str(chunk_id)))

        if chunk_ids_to_resolve:
            ids = [cid for _, cid in chunk_ids_to_resolve]
            with conn.cursor() as cur:
                cur.execute("SELECT chunk_id, file_id FROM chunks WHERE chunk_id = ANY(%s)", (ids,))
                mapping = {row[0]: row[1] for row in cur.fetchall()}
            for i, chunk_id in chunk_ids_to_resolve:
                fid = mapping.get(chunk_id)
                if fid:
                    file_vecs[int(fid)].append(all_vecs[i])

        emb_map = {
            fid: np.mean(np.vstack(vecs), axis=0).astype(np.float32)
            for fid, vecs in file_vecs.items() if len(vecs) > 0
        }
        print(f" → {len(emb_map):,} file embeddings created")
        return emb_map
    except Exception as e:
        logging.error(f"FAISS load failed: {e}")
        print(f" → FAISS failed: {e} — S7/S9 = 0")
        return {}


def _fetch_top_chunks(conn, file_ids: list) -> dict:
    if not file_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                file_id,
                array_to_string(array_agg(chunk_text ORDER BY chunk_index), ' ') AS chunk_text
            FROM (
                SELECT 
                    file_id,
                    clean_text AS chunk_text,
                    chunk_index,
                    ROW_NUMBER() OVER (PARTITION BY file_id ORDER BY chunk_index ASC) AS rn
                FROM chunks
                WHERE file_id = ANY(%s)
                  AND clean_status = 'SUCCESS'
                  AND clean_text IS NOT NULL
            ) t
            WHERE rn <= 3
            GROUP BY file_id
        """, (file_ids,))
        return {fid: text for fid, text in cur.fetchall()}


# ========================= SIGNALS =========================
def _s_content_richness(word_count) -> float:
    if not word_count or word_count <= 0:
        return 0.0
    return float(min(np.log1p(word_count) / np.log1p(20_000), 1.0))


def _s_recency(modified, accessed) -> float:
    now = datetime.now(timezone.utc)
    best = max((dt for dt in (modified, accessed) if dt), default=None)
    if best is None:
        return 0.0
    if best.tzinfo is None:
        best = best.replace(tzinfo=timezone.utc)
    return float(1.0 - min((now - best).days / MAX_STALENESS_DAYS, 1.0))


def _s_type_importance(ext: str) -> float:
    e = (ext or "").lower().lstrip(".")
    if e in {"pdf","docx","doc","odt","rtf","txt","md"}: return 1.00
    if e in {"xlsx","xls","csv","ods"}: return 0.95
    if e in {"pem","key","crt","p12","pfx"}: return 0.90
    if e in {"pptx","ppt","odp"}: return 0.85
    if e in {"py","js","ts","java","c","cpp","cs","go","rs","sql","sh"}: return 0.80
    if e in {"json","xml","yaml","yml","toml"}: return 0.75
    if e in {"jpg","jpeg","png","gif","bmp","svg","tiff"}: return 0.50
    if e in {"zip","tar","gz","7z","rar"}: return 0.40
    if e in {"tmp","temp","bak","log","cache"}: return 0.00
    return 0.30


def _s_uniqueness(path: str, dup_paths: set) -> float:
    return 0.0 if path in dup_paths else 1.0


def _s_extraction_quality(status, ocr) -> float:
    s = (status or "").upper()
    if s == "SUCCESS": return 1.0 if not ocr else 0.6
    if s == "PARTIAL": return 0.3
    return 0.0


def _s_content_depth(chunk_total) -> float:
    if not chunk_total or chunk_total <= 0:
        return 0.0
    return float(min(np.log1p(chunk_total) / np.log1p(20), 1.0))


def _compute_traditional_signals(df: pd.DataFrame, dup_paths: set) -> pd.DataFrame:
    print(" [4/7] Computing traditional signals S1–S6 …")
    df["s_content_richness"] = df.apply(
        lambda r: _s_content_richness(r.get("clean_word_count") or r.get("extracted_words") or 0), axis=1)
    df["s_recency"] = df.apply(lambda r: _s_recency(r["modified_time"], r["access_time"]), axis=1)
    df["s_type_importance"] = df["extension"].apply(_s_type_importance)
    df["s_uniqueness"] = df["path"].apply(lambda p: _s_uniqueness(p, dup_paths))
    df["s_extraction_quality"] = df.apply(
        lambda r: _s_extraction_quality(r["extraction_status"], r["ocr_applied"]), axis=1)
    df["s_content_depth"] = df["chunk_total"].apply(_s_content_depth)
    return df


def _compute_cluster_density(df: pd.DataFrame, emb_map: dict) -> pd.DataFrame:
    print(" [5/7] Computing S7 — cluster density (DBSCAN) …")
    has_emb = [fid for fid in df["file_id"] if fid in emb_map]
    if not has_emb:
        df["s_cluster_density"] = 0.0
        return df

    matrix = normalize(np.vstack([emb_map[fid] for fid in has_emb]))
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES,
                    metric="cosine", n_jobs=-1).fit_predict(matrix)

    counts = Counter(labels)
    max_size = max((v for k, v in counts.items() if k != -1), default=1)
    density_map = {fid: (0.0 if lbl == -1 else counts[lbl] / max_size)
                   for fid, lbl in zip(has_emb, labels)}

    df["s_cluster_density"] = df["file_id"].map(density_map).fillna(0.0)
    print(f" → {len(set(labels) - {-1})} clusters, {sum(1 for l in labels if l == -1)} noise")
    return df


# LLM
_LLM_SYSTEM = """You are an enterprise data analyst. Rate the following document excerpt for business relevance and information density on a scale of 1 to 5:
1 = useless (empty, gibberish, log noise, temp file)
2 = low value (boilerplate, auto-generated, trivial)
3 = moderate value (some useful info but generic)
4 = high value (clear business content, decisions, data)
5 = critical (contracts, financials, strategy, unique knowledge)
Reply with a SINGLE integer 1-5. Nothing else."""


def _load_llm_cache():
    if LLM_CACHE_PATH.exists():
        try:
            return json.loads(LLM_CACHE_PATH.read_text(encoding="utf-8"))
        except:
            return {}
    return {}


def _save_llm_cache(cache):
    LLM_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _call_groq_batch(texts: list[str]) -> list[float]:
    if not GROQ_API_KEY:
        return [0.5] * len(texts)

    cache = _load_llm_cache()
    results = []
    new_entries = {}

    for text in texts:
        key = hash(text[:800])
        if key in cache:
            results.append(cache[key])
            continue

        snippet = " ".join(text.split()[:LLM_MAX_WORDS])
        try:
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": _LLM_SYSTEM},
                        {"role": "user", "content": snippet}
                    ],
                    "max_tokens": 5,
                    "temperature": 0.0,
                },
                timeout=18,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            score = int("".join(c for c in raw if c.isdigit())[:1] or 3)
            norm_score = float(max(1, min(score, 5)) - 1) / 4.0
            results.append(norm_score)
            new_entries[key] = norm_score
        except Exception as e:
            logging.warning(f"Groq error: {e}")
            results.append(0.5)
            new_entries[key] = 0.5

    cache.update(new_entries)
    _save_llm_cache(cache)
    return results


def _compute_llm_quality(df: pd.DataFrame, dup_paths: set, conn) -> pd.DataFrame:
    print(" [6/7] Computing S8 — LLM quality …")
    if not GROQ_API_KEY:
        print("   → No GROQ key, using heuristic fallback")
        df["s_llm_quality"] = (0.3 * df["s_content_richness"] +
                               0.4 * df["s_content_depth"] +
                               0.3 * df["s_extraction_quality"]).clip(0.0, 1.0).fillna(0.2)
        return df

    candidates = df[(~df["path"].isin(dup_paths)) & (df["extracted_words"].fillna(0) > 20)]["file_id"].tolist()
    print(f"   → Scoring {len(candidates):,} files via Groq (batched)")

    chunk_map = _fetch_top_chunks(conn, candidates)
    texts = [chunk_map.get(fid, "") for fid in candidates]

    scores_list = _call_groq_batch(texts)
    score_map = dict(zip(candidates, scores_list))

    df["s_llm_quality"] = df["file_id"].map(score_map).fillna(0.2)
    return df


def _compute_semantic_proximity(df: pd.DataFrame, emb_map: dict) -> pd.DataFrame:
    print(" [7/7] Computing S9 — semantic proximity to KEEP centroid …")
    if not emb_map:
        df["s_semantic_proximity"] = 0.0
        return df

    boot = sum(_BOOT_WEIGHTS[s] * df[s] for s in SIGNAL_COLS if s != "s_semantic_proximity")
    df["_boot"] = boot

    top_ids = (df[df["file_id"].isin(emb_map)]
               .nlargest(KEEP_CENTROID_N, "_boot")["file_id"]
               .tolist())

    if not top_ids:
        df["s_semantic_proximity"] = 0.0
        df.drop(columns=["_boot"], errors='ignore', inplace=True)
        return df

    centroid = normalize(np.mean(np.vstack([emb_map[fid] for fid in top_ids]), axis=0, keepdims=True))
    has_emb = [fid for fid in df["file_id"] if fid in emb_map]
    matrix = normalize(np.vstack([emb_map[fid] for fid in has_emb]))

    sims = cosine_similarity(matrix, centroid).flatten()
    sim_map = dict(zip(has_emb, sims.tolist()))

    df["s_semantic_proximity"] = df["file_id"].map(sim_map).fillna(0.0).clip(-1, 1).add(1).div(2)
    df.drop(columns=["_boot"], errors='ignore', inplace=True)
    print(f"   → Centroid from top {len(top_ids)} files")
    return df


# ========================= RF =========================
def _bootstrap_labels(df: pd.DataFrame) -> np.ndarray:
    raw = sum(_BOOT_WEIGHTS[s] * df[s] for s in SIGNAL_COLS) * 100
    lo, hi = raw.min(), raw.max()
    if hi > lo:
        raw = (raw - lo) / (hi - lo) * 100
    return np.where(raw >= 80, 3,
                    np.where(raw >= 50, 2,
                             np.where(raw >= 20, 1, 0)))


def _get_model(df: pd.DataFrame, X: np.ndarray) -> RandomForestClassifier:
    if Path(MODEL_PATH).exists():
        print(f" Loading existing model from {MODEL_PATH}")
        return joblib.load(MODEL_PATH)

    print(" Training new Random Forest (bootstrapped labels)…")
    y = _bootstrap_labels(df)
    rf = RandomForestClassifier(**RF_PARAMS)
    if len(X) >= 10:
        cv = cross_val_score(rf, X, y, cv=min(5, len(X)//2), scoring="accuracy")
        print(f" Cross-val accuracy: {cv.mean():.3f} ± {cv.std():.3f}")
    rf.fit(X, y)
    joblib.dump(rf, MODEL_PATH)
    print(f" Model saved → {MODEL_PATH}")
    return rf


def _rf_score(rf, X):
    proba = rf.predict_proba(X)
    scores = np.clip(proba.dot(CLASS_MIDPOINTS), 0, 100).round(2)
    labels = np.array([IDX_TO_LABEL[i] for i in np.argmax(proba, axis=1)])
    return scores, labels


def _save_to_db(conn, df: pd.DataFrame):
    print(" Saving scores to database …")
    rows = [
        (
            int(r.file_id), r.path, r.name, r.extension, None,
            float(r.s_content_richness), float(r.s_recency),
            float(r.s_type_importance), float(r.s_uniqueness),
            float(r.s_extraction_quality), float(r.s_content_depth),
            float(r.s_cluster_density), float(r.s_llm_quality),
            float(r.s_semantic_proximity),
            float(r.importance_score), r.label
        )
        for r in df.itertuples(index=False)
    ]

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO file_scores (
                file_id, path, name, ext, category,
                s_content_richness, s_recency, s_type_importance, s_uniqueness,
                s_extraction_quality, s_content_depth, s_cluster_density,
                s_llm_quality, s_semantic_proximity,
                importance_score, label
            ) VALUES %s
            ON CONFLICT (file_id) DO UPDATE SET
                s_content_richness = EXCLUDED.s_content_richness,
                s_recency = EXCLUDED.s_recency,
                s_type_importance = EXCLUDED.s_type_importance,
                s_uniqueness = EXCLUDED.s_uniqueness,
                s_extraction_quality = EXCLUDED.s_extraction_quality,
                s_content_depth = EXCLUDED.s_content_depth,
                s_cluster_density = EXCLUDED.s_cluster_density,
                s_llm_quality = EXCLUDED.s_llm_quality,
                s_semantic_proximity = EXCLUDED.s_semantic_proximity,
                importance_score = EXCLUDED.importance_score,
                label = EXCLUDED.label,
                scored_at = NOW()
        """, rows)
        conn.commit()
    print(f" → {len(rows):,} files scored")


def _build_report(df: pd.DataFrame, rf, elapsed: float) -> dict:
    fi = dict(zip(SIGNAL_COLS, rf.feature_importances_.tolist()))
    return {
        "phase": 7,
        "elapsed_seconds": round(elapsed, 2),
        "total_files_scored": len(df),
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": rf.n_estimators,
            "max_depth": rf.max_depth,
            "model_path": MODEL_PATH,
            "feature_importance": {k: round(v, 4) for k, v in fi.items()},
        },
        "score_distribution": {
            "mean": round(float(df["importance_score"].mean()), 2),
            "median": round(float(df["importance_score"].median()), 2),
            "std": round(float(df["importance_score"].std()), 2),
            "min": round(float(df["importance_score"].min()), 2),
            "max": round(float(df["importance_score"].max()), 2),
        },
        "label_counts": df["label"].value_counts().to_dict(),
        "top_10_files": df.nlargest(10, "importance_score")[[
            "name", "extension", "importance_score", "label",
            "s_llm_quality", "s_cluster_density", "s_semantic_proximity"
        ]].to_dict(orient="records"),
        "bottom_10_files": df.nsmallest(10, "importance_score")[[
            "name", "extension", "importance_score", "label",
            "s_llm_quality", "s_cluster_density", "s_semantic_proximity"
        ]].to_dict(orient="records"),
    }


# ========================= MAIN =========================
def run_phase7(max_files: int = None, dry_run: bool = False, force_retrain: bool = False):
    t0 = time.time()
    print("\n" + "="*70)
    print("Phase 7 — File Importance Scoring (Dash / Gemini-style)")
    print("="*70 + "\n")

    conn = _conn()
    _create_tables(conn)

    df = _fetch_all_data(conn)
    if max_files:
        df = df.head(max_files)

    dup_paths = _fetch_duplicate_paths(conn)
    emb_map = _fetch_embeddings(conn)

    df = _compute_traditional_signals(df, dup_paths)
    df = _compute_cluster_density(df, emb_map)
    df = _compute_llm_quality(df, dup_paths, conn)
    df = _compute_semantic_proximity(df, emb_map)

    X = df[SIGNAL_COLS].values.astype(np.float32)

    if force_retrain and Path(MODEL_PATH).exists():
        Path(MODEL_PATH).unlink()

    rf = _get_model(df, X)

    print(" Scoring with Random Forest …")
    scores, labels = _rf_score(rf, X)
    df["importance_score"] = scores
    df["label"] = labels

    if not dry_run:
        _save_to_db(conn, df)

    conn.close()

    elapsed = time.time() - t0
    report = _build_report(df, rf, elapsed)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # Pretty print summary
    print(f"\n{'='*70}")
    print(f"Files scored     : {report['total_files_scored']:,}")
    sd = report["score_distribution"]
    print(f"Score mean/median: {sd['mean']:.1f} / {sd['median']:.1f}")
    print(f"Label distribution:")
    for label, count in sorted(report["label_counts"].items(), key=lambda x: -x[1]):
        pct = count / report["total_files_scored"] * 100
        print(f"  {label:<18} {count:6,} ({pct:5.1f}%)")
    print(f"\nTop file : {report['top_10_files'][0]['name'] if report['top_10_files'] else 'N/A'}")
    print(f"Report   : {REPORT_PATH}")
    print(f"Elapsed  : {elapsed:.1f}s")
    print("="*70)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files", type=int, help="Limit number of files for testing")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    parser.add_argument("--force-retrain", action="store_true", help="Ignore saved model")
    args = parser.parse_args()

    run_phase7(
        max_files=args.max_files,
        dry_run=args.dry_run,
        force_retrain=args.force_retrain
    )
