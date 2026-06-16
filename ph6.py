"""
Phase 6 — Duplicate & Redundancy Detection

Reads from : chunks table          (ph1 → ph3)
             find_duplicate_chunks  (py4_5 — FAISS-based near-dup scan)
Writes to  : duplicate_files, redundant_sections,
             file_redundancy, merge_recommendations  (PostgreSQL)
             phase6_report.json

Steps:
  1. Exact duplicates  — hash match on chunks.hash (set by ph1)
  2. Near-duplicates   — import find_duplicate_chunks() from py4_5 (FAISS)
  3. File-level groups — aggregate chunk pairs → file pairs
  4. Recommendations   — action + severity per pair
  5. JSON report
"""

import json
import logging
import time
import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from dotenv import load_dotenv

from ph4 import find_duplicate_chunks   # ← ph4/5 FAISS scan

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────

DB_URL = os.getenv("DB_URL")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.90))
FILE_DUP_THRESHOLD = float(os.getenv("FILE_DUP_THRESHOLD", 0.85))
REPORT_PATH = os.getenv("REPORT_PATH", "phase6_report.json")

logging.basicConfig(
    filename="phase6_errors.log",
    level=logging.WARNING,
    format="%(asctime)s — %(levelname)s — %(message)s",
)

# ── DB ────────────────────────────────────────────────────────────────────────

def _conn():
    if not DB_URL:
        raise RuntimeError("DB_URL not set. Please set DB_URL in your .env or environment before running Phase 6.")
    return psycopg2.connect(DB_URL)


def _create_tables(conn):
    with conn.cursor() as cur:
        cur.execute("""
            DROP TABLE IF EXISTS duplicate_files;
            DROP TABLE IF EXISTS redundant_sections;
            DROP TABLE IF EXISTS file_redundancy;
            DROP TABLE IF EXISTS merge_recommendations;
            
            CREATE TABLE duplicate_files (
                hash        TEXT,
                path_1      TEXT,
                path_2      TEXT,
                detected_at TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (path_1, path_2)
            );
            CREATE TABLE redundant_sections (
                chunk_id_1  TEXT,
                chunk_id_2  TEXT,
                path_1      TEXT,
                path_2      TEXT,
                similarity  FLOAT,
                detected_at TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (chunk_id_1, chunk_id_2)
            );
            CREATE TABLE file_redundancy (
                path_1           TEXT,
                path_2           TEXT,
                shared_chunks    INTEGER,
                avg_similarity   FLOAT,
                redundancy_ratio FLOAT,
                action           TEXT,
                detected_at      TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (path_1, path_2)
            );
            CREATE TABLE merge_recommendations (
                id           SERIAL PRIMARY KEY,
                rec_type     TEXT,
                severity     TEXT,
                primary_id   TEXT,
                duplicate_id TEXT,
                action       TEXT,
                details      TEXT,
                detected_at  TIMESTAMP DEFAULT NOW()
            );
        """)
    conn.commit()
# ── Step 1 — Exact duplicates ─────────────────────────────────────────────────

def _exact_duplicates(conn) -> pd.DataFrame:
    print("\n[1] Exact duplicates (hash match)...")
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT a.hash, a.path AS path_1, b.path AS path_2
            FROM (SELECT DISTINCT hash, path FROM chunks
                  WHERE hash IS NOT NULL AND hash NOT LIKE 'error%') a
            JOIN (SELECT DISTINCT hash, path FROM chunks
                  WHERE hash IS NOT NULL AND hash NOT LIKE 'error%') b
              ON a.hash = b.hash AND a.path < b.path
            ORDER BY a.hash
        """)
        rows = cur.fetchall()

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["hash", "path_1", "path_2"])
    if not df.empty:
        with conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO duplicate_files (hash, path_1, path_2) VALUES %s ON CONFLICT DO NOTHING",
                df.itertuples(index=False, name=None),
            )
        conn.commit()
    print(f"  → {len(df):,} exact duplicate pair(s)")
    return df

# ── Step 2 — Near-duplicates (imported from py4_5) ────────────────────────────

def _near_duplicates(conn) -> pd.DataFrame:
    print(f"\n[2] Near-duplicate chunks (FAISS, threshold={SIMILARITY_THRESHOLD})...")

    # find_duplicate_chunks() does the FAISS scan — no logic duplicated here
    dup_df = find_duplicate_chunks(similarity_threshold=SIMILARITY_THRESHOLD, store=False)
    if dup_df.empty:
        print("  → 0 near-duplicate pair(s)")
        return pd.DataFrame(columns=["chunk_id_1", "chunk_id_2", "path_1", "path_2", "similarity"])

    # Attach file paths from chunks table in one query
    all_ids = list(set(dup_df["chunk_id_1"].tolist() + dup_df["chunk_id_2"].tolist()))
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT DISTINCT chunk_id, path FROM chunks WHERE chunk_id = ANY(%s)", (all_ids,)
        )
        id_to_path = {r["chunk_id"]: r["path"] for r in cur.fetchall()}

    dup_df["path_1"] = dup_df["chunk_id_1"].map(id_to_path).fillna("")
    dup_df["path_2"] = dup_df["chunk_id_2"].map(id_to_path).fillna("")
    dup_df = dup_df.rename(columns={"similarity_score": "similarity"})

    # Keep cross-file pairs only
    dup_df = dup_df[dup_df["path_1"] != dup_df["path_2"]].reset_index(drop=True)

    if not dup_df.empty:
        with conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO redundant_sections "
                "(chunk_id_1, chunk_id_2, path_1, path_2, similarity) "
                "VALUES %s ON CONFLICT DO NOTHING",
                dup_df[["chunk_id_1", "chunk_id_2", "path_1", "path_2", "similarity"]]
                    .itertuples(index=False, name=None),
            )
        conn.commit()

    print(f"  → {len(dup_df):,} near-duplicate chunk pair(s)")
    return dup_df

# ── Step 3 — File-level aggregation ──────────────────────────────────────────

def _file_redundancy(conn, chunk_df: pd.DataFrame) -> pd.DataFrame:
    print("\n[3] File-level redundancy aggregation...")
    if chunk_df.empty:
        print("  → No pairs to aggregate")
        return pd.DataFrame()

    df = chunk_df.copy()
    df["fp1"] = df[["path_1", "path_2"]].min(axis=1)
    df["fp2"] = df[["path_1", "path_2"]].max(axis=1)
    df = df[df["fp1"] != df["fp2"]]

    agg = (
        df.groupby(["fp1", "fp2"])
        .agg(shared_chunks=("similarity", "count"), avg_similarity=("similarity", "mean"))
        .reset_index()
        .rename(columns={"fp1": "path_1", "fp2": "path_2"})
    )

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT path, COUNT(*) AS cnt FROM chunks WHERE path IS NOT NULL GROUP BY path")
        counts = {r["path"]: r["cnt"] for r in cur.fetchall()}

    agg["redundancy_ratio"] = agg.apply(
        lambda r: r["shared_chunks"] / max(counts.get(r["path_1"], 1), counts.get(r["path_2"], 1)),
        axis=1,
    )
    agg["action"] = agg["avg_similarity"].apply(
        lambda s: "DELETE duplicate" if s >= FILE_DUP_THRESHOLD else "REVIEW"
    )

    if not agg.empty:
        with conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO file_redundancy "
                "(path_1, path_2, shared_chunks, avg_similarity, redundancy_ratio, action) "
                "VALUES %s ON CONFLICT DO NOTHING",
                [
                    (r.path_1, r.path_2, int(r.shared_chunks),
                     float(r.avg_similarity), float(r.redundancy_ratio), r.action)
                    for r in agg.itertuples()
                ],
            )
        conn.commit()

    print(f"  → {len(agg):,} redundant file pair(s)")
    return agg

# ── Step 4 — Recommendations ──────────────────────────────────────────────────

def _recommendations(conn, exact_df: pd.DataFrame, file_df: pd.DataFrame) -> list[dict]:
    print("\n[4] Building recommendations...")
    recs = []

    for r in exact_df.itertuples():
        recs.append({
            "rec_type": "exact_duplicate", "severity": "HIGH",
            "primary_id": r.path_1, "duplicate_id": r.path_2,
            "action": "Delete one copy",
            "details": f"Identical files (hash={r.hash[:12]}…)",
        })

    for r in file_df.itertuples():
        recs.append({
            "rec_type": "near_duplicate_file",
            "severity": "HIGH" if r.avg_similarity >= FILE_DUP_THRESHOLD else "MEDIUM",
            "primary_id": r.path_1, "duplicate_id": r.path_2,
            "action": r.action,
            "details": (
                f"Shared chunks: {r.shared_chunks}, "
                f"avg sim: {r.avg_similarity:.2f}, "
                f"overlap: {r.redundancy_ratio:.0%}"
            ),
        })

    if recs:
        with conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO merge_recommendations "
                "(rec_type, severity, primary_id, duplicate_id, action, details) VALUES %s",
                [(r["rec_type"], r["severity"], r["primary_id"],
                  r["duplicate_id"], r["action"], r["details"]) for r in recs],
            )
        conn.commit()

    print(f"  → {len(recs):,} recommendation(s)")
    return recs

# ── Entry point ───────────────────────────────────────────────────────────────

def run_phase6():
    t0 = time.time()
    print("\n" + "=" * 56)
    print("  Phase 6 — Duplicate & Redundancy Detection")
    print("=" * 56)

    conn = _conn()
    _create_tables(conn)

    exact_df = _exact_duplicates(conn)
    chunk_df = _near_duplicates(conn)
    file_df  = _file_redundancy(conn, chunk_df)
    recs     = _recommendations(conn, exact_df, file_df)
    conn.close()

    elapsed = round(time.time() - t0, 2)
    summary = {
        "phase": 6,
        "elapsed_seconds": elapsed,
        "exact_duplicate_pairs":      len(exact_df),
        "near_duplicate_chunk_pairs": len(chunk_df),
        "redundant_file_pairs":       len(file_df),
        "recommendations":            len(recs),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*56}")
    print(f"  Exact duplicate pairs      : {len(exact_df):,}")
    print(f"  Near-duplicate chunk pairs : {len(chunk_df):,}")
    print(f"  Redundant file pairs       : {len(file_df):,}")
    print(f"  Recommendations            : {len(recs):,}")
    print(f"  Elapsed                    : {elapsed}s")
    print(f"  Report → {REPORT_PATH}")
    print(f"{'='*56}\nPhase 6 complete.")
    return summary


if __name__ == "__main__":
    run_phase6()