"""
Phase 4 — Embeddings  (sentence-transformers → FAISS)
Phase 5 — Vector Storage, Search & Duplicate Detection  (FAISS)

Reads from : chunks table  (written by ph3.py)
Writes to  : embeddings_log table  +  vector_store.faiss  +  chunk_index.json

Index auto-selection:
  ≤ 500 k chunks → IndexFlatIP   (exact cosine, no training, instant)
  > 500 k chunks → IndexIVFFlat  (approximate, trained, scales to millions)

Install:
    pip install faiss-cpu sentence-transformers psycopg2-binary numpy pandas
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor

from hf_config import use_hf_token

load_dotenv()
use_hf_token()

# ── Config ───────────────────────────────────────────────────────────────────

DB_URL = os.getenv("DB_URL")
EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 128))
EMBED_MAX_CHARS = int(os.getenv("EMBED_MAX_CHARS", 2000))
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "vector_store.faiss")
CHUNK_INDEX_PATH = os.getenv("CHUNK_INDEX_PATH", "chunk_index.json")
IVF_THRESHOLD = int(os.getenv("IVF_THRESHOLD", 500_000))
IVF_NLIST = int(os.getenv("IVF_NLIST", 256))
IVF_NPROBE = int(os.getenv("IVF_NPROBE", 32))

logging.basicConfig(
    filename="embedding_errors.log",
    level=logging.WARNING,
    format="%(asctime)s — %(levelname)s — %(message)s",
)

# ── stdout safety ────────────────────────────────────────────────────────────

def _safe_print(*args, sep=" ", end="\n", file=None, flush=False):
    if file is None:
        file = sys.stdout
    text = sep.join(str(a) for a in args) + end
    try:
        file.write(text)
    except Exception:
        sys.__stdout__.buffer.write(text.encode("utf-8", errors="backslashreplace"))
    if flush:
        try: file.flush()
        except Exception: pass

print = _safe_print

# ── DB ────────────────────────────────────────────────────────────────────────

def _conn():
    if not DB_URL:
        raise RuntimeError("DB_URL not set. Please set DB_URL in your .env or environment before running Phase 4.")
    return psycopg2.connect(DB_URL)


def _setup_log_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings_log (
                chunk_id    TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                faiss_row   INTEGER NOT NULL,
                embedded_at TIMESTAMP DEFAULT NOW()
            );
        """)
    conn.commit()


def _fetch_pending(conn) -> list[dict]:
    """All chunks written by ph3 that are not yet embedded."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT c.chunk_id, c.clean_text
            FROM   chunks c
            LEFT   JOIN embeddings_log el ON c.chunk_id = el.chunk_id
            WHERE  el.chunk_id IS NULL
              AND  c.clean_status = 'SUCCESS'
            ORDER  BY c.chunk_id
        """)
        return cur.fetchall()


def _log(conn, rows: list[tuple]):
    if not rows:
        return
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO embeddings_log (chunk_id, faiss_row) VALUES %s ON CONFLICT DO NOTHING",
            rows,
        )
    conn.commit()

# ── FAISS index helpers ───────────────────────────────────────────────────────

def _build_index(dim: int, total_chunks: int):
    """
    Auto-select index type based on corpus size:
      ≤ IVF_THRESHOLD → IndexFlatIP  (exact, zero setup)
      >  IVF_THRESHOLD → IndexIVFFlat (approximate, needs training)
    Returns (index, needs_training: bool)
    """
    import faiss
    if total_chunks <= IVF_THRESHOLD:
        print(f"  Index type : IndexFlatIP (exact, corpus={total_chunks:,})")
        return faiss.IndexFlatIP(dim), False
    else:
        print(f"  Index type : IndexIVFFlat nlist={IVF_NLIST} (corpus={total_chunks:,})")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, IVF_NLIST, faiss.METRIC_INNER_PRODUCT)
        return index, True


def _load_index():
    import faiss
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNK_INDEX_PATH, encoding="utf-8") as f:
        chunk_index = json.load(f)
    # Restore probe setting on IVFFlat indexes
    if hasattr(index, "nprobe"):
        index.nprobe = IVF_NPROBE
    return index, chunk_index


def _save_index(index, chunk_index: list):
    import faiss
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNK_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_index, f)

# ── Phase 4 — Embed & index ──────────────────────────────────────────────────

def run_phase4(model_name: str = EMBED_MODEL) -> int:
    print(f"\n{'='*60}")
    print("  Phase 4 — Embeddings & FAISS Vector Index")
    print(f"{'='*60}")

    try:
        import torch
        torch.set_num_threads(os.cpu_count() or 4)
    except Exception:
        pass

    import faiss
    from sentence_transformers import SentenceTransformer

    conn = _conn()
    _setup_log_table(conn)

    pending = _fetch_pending(conn)
    total   = len(pending)
    print(f"  Pending chunks : {total:,}")
    if total == 0:
        print("  Nothing to embed — already up to date.")
        conn.close()
        return 0

    # Load model
    print(f"  Loading model  : {model_name}")
    t0    = time.time()
    model = SentenceTransformer(model_name)
    dim   = model.get_sentence_embedding_dimension()
    print(f"  Model ready    : {time.time()-t0:.1f}s  dim={dim}")

    # Load or create index
    if Path(FAISS_INDEX_PATH).exists() and Path(CHUNK_INDEX_PATH).exists():
        index, chunk_index = _load_index()
        print(f"  Loaded index   : {index.ntotal:,} existing vectors")
        needs_training = False   # existing index is already trained
    else:
        index, needs_training = _build_index(dim, total)
        chunk_index = []

    # If IVFFlat and not yet trained, collect all vectors first then train
    if needs_training:
        print("  Training IVFFlat index on full corpus …")
        all_vecs = []
        for start in range(0, total, EMBED_BATCH_SIZE):
            batch = pending[start : start + EMBED_BATCH_SIZE]
            texts = [r["clean_text"][:EMBED_MAX_CHARS] for r in batch]
            vecs  = model.encode(texts, batch_size=EMBED_BATCH_SIZE,
                                 normalize_embeddings=True, convert_to_numpy=True).astype("float32")
            all_vecs.append(vecs)
            done = min(start + EMBED_BATCH_SIZE, total)
            pct  = done / total * 100
            bar  = "#" * int(pct // 5) + "-" * (20 - int(pct // 5))
            print(f"  [encode {bar}] {pct:5.1f}%", end="\r", flush=True)
        print()
        all_vecs = np.vstack(all_vecs)
        index.train(all_vecs)
        print(f"  Training done.")

        # Now add all at once
        log_rows = []
        for i, row in enumerate(pending):
            chunk_index.append(row["chunk_id"])
            log_rows.append((row["chunk_id"], i))
        index.add(all_vecs)
        _log(conn, log_rows)
        inserted = total

    else:
        # Flat index or already-trained IVFFlat: add batch by batch
        t0 = time.time()
        inserted = 0
        log_rows = []

        for start in range(0, total, EMBED_BATCH_SIZE):
            batch = pending[start : start + EMBED_BATCH_SIZE]
            texts = [r["clean_text"][:EMBED_MAX_CHARS] for r in batch]
            ids   = [r["chunk_id"] for r in batch]
            vecs  = model.encode(texts, batch_size=EMBED_BATCH_SIZE,
                                 normalize_embeddings=True, convert_to_numpy=True).astype("float32")
            faiss_start = len(chunk_index)
            index.add(vecs)
            chunk_index.extend(ids)
            log_rows.extend((cid, faiss_start + i) for i, cid in enumerate(ids))
            inserted += len(batch)

            if len(log_rows) >= 1000:
                _log(conn, log_rows)
                log_rows.clear()

            done = min(start + EMBED_BATCH_SIZE, total)
            pct  = done / total * 100
            bar  = "#" * int(pct // 5) + "-" * (20 - int(pct // 5))
            print(f"  [{bar}] {pct:5.1f}%  {done}/{total}", end="\r", flush=True)

        print()
        _log(conn, log_rows)

    _save_index(index, chunk_index)
    conn.close()

    print(f"\n{'='*60}")
    print("  Phase 4 Summary")
    print(f"{'='*60}")
    print(f"  Embedded       : {inserted:,} chunks")
    print(f"  FAISS total    : {index.ntotal:,} vectors")
    print(f"  Index file     : {FAISS_INDEX_PATH}")
    print(f"  Ready for Phase 5 / 6  (search & duplicate detection)")
    print(f"{'='*60}\nPhase 4 complete")
    return inserted

# ── Phase 5 — Search ─────────────────────────────────────────────────────────

def search(
    query: str,
    top_k: int = 10,
    model_name: str = EMBED_MODEL,
    # metadata filters (all optional)
    language: str = None, extension: str = None, category: str = None,
    name: str = None, folder: str = None, path: str = None,
    source_archive: str = None,
    created_after: str = None, created_before: str = None,
    modified_after: str = None, modified_before: str = None,
) -> pd.DataFrame:
    """
    Semantic search over the FAISS index with optional metadata pre-filtering.
    Reads chunks table (ph3) for metadata; reads FAISS index (ph4) for vectors.
    """
    from sentence_transformers import SentenceTransformer

    index, chunk_index = _load_index()

    q_vec = SentenceTransformer(model_name).encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    # Optional metadata pre-filter via PG
    allowed_ids = None
    filters, params = [], []
    if language:        filters.append("language = %s");        params.append(language)
    if extension:       filters.append("extension = %s");       params.append(extension)
    if category:        filters.append("category = %s");        params.append(category)
    if name:            filters.append("name = %s");            params.append(name)
    if folder:          filters.append("folder = %s");          params.append(folder)
    if path:            filters.append("path = %s");            params.append(path)
    if source_archive:  filters.append("source_archive = %s");  params.append(source_archive)
    if created_after:   filters.append("created_time >= %s");   params.append(created_after)
    if created_before:  filters.append("created_time <= %s");   params.append(created_before)
    if modified_after:  filters.append("modified_time >= %s");  params.append(modified_after)
    if modified_before: filters.append("modified_time <= %s");  params.append(modified_before)

    if filters:
        conn = _conn()
        with conn.cursor() as cur:
            cur.execute(f"SELECT chunk_id FROM chunks WHERE {' AND '.join(filters)}", params)
            allowed_ids = {r[0] for r in cur.fetchall()}
        conn.close()
        if not allowed_ids:
            return pd.DataFrame()

    search_k = min(top_k * 20 if allowed_ids else top_k, index.ntotal)
    scores, faiss_rows = index.search(q_vec, search_k)

    hits = []
    for score, row_idx in zip(scores[0], faiss_rows[0]):
        if row_idx < 0:
            continue
        cid = chunk_index[row_idx]
        if allowed_ids and cid not in allowed_ids:
            continue
        hits.append((cid, float(score)))
        if len(hits) >= top_k:
            break

    if not hits:
        return pd.DataFrame()

    score_map = {cid: s for cid, s in hits}
    conn = _conn()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM chunks WHERE chunk_id = ANY(%s)", ([h[0] for h in hits],))
        rows = cur.fetchall()
    conn.close()

    df = pd.DataFrame(rows)
    df["score"] = df["chunk_id"].map(score_map)
    return df.sort_values("score", ascending=False).reset_index(drop=True)

# ── Phase 5 — Duplicate helpers (used by ph6) ────────────────────────────────

def find_duplicate_chunks(similarity_threshold: float = 0.95, store: bool = False) -> pd.DataFrame:
    """
    Batch FAISS search to find near-duplicate chunk pairs.
    Called by ph6 — result feeds into file-level grouping.
    """
    index, chunk_index = _load_index()
    total = index.ntotal
    if total == 0:
        return pd.DataFrame(columns=["chunk_id_1", "chunk_id_2", "similarity_score"])

    print(f"  Scanning {total:,} vectors (threshold≥{similarity_threshold}) …")
    all_vecs = np.zeros((total, index.d), dtype="float32")
    index.reconstruct_n(0, total, all_vecs)

    k = min(50, total)
    scores_all, rows_all = index.search(all_vecs, k)

    pairs: dict[tuple, float] = {}
    for anchor in range(total):
        a_id = chunk_index[anchor]
        for rank in range(1, k):
            nb = int(rows_all[anchor, rank])
            if nb < 0: break
            score = float(scores_all[anchor, rank])
            if score < similarity_threshold: break
            nb_id = chunk_index[nb]
            key = (min(a_id, nb_id), max(a_id, nb_id))
            if score > pairs.get(key, -1):
                pairs[key] = score

    if not pairs:
        return pd.DataFrame(columns=["chunk_id_1", "chunk_id_2", "similarity_score"])

    df = pd.DataFrame(
        [{"chunk_id_1": k[0], "chunk_id_2": k[1], "similarity_score": v} for k, v in pairs.items()]
    ).sort_values("similarity_score", ascending=False).reset_index(drop=True)

    if store:
        conn = _conn()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS duplicate_chunks (
                    chunk_id_1 TEXT, chunk_id_2 TEXT, similarity FLOAT,
                    detected_at TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (chunk_id_1, chunk_id_2)
                );
            """)
            execute_values(
                cur,
                "INSERT INTO duplicate_chunks (chunk_id_1, chunk_id_2, similarity) VALUES %s ON CONFLICT DO NOTHING",
                [(r.chunk_id_1, r.chunk_id_2, r.similarity_score) for r in df.itertuples()],
            )
        conn.commit(); conn.close()

    print(f"  Found {len(df):,} near-duplicate pairs")
    return df


def find_similar_chunks(chunk_id: str, similarity_threshold: float = 0.90, top_k: int = 20) -> pd.DataFrame:
    """Return chunks most similar to a given chunk_id."""
    index, chunk_index = _load_index()
    try:
        anchor_row = chunk_index.index(chunk_id)
    except ValueError:
        return pd.DataFrame()

    anchor_vec = np.zeros((1, index.d), dtype="float32")
    index.reconstruct(anchor_row, anchor_vec[0])

    k = min(top_k + 1, index.ntotal)
    scores, rows = index.search(anchor_vec, k)

    hits = []
    for score, row_idx in zip(scores[0], rows[0]):
        if row_idx < 0 or row_idx == anchor_row: continue
        s = float(score)
        if s < similarity_threshold: break
        hits.append((chunk_index[row_idx], s))
        if len(hits) >= top_k: break

    if not hits:
        return pd.DataFrame()

    score_map = {cid: s for cid, s in hits}
    conn = _conn()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM chunks WHERE chunk_id = ANY(%s)", ([h[0] for h in hits],))
        rows_pg = cur.fetchall()
    conn.close()

    df = pd.DataFrame(rows_pg)
    df["similarity_score"] = df["chunk_id"].map(score_map)
    return df.sort_values("similarity_score", ascending=False).reset_index(drop=True)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_phase4()