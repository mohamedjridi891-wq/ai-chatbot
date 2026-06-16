from pathlib import Path
import pandas as pd
import re
import os
import sys
import time
import logging
import unicodedata
import warnings
import threading
from html.parser import HTMLParser
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DB_URL")
logging.basicConfig(
    filename=os.getenv("CLEANING_ERRORS", "cleaning_errors.log"),
    level=logging.WARNING,
    format="%(asctime)s — %(levelname)s — %(message)s",
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
MIN_CHUNK_WORDS = int(os.getenv("MIN_CHUNK_WORDS", 20))
MIN_LANG_CHARS = int(os.getenv("MIN_LANG_CHARS", 50))

def _safe_print(*args, sep=" ", end="\n", file=None, flush=False):
    if file is None:
        file = sys.stdout
    text = sep.join(str(a) for a in args) + end
    try:
        file.write(text)
    except Exception:
        data = text.encode("utf-8", errors="backslashreplace")
        try:
            file.buffer.write(data)
        except Exception:
            sys.__stdout__.buffer.write(data)
    if flush:
        try:
            file.flush()
        except Exception:
            pass

print = _safe_print


def _try_import(module, pip_name=None):
    try:
        import importlib
        return importlib.import_module(module)
    except ImportError:
        logging.warning(f"Optional: {pip_name or module} not installed")
        print(f"  ⚠  Missing: {pip_name or module}  →  pip install {pip_name or module}")
        return None

langdetect_mod = _try_import("langdetect", "langdetect")

class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def get_text(self):
        return " ".join(self.parts)


def strip_html(text: str) -> str:
    try:
        s = _HTMLStripper()
        s.feed(text)
        return s.get_text()
    except Exception:
        return re.sub(r"<[^>]+>", " ", text)


def _safe_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)

_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$", re.I),
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^\s*confidential\s*$", re.I),
    re.compile(r"^\s*all rights reserved\s*\.?\s*$", re.I),
    re.compile(r"^\s*www\.")
]

_OCR_NOISE = [
    re.compile(r"[|]{2,}"),
    re.compile(r"[_]{4,}"),
    re.compile(r"\.{4,}"),
    re.compile(r"[^\S\n]{3,}"),
]


def _remove_boilerplate_lines(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(
        line for line in lines
        if not any(p.match(line) for p in _BOILERPLATE_PATTERNS)
    )


def _fix_encoding_artifacts(text: str) -> str:
    replacements = {
        "\x00": "",
        "\ufffd": "",
        "\u00a0": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\u2028": "\n",
        "\u2029": "\n",
        "\u00ad": "",
        "â€™": "'",
        "â€œ": '"',
        "Ã©": "é",
        "Ã¨": "è",
        "Ã ": "à",
        "Ã§": "ç",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def _normalize_unicode(text: str) -> str:
    try:
        return unicodedata.normalize("NFC", text)
    except Exception:
        return text


def _remove_ocr_noise(text: str) -> str:
    for pattern in _OCR_NOISE:
        text = pattern.sub(" ", text)
    return text


def clean_text(raw_text: str, extraction_method: str = "") -> str:
    if not raw_text or not isinstance(raw_text, str):
        return ""
    text = raw_text
    text = _fix_encoding_artifacts(text)
    html_methods = {"unstructured", "plain_text_read"}
    if any(m in extraction_method.lower() for m in html_methods):
        if re.search(r"<[a-zA-Z][^>]*>", text):
            text = strip_html(text)
    if "ocr" in extraction_method.lower():
        text = _remove_ocr_noise(text)
    text = _normalize_unicode(text)
    text = _remove_boilerplate_lines(text)
    text = _normalize_whitespace(text)
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if not text:
        return []
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if len(words) >= MIN_CHUNK_WORDS else []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) > 1:
        chunks = []
        current_words = []
        for para in paragraphs:
            para_words = para.split()
            if len(para_words) > chunk_size:
                if len(current_words) >= MIN_CHUNK_WORDS:
                    chunks.append(" ".join(current_words))
                for start in range(0, len(para_words), chunk_size - overlap):
                    sub = para_words[start:start + chunk_size]
                    if len(sub) >= MIN_CHUNK_WORDS:
                        chunks.append(" ".join(sub))
                current_words = para_words[-overlap:] if overlap else []
                continue
            if len(current_words) + len(para_words) > chunk_size:
                if len(current_words) >= MIN_CHUNK_WORDS:
                    chunks.append(" ".join(current_words))
                current_words = current_words[-overlap:] + para_words
            else:
                current_words.extend(para_words)
        if len(current_words) >= MIN_CHUNK_WORDS:
            chunks.append(" ".join(current_words))
        if chunks:
            return chunks
    chunks = []
    for start in range(0, len(words), chunk_size - overlap):
        chunk_words = words[start:start + chunk_size]
        if len(chunk_words) >= MIN_CHUNK_WORDS:
            chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def detect_language(text: str) -> str:
    if len(text) < MIN_LANG_CHARS:
        return ""
    if not langdetect_mod:
        return ""
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return ""


def _build_hash_index(rows: list[dict]) -> dict:
    seen = {}
    index = {}
    for i, row in enumerate(rows):
        h = str(row.get("hash", ""))
        if h and not h.startswith("error") and h not in seen:
            seen[h] = i
            index[i] = None
        elif h in seen:
            index[i] = seen[h]
        else:
            index[i] = None
    return index


def _clean_text_for_db(text):
    if not isinstance(text, str):
        return text
    text = text.replace("\x00", "")
    return text


def get_db_connection():
    if not DB_URL:
        raise RuntimeError("DB_URL not set. Please set DB_URL in your .env or environment before running Phase 3.")
    return psycopg2.connect(DB_URL)


def create_tables(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                chunk_id TEXT UNIQUE,
                extracted_content_id INTEGER NOT NULL REFERENCES extracted_content(id) ON DELETE CASCADE,
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                chunk_index INTEGER,
                chunk_total INTEGER,
                clean_text TEXT,
                clean_char_count INTEGER,
                clean_word_count INTEGER,
                language TEXT,
                phase1_status TEXT,
                clean_status TEXT,
                name TEXT,
                stem TEXT,
                extension TEXT,
                category TEXT,
                path TEXT,
                folder TEXT,
                depth INTEGER,
                size_mb REAL,
                size_bytes BIGINT,
                is_empty BOOLEAN,
                created_time TIMESTAMP,
                modified_time TIMESTAMP,
                access_time TIMESTAMP,
                hash TEXT,
                status TEXT,
                source_archive TEXT,
                extracted_text TEXT,
                extraction_method TEXT,
                extraction_status TEXT,
                ocr_applied BOOLEAN,
                language_hint TEXT
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_extracted_content_id ON chunks(extracted_content_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);")
    conn.commit()


def fetch_existing_clean_texts(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT hash, clean_text
            FROM chunks
            WHERE chunk_index = 0
              AND clean_status = 'SUCCESS'
              AND hash IS NOT NULL
              AND hash NOT LIKE 'error%'
            """
        )
        return {row[0]: row[1] for row in cur.fetchall() if row[0]}


def fetch_unprocessed_extracted_rows(conn):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                e.id AS extracted_content_id,
                f.id AS file_id,
                f.name,
                f.stem,
                f.extension,
                f.category,
                f.path,
                f.folder,
                f.depth,
                f.size_mb,
                f.size_bytes,
                f.is_empty,
                f.created_time,
                f.modified_time,
                f.access_time,
                f.hash,
                f.status,
                f.source_archive,
                e.extracted_text,
                e.char_count,
                e.word_count,
                e.extraction_method,
                e.extraction_status,
                e.ocr_applied,
                e.language_hint,
                e.phase1_status
            FROM extracted_content e
            JOIN files f ON e.file_id = f.id
            LEFT JOIN chunks c ON e.id = c.extracted_content_id
            WHERE c.extracted_content_id IS NULL
            ORDER BY CASE WHEN f.status = 'KEEP' THEN 0 ELSE 1 END, e.id
            """
        )
        return cur.fetchall()


def insert_chunk_rows(conn, rows):
    if not rows:
        return
    sql = """
        INSERT INTO chunks (
            chunk_id, chunk_index, chunk_total, clean_text,
            clean_char_count, clean_word_count, language,
            phase1_status, clean_status, file_id, extracted_content_id,
            name, stem, extension, category, path, folder,
            depth, size_mb, size_bytes, is_empty,
            created_time, modified_time, access_time,
            hash, status, source_archive,
            extracted_text, extraction_method, extraction_status,
            ocr_applied, language_hint
        ) VALUES %s
        ON CONFLICT (chunk_id) DO NOTHING
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()


def run_phase3(extracted_csv="extracted_content.csv", output_csv="cleaned_chunks.csv"):
    print(f"\n{'='*60}")
    print("  Phase 3 - Normalization, Cleaning & Chunking")
    print(f"{'='*60}")
    print("  Using PostgreSQL extracted_content/chunks tables")

    conn = get_db_connection()
    create_tables(conn)

    rows = fetch_unprocessed_extracted_rows(conn)
    total = len(rows)
    print(f"  {total:,} rows loaded from extracted_content table\n")

    if total == 0:
        print("  No new rows to clean.")
        conn.close()
        return 0

    hash_index = _build_hash_index(rows)
    dupes_skipped = sum(1 for v in hash_index.values() if v is not None)
    print(f"  Duplicate detection: {dupes_skipped:,} files may reuse clean text")

    reuse_map = fetch_existing_clean_texts(conn)
    success = failed = skipped = 0
    total_chunks = 0
    total_words = 0
    language_counts = {}
    batch = []

    def flush_batch():
        nonlocal batch
        if not batch:
            return
        insert_chunk_rows(conn, batch)
        batch = []

    for i, row in enumerate(rows):
        raw_text = _safe_str(row.get("extracted_text", ""))
        ext_method = _safe_str(row.get("extraction_method", ""))
        ext_status = _safe_str(row.get("extraction_status", ""))
        extension = _safe_str(row.get("extension", "")).lower()
        file_hash = _safe_str(row.get("hash", ""))
        phase1_st = _safe_str(row.get("phase1_status", ""))

        if ext_status in ("FAILED", "SKIPPED") or not raw_text.strip():
            chunk_id = f"{row['file_id']}_0"
            batch.append((
                chunk_id,
                0,
                0,
                "",
                0,
                0,
                "",
                phase1_st,
                "SKIPPED",
                row["file_id"],
                row["extracted_content_id"],
                row.get("name", ""),
                row.get("stem", ""),
                row.get("extension", ""),
                row.get("category", ""),
                row.get("path", ""),
                row.get("folder", ""),
                row.get("depth", 0),
                row.get("size_mb", 0.0),
                row.get("size_bytes", 0),
                row.get("is_empty", False),
                row.get("created_time"),
                row.get("modified_time"),
                row.get("access_time"),
                row.get("hash", ""),
                row.get("status", ""),
                row.get("source_archive", ""),
                row.get("extracted_text", ""),
                row.get("extraction_method", ""),
                row.get("extraction_status", ""),
                row.get("ocr_applied", False),
                row.get("language_hint", ""),
            ))
            skipped += 1
            if len(batch) >= 100:
                flush_batch()
            continue

        if file_hash and file_hash in reuse_map:
            cleaned = reuse_map[file_hash]
        else:
            cleaned = clean_text(raw_text, ext_method)
            if file_hash and not file_hash.startswith("error"):
                reuse_map[file_hash] = cleaned

        if not cleaned.strip():
            chunk_id = f"{row['file_id']}_0"
            batch.append((
                chunk_id,
                0,
                0,
                "",
                0,
                0,
                "",
                phase1_st,
                "FAILED",
                row["file_id"],
                row["extracted_content_id"],
                row.get("name", ""),
                row.get("stem", ""),
                row.get("extension", ""),
                row.get("category", ""),
                row.get("path", ""),
                row.get("folder", ""),
                row.get("depth", 0),
                row.get("size_mb", 0.0),
                row.get("size_bytes", 0),
                row.get("is_empty", False),
                row.get("created_time"),
                row.get("modified_time"),
                row.get("access_time"),
                row.get("hash", ""),
                row.get("status", ""),
                row.get("source_archive", ""),
                row.get("extracted_text", ""),
                row.get("extraction_method", ""),
                row.get("extraction_status", ""),
                row.get("ocr_applied", False),
                row.get("language_hint", ""),
            ))
            failed += 1
            if len(batch) >= 100:
                flush_batch()
            continue

        chunks = chunk_text(cleaned) or [cleaned]
        for c_idx, chunk in enumerate(chunks):
            existing_lang = _safe_str(row.get("language_hint", "")).strip()
            language = existing_lang or detect_language(chunk)
            batch.append((
                f"{row['file_id']}_{c_idx}",
                c_idx,
                len(chunks),
                _clean_text_for_db(chunk),
                len(chunk),
                len(chunk.split()),
                language,
                phase1_st,
                "SUCCESS",
                row["file_id"],
                row["extracted_content_id"],
                row.get("name", ""),
                row.get("stem", ""),
                row.get("extension", ""),
                row.get("category", ""),
                row.get("path", ""),
                row.get("folder", ""),
                row.get("depth", 0),
                row.get("size_mb", 0.0),
                row.get("size_bytes", 0),
                row.get("is_empty", False),
                row.get("created_time"),
                row.get("modified_time"),
                row.get("access_time"),
                row.get("hash", ""),
                row.get("status", ""),
                row.get("source_archive", ""),
                row.get("extracted_text", ""),
                row.get("extraction_method", ""),
                row.get("extraction_status", ""),
                row.get("ocr_applied", False),
                row.get("language_hint", ""),
            ))
            total_chunks += 1
            total_words += len(chunk.split())
            if language:
                language_counts[language] = language_counts.get(language, 0) + 1
        success += 1
        if len(batch) >= 100:
            flush_batch()

        if (i + 1) % 100 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            bar = "#" * int(pct // 5) + "-" * (20 - int(pct // 5))
            print(
                f"  [{bar}] {pct:5.1f}%  done={success} failed={failed} skipped={skipped} chunks={total_chunks}",
                end="\r"
            )
    flush_batch()
    print()
    conn.close()

    print(f"\n{'='*60}")
    print("  Phase 3 Summary")
    print(f"{'='*60}")
    print(f"  Source files       : {total:,}")
    print(f"  Cleaned          : {success:,}  ({(success/total*100) if total else 0:.1f}%)")
    print(f"  Failed           : {failed:,}   ({(failed/total*100) if total else 0:.1f}%)")
    print(f"  Skipped          : {skipped:,}  ({(skipped/total*100) if total else 0:.1f}%)")
    print(f"  Dupes reused      : {dupes_skipped:,}")
    print(f"\n  Total chunks       : {total_chunks:,}")
    if total_words:
        print(f"  Total words        : {total_words:,}")
    if language_counts:
        print(f"\n  Languages:")
        for lang, count in sorted(language_counts.items(), key=lambda x: -x[1]):
            print(f"    {lang}: {count}")
    print(f"\n  Output -> chunks table")
    print("  Ready for Phase 4 (Embeddings)")
    print(f"{'='*60}")
    print("\nPhase 3 complete")
    return total


if __name__ == "__main__":
    run_phase3()
