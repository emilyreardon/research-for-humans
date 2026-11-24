#!/usr/bin/env python3
"""
Step 2: Build enriched passage collection.
Source: isls_papers_grobid_candidate (106,065 chunks)
Target: isls_papers_enriched
"""

import os
import chromadb
import requests
import json
import time
import logging
import sys
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────
CHROMA_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
OLLAMA_URL    = "http://localhost:11434"
EMBED_MODEL   = "nomic-embed-text"
SOURCE_COLL   = "isls_papers_grobid_candidate"
TARGET_COLL   = "isls_papers_enriched"
BATCH_SIZE    = 500    # chunks fetched from source per page
EMBED_BATCH   = 32     # passages sent per embedding call (if batch available)
LOG_INTERVAL  = 1000   # log progress every N chunks
RETRY_LIMIT   = 3      # embedding attempts before marking a chunk as failed
RETRY_SLEEP   = 5      # seconds between retry attempts
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler("precompute_enriched.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def build_label(meta: dict) -> str:
    """
    Construct context label from chunk metadata.
    Edit the field names below to match the schema confirmed in pre-flight.
    Missing fields are omitted gracefully — no placeholders inserted.
    """
    # ── Edit these field names to match your metadata schema ──────────────────
    title   = meta.get("title", "")         # e.g. "title" or "paper_title"
    authors = meta.get("authors", "")        # e.g. "authors" — used as-is
    year    = meta.get("year", "")           # e.g. "year" or "pub_year"
    section = meta.get("section_type", "")   # e.g. "section_type"
    # ─────────────────────────────────────────────────────────────────────────

    parts = []
    if title:
        yr_str = f" ({year})" if year else ""
        parts.append(f"{title}{yr_str}")
    if authors:
        parts.append(str(authors))
    if section:
        parts.append(str(section).title())

    if not parts:
        return ""
    return " — ".join(parts) + ": "


def detect_batch_embed() -> bool:
    """Return True if Ollama /api/embed accepts array input."""
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": ["test"]},
            timeout=10,
        )
        return r.status_code == 200 and "embeddings" in r.json()
    except Exception:
        return False


def embed_single_with_retry(text: str) -> list[float]:
    """Embed one text with retry-with-backoff. Raises on final failure."""
    last_exc = None
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": f"search_document: {text}"},
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["embedding"]
        except Exception as e:
            last_exc = e
            if attempt < RETRY_LIMIT:
                log.warning(f"Embed attempt {attempt} failed: {e} — retrying in {RETRY_SLEEP}s")
                time.sleep(RETRY_SLEEP)
    raise last_exc


def embed_batch_with_retry(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with retry-with-backoff. Raises on final failure."""
    prefixed = [f"search_document: {t}" for t in texts]
    last_exc = None
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": prefixed},
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["embeddings"]
        except Exception as e:
            last_exc = e
            if attempt < RETRY_LIMIT:
                log.warning(f"Batch embed attempt {attempt} failed: {e} — retrying in {RETRY_SLEEP}s")
                time.sleep(RETRY_SLEEP)
    raise last_exc


def main(test_mode: bool = False):
    start_time = time.time()
    log.info("=== precompute_enriched.py starting ===")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    src    = client.get_collection(SOURCE_COLL)
    total  = src.count()
    log.info(f"Source collection: {SOURCE_COLL} — {total} chunks")

    # Create or resume target collection
    try:
        tgt = client.get_collection(TARGET_COLL)
        existing_count = tgt.count()
        log.info(f"Resuming — target has {existing_count} docs already")
    except Exception:
        tgt = client.create_collection(
            TARGET_COLL,
            metadata={"hnsw:space": "cosine"},
        )
        existing_count = 0
        log.info(f"Created target collection: {TARGET_COLL}")

    # Build set of IDs already in target to skip on resume
    if existing_count > 0:
        log.info("Loading existing IDs for resume...")
        existing_ids = set()
        skip_offset = 0
        while skip_offset < existing_count:
            batch = tgt.get(limit=BATCH_SIZE, offset=skip_offset, include=[])
            existing_ids.update(batch["ids"])
            skip_offset += len(batch["ids"])
        log.info(f"Will skip {len(existing_ids)} already-processed chunks")
    else:
        existing_ids = set()

    use_batch_embed = detect_batch_embed()
    log.info(f"Batch embedding: {'enabled' if use_batch_embed else 'disabled (single-call mode)'}")

    limit = 50 if test_mode else total
    processed = 0
    errors = 0
    offset = 0

    while offset < limit:
        fetch_size = min(BATCH_SIZE, limit - offset)
        page = src.get(
            limit=fetch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        if not page["ids"]:
            break

        # Filter to chunks not yet in target
        new_ids, new_docs, new_metas = [], [], []
        for chunk_id, doc, meta in zip(page["ids"], page["documents"], page["metadatas"]):
            if chunk_id in existing_ids:
                continue
            label = build_label(meta)
            enriched_text = label + doc if label else doc
            new_meta = dict(meta)
            new_meta["context_enriched"] = True
            new_ids.append(chunk_id)
            new_docs.append(enriched_text)
            new_metas.append(new_meta)

        if not new_ids:
            offset += fetch_size
            continue

        # Embed in sub-batches with retry
        all_embeddings = []
        sub_batch_size = EMBED_BATCH if use_batch_embed else 1

        for i in range(0, len(new_docs), sub_batch_size):
            sub_texts = new_docs[i : i + sub_batch_size]
            sub_ids   = new_ids[i : i + sub_batch_size]
            try:
                if use_batch_embed:
                    vecs = embed_batch_with_retry(sub_texts)
                else:
                    vecs = [embed_single_with_retry(t) for t in sub_texts]
                all_embeddings.extend(vecs)
            except Exception as e:
                log.error(
                    f"Embedding failed after {RETRY_LIMIT} attempts at offset {offset + i} "
                    f"(chunk IDs: {sub_ids}): {e} — skipping {len(sub_texts)} chunks"
                )
                errors += len(sub_texts)
                all_embeddings.extend([None] * len(sub_texts))

        # Filter out permanently failed embeddings before insert
        valid = [
            (cid, doc, meta, emb)
            for cid, doc, meta, emb in zip(new_ids, new_docs, new_metas, all_embeddings)
            if emb is not None
        ]
        if valid:
            v_ids, v_docs, v_metas, v_embs = zip(*valid)
            tgt.add(
                ids=list(v_ids),
                documents=list(v_docs),
                metadatas=list(v_metas),
                embeddings=list(v_embs),
            )

        processed += len(valid)
        offset += fetch_size

        if processed % LOG_INTERVAL < fetch_size or test_mode:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (total - processed) / rate if rate > 0 else 0
            log.info(
                f"Progress: {processed:,}/{total:,} chunks | "
                f"errors: {errors} | "
                f"elapsed: {elapsed/3600:.1f}h | "
                f"ETA: {remaining/3600:.1f}h"
            )

        if test_mode and processed >= 50:
            break

    elapsed_total = time.time() - start_time
    log.info(
        f"=== Done === processed: {processed:,} | errors: {errors} | "
        f"elapsed: {elapsed_total/3600:.2f}h"
    )

    # Verify final count
    final_count = tgt.count()
    log.info(f"Target collection count: {final_count} (source: {total})")
    if final_count < total - errors:
        log.warning(f"Count mismatch — expected ~{total - errors}, got {final_count}")
    else:
        log.info("Count verified ✓")


if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    if test_mode:
        print("=== TEST MODE: processing first 50 chunks only ===")
    main(test_mode=test_mode)
