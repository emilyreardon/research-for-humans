#!/usr/bin/env python3
"""
Step 1: Build paper abstract catalog.
Source:  isls_papers_grobid_candidate
Target:  isls_papers_abstract_catalog (one document per paper)
"""

import chromadb
import requests
import os
import sys
import logging
import time

HERE        = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(HERE, "chroma_db")
OLLAMA_URL  = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
SOURCE_COLL = "isls_papers_grobid_candidate"
TARGET_COLL = "isls_papers_abstract_catalog"
BATCH_SIZE  = 500
MISSING_LOG = os.path.join(HERE, "precompute_abstracts_missing.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(HERE, "precompute_abstracts.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def _embed(text):
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def detect_batch_embed():
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": ["test"]},
            timeout=10,
        )
        return r.status_code == 200 and "embeddings" in r.json()
    except Exception:
        return False


def embed_batch(texts):
    r = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": [f"search_document: {t}" for t in texts]},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embeddings"]


def main():
    start = time.time()
    log.info("=== precompute_abstracts.py starting ===")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Bail out if target already exists — do not overwrite
    existing = [c.name for c in client.list_collections()]
    if TARGET_COLL in existing:
        log.warning(f"Collection '{TARGET_COLL}' already exists ({client.get_collection(TARGET_COLL).count()} docs). Exiting without changes.")
        sys.exit(0)

    src   = client.get_collection(SOURCE_COLL)
    total = src.count()
    log.info(f"Source: {SOURCE_COLL} — {total} chunks")

    # ── Pass 1: collect all abstract chunks, grouped by paper_id ─────────────
    log.info("Pass 1: scanning for abstract chunks...")
    abstracts_by_paper = {}   # paper_id -> list of (chunk_idx, text, meta)
    all_paper_meta     = {}   # paper_id -> representative metadata (paper-level fields)
    offset = 0
    while offset < total:
        page = src.get(limit=BATCH_SIZE, offset=offset, include=["documents", "metadatas"])
        if not page["ids"]:
            break
        for doc, meta in zip(page["documents"], page["metadatas"]):
            pid = meta.get("paper_id", "")
            if not pid:
                offset += 1
                continue
            # Track paper-level metadata regardless of section
            if pid not in all_paper_meta:
                all_paper_meta[pid] = meta
            # Collect abstract chunks
            if meta.get("section_type") == "abstract":
                idx = meta.get("chunk_idx", 0)
                abstracts_by_paper.setdefault(pid, []).append((idx, doc, meta))
        offset += len(page["ids"])

    log.info(f"Pass 1 complete: {len(all_paper_meta)} unique papers, "
             f"{len(abstracts_by_paper)} have abstract chunks")

    # ── Find papers with no abstract ─────────────────────────────────────────
    missing = [pid for pid in all_paper_meta if pid not in abstracts_by_paper]
    log.info(f"Papers with no abstract: {len(missing)}")
    if missing:
        with open(MISSING_LOG, "w", encoding="utf-8") as f:
            for pid in sorted(missing):
                m = all_paper_meta[pid]
                f.write(f"{pid}\t{m.get('title','')}\t{m.get('authors','')}\t{m.get('year','')}\n")
        log.info(f"Missing abstracts logged to {MISSING_LOG}")

    # ── Build one document per paper ─────────────────────────────────────────
    # Sort multi-chunk abstracts by chunk_idx then join with space
    papers = []  # list of (paper_id, abstract_text, metadata)
    for pid, chunks in abstracts_by_paper.items():
        chunks_sorted = sorted(chunks, key=lambda x: x[0])
        text = " ".join(c[1] for c in chunks_sorted).strip()
        # Use the meta from the first chunk but drop chunk-level fields
        base_meta = dict(chunks_sorted[0][2])
        for drop in ("chunk_idx", "chunk_type", "section", "section_type"):
            base_meta.pop(drop, None)
        base_meta["source"] = SOURCE_COLL
        papers.append((pid, text, base_meta))

    log.info(f"Papers to insert: {len(papers)}")

    # ── Create target collection ──────────────────────────────────────────────
    tgt = client.create_collection(TARGET_COLL, metadata={"hnsw:space": "cosine"})
    log.info(f"Created collection: {TARGET_COLL}")

    use_batch = detect_batch_embed()
    log.info(f"Batch embedding: {'enabled' if use_batch else 'disabled'}")

    # ── Embed and insert in batches ───────────────────────────────────────────
    inserted = 0
    errors   = 0
    EMBED_BATCH = 32

    for batch_start in range(0, len(papers), BATCH_SIZE):
        batch = papers[batch_start : batch_start + BATCH_SIZE]
        ids    = [p[0] for p in batch]
        texts  = [p[1] for p in batch]
        metas  = [p[2] for p in batch]

        # Embed in sub-batches
        all_embeddings = []
        for i in range(0, len(texts), EMBED_BATCH):
            sub = texts[i : i + EMBED_BATCH]
            sub_ids = ids[i : i + EMBED_BATCH]
            try:
                if use_batch:
                    vecs = embed_batch(sub)
                else:
                    vecs = [_embed(f"search_document: {t}") for t in sub]
                all_embeddings.extend(vecs)
            except Exception as e:
                log.error(f"Embedding error at batch offset {batch_start + i} "
                          f"(IDs: {sub_ids}): {e} — skipping {len(sub)} papers")
                errors += len(sub)
                all_embeddings.extend([None] * len(sub))

        # Filter failed embeddings
        valid = [(cid, txt, meta, emb)
                 for cid, txt, meta, emb in zip(ids, texts, metas, all_embeddings)
                 if emb is not None]
        if valid:
            v_ids, v_docs, v_metas, v_embs = zip(*valid)
            tgt.add(
                ids=list(v_ids),
                documents=list(v_docs),
                metadatas=list(v_metas),
                embeddings=list(v_embs),
            )
        inserted += len(valid)

        elapsed = time.time() - start
        rate = inserted / elapsed if elapsed > 0 else 0
        remaining = (len(papers) - inserted) / rate if rate > 0 else 0
        log.info(f"Progress: {inserted}/{len(papers)} papers | errors: {errors} | "
                 f"elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m")

    elapsed_total = time.time() - start
    log.info(f"=== Done === inserted: {inserted} | skipped (no abstract): {len(missing)} | "
             f"errors: {errors} | elapsed: {elapsed_total/60:.1f}m")

    final = tgt.count()
    log.info(f"Final collection count: {final}")
    if final != inserted:
        log.warning(f"Count mismatch: inserted={inserted}, collection={final}")
    else:
        log.info("Count verified ✓")

    print(f"\nTotal papers added: {inserted}")
    print(f"Total papers skipped (no abstract): {len(missing)}")
    if missing:
        print(f"Missing abstracts log: {MISSING_LOG}")


if __name__ == "__main__":
    main()
