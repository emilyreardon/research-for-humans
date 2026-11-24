#!/usr/bin/env python3
"""
Step 3: Extract tension candidates from the ISLS abstract catalog.
Runs unattended end-to-end. Checkpoints continuously. Resumes on restart. Pushes to GitHub when done.

Configuration discovered from app.py — no values guessed.
"""

import json, os, sys, time, re, random, math, subprocess, traceback, datetime
import numpy as np
import requests
import chromadb
from sklearn.cluster import KMeans

# ── Config discovered from app.py ────────────────────────────────────────────
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(REPO_ROOT, "chroma_db")
CATALOG_COL = "isls_papers_abstract_catalog"
OLMO_URL    = "http://localhost:11434/api/generate"
OLMO_MODEL  = "olmo3-instruct"

STEP3_DIR   = os.path.join(REPO_ROOT, "outputs", "step3")
OUTPUTS_DIR = os.path.join(REPO_ROOT, "outputs")

# Clustering
K_CLUSTERS       = 150
RANDOM_STATE     = 42
MIN_CLUSTER_SIZE = 4
PILOT_N          = 10

# Generation
PRIMARY_BATCH  = 12
FALLBACK_BATCH = 6
GEN_OPTIONS    = {"temperature": 0.2, "num_ctx": 16384, "num_predict": 1200}
CALL_TIMEOUT   = 180
MAX_RETRIES    = 2

ALLOWED_TYPES = {"opposition", "double_edge", "incompatible_operationalization", "convergence"}

# ── Prompt templates ─────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """\
You are reading {n} research paper abstracts from the learning sciences. They share a subject area. Read all of them, then identify up to 2 cases where papers in this set stand in a genuine relation to each other.

The relation must be one of these four types:
- opposition: two papers reach findings or positions that conflict
- double_edge: the same factor or intervention is shown to help in one paper and to harm or cost in another
- incompatible_operationalization: two papers define or measure the same concept in ways that cannot both hold
- convergence: two papers using clearly different methods or studying clearly different contexts arrive at the same finding. Routine agreement between similar studies does not count.

Rules:
- Every claim you attribute to a paper must actually be stated in that paper's abstract. Do not infer beyond what is written.
- For each pole, copy one sentence from the abstract verbatim as the snippet. Copy it exactly.
- If no genuine relation exists in this set, return an empty list. That is a correct and common answer. Do not force a relation.

Return JSON only, in exactly this structure:
{{"candidates": [{{"relation_statement": "<one sentence naming the specific relation between the two findings>", "relation_type": "<one of the four types>", "poles": [{{"paper_id": "<id>", "claim": "<one sentence stating what this paper found, specific to this paper>", "snippet": "<verbatim sentence copied from this paper's abstract>"}}, {{"paper_id": "<id>", "claim": "<one sentence>", "snippet": "<verbatim sentence>"}}]}}]}}

The abstracts:

{abstracts}"""

FALLBACK_PROSE_PROMPT = """\
You are reading {n} research paper abstracts from the learning sciences. They share a subject area.

Read all of them. In plain prose (2–4 sentences), describe what these papers disagree about or unexpectedly agree on. Focus on genuine tensions — conflicting findings, the same factor shown to help in one context and hurt in another, incompatible definitions of the same concept, or very different methods arriving at the same conclusion. If you see nothing genuine, say so.

The abstracts:

{abstracts}"""

FALLBACK_JSON_PROMPT = """\
You are reading {n} research paper abstracts from the learning sciences.

A preliminary reading noted the following possible relations:
{prose}

Now read the abstracts carefully and produce the structured JSON output. Apply the rules strictly.

The relation must be one of these four types:
- opposition: two papers reach findings or positions that conflict
- double_edge: the same factor or intervention is shown to help in one paper and to harm or cost in another
- incompatible_operationalization: two papers define or measure the same concept in ways that cannot both hold
- convergence: two papers using clearly different methods or studying clearly different contexts arrive at the same finding. Routine agreement between similar studies does not count.

Rules:
- Every claim you attribute to a paper must actually be stated in that paper's abstract. Do not infer beyond what is written.
- For each pole, copy one sentence from the abstract verbatim as the snippet. Copy it exactly.
- If no genuine relation exists in this set, return an empty list. That is a correct and common answer. Do not force a relation.

Return JSON only, in exactly this structure:
{{"candidates": [{{"relation_statement": "<one sentence naming the specific relation between the two findings>", "relation_type": "<one of the four types>", "poles": [{{"paper_id": "<id>", "claim": "<one sentence stating what this paper found, specific to this paper>", "snippet": "<verbatim sentence copied from this paper's abstract>"}}, {{"paper_id": "<id>", "claim": "<one sentence>", "snippet": "<verbatim sentence>"}}]}}]}}

The abstracts:

{abstracts}"""


# ── Utilities ────────────────────────────────────────────────────────────────

def norm_ws(s):
    return re.sub(r'\s+', ' ', s or '').strip()

def _words(text):
    """Lowercase word set, punctuation stripped — for fuzzy snippet check."""
    return set(re.sub(r'[^\w\s]', ' ', text.lower()).split())

def snippet_overlap(snippet, abstract):
    """Fraction of snippet words that appear anywhere in the abstract."""
    sw = _words(snippet)
    if not sw:
        return 0.0
    aw = _words(abstract)
    return len(sw & aw) / len(sw)

def cosine_sim(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a / na, b / nb))


# ── Data loading ─────────────────────────────────────────────────────────────

def load_catalog():
    print("Loading abstract catalog from ChromaDB...", flush=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(name=CATALOG_COL)
    count = col.count()
    print(f"  Count: {count}", flush=True)
    assert count == 7220, f"Expected 7220 documents, got {count}"
    result = col.get(limit=count, include=["embeddings", "documents", "metadatas"])
    ids   = result["ids"]
    docs  = result["documents"]
    embs  = np.array(result["embeddings"], dtype=np.float32)
    metas = result["metadatas"]
    print(f"  Loaded. Embedding shape: {embs.shape}", flush=True)
    # Build global lookup: paper_id → {document, metadata, embedding}
    lookup = {}
    for pid, doc, emb, meta in zip(ids, docs, embs, metas):
        lookup[pid] = {"paper_id": pid, "document": doc or "", "embedding": emb, "metadata": meta or {}}
    return ids, docs, embs, metas, lookup


# ── Clustering ───────────────────────────────────────────────────────────────

def cluster_papers(ids, embs):
    print(f"K-means clustering (k={K_CLUSTERS}, random_state={RANDOM_STATE})...", flush=True)
    km = KMeans(n_clusters=K_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(embs)
    centroids = km.cluster_centers_
    print("  Done.", flush=True)
    return labels.tolist(), centroids


def build_cluster_assignments(ids, labels, centroids):
    raw = {}
    for pid, lbl in zip(ids, labels):
        lbl = int(lbl)
        if lbl not in raw:
            raw[lbl] = []
        raw[lbl].append(pid)

    valid, skipped = {}, {}
    for cid, pids in raw.items():
        if len(pids) < MIN_CLUSTER_SIZE:
            skipped[str(cid)] = pids
        else:
            valid[str(cid)] = {
                "paper_ids": pids,
                "centroid": centroids[cid].tolist(),
                "count": len(pids)
            }
    return valid, skipped


# ── Sampling ─────────────────────────────────────────────────────────────────

def sample_cluster(paper_ids, centroid, lookup, n):
    """Max-min diversity: start nearest centroid, greedily add most distant."""
    centroid = np.asarray(centroid, dtype=np.float32)
    if len(paper_ids) <= n:
        return list(paper_ids)

    sims = [cosine_sim(lookup[pid]["embedding"], centroid) for pid in paper_ids]
    start = int(np.argmax(sims))
    selected = [start]
    remaining = list(range(len(paper_ids)))
    remaining.remove(start)

    while len(selected) < n and remaining:
        best_i, best_dist = None, -1.0
        for i in remaining:
            min_dist = min(
                1.0 - cosine_sim(lookup[paper_ids[i]]["embedding"], lookup[paper_ids[j]]["embedding"])
                for j in selected
            )
            if min_dist > best_dist:
                best_dist, best_i = min_dist, i
        selected.append(best_i)
        remaining.remove(best_i)

    return [paper_ids[i] for i in selected]


def build_samples(valid_clusters, lookup, batch_size):
    print(f"Sampling clusters (batch_size={batch_size})...", flush=True)
    samples = {}
    for cid, cdata in valid_clusters.items():
        samples[cid] = sample_cluster(
            cdata["paper_ids"], cdata["centroid"], lookup, n=batch_size
        )
    print(f"  Done. {len(samples)} clusters sampled.", flush=True)
    return samples


# ── Prompt building ──────────────────────────────────────────────────────────

def format_abstracts(selected_ids, lookup):
    parts = []
    for i, pid in enumerate(selected_ids, 1):
        p = lookup.get(pid, {})
        meta = p.get("metadata", {}) or {}
        title = meta.get("title", "") or ""
        doc   = p.get("document", "") or ""
        parts.append(f"[{i}] PAPER_ID: {pid}\nTITLE: {title}\n{doc}")
    return "\n\n".join(parts)


# ── Ollama calls ─────────────────────────────────────────────────────────────

def ollama_call(prompt, use_json_format):
    payload = {
        "model": OLMO_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": GEN_OPTIONS,
    }
    if use_json_format:
        payload["format"] = "json"
    t0 = time.time()
    r = requests.post(OLMO_URL, json=payload, timeout=CALL_TIMEOUT)
    elapsed = time.time() - t0
    r.raise_for_status()
    return r.json()["response"], elapsed


def call_primary(selected_ids, lookup):
    abstracts = format_abstracts(selected_ids, lookup)
    prompt = EXTRACTION_PROMPT.format(n=len(selected_ids), abstracts=abstracts)
    return ollama_call(prompt, use_json_format=True)


def call_fallback(selected_ids, lookup):
    abstracts = format_abstracts(selected_ids, lookup)
    n = len(selected_ids)

    prose_prompt = FALLBACK_PROSE_PROMPT.format(n=n, abstracts=abstracts)
    prose, t1 = ollama_call(prose_prompt, use_json_format=False)

    json_prompt = FALLBACK_JSON_PROMPT.format(n=n, prose=prose.strip(), abstracts=abstracts)
    result, t2 = ollama_call(json_prompt, use_json_format=True)
    return result, t1 + t2


def parse_response(raw_text):
    """Returns list of candidate dicts, or None if unparseable."""
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict) and "candidates" in data and isinstance(data["candidates"], list):
            return data["candidates"]
    except Exception:
        pass
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data.get("candidates"), list):
                return data["candidates"]
        except Exception:
            pass
    return None


# ── Cluster run (with retry) ──────────────────────────────────────────────────

def run_cluster(cluster_id, selected_ids, lookup, use_fallback):
    """
    Returns a record dict:
      cluster_id, status, selected_ids, candidates, raw_output, elapsed, retries, error
    """
    total_elapsed = 0.0
    retries = 0
    last_raw = ""
    last_error = ""

    for attempt in range(MAX_RETRIES + 1):
        try:
            if use_fallback:
                raw, elapsed = call_fallback(selected_ids, lookup)
            else:
                raw, elapsed = call_primary(selected_ids, lookup)

            total_elapsed += elapsed
            last_raw = raw
            candidates = parse_response(raw)

            if candidates is None:
                last_error = "unparseable_json"
                retries += 1
                continue

            return {
                "cluster_id": cluster_id,
                "status": "ok",
                "selected_ids": selected_ids,
                "candidates": candidates,
                "raw_output": raw,
                "elapsed": round(total_elapsed, 2),
                "retries": retries,
                "error": None,
            }

        except requests.exceptions.Timeout:
            last_error = "timeout"
            total_elapsed += CALL_TIMEOUT
            retries += 1
        except Exception as e:
            last_error = str(e)[:200]
            total_elapsed += 0
            retries += 1

    return {
        "cluster_id": cluster_id,
        "status": "failed",
        "selected_ids": selected_ids,
        "candidates": [],
        "raw_output": last_raw,
        "elapsed": round(total_elapsed, 2),
        "retries": retries,
        "error": last_error,
    }


# ── Validation ────────────────────────────────────────────────────────────────

def validate_candidate(cand, selected_ids_set, lookup):
    """Returns (True, None) or (False, reason_string)."""
    if not isinstance(cand, dict):
        return False, "not_a_dict"
    poles = cand.get("poles")
    if not isinstance(poles, list) or len(poles) != 2:
        return False, "wrong_pole_count"

    rtype     = cand.get("relation_type", "")
    rstatement = cand.get("relation_statement", "")

    if not norm_ws(rstatement):
        return False, "empty_relation_statement"
    if rtype not in ALLOWED_TYPES:
        return False, f"invalid_relation_type"

    for pole in poles:
        pid     = pole.get("paper_id", "")
        claim   = pole.get("claim", "")
        snippet = pole.get("snippet", "")

        if not norm_ws(claim):
            return False, "empty_claim"
        if not norm_ws(snippet):
            return False, "empty_snippet"
        if pid not in selected_ids_set:
            return False, "paper_id_not_in_batch"
        if pid not in lookup:
            return False, "paper_id_not_found"

        abstract = norm_ws(lookup[pid]["document"])
        if snippet_overlap(snippet, abstract) < 0.60:
            return False, "snippet_low_overlap"

    return True, None


def gate_validate(cand, selected_ids_set, lookup):
    """Structure-only check for the pilot gate — no snippet quality requirement."""
    if not isinstance(cand, dict):
        return False
    poles = cand.get("poles")
    if not isinstance(poles, list) or len(poles) != 2:
        return False
    if cand.get("relation_type", "") not in ALLOWED_TYPES:
        return False
    if not norm_ws(cand.get("relation_statement", "")):
        return False
    for pole in poles:
        if pole.get("paper_id", "") not in selected_ids_set:
            return False
        if not norm_ws(pole.get("claim", "")):
            return False
        if not norm_ws(pole.get("snippet", "")):
            return False
    return True


# ── Gate check ────────────────────────────────────────────────────────────────

def gate_check(results, lookup):
    parseable = sum(1 for r in results if r["status"] == "ok")
    total     = len(results)

    raw_candidates, accepted = [], []
    for r in results:
        if r["status"] != "ok":
            continue
        sid_set = set(r["selected_ids"])
        for c in r["candidates"]:
            raw_candidates.append(c)
            if gate_validate(c, sid_set, lookup):
                accepted.append(c)

    parseable_rate = parseable / total if total else 0
    n_raw   = len(raw_candidates)
    n_ok    = len(accepted)
    attr_rate = n_ok / n_raw if n_raw > 0 else 0.0

    scores = {
        "parseable":        parseable,
        "total_calls":      total,
        "parseable_rate":   round(parseable_rate, 3),
        "total_candidates": n_raw,
        "accepted":         n_ok,
        "attribution_rate": round(attr_rate, 3),
        "gate_parseable":   parseable >= 7,
        "gate_candidates":  n_raw >= 1,
        "gate_attribution": n_raw == 0 or attr_rate >= 0.70,
    }
    passed = scores["gate_parseable"] and scores["gate_candidates"] and scores["gate_attribution"]
    return passed, scores


# ── Checkpoint helpers ────────────────────────────────────────────────────────

CHECKPOINT = os.path.join(STEP3_DIR, "raw_candidates.jsonl")

def load_done_clusters():
    done = {}
    if not os.path.exists(CHECKPOINT):
        return done
    with open(CHECKPOINT) as f:
        for line in f:
            try:
                rec = json.loads(line)
                done[str(rec["cluster_id"])] = rec
            except Exception:
                pass
    return done

def append_record(rec):
    with open(CHECKPOINT, "a") as f:
        f.write(json.dumps(rec) + "\n")

def rewrite_checkpoint(records):
    with open(CHECKPOINT, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ── Git helpers ───────────────────────────────────────────────────────────────

def git_commit_push(message, paths):
    os.chdir(REPO_ROOT)
    existing = [p for p in paths if os.path.exists(p)]
    if not existing:
        print("Nothing to commit.", flush=True)
        return
    subprocess.run(["git", "add"] + existing, check=True)
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        capture_output=True
    )
    if result.returncode == 0:
        print("No staged changes — nothing to commit.", flush=True)
        return
    subprocess.run(["git", "commit", "-m", message], check=True)
    subprocess.run(["git", "push"], check=True)
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    print(f"Commit hash: {out.stdout.strip()}", flush=True)


# ── Failure report ────────────────────────────────────────────────────────────

def write_failure_report(scores_by_config, raw_outputs_by_config, pilot_ids):
    lines = [
        "# Step 3 Pilot Failure Report",
        f"Generated: {datetime.datetime.now().isoformat()}",
        "",
        "Both configurations failed the pilot gate. Full run was not started.",
        "",
    ]
    for cfg, scores in scores_by_config.items():
        passed = scores["gate_parseable"] and scores["gate_candidates"] and scores["gate_attribution"]
        lines += [
            f"## {cfg.capitalize()} configuration — {'PASSED' if passed else 'FAILED'}",
            "```json",
            json.dumps(scores, indent=2),
            "```",
            "",
            f"### Failure reasons:",
        ]
        if not scores["gate_parseable"]:
            lines.append(f"- gate_parseable: {scores['parseable']}/{scores['total_calls']} parseable (need ≥7)")
        if not scores["gate_candidates"]:
            lines.append(f"- gate_candidates: {scores['total_candidates']} candidates (need ≥1)")
        if not scores["gate_attribution"]:
            lines.append(f"- gate_attribution: {scores['attribution_rate']:.1%} (need ≥70%)")
        lines.append("")

    for cfg, raw_outputs in raw_outputs_by_config.items():
        lines += [f"## {cfg.capitalize()} raw outputs", ""]
        for cid, raw in raw_outputs.items():
            lines += [f"### Cluster {cid}", "```", raw[:3000], "```", ""]

    path = os.path.join(OUTPUTS_DIR, "step3_pilot_failure.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    os.makedirs(STEP3_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("=== Step 3: Extract Tension Candidates ===", flush=True)
    print(f"Repo: {REPO_ROOT}  Model: {OLMO_MODEL}  Collection: {CATALOG_COL}", flush=True)

    # ── Task 0: Load collection ───────────────────────────────────────────────
    ids, docs, embs, metas, lookup = load_catalog()

    # ── Task 1: Cluster ───────────────────────────────────────────────────────
    assignments_path = os.path.join(STEP3_DIR, "cluster_assignments.json")

    if os.path.exists(assignments_path):
        print("Checkpoint: loading cluster assignments.", flush=True)
        with open(assignments_path) as f:
            asgn = json.load(f)
        valid_clusters   = asgn["valid_clusters"]
        skipped_clusters = asgn["skipped_clusters"]
    else:
        labels, centroids = cluster_papers(ids, embs)
        valid_clusters, skipped_clusters = build_cluster_assignments(ids, labels, centroids)
        asgn = {"valid_clusters": valid_clusters, "skipped_clusters": skipped_clusters}
        with open(assignments_path, "w") as f:
            json.dump(asgn, f, indent=2)
        print(f"Saved cluster assignments. Valid: {len(valid_clusters)}, Skipped: {len(skipped_clusters)}", flush=True)

    sorted_cluster_ids = sorted(valid_clusters.keys(), key=lambda x: int(x))
    pilot_ids = sorted_cluster_ids[:PILOT_N]

    # ── Task 2: Primary samples ───────────────────────────────────────────────
    samples_path = os.path.join(STEP3_DIR, "cluster_samples.json")

    if os.path.exists(samples_path):
        print("Checkpoint: loading cluster samples.", flush=True)
        with open(samples_path) as f:
            primary_samples = json.load(f)
    else:
        primary_samples = build_samples(valid_clusters, lookup, PRIMARY_BATCH)
        with open(samples_path, "w") as f:
            json.dump(primary_samples, f, indent=2)
        print(f"Saved primary samples.", flush=True)

    # ── Task 3: Pilot gate ────────────────────────────────────────────────────
    config_path = os.path.join(STEP3_DIR, "run_config.json")
    done_clusters = load_done_clusters()

    if os.path.exists(config_path):
        with open(config_path) as f:
            run_cfg = json.load(f)
        use_fallback   = run_cfg["config"] == "fallback"
        active_samples = (
            build_samples(valid_clusters, lookup, FALLBACK_BATCH) if use_fallback else primary_samples
        )
        print(f"Checkpoint: run config = {run_cfg['config']}. Pilot already passed.", flush=True)
        pilot_scores = run_cfg.get("pilot_scores", {})
    else:
        # Run pilot (primary config)
        print(f"\n--- PILOT (primary, batch={PRIMARY_BATCH}) ---", flush=True)
        pilot_results_primary = {}
        raw_outputs_primary   = {}

        for cid in pilot_ids:
            if cid in done_clusters:
                pilot_results_primary[cid] = done_clusters[cid]
                raw_outputs_primary[cid]   = done_clusters[cid].get("raw_output", "")
                print(f"  Cluster {cid}: already done (checkpoint)", flush=True)
                continue

            selected = primary_samples[cid]
            print(f"  Cluster {cid} ({valid_clusters[cid]['count']} papers, {len(selected)} sampled)...",
                  end=" ", flush=True)
            rec = run_cluster(int(cid), selected, lookup, use_fallback=False)
            pilot_results_primary[cid] = rec
            raw_outputs_primary[cid]   = rec["raw_output"]
            print(f"{rec['status']} | {len(rec['candidates'])} candidates | {rec['elapsed']:.1f}s", flush=True)

        primary_passed, primary_scores = gate_check(list(pilot_results_primary.values()), lookup)
        print(f"\nPrimary pilot gate: {'PASSED' if primary_passed else 'FAILED'}", flush=True)
        print(f"  {json.dumps(primary_scores)}", flush=True)

        if primary_passed:
            use_fallback   = False
            active_samples = primary_samples
            pilot_scores   = {"primary": primary_scores}
            # Persist pilot results to checkpoint
            for cid, rec in pilot_results_primary.items():
                if cid not in done_clusters:
                    append_record(rec)
            done_clusters = load_done_clusters()

        else:
            print(f"\n--- PILOT (fallback, batch={FALLBACK_BATCH}, two-stage) ---", flush=True)
            fallback_samples  = build_samples(valid_clusters, lookup, FALLBACK_BATCH)
            pilot_results_fb  = {}
            raw_outputs_fb    = {}

            for cid in pilot_ids:
                selected = fallback_samples[cid]
                print(f"  Cluster {cid} ({valid_clusters[cid]['count']} papers, {len(selected)} sampled)...",
                      end=" ", flush=True)
                rec = run_cluster(int(cid), selected, lookup, use_fallback=True)
                pilot_results_fb[cid] = rec
                raw_outputs_fb[cid]   = rec["raw_output"]
                print(f"{rec['status']} | {len(rec['candidates'])} candidates | {rec['elapsed']:.1f}s", flush=True)

            fb_passed, fb_scores = gate_check(list(pilot_results_fb.values()), lookup)
            print(f"\nFallback pilot gate: {'PASSED' if fb_passed else 'FAILED'}", flush=True)
            print(f"  {json.dumps(fb_scores)}", flush=True)

            if not fb_passed:
                print("\n=== BOTH CONFIGURATIONS FAILED THE PILOT GATE. STOPPING. ===", flush=True)
                scores_by_config = {"primary": primary_scores, "fallback": fb_scores}
                raw_by_config    = {"primary": raw_outputs_primary, "fallback": raw_outputs_fb}
                failure_path = write_failure_report(scores_by_config, raw_by_config, pilot_ids)
                git_commit_push(
                    "Step 3: pilot failure — both configurations failed gate",
                    [failure_path, os.path.abspath(__file__)]
                )
                sys.exit(1)

            use_fallback   = True
            active_samples = fallback_samples
            pilot_scores   = {"primary": primary_scores, "fallback": fb_scores}

            # Overwrite checkpoint with fallback pilot results (discard primary attempts)
            rewrite_checkpoint(list(pilot_results_fb.values()))
            done_clusters = load_done_clusters()

        run_cfg = {
            "config":        "fallback" if use_fallback else "primary",
            "batch_size":    FALLBACK_BATCH if use_fallback else PRIMARY_BATCH,
            "pilot_scores":  pilot_scores,
        }
        with open(config_path, "w") as f:
            json.dump(run_cfg, f, indent=2)

    # ── Task 4: Full run ──────────────────────────────────────────────────────
    remaining = [cid for cid in sorted_cluster_ids if cid not in done_clusters]
    print(f"\n--- FULL RUN: {len(remaining)} clusters remaining ---", flush=True)

    processed = total_cands = total_failures = 0

    for cid in remaining:
        selected = active_samples.get(cid) or sample_cluster(
            valid_clusters[cid]["paper_ids"], valid_clusters[cid]["centroid"], lookup,
            n=FALLBACK_BATCH if use_fallback else PRIMARY_BATCH
        )
        rec = run_cluster(int(cid), selected, lookup, use_fallback=use_fallback)
        append_record(rec)

        total_cands    += len(rec.get("candidates", []))
        total_failures += 1 if rec["status"] == "failed" else 0
        processed      += 1

        if processed % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  [{processed}/{len(remaining)}] clusters | {total_cands} candidates | "
                  f"{total_failures} failures | {elapsed/60:.1f}m", flush=True)

    # ── Task 5: Validate and consolidate ─────────────────────────────────────
    print("\nValidating candidates...", flush=True)
    all_records = list(load_done_clusters().values())

    accepted        = []
    rejected_by     = {}
    raw_total       = 0
    candidate_num   = 0

    for rec in all_records:
        if rec["status"] != "ok":
            continue
        sid_set = set(rec.get("selected_ids", []))
        cid     = rec["cluster_id"]

        for cand in rec.get("candidates", []):
            raw_total += 1
            ok, reason = validate_candidate(cand, sid_set, lookup)
            if not ok:
                rejected_by[reason] = rejected_by.get(reason, 0) + 1
                continue

            candidate_num += 1
            poles = cand["poles"]
            p0, p1 = poles[0], poles[1]

            def get_title(pid):
                return lookup.get(pid, {}).get("metadata", {}).get("title", "") or pid

            accepted.append({
                "id":           f"c{candidate_num:04d}",
                "type":         cand["relation_type"],
                "relation":     cand["relation_statement"],
                "claim_A":      p0["claim"],
                "claim_B":      p1["claim"],
                "paper_A":      p0["paper_id"],
                "paper_B":      p1["paper_id"],
                "paper_A_title": get_title(p0["paper_id"]),
                "paper_B_title": get_title(p1["paper_id"]),
                "snippet_A":    p0["snippet"],
                "snippet_B":    p1["snippet"],
                "instance":     "",
                "example_question": "",
                "status":       "candidate",
                "source":       "step3_extraction",
                "cluster_id":   int(cid),
            })

    candidates_out = os.path.join(OUTPUTS_DIR, "tension_candidates.json")
    with open(candidates_out, "w") as f:
        json.dump({"version": "step3_candidates", "candidates": accepted}, f, indent=2)
    print(f"Saved {len(accepted)} accepted candidates → {candidates_out}", flush=True)

    # ── Task 6: Report and push ───────────────────────────────────────────────
    elapsed_total = time.time() - t_start

    n_skipped   = len(skipped_clusters)
    n_failed    = sum(1 for r in all_records if r["status"] == "failed")
    n_processed = len(all_records)
    all_retries = sum(r.get("retries", 0) for r in all_records)
    all_timeouts= sum(1 for r in all_records if r.get("error") == "timeout")
    elapsed_vals= [r.get("elapsed", 0) for r in all_records if r.get("elapsed", 0) > 0]
    avg_secs    = sum(elapsed_vals) / len(elapsed_vals) if elapsed_vals else 0

    type_dist  = {}
    for c in accepted:
        type_dist[c["type"]] = type_dist.get(c["type"], 0) + 1

    zero_cand  = sum(1 for r in all_records if r["status"] == "ok" and not r.get("candidates"))
    cand_by_cluster = {str(r["cluster_id"]): len(r.get("candidates", [])) for r in all_records if r["status"] == "ok"}
    top5 = sorted(cand_by_cluster.items(), key=lambda x: -x[1])[:5]

    sample10 = random.sample(accepted, min(10, len(accepted)))

    config_used = "fallback" if use_fallback else "primary"

    report = f"""# Step 3 Run Report
Generated: {datetime.datetime.now().isoformat()}
Config used: **{config_used}** (batch_size={FALLBACK_BATCH if use_fallback else PRIMARY_BATCH})

## Pilot Results
"""
    for cfg_name, scores in (pilot_scores or {}).items():
        passed = scores["gate_parseable"] and scores["gate_candidates"] and scores["gate_attribution"]
        report += f"""
### {cfg_name.capitalize()} — {"PASSED" if passed else "FAILED"}
- Parseable calls: {scores['parseable']}/{scores['total_calls']} (≥7: {"✓" if scores['gate_parseable'] else "✗"})
- Candidates produced: {scores['total_candidates']} (≥1: {"✓" if scores['gate_candidates'] else "✗"})
- Attribution rate: {scores['attribution_rate']:.1%} (≥70%: {"✓" if scores['gate_attribution'] else "✗"})
"""

    report += f"""
## Cluster Stats
- Total valid clusters: {n_processed + n_failed}
- Processed: {n_processed}
- Skipped (<{MIN_CLUSTER_SIZE} papers): {n_skipped}
- Failed after {MAX_RETRIES} retries: {n_failed}

## Call Stats
- Total calls: {n_processed}
- Total retries: {all_retries}
- Timeouts: {all_timeouts}
- Average seconds/call: {avg_secs:.1f}
- Total runtime: {elapsed_total/60:.1f} minutes

## Candidate Stats
- Raw produced: {raw_total}
- Accepted: {len(accepted)}
- Rejected by reason:
"""
    for reason, count in sorted(rejected_by.items(), key=lambda x: -x[1]):
        report += f"  - {reason}: {count}\n"

    report += "\n### Relation type distribution\n"
    for t, count in sorted(type_dist.items(), key=lambda x: -x[1]):
        report += f"- {t}: {count}\n"

    report += f"\n- Zero-candidate clusters: {zero_cand}\n"
    report += "\n### Top 5 clusters by candidate count\n"
    for cid, count in top5:
        report += f"- Cluster {cid}: {count}\n"

    report += "\n## Sample Candidates (10 random)\n\n"
    for i, c in enumerate(sample10, 1):
        report += f"""### {i}. [{c['type']}] (cluster {c['cluster_id']})
**Relation:** {c['relation']}
**Paper A** `{c['paper_A']}` — {c['paper_A_title']}
  Claim: {c['claim_A']}
  Snippet: *{c['snippet_A']}*
**Paper B** `{c['paper_B']}` — {c['paper_B_title']}
  Claim: {c['claim_B']}
  Snippet: *{c['snippet_B']}*

"""

    report_path = os.path.join(OUTPUTS_DIR, "step3_run_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    # Append to automaton-state.md
    state_path = os.path.join(REPO_ROOT, "automaton-state.md")
    state_line = (
        f"\n- Step 3 complete {datetime.datetime.now().strftime('%Y-%m-%d')}: "
        f"{len(accepted)} accepted candidates, config={config_used}\n"
    )
    with open(state_path, "a") as f:
        f.write(state_line)

    commit_paths = [
        os.path.abspath(__file__),
        assignments_path,
        samples_path,
        config_path,
        CHECKPOINT,
        candidates_out,
        report_path,
        state_path,
    ]
    git_commit_push("Step 3: tension candidate extraction complete", commit_paths)

    print(f"\n=== Done. {len(accepted)} accepted candidates. {elapsed_total/60:.1f}m total. ===", flush=True)


if __name__ == "__main__":
    main()
