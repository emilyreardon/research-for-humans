
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import json, re, chromadb, requests as req, os, tempfile, subprocess, threading, queue, datetime, uuid, socket, glob
import qrcode as qrcode_lib
from PIL import Image, ImageDraw, ImageFont
try:
    import serial as _serial
    _SERIAL_OK = True
except ImportError:
    _SERIAL_OK = False

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Circuit Playground Express (NeoPixel lights) ──────────────────────────────
_cpx_port = None
_cpx_lock = threading.Lock()
import time as _time

def _cpx_find():
    for pat in ["/dev/cu.usbmodem*", "/dev/tty.usbmodem*"]:
        found = glob.glob(pat)
        if found:
            return found[0]
    return None

def _cpx_connect():
    """Open serial port at startup and wait for CPX to boot. Runs in background thread."""
    global _cpx_port
    if not _SERIAL_OK:
        return
    dev = _cpx_find()
    if not dev:
        print("CPX: device not found")
        return
    try:
        port = _serial.Serial(dev, 115200, timeout=1)
        _time.sleep(3.0)  # wait for CPX to boot after DTR reset on open
        with _cpx_lock:
            _cpx_port = port
        print(f"CPX: ready on {dev}", flush=True)
        _cpx_send("OFF")  # clear any leftover animation from a previous run
    except Exception as e:
        print(f"CPX: connect failed: {e}", flush=True)

def _cpx_send(cmd):
    global _cpx_port
    if not _SERIAL_OK:
        return
    if _cpx_port is None:
        _cpx_connect()
    with _cpx_lock:
        if _cpx_port is None:
            return
        try:
            _cpx_port.write((cmd + "\r\n").encode())
            _cpx_port.flush()
            _time.sleep(0.4)
            waiting = _cpx_port.in_waiting
            if waiting:
                resp = _cpx_port.read(waiting)
                print(f"CPX response: {resp!r}", flush=True)
            else:
                print(f"CPX: no response to {cmd}", flush=True)
        except Exception as e:
            print(f"CPX send error: {e}", flush=True)
            try: _cpx_port.close()
            except: pass
            _cpx_port = None

def cpx_on():
    print("CPX: sending ON", flush=True)
    _cpx_send("ON")

def cpx_off():
    print("CPX: sending OFF", flush=True)
    _cpx_send("OFF")

def _get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

LOCAL_IP = _get_local_ip()

def _qr_base_url():
    return f"http://{LOCAL_IP}:5050"

def _load_section(filename, tag, default=""):
    """Read content between [TAG] and [END TAG] in a framework txt file."""
    path = os.path.join(_HERE, filename)
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        start_tag = f"[{tag}]"
        end_tag   = f"[END {tag}]"
        s = content.find(start_tag)
        e = content.find(end_tag)
        if s == -1 or e == -1:
            print(f"Warn: [{tag}] not found in {filename} — using default")
            return default
        text = content[s + len(start_tag):e].strip()
        print(f"Loaded [{tag}] from {filename}")
        return text
    except FileNotFoundError:
        print(f"Warn: {filename} not found — using default")
        return default
    except Exception as e:
        print(f"Warn: could not read {filename}: {e} — using default")
        return default

def _load_params(filename, defaults):
    """Parse KEY = VALUE pairs from [CONFIG] block of a framework txt file."""
    raw = _load_section(filename, "CONFIG", "")
    params = dict(defaults)
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.split("#")[0].strip()
            if key not in params:
                continue
            ref = defaults[key]
            try:
                params[key] = float(val) if isinstance(ref, float) else int(val)
            except ValueError:
                params[key] = val
    return params

# ── config ───────────────────────────────────────────────────────────────────
CHROMA_PATH  = os.path.join(_HERE, "chroma_db")
COLLECTION   = "isls_papers_cosine_prefixed"
OLMO_URL     = "http://localhost:11434/api/generate"
OLMO_MODEL   = "olmo3-instruct"
PRINTER_NAME = "PTI_Color_Label_500"  # CUPS name (lpstat -p to verify)
TESTING      = True              # True = on-screen + print button; False = auto-print only
LIVE_RELOAD  = True              # True = reload framework txt files on every request (for editing); False = load once at startup
ICON_PATH    = os.path.join(_HERE, "icon.png")
LOG_PATH     = os.path.join(_HERE, "questions.log")
LOG_KEY      = "rfh2026"    # password for /log viewer — change before sharing

# Search & Synthesis — retrieval and response parameters from framework doc
_ss = _load_params("Research for Humans Search & Synthesis.txt", {
    "N_RESULTS": 7, "N_REFS": 20, "MAX_CHUNK_CHARS": 700, "MAX_RESPONSE_CHARS": 480, "MAX_CITATIONS": 3,
})
N_RESULTS          = _ss["N_RESULTS"]
N_REFS             = _ss["N_REFS"]
MAX_CHUNK_CHARS    = _ss["MAX_CHUNK_CHARS"]
MAX_RESPONSE_CHARS = _ss["MAX_RESPONSE_CHARS"]
MAX_CITATIONS      = _ss["MAX_CITATIONS"]
REGEN_MAX_ATTEMPTS = 12

# Prompt version management — see CONVENTIONS.md
# To roll back: change ACTIVE_PROMPT_VERSION to any earlier version name and restart.
# NEVER remove a version from PROMPT_VERSIONS. Add new versions; keep old ones.
PROMPT_VERSIONS = ["v1_baseline", "v2_tension_anchor", "v3_direct_anchor", "v4_anchored_form", "v5_passage_grounded"]
ACTIVE_PROMPT_VERSION = "v4_anchored_form"
PROMPTS_DIR = os.path.join(_HERE, "prompts")

_CLOSING_MOVES = [
    "Make it put two paths to the asker and ask which of the two they are on — not what would follow from either.",
    "Make it ask what is risked, or lost, if the saying holds.",
    "Make it turn the saying back on the asker, asked of them directly.",
    "Make it ask who is served by this and who is not.",
]
_closing_move_counter = 0

# Opener-diversity backstop — sliding window of the last 4 accepted card openers.
# Opener = first 3 words (lowercased, punct-stripped) of the closing question.
# Shared across all threads; protected by _opener_lock.
_RECENT_OPENERS: list[str] = []
_opener_lock = threading.Lock()

def _extract_opener(card: str) -> str:
    """Return first 3 words of the last '?'-terminated sentence, lowercased, punct stripped.
    Returns '' if no closing question is found."""
    sentences = re.split(r'(?<=[.!?])\s+', card.strip())
    for sent in reversed(sentences):
        if sent.strip().endswith("?"):
            words = re.findall(r"[a-zA-Z']+", sent.lower())
            return " ".join(words[:3])
    return ""

# Validator rules — banned terms for regeneration loop
try:
    with open(os.path.join(_HERE, "card-validator-rules.json")) as _vf:
        _vrules = json.load(_vf)
    _BANNED_TERMS = _vrules.get("banned_terms", [])
    print(f"Validator loaded: {len(_BANNED_TERMS)} banned terms")
except Exception as _ve:
    _BANNED_TERMS = []
    print(f"Validator not loaded: {_ve}")

# Format Specifications — label and typography parameters from framework doc
_fmt = _load_params("Research for Humans Format Specifications.txt", {
    "LABEL_DPI": 300, "LABEL_COPIES": 2,
    "BODY_FONT_SIZE": 44, "BODY_LINE_HEIGHT": 56,
    "REF_FONT_SIZE": 24,  "REF_LINE_HEIGHT": 30,
})
LABEL_DPI        = _fmt["LABEL_DPI"]
LABEL_COPIES     = _fmt["LABEL_COPIES"]
BODY_FONT_SIZE   = _fmt["BODY_FONT_SIZE"]
BODY_LINE_HEIGHT = _fmt["BODY_LINE_HEIGHT"]
REF_FONT_SIZE    = _fmt["REF_FONT_SIZE"]
REF_LINE_HEIGHT  = _fmt["REF_LINE_HEIGHT"]

_DEFAULT_SYSTEM_PROMPT = """You write oracle cards. Here are two examples. Match the format exactly.

Example A:

**Trust in collaborative research may move through practices that formal methods have no name for.**

Work that seems to circle this question finds students in joint inquiry sometimes protecting shared uncertainty rather than resolving it. A paper that may be tracking this shows trust consolidating around what a group agrees not to settle. What do the unwritten agreements inside a study tell you that the methodology section cannot?

Example B:

**Wellbeing keeps appearing in the learning sciences without ever quite landing.**

Research that may be circling this shows the term attached to productivity in one decade and to belonging in the next. The gap between them is not a gap in understanding. What would it mean to stop moving the target?

Now write one new oracle card using the research excerpts in the user message. The card has two paragraphs separated by a blank line. First paragraph is one short bold sentence containing a theme from the input. Second paragraph is 1-3 sentences of evidence followed by a question. The card ends with a question mark. No section labels. No citation. Periods, commas, and question marks only. The card must be self-contained and legible on one read — a visitor with no context should be able to follow it without rereading."""

# Style Guide — system prompt loaded from framework doc
SYSTEM_PROMPT = _load_section(
    "Research for Humans Style Guide.txt", "SYSTEM PROMPT", _DEFAULT_SYSTEM_PROMPT
)

def _reload_framework():
    global SYSTEM_PROMPT, N_RESULTS, N_REFS, MAX_CHUNK_CHARS, MAX_RESPONSE_CHARS, MAX_CITATIONS
    ss = _load_params("Research for Humans Search & Synthesis.txt", {
        "N_RESULTS": 7, "N_REFS": 20, "MAX_CHUNK_CHARS": 700, "MAX_RESPONSE_CHARS": 480, "MAX_CITATIONS": 3,
    })
    N_RESULTS          = ss["N_RESULTS"]
    N_REFS             = ss["N_REFS"]
    MAX_CHUNK_CHARS    = ss["MAX_CHUNK_CHARS"]
    MAX_RESPONSE_CHARS = ss["MAX_RESPONSE_CHARS"]
    MAX_CITATIONS      = ss["MAX_CITATIONS"]
    SYSTEM_PROMPT      = _load_section(
        "Research for Humans Style Guide.txt", "SYSTEM PROMPT", _DEFAULT_SYSTEM_PROMPT
    )

# Refs cache — persisted to disk so QR codes survive app restarts
_REF_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ref_cache.json")
try:
    with open(_REF_CACHE_PATH, encoding="utf-8") as _f:
        _ref_cache = json.load(_f)
    print(f"Refs cache: loaded {len(_ref_cache)} entries from disk", flush=True)
except FileNotFoundError:
    _ref_cache = {}
except Exception as _e:
    print(f"Refs cache: load failed ({_e}), starting empty", flush=True)
    _ref_cache = {}

def _save_ref_entry(uid, entry):
    _ref_cache[uid] = entry
    try:
        with open(_REF_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_ref_cache, f, ensure_ascii=False)
    except Exception as e:
        print(f"Refs cache: write failed: {e}", flush=True)

# Git version — captured once at startup for the question log
try:
    GIT_VERSION = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=_HERE, stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    GIT_VERSION = "unknown"

# ── corpus ────────────────────────────────────────────────────────────────────
EMBED_URL   = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

try:
    col = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(name=COLLECTION)
    print(f"Corpus ready: {col.count()} docs")
except Exception as e:
    col = None
    print(f"ChromaDB: {e}")

def _embed(text):
    r = req.post(EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=30)
    r.raise_for_status()
    return r.json()["embedding"]

# ── runtime controls ──────────────────────────────────────────────────────────
_RUNTIME_CONFIG_PATH = os.path.join(_HERE, "runtime_config.json")
_SPEED_ATTEMPTS      = {"quality": 8, "balanced": 4, "fast": 2, "single": 1}
_RUNTIME_SPEED       = "quality"
_RUNTIME_SOURCES     = "standard"

def _load_runtime_config():
    global _RUNTIME_SPEED, _RUNTIME_SOURCES
    try:
        with open(_RUNTIME_CONFIG_PATH) as f:
            cfg = json.load(f)
        speed   = cfg.get("speed",   "quality")
        sources = cfg.get("sources", "standard")
        _RUNTIME_SPEED   = speed   if speed   in _SPEED_ATTEMPTS         else "quality"
        _RUNTIME_SOURCES = sources if sources in ("standard", "focused") else "standard"
        if speed not in _SPEED_ATTEMPTS or sources not in ("standard", "focused"):
            print("runtime_config: unknown value(s) — using defaults", flush=True)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"runtime_config: malformed ({e}), using defaults (quality, standard)", flush=True)
        _RUNTIME_SPEED, _RUNTIME_SOURCES = "quality", "standard"

def _save_runtime_config():
    cfg = {"speed": _RUNTIME_SPEED, "sources": _RUNTIME_SOURCES}
    fd, tmp = tempfile.mkstemp(dir=_HERE, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f)
        os.replace(tmp, _RUNTIME_CONFIG_PATH)
    except Exception as e:
        print(f"runtime_config: save failed ({e})", flush=True)
        try:
            os.unlink(tmp)
        except Exception:
            pass

_load_runtime_config()
print(f"Runtime config: speed={_RUNTIME_SPEED}, sources={_RUNTIME_SOURCES}", flush=True)

# ── corpus mean (for focused centering) ───────────────────────────────────────
_CORPUS_MEAN      = None
_CORPUS_MEAN_LOCK = threading.Lock()
_CORPUS_MEAN_PATH = os.path.join(_HERE, "data", "corpus_mean.json")

def _load_or_compute_corpus_mean():
    global _CORPUS_MEAN
    try:
        with open(_CORPUS_MEAN_PATH) as f:
            mean = json.load(f)
        with _CORPUS_MEAN_LOCK:
            _CORPUS_MEAN = mean
        print(f"Corpus mean: loaded from cache ({len(mean)}-dim)", flush=True)
        return
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Corpus mean: cache read error ({e})", flush=True)
    try:
        if col is None:
            print("Corpus mean: ChromaDB unavailable, focused will behave as standard", flush=True)
            return
        sample   = col.get(limit=10000, include=["embeddings"])
        embs_raw = sample.get("embeddings")
        # ChromaDB may return a numpy array; convert explicitly to avoid truth-value errors
        embs = list(embs_raw) if embs_raw is not None else []
        if not embs:
            raise ValueError("no embeddings returned from sample")
        dim  = len(embs[0])
        mean = [sum(float(e[i]) for e in embs) / len(embs) for i in range(dim)]
        with _CORPUS_MEAN_LOCK:
            _CORPUS_MEAN = mean
        os.makedirs(os.path.dirname(_CORPUS_MEAN_PATH), exist_ok=True)
        with open(_CORPUS_MEAN_PATH, "w") as f:
            json.dump(mean, f)
        print(f"Corpus mean: computed and cached ({dim}-dim, {len(embs)} docs)", flush=True)
    except Exception as e:
        print(f"Corpus mean: computation failed ({e}), focused will behave as standard", flush=True)

threading.Thread(target=_load_or_compute_corpus_mean, daemon=True).start()

# ── topical grounding helpers ─────────────────────────────────────────────────
_CONTENT_STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being","and","or","but",
    "so","in","on","at","to","for","of","with","that","this","it","its","not",
    "what","how","who","when","why","which","from","by","as","if","into","than",
    "more","their","they","them","can","could","may","might","do","does","did",
    "have","has","had","will","would","should","about","only","just","one","two",
    "some","any","all","no","new","such","each","both","same","other","out","up",
    "he","she","we","you","his","her","our","your","my","very","even","still",
    "also","then","once","here","there","where","these","those","after","before",
    "between","through","during","while","because","since","though","without",
}

def _content_tokens(text, min_len=4):
    """Lower-case alpha content words, ≥min_len chars, not in stopword list."""
    return {w for w in re.findall(r'\b[a-zA-Z]+\b', text.lower())
            if len(w) >= min_len and w not in _CONTENT_STOPWORDS}

def _topical_candidate_set(passage_text, tension_block, question):
    """Passage content tokens minus tension/question/banned — valid grounding terms."""
    p = _content_tokens(passage_text)
    t = _content_tokens(tension_block)
    q = _content_tokens(question)
    b = {term.lower() for term in _BANNED_TERMS if len(term) >= 4}
    return p - t - q - b

# ── tensions ──────────────────────────────────────────────────────────────────
_TENSIONS           = []
_TENSION_EMBEDDINGS = []

def _load_tensions():
    global _TENSIONS, _TENSION_EMBEDDINGS
    path = os.path.join(_HERE, "data", "tensions.json")
    try:
        with open(path) as f:
            data = json.load(f)
        _TENSIONS = data.get("tensions", [])
        _TENSION_EMBEDDINGS = []
        for t in _TENSIONS:
            text = f"search_document: {t['relation']} {t['claim_A']} {t['claim_B']}"
            _TENSION_EMBEDDINGS.append(_embed(text))
        print(f"Tensions loaded: {len(_TENSIONS)}")
    except Exception as e:
        print(f"Tensions not loaded: {e}")

_load_tensions()

# Calibrated against 8-tension set (Step 6). Revisit after tension set expansion (Step 5).
# Conservative: errs toward tension-grounded cards rather than weak-match.
# Scores below this threshold indicate the question doesn't map well onto any available tension.
TENSION_RELEVANCE_THRESHOLD = 0.54

WEAK_MATCH_PROMPT = (
    "This question touches learning, struggle, or the life of the mind, but it does not land "
    "on any single tension the oracle holds. Speak from the wider territory instead.\n\n"
    "Say plainly what the question reaches for, and answer from the broad ground within the oracle's sight, "
    "how people learn, struggle, change, and come to know, without claiming any specific finding, "
    "study, or relation. Do not cite, quote, or attribute.\n\n"
    "Write one bold opening sentence that makes a specific claim or asks a real question, "
    "then one to three sentences in the same voice with real weight, not a bare question. "
    "End with a single question that does work the opening did not.\n\n"
    "Never refer to the card, the research, the sources, or your own writing. "
    "Do not include examples. Only speak.\n\n"
    "VISITOR QUESTION: {question}"
)

def _cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def find_nearest_tension(question):
    if not _TENSIONS or not _TENSION_EMBEDDINGS:
        return None, 0.0
    q_emb = _embed("search_query: " + question)
    best_idx, best_sim = 0, -1.0
    for i, t_emb in enumerate(_TENSION_EMBEDDINGS):
        sim = _cosine_sim(q_emb, t_emb)
        if sim > best_sim:
            best_sim, best_idx = sim, i
    print(f"Nearest tension: {_TENSIONS[best_idx]['id']} (sim={best_sim:.3f})")
    return _TENSIONS[best_idx], best_sim

# ── prompt version management ───────────────────────────────────────────────
# Templates live in prompts/<version>.txt. The first lines starting with "#" are
# a header (version / description / author); the rest is the prompt text with
# {question}, {tension}, {passages} substitution slots. See CONVENTIONS.md.

def load_prompt_template(version_name):
    """Read prompts/<version_name>.txt, strip header lines beginning with '#',
    return the template string (slots intact)."""
    path = os.path.join(PROMPTS_DIR, f"{version_name}.txt")
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    body = [ln for ln in lines if not ln.lstrip().startswith("#")]
    return "\n".join(body).strip()

def load_prompt_description(version_name):
    """Return the text after '# description:' in the template header, or ''."""
    path = os.path.join(PROMPTS_DIR, f"{version_name}.txt")
    try:
        with open(path, encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if s.lower().startswith("# description:"):
                    return s.split(":", 1)[1].strip()
    except Exception:
        pass
    return ""

def render_prompt(template, question, tension_text, passages_text):
    """Substitute the slots into the template. Uses str.replace (not .format)
    because passage text may contain literal braces. Slot -> argument mapping:
    {question} -> question, {tension} -> tension_text, {passages} -> passages_text."""
    return (template
            .replace("{tension}", tension_text)
            .replace("{question}", question)
            .replace("{passages}", passages_text))

def _format_tension_block(tension):
    """Build the TENSION block. No trailing newline — template supplies the blank line."""
    return (
        f"TENSION\n"
        f"Relation: {tension['relation']}\n"
        f"Pole A: {tension['claim_A']}\n"
        f"Pole B: {tension['claim_B']}\n"
        f"Instance: {tension['instance']}"
    )

def get_tension_with_score(question):
    """
    Wraps find_nearest_tension() to handle both return signatures:
    - Post-Step-6: returns (tension, score)
    - Pre-Step-6: returns tension only
    Returns (tension, score_or_None).
    """
    result = find_nearest_tension(question)
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, None

def _tension_label(tension):
    """Human-readable label for a tension. The objects have no dedicated name
    field; 'relation' is the descriptive sentence that names the tension."""
    if not tension:
        return None
    return tension.get("relation")

def _current_tensions_version():
    try:
        with open(os.path.join(_HERE, "data", "tensions.json"), encoding="utf-8") as f:
            return json.load(f).get("version")
    except Exception:
        return None

# ── question log + comparison cache (outputs/) ──────────────────────────────
OUTPUTS_DIR     = os.path.join(_HERE, "outputs")
BETA_LOG_PATH   = os.path.join(OUTPUTS_DIR, "beta_questions.jsonl")
COMPARISON_DIR  = os.path.join(OUTPUTS_DIR, "comparison")
os.makedirs(COMPARISON_DIR, exist_ok=True)

def log_question(entry):
    """Append one JSON line to outputs/beta_questions.jsonl."""
    try:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        with open(BETA_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"log_question error: {e}")

def _normalize_q(question):
    return question.strip().lower()

def get_cached_result(version_name, question):
    """Return the cached comparison result for (version, question), or None."""
    path = os.path.join(COMPARISON_DIR, f"{version_name}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:
        return None
    return cache.get(_normalize_q(question))

def save_cached_result(version_name, question, result):
    """Store result under the normalized question key in the version's cache file."""
    os.makedirs(COMPARISON_DIR, exist_ok=True)
    path = os.path.join(COMPARISON_DIR, f"{version_name}.json")
    cache = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    cache[_normalize_q(question)] = result
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def is_on_domain(question):
    domain_prompt = (
        f'A visitor asked: "{question}"\n\n'
        "Decide whether this question has any human dimension — any connection to learning, growth, curiosity, "
        "skill, understanding, work, relationships, or the inner life.\n\n"
        "The bar for yes is very low. Almost any question about a person's situation, choices, or experience qualifies — "
        "including questions about tools, technology, or AI as they relate to skill or growth.\n\n"
        "Answer no ONLY for purely logistical requests with zero human dimension: asking for a time, "
        "a schedule, a location, directions, or a weather forecast.\n\n"
        'Examples of yes: "Is using AI making me a worse writer?", "Why is this hard?", "How do I motivate my students?"\n'
        'Examples of no: "What time does the reception start?", "Where is room 204?", "What\'s the WiFi password?"\n\n'
        "Answer with one word only: yes or no."
    )
    try:
        r = req.post(OLMO_URL, json={"model": OLMO_MODEL, "prompt": domain_prompt, "stream": False}, timeout=30)
        r.raise_for_status()
        answer = r.json().get("response", "").strip().lower()
        result = "yes" in answer
        print(f"is_on_domain: '{answer}' → {result}")
        return result
    except Exception as e:
        print(f"is_on_domain error: {e}")
        return True

def _truncate_reply(text, max_chars=480):
    if len(text) <= max_chars:
        result = text
    else:
        chunk = text[:max_chars]
        result = chunk
        for punct in (". ", "! ", "? "):
            idx = chunk.rfind(punct)
            if idx != -1:
                result = chunk[:idx + 1]
                break
    if result.count("**") % 2 != 0:
        result = result.replace("**", "")
    return result.strip()

def _cleanup_card(text):
    """Non-length cleanup only: fix unbalanced bold markers. No shortening."""
    result = text.strip()
    if result.count("**") % 2 != 0:
        result = result.replace("**", "")
    return result

def _validate_card(text, chunks_text=""):
    """Check card body against validator rules. Returns (passes, failures, score).
    Score = 4 - len(failures); used to rank fallbacks (higher is better).
    Checks: min_length (<200), length (>MAX_RESPONSE_CHARS), banned term, no_question.
    chunks_text: if provided, a banned term that also appears in the retrieved chunks
    is allowed through (corpus-word exception)."""
    failures = []
    if len(text) < 200:
        failures.append("min_length")
    if len(text) > MAX_RESPONSE_CHARS:
        failures.append("length")
    t_lower = text.lower()
    chunks_lower = chunks_text.lower() if chunks_text else ""
    for term in _BANNED_TERMS:
        term_lower = term.lower()
        if re.search(r'\b' + re.escape(term_lower) + r'\b', t_lower):
            if chunks_lower and re.search(r'\b' + re.escape(term_lower) + r'\b', chunks_lower):
                continue  # corpus-word exception: term came from the source material
            failures.append(f"banned:{term}")
            break
    if not text.strip().endswith("?"):
        failures.append("no_question")
    score = 4 - len(failures)
    return (len(failures) == 0), failures, score

def _enforce_format(text):
    """Ensure two-paragraph oracle card format and remove banned punctuation."""
    # Strip section labels the model sometimes outputs
    text = re.sub(r'^\s*\*?\*?Opening\s+[Ss]entence\*?\*?\s*[,:]?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\*?\*?Body\*?\*?\s*[,:]?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\*?\*?Closing\s+[Qq]uestion\*?\*?\s*[,:]?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\*?\*?Card\*?\*?\s*[,:]?\s*', '', text, flags=re.MULTILINE)

    # Strip banned punctuation everywhere first
    text = re.sub(r'\(([^)]*)\)', r'\1', text)       # parentheses → content only
    text = re.sub(r';', ',', text)                    # semicolons → commas
    text = re.sub(r'(?<=\w):\s', ', ', text)          # mid-sentence colon → comma
    text = re.sub(r'[–—]', ',', text)                 # dashes → commas
    text = re.sub(r',\s*,', ',', text)                # double commas → single
    text = re.sub(r'\s+,', ',', text)                 # remove space(s) before comma
    text = re.sub(r',(?!\s|$)', ', ', text)           # add space after comma if missing
    text = re.sub(r',[ \t]{2,}', ', ', text)          # collapse multiple spaces after comma
    text = re.sub(r'\*\*\s*\*\*', '', text)           # empty bold spans

    paras = [p.strip() for p in text.strip().split('\n\n') if p.strip()]
    if not paras:
        return text

    # If only one paragraph (model put everything together), split at first sentence end
    if len(paras) == 1:
        m = re.search(r'[.?!]\s+', paras[0][20:])    # skip first 20 chars
        if m:
            split = 20 + m.start() + 1
            paras = [paras[0][:split].strip(), paras[0][split:].strip()]

    # Bold the opening sentence only (strip any existing ** first)
    first = re.sub(r'\*\*', '', paras[0]).strip()
    if not first[-1:] in '.?!':
        first += '.'
    paras[0] = f'**{first}**'

    # Merge any extra paragraphs into the second; strip bold from body
    if len(paras) > 2:
        paras = [paras[0], ' '.join(paras[1:])]
    if len(paras) > 1:
        paras[1] = re.sub(r'\*\*', '', paras[1]).strip()

    return '\n\n'.join(paras)

def _trim_citation(citation):
    citation = re.sub(r',?\s*\(pp\.\s*\d+[–-]\d+\)', '', citation)
    citation = re.sub(r',?\s*pp\.\s*\d+[–-]\d+', '', citation)
    citation = re.sub(r'\.\s+In\s+.+$', '.', citation, flags=re.DOTALL)
    citation = citation.strip().rstrip(',').strip()
    if citation and not citation.endswith('.'):
        citation += '.'
    return citation

def _format_citation_display(citations):
    citations = [c for c in citations if c]  # drop empty strings (empty metadata field bypasses fallback)
    if not citations:
        return ""
    if len(citations) == 1:
        return citations[0]
    first_cit = citations[0]
    first_author = re.split(r'[,.(]', first_cit)[0].strip()
    year_match = re.search(r'\((\d{4})\)', first_cit)
    year = year_match.group(1) if year_match else ""
    return f"{first_author} et al. ({year})" if year else f"{first_author} et al."

def _format_ref_author(authors_str):
    """Return 'First Last[, et al.]' from a metadata authors string."""
    if not authors_str or authors_str.lower() == 'unknown':
        return "Unknown"
    # Authors stored as "Last, FirstName | Last2, FirstName2 | ..."
    parts = [a.strip() for a in authors_str.split('|') if a.strip()]
    if not parts:
        return "Unknown"
    multi = len(parts) > 1
    first = parts[0]
    if ',' in first:
        last, given = first.split(',', 1)
        name = f"{given.strip()} {last.strip()}"
    else:
        name = first
    if multi:
        name += ", et al."
    return name

def _format_ref_name_lines(authors_str):
    """Return stacked label lines: ['First M.', 'Last'] or ['First M.', 'Last', 'et al.']"""
    if not authors_str or authors_str.lower() == 'unknown':
        return ["Unknown"]
    parts = [a.strip() for a in authors_str.split('|') if a.strip()]
    if not parts:
        return ["Unknown"]
    multi = len(parts) > 1
    first_author = parts[0]
    if ',' in first_author:
        last, given = first_author.split(',', 1)
        last  = last.strip()
        given_parts = given.strip().split()
        if given_parts:
            first_name = given_parts[0]
            if len(given_parts) > 1:
                mid = given_parts[1]
                mid_init = (mid[0].upper() + '.') if not mid.endswith('.') else mid
                line1 = f"{first_name} {mid_init}"
            else:
                line1 = first_name
        else:
            line1 = last
            last = ""
    else:
        name_parts = first_author.split()
        if len(name_parts) >= 2:
            last = name_parts[-1]
            gp   = name_parts[:-1]
            first_name = gp[0]
            if len(gp) > 1:
                mid = gp[1]
                mid_init = (mid[0].upper() + '.') if not mid.endswith('.') else mid
                line1 = f"{first_name} {mid_init}"
            else:
                line1 = first_name
        else:
            line1 = first_author
            last  = ""
    lines = [l for l in [line1, last] if l]
    if multi:
        lines.append("et al.")
    return lines

def get_context(question):
    if not col:
        return None, [], []
    try:
        embedding = _embed("search_query: " + question)

        # Focused centering: subtract corpus mean from query, re-normalise
        if _RUNTIME_SOURCES == "focused":
            with _CORPUS_MEAN_LOCK:
                mean = _CORPUS_MEAN
            if mean is not None:
                try:
                    centered = [e - m for e, m in zip(embedding, mean)]
                    norm = sum(c * c for c in centered) ** 0.5
                    if norm > 0:
                        embedding = [c / norm for c in centered]
                    print("Retrieval: focused centering applied", flush=True)
                except Exception as ce:
                    print(f"Centering failed ({ce}), using standard embedding", flush=True)
            else:
                print("Retrieval: focused requested but corpus mean unavailable, using standard", flush=True)

        n_retrieve = max(N_RESULTS, N_REFS)
        res = col.query(
            query_embeddings=[embedding], n_results=n_retrieve,
            include=["documents", "metadatas", "distances"]
        )
        docs_all  = res["documents"][0]
        metas_all = res["metadatas"][0]
        dists_all = res["distances"][0]

        # Per-paper cap + backfill: take at most 1 chunk per paper (best cosine),
        # then backfill remaining N_RESULTS slots from leftover same-paper chunks.
        seen_papers  = set()
        context_rows = []   # (doc, meta, dist) for the N_RESULTS context slots
        leftover     = []
        for doc, meta, dist in zip(docs_all, metas_all, dists_all):
            title = meta.get("title", "")
            if title not in seen_papers:
                seen_papers.add(title)
                context_rows.append((doc, meta, dist))
            else:
                leftover.append((doc, meta, dist))
            if len(context_rows) >= N_RESULTS:
                break
        for doc, meta, dist in leftover:
            if len(context_rows) >= N_RESULTS:
                break
            context_rows.append((doc, meta, dist))

        # Context string for the model (N_RESULTS chunks)
        parts = []
        for doc, meta, dist in context_rows:
            snippet = doc[:MAX_CHUNK_CHARS] if len(doc) > MAX_CHUNK_CHARS else doc
            parts.append(
                f"{meta.get('authors','Unknown')} ({str(meta.get('date',''))[:4]}), "
                f"{meta.get('title','Untitled')}:\n{snippet}"
            )
        context = "\n---\n".join(parts)

        # Cited papers — deduplicated by title from context_rows, for label footer
        papers = []
        seen_titles = set()
        for doc, meta, dist in context_rows:
            if len(papers) >= MAX_CITATIONS:
                break
            title = meta.get("title", "")
            if title in seen_titles:
                continue
            seen_titles.add(title)
            cit = _trim_citation(
                meta.get("citation") or
                f"{meta.get('authors','')} ({str(meta.get('date',''))[:4]}). {meta.get('title','')}."
            )
            papers.append({
                "citation": cit,
                "url":      meta.get("item_url", ""),
                "authors":  meta.get("authors", ""),
                "title":    title,
                "year":     str(meta.get("date", ""))[:4],
            })

        # All unique papers for /refs — full n_retrieve results
        ref_map = {}
        for doc, meta, dist in zip(docs_all, metas_all, dists_all):
            title = meta.get("title", "")
            snippet = doc[:MAX_CHUNK_CHARS] if len(doc) > MAX_CHUNK_CHARS else doc
            if title not in ref_map:
                ref_map[title] = {
                    "title":      title,
                    "authors":    meta.get("authors", ""),
                    "year":       str(meta.get("date", ""))[:4],
                    "url":        meta.get("item_url", ""),
                    "conference": meta.get("conference", ""),
                    "citation":   meta.get("citation", ""),
                    "chunks":     [],
                    "best_score": dist,
                }
            ref_map[title]["chunks"].append(snippet)
            ref_map[title]["best_score"] = min(ref_map[title]["best_score"], dist)
        ref_papers = sorted(ref_map.values(), key=lambda x: x["best_score"])

        unique_in_ctx = len({meta.get("title","") for _,meta,_ in context_rows})
        print(f"Context: {len(parts)} chunks ({unique_in_ctx} unique papers) to model, "
              f"{len(ref_map)} unique papers for refs, {sum(len(p) for p in parts)} chars")
        return context, papers, ref_papers
    except Exception as e:
        print(f"Query error: {e}")
        return None, [], []

# ── generation + comparison ────────────────────────────────────────────────
def generate_card(user_prompt, chunks_text=""):
    """Non-streaming generation with the same regenerate-until-valid loop and
    fallback selection as /ask (REGEN_MAX_ATTEMPTS, _enforce_format,
    _cleanup_card, _validate_card all used unchanged).
    Returns (card_text_or_None, attempt_count, failures)."""
    best_card = None   # (score, len, text, failures)
    attempts  = 0
    _card_t0  = _time.monotonic()
    for attempt in range(1, REGEN_MAX_ATTEMPTS + 1):
        attempts  = attempt
        _att_t0   = _time.monotonic()
        try:
            r = req.post(
                OLMO_URL,
                json={"model": OLMO_MODEL, "system": SYSTEM_PROMPT,
                      "prompt": user_prompt, "stream": False},
                timeout=180,
            )
            r.raise_for_status()
            raw = r.json().get("response", "")
        except Exception as e:
            print(f"generate_card attempt {attempt} error: {e}")
            raw = ""
        if not raw.strip():
            continue
        card = _enforce_format(raw)
        card = _cleanup_card(card)
        passes, failures, score = _validate_card(card, chunks_text)
        _att_s = _time.monotonic() - _att_t0
        print(f"compare attempt {attempt}: len={len(card)} score={score} failures={failures} elapsed={_att_s:.1f}s")
        if passes:
            _card_s = _time.monotonic() - _card_t0
            print(f"compare card done: {attempt} attempt(s) total={_card_s:.1f}s")
            return card, attempt, []
        if best_card is None or score > best_card[0] or (score == best_card[0] and len(card) < best_card[1]):
            best_card = (score, len(card), card, failures)
    _card_s = _time.monotonic() - _card_t0
    if best_card is None:
        print(f"compare card done (empty): {attempts} attempt(s) total={_card_s:.1f}s")
        return None, attempts, ["empty"]
    print(f"compare card done (fallback): {attempts} attempt(s) total={_card_s:.1f}s")
    return best_card[2], attempts, best_card[3]

def run_comparison(question, versions, force_regen=False):
    """Run retrieval + tension matching once, then generate (or reuse cache) a
    card per version. Logs each version result. Returns a list of result dicts
    shaped for the /compare-ui page."""
    q = question.strip()
    if not q:
        return []
    context, papers, ref_papers = get_context(q)
    context = context or ""
    tension, tension_score = get_tension_with_score(q)
    tension_id   = tension.get("id") if tension else None
    tension_name = _tension_label(tension)
    tension_text = _format_tension_block(tension) if tension else ""
    tensions_now = _current_tensions_version()

    results = []
    for version_name in versions:
        cached = None if force_regen else get_cached_result(version_name, q)
        if cached:
            result = cached
        else:
            template    = load_prompt_template(version_name)
            template    = template.replace("{closing_move}", "")
            user_prompt = render_prompt(template, q, tension_text, context)
            card, attempts, _failures = generate_card(user_prompt, context)
            result = {
                "card":             card,
                "tension_id":       tension_id,
                "tension_name":     tension_name,
                "tension_score":    tension_score,
                "attempt_count":    attempts,
                "timestamp":        datetime.datetime.now().isoformat(),
                "tensions_version": tensions_now,
                "char_count":       len(card) if card else 0,
            }
            save_cached_result(version_name, q, result)

        log_question({
            "timestamp":      datetime.datetime.now().isoformat(),
            "question":       q,
            "source":         "compare",
            "path":           "compare",
            "tension_id":     result.get("tension_id"),
            "tension_name":   result.get("tension_name"),
            "tension_score":  result.get("tension_score"),
            "prompt_version": version_name,
            "card":           result.get("card"),
            "attempt_count":  result.get("attempt_count"),
        })

        results.append({
            "version_name":             version_name,
            "description":              load_prompt_description(version_name),
            "tension_id":               result.get("tension_id"),
            "tension_name":             result.get("tension_name"),
            "tension_score":            result.get("tension_score"),
            "card":                     result.get("card"),
            "char_count":               result.get("char_count"),
            "tensions_version":         result.get("tensions_version"),
            "current_tensions_version": tensions_now,
        })
    return results

# ── print label ───────────────────────────────────────────────────────────────
ICON_LABEL_W = int(2.0 / 2.54 * LABEL_DPI)   # 2.0 cm ≈ 236 px
QR_REF       = int(1.5 / 2.54 * LABEL_DPI)   # 1.5 cm QR codes ≈ 177 px
REF_SLOT_GAP = 12                             # px between horizontal reference slots

def _find_font(size):
    for path in [
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()

def _find_font_bold(size):
    for path in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return _find_font(size)

def _parse_inline(text):
    result = []
    for chunk in re.split(r'(\*\*.*?\*\*)', text):
        if chunk.startswith('**') and chunk.endswith('**') and len(chunk) > 4:
            result.append((chunk[2:-2], True))
        else:
            result.append((chunk, False))
    return result

def _wrap_rich(draw, text, font_r, font_b, max_w):
    lines = []
    space_w = draw.textlength(' ', font=font_r)
    for para in re.split(r'\n\n+', text):
        words = []
        for chunk in re.split(r'(\*\*.*?\*\*)', para):
            is_bold = chunk.startswith('**') and chunk.endswith('**') and len(chunk) > 4
            inner = chunk[2:-2] if is_bold else chunk
            for w in inner.split():
                words.append((w, is_bold))
        cur, cur_w = [], 0
        for word, is_bold in words:
            font = font_b if is_bold else font_r
            w = draw.textlength(word, font=font)
            if cur and cur_w + space_w + w > max_w:
                lines.append(cur)
                cur, cur_w = [(word, is_bold)], w
            else:
                if cur:
                    cur.append((' ', False))
                    cur_w += space_w
                cur.append((word, is_bold))
                cur_w += w
        if cur:
            lines.append(cur)
        lines.append(None)  # paragraph break
    while lines and lines[-1] is None:
        lines.pop()
    return lines

def _render_rich_line(draw, x, y, segs, font_r, font_b, fill):
    for text, is_bold in segs:
        font = font_b if is_bold else font_r
        draw.text((x, y), text, font=font, fill=fill)
        x += int(draw.textlength(text, font=font))

def _reset_printer():
    subprocess.run(["cancel", "-a", PRINTER_NAME], capture_output=True)
    subprocess.run(["cupsenable", PRINTER_NAME], capture_output=True)

def print_label(body_text, papers, qr_url="", copies=None):
    copies = LABEL_COPIES if copies is None else copies
    _reset_printer()
    W = int(4 * LABEL_DPI)
    H = int(3 * LABEL_DPI)
    M = int(0.15 * LABEL_DPI)

    font_r   = _find_font(BODY_FONT_SIZE)
    font_bld = _find_font_bold(BODY_FONT_SIZE)
    font_ref = _find_font(REF_FONT_SIZE)
    lh_b, lh_ref = BODY_LINE_HEIGHT, REF_LINE_HEIGHT

    img  = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    # Reference strip — icon bottom-left (same height as QR), author + QR bottom-right
    ref_y = H - M - QR_REF

    if qr_url:
        qr = qrcode_lib.QRCode(
            error_correction=qrcode_lib.constants.ERROR_CORRECT_M,
            box_size=3, border=1
        )
        qr.add_data(qr_url)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        qr_img = qr_img.resize((QR_REF, QR_REF), Image.LANCZOS)
        img.paste(qr_img, (W - M - QR_REF, ref_y))

    if qr_url:
        label_text = "references available here:"
        lw = draw.textlength(label_text, font=font_ref)
        text_right = W - M - QR_REF - 8
        ty = ref_y + (QR_REF - lh_ref) // 2
        draw.text((text_right - lw, ty), label_text, font=font_ref, fill=(85, 85, 85))

    # Body — rich text, starts at top margin, capped above reference strip
    body_max_w  = W - 2 * M
    body_max_y  = ref_y - lh_b          # stop before references
    rich_lines  = _wrap_rich(draw, body_text, font_r, font_bld, body_max_w)
    y = M
    for line in rich_lines:
        if y + lh_b > body_max_y:
            break
        if line is None:
            y += lh_b // 2
        else:
            _render_rich_line(draw, M, y, line, font_r, font_bld, (0, 0, 0))
            y += lh_b

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    img.save(path, dpi=(LABEL_DPI, LABEL_DPI))
    job_ids = []
    for idx in range(copies):
        r = subprocess.run(
            ["lp", "-d", PRINTER_NAME, "-o", "PageSize=4x3", path],
            check=False, capture_output=True, text=True
        )
        m = re.search(r'request id is (\S+)', r.stdout or "")
        if m:
            job_ids.append(m.group(1))
        print(f"Print job: {r.stdout.strip()}", flush=True)
    os.unlink(path)
    # kill the macOS printer queue window that CUPS opens after each job;
    # fire immediately and again after 1.5 s (window often opens late),
    # then reactivate Chrome so focus returns to the kiosk
    subprocess.Popen(
        ["pkill", "-f", PRINTER_NAME],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    def _delayed_refocus():
        _time.sleep(1.5)
        subprocess.Popen(["pkill", "-f", PRINTER_NAME],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _time.sleep(0.2)
        subprocess.Popen(["osascript", "-e", 'tell application "Google Chrome" to activate'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    threading.Thread(target=_delayed_refocus, daemon=True).start()
    return job_ids

# ── HTML ──────────────────────────────────────────────────────────────────────
_TESTING_JS      = "true" if TESTING else "false"
_PREVIEW_FONT_PX = str(round(BODY_FONT_SIZE * 96 / 300, 1))
_PREVIEW_LH      = str(round(BODY_LINE_HEIGHT / BODY_FONT_SIZE, 2))

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Research for Humans</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:Arial,sans-serif; background:#fff; color:#000; display:flex; align-items:flex-start; justify-content:center; min-height:100vh; padding:0 20px; }
  #wrap { width:100%; max-width:900px; }
  #qrow { width:80%; margin:45vh auto 0; border-bottom:1px solid #000; }
  #q { width:100%; font-family:Arial,sans-serif; font-size:28px; font-weight:normal; line-height:1.4; color:#555; background:#fff; border:none; outline:none; padding:8px 0; text-align:left; resize:none; overflow:hidden; display:block; white-space:pre-wrap; word-break:break-word; overflow-wrap:break-word; }
  #q::placeholder { font-size:22px; color:#bbb; }
  #spin { margin-top:48px; display:none; text-align:center; height:32px; }
  .dot { display:inline-block; width:18px; height:18px; border-radius:50%; background:#000; margin:0 8px; animation:pulse 1.4s infinite both; opacity:0; }
  .dot:nth-child(2) { animation-delay:0.2s; }
  .dot:nth-child(3) { animation-delay:0.4s; }
  @keyframes pulse { 0%,80%,100%{opacity:0} 40%{opacity:1} }
  /* Full card — visible below the fold (scroll down to see); not in the kiosk viewport */
  #card-spacer { display:none; height:1000px; }
  #resp { display:none; flex-direction:column; font-family:Arial,sans-serif; font-size:__PREVIEW_FONT_PX__px; color:#000; line-height:__PREVIEW_LH__; text-align:left; width:4in; height:3in; overflow:hidden; padding:0.15in; box-sizing:border-box; background:#fff; border:1px solid #ccc; margin:0 auto; }
  #resp-body { flex:1; overflow:hidden; white-space:pre-wrap; word-wrap:break-word; }
  #resp-strip { display:flex; align-items:flex-start; justify-content:space-between; height:38px; flex-shrink:0; margin-top:6px; }
  #resp-strip-icon img { height:38px; width:auto; display:block; }
  #resp-authors { font-size:8px; color:#777; line-height:1.4; text-align:right; padding-right:6px; align-self:flex-start; }
  #resp-authors a { color:#555; text-decoration:underline; }
  #resp-qr { width:38px; height:38px; flex-shrink:0; }
  #resp-qr img { width:38px; height:38px; display:block; }
</style>
</head>
<body>
<div id="wrap">
  <div id="qrow">
    <textarea id="q" rows="1" placeholder="Your question" autofocus></textarea>
  </div>
  <div id="spin"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>
  <div id="card-spacer"></div>
  <div id="resp">
    <div id="resp-body"></div>
    <div id="resp-strip">
      <div id="resp-strip-icon"><img src="/icon" alt=""></div>
      <div style="display:flex;align-items:flex-start;">
        <div id="resp-authors"></div>
        <div id="resp-qr"></div>
      </div>
    </div>
  </div>
</div>
<script>
  var q       = document.getElementById("q");
  var spinner = document.getElementById("spin");
  var rb      = document.getElementById("resp-body");
  var respEl  = document.getElementById("resp");
  var pendingBody = "", pendingCitations = [], pendingRefUid = "";

  q.addEventListener("input", function() { q.style.height="auto"; q.style.height=q.scrollHeight+"px"; });
  q.addEventListener("keydown", function(e) { if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();submit();} });

  // Aggressive refocus for Bluetooth keyboard / second-screen focus drift
  window.addEventListener("focus", function() { q.focus(); });
  document.addEventListener("visibilitychange", function() { if(!document.hidden) q.focus(); });
  document.addEventListener("click", function() { q.focus(); });
  document.addEventListener("keydown", function(e) {
    if (e.key === "Enter" && !e.shiftKey && document.activeElement !== q) {
      e.preventDefault();
      q.focus();
      submit();
    }
  });
  // Reclaim focus whenever the textarea is blurred — DOM focus + OS window activation
  q.addEventListener("blur", function() {
    fetch("/refocus", {method:"POST"}).catch(function(){});
    setTimeout(function() { q.focus(); }, 50);
  });
  // Poll every 500 ms as backstop
  setInterval(function() {
    if (!spinner.style.display || spinner.style.display === "none") {
      if (document.activeElement !== q) q.focus();
    }
  }, 500);

  // Kiosk: block shortcuts that open new tabs/windows or navigate away (capture phase)
  document.addEventListener("keydown", function(e) {
    var meta = e.metaKey || e.ctrlKey;
    if (meta && "tnwrpf".indexOf(e.key.toLowerCase()) !== -1) {
      e.preventDefault(); e.stopPropagation();
    }
    if (e.key === "F5" || e.key === "F12") {
      e.preventDefault(); e.stopPropagation();
    }
  }, true);
  document.addEventListener("contextmenu", function(e) { e.preventDefault(); });
  window.open = function() { return null; };

  // Kiosk: hide cursor after 3 s of inactivity, restore on movement
  (function() {
    var _cursorTimer;
    function _showCursor() {
      document.body.style.cursor = "";
      clearTimeout(_cursorTimer);
      _cursorTimer = setTimeout(function() { document.body.style.cursor = "none"; }, 3000);
    }
    document.addEventListener("mousemove", _showCursor);
    _cursorTimer = setTimeout(function() { document.body.style.cursor = "none"; }, 3000);
  })();

  function esc(t) { return t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

  function clearField() {
    q.value = "";
    q.style.height = "auto";
    pendingBody = ""; pendingCitations = []; pendingRefUid = "";
    q.focus();
  }

  function showCard(body, refUid) {
    rb.innerHTML = mdToHtml(body);
    var qrEl = document.getElementById("resp-qr");
    qrEl.innerHTML = "";
    if (refUid) {
      var img = document.createElement("img");
      img.src = "/qr?url=" + encodeURIComponent(window.location.origin + "/refs/" + refUid);
      img.alt = "Sources";
      qrEl.appendChild(img);
    }
    document.getElementById("card-spacer").style.display = "block";
    respEl.style.display = "flex";
  }

  function mdToHtml(text) {
    var h = esc(text);
    h = h.replace(/[*][*]([^*]+)[*][*]/g, "<strong>$1</strong>");
    h = h.split("\\n\\n").join("<br><br>");
    h = h.split("\\n").join("<br>");
    return h;
  }

  function doPrint() {
    var refsUrl = pendingRefUid ? window.location.origin + '/refs/' + pendingRefUid : '';
    fetch("/print", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({body: pendingBody, citations: pendingCitations, ref_url: refsUrl})
    }).then(function() {
      setTimeout(clearField, 800);
      setTimeout(function() { q.focus(); }, 1800);
      setTimeout(function() { q.focus(); }, 3000);
    }).catch(function(e) { console.error("Print error:", e); setTimeout(clearField, 800); });
  }

  var currentReader = null;
  var currentAbort  = null;
  var activeReqId   = 0;

  function playSubmitSfx() {
    try {
      var ctx = new (window.AudioContext || window.webkitAudioContext)();
      var comp = ctx.createDynamicsCompressor();
      comp.threshold.value = -3;
      comp.knee.value = 3;
      comp.ratio.value = 4;
      comp.attack.value = 0;
      comp.release.value = 0.1;
      comp.connect(ctx.destination);
      var osc = ctx.createOscillator();
      var gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(comp);
      osc.type = "sine";
      osc.frequency.setValueAtTime(880, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(660, ctx.currentTime + 0.12);
      gain.gain.setValueAtTime(1.5, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.22);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.22);
    } catch(e) {}
  }

  // ── ambient + thinking audio ──────────────────────────────────────────────
  var _ambEl = null;
  var _ambReady = false;
  var _thinkCtx = null;
  var _thinkTimer = null;

  function _ensureAmb() {
    if (!_ambEl) {
      _ambEl = new Audio("/ambient");
      _ambEl.loop = true;
      _ambEl.volume = 0.18;
    }
  }

  function resumeAmb() {
    _ensureAmb();
    _ambEl.play().catch(function(){});
    _ambReady = true;
  }

  function pauseAmb() {
    if (_ambEl) _ambEl.pause();
  }

  function startThinking() {
    pauseAmb();
    try {
      _thinkCtx = new (window.AudioContext || window.webkitAudioContext)();
      _thinkCtx.resume();
      // C major pentatonic, 5th octave — gentle music-box pings
      var scale = [523.3, 587.3, 659.3, 784.0, 880.0];
      var idx = 0;
      function ping() {
        if (!_thinkCtx) return;
        var freq = scale[idx % scale.length];
        idx++;
        // Two slightly detuned sines for bell-like richness
        [freq, freq * 1.003].forEach(function(f) {
          var o = _thinkCtx.createOscillator();
          var g = _thinkCtx.createGain();
          o.type = "sine";
          o.frequency.value = f;
          o.connect(g);
          g.connect(_thinkCtx.destination);
          var t = _thinkCtx.currentTime;
          g.gain.setValueAtTime(0.55, t);
          g.gain.exponentialRampToValueAtTime(0.001, t + 1.4);
          o.start(t);
          o.stop(t + 1.4);
        });
        _thinkTimer = setTimeout(ping, 420 + Math.random() * 260);
      }
      ping();
    } catch(e) {}
  }

  function stopThinking() {
    if (_thinkTimer) { clearTimeout(_thinkTimer); _thinkTimer = null; }
    if (_thinkCtx) {
      setTimeout(function() {
        try { _thinkCtx.close(); } catch(e) {}
        _thinkCtx = null;
      }, 300);
    }
    resumeAmb();
  }

  // Start ambient on first keydown (browser autoplay policy requires user gesture)
  document.addEventListener("keydown", function _startAmb() {
    resumeAmb();
    document.removeEventListener("keydown", _startAmb);
  });

  function submit() {
    var v = q.value.trim();
    if (!v) return;
    playSubmitSfx();

    if (currentAbort)  { currentAbort.abort();  currentAbort  = null; }
    if (currentReader) { try { currentReader.cancel(); } catch(e) {} currentReader = null; }

    var myId = ++activeReqId;

    rb.textContent = "";
    pendingBody = ""; pendingCitations = []; pendingRefUid = "";
    respEl.style.display = "none";
    document.getElementById("card-spacer").style.display = "none";
    spinner.style.display = "block";
    startThinking();
    q.focus();

    currentAbort = new AbortController();

    fetch("/ask", {
      method: "POST",
      signal: currentAbort.signal,
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({question: v})
    })
    .then(function(res) {
      if (myId !== activeReqId) return;
      var reader = res.body.getReader();
      currentReader = reader;
      var decoder = new TextDecoder();
      var buf = "";

      function read() {
        if (myId !== activeReqId) { try { reader.cancel(); } catch(e) {} return; }
        reader.read().then(function(chunk) {
          if (myId !== activeReqId) return;
          if (chunk.done) return;
          buf += decoder.decode(chunk.value, {stream:true});
          var lines = buf.split("\\n");
          buf = lines.pop();
          lines.forEach(function(line) {
            if (!line.startsWith("data: ")) return;
            var d;
            try { d = JSON.parse(line.slice(6)); } catch(e) { return; }
            if (d.done) {
              if (myId !== activeReqId) return;
              spinner.style.display = "none";
              stopThinking();
              currentReader = null;
              if (!d.answer) return;
              var idx = d.answer.indexOf("\\n\\n— ");
              pendingBody      = idx === -1 ? d.answer : d.answer.slice(0, idx);
              pendingCitations = d.citations || [];
              pendingRefUid    = d.ref_uid || "";
              showCard(pendingBody, pendingRefUid);
              q.focus();
              doPrint();
            }
          });
          read();
        }).catch(function() {
          if (myId !== activeReqId) return;
          spinner.style.display = "none";
          stopThinking();
        });
      }
      read();
    })
    .catch(function() {
      if (myId !== activeReqId) return;
      spinner.style.display = "none";
      stopThinking();
    });
  }
  q.focus();
</script>
</body>
</html>""".replace("__PREVIEW_FONT_PX__", _PREVIEW_FONT_PX) \
          .replace("__PREVIEW_LH__", _PREVIEW_LH)

# ── comparison dashboard page (dev tool) ──────────────────────────────────────
COMPARE_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RfH — Prompt Comparison</title>
<style>
  * { box-sizing:border-box; }
  body { font-family:Arial,sans-serif; margin:0; padding:24px; color:#222; background:#fff; }
  h1 { font-size:18px; margin:0 0 4px; }
  .sub { font-size:12px; color:#999; margin-bottom:18px; }
  #bar { display:flex; gap:8px; margin-bottom:20px; max-width:900px; }
  #q { flex:1; font-size:14px; padding:10px; border:1px solid #ccc; border-radius:4px; }
  #go { background:#C8FF00; border:none; padding:10px 22px; font-size:13px; font-weight:bold; cursor:pointer; border-radius:4px; }
  #go:disabled { opacity:0.5; cursor:wait; }
  #status { font-size:12px; color:#888; margin-bottom:16px; min-height:16px; }
  .qrow { margin-bottom:28px; border-bottom:1px solid #eee; padding-bottom:24px; }
  .qlabel { font-size:13px; font-weight:bold; color:#444; background:#f6f6f6; border-left:3px solid #C8FF00; padding:6px 10px; border-radius:3px; margin-bottom:12px; }
  .cells { display:flex; gap:16px; flex-wrap:wrap; align-items:stretch; }
  .col { flex:1 1 320px; min-width:300px; border:1px solid #e0e0e0; border-radius:6px; padding:16px; display:flex; flex-direction:column; }
  .vname { font-size:14px; font-weight:bold; margin-bottom:2px; }
  .vdesc { font-size:11px; font-style:italic; color:#888; margin-bottom:10px; line-height:1.4; }
  .tension { font-size:11px; color:#555; background:#fafafa; border-left:3px solid #C8FF00; padding:6px 8px; margin-bottom:6px; line-height:1.4; }
  .tlabel { color:#aaa; text-transform:uppercase; letter-spacing:0.06em; font-size:9px; display:block; margin-bottom:2px; }
  .stale { font-size:10px; color:#c08000; margin-bottom:8px; }
  .card { font-size:15px; line-height:1.55; color:#111; white-space:pre-wrap; word-wrap:break-word; flex:1; margin:8px 0; }
  .card strong { font-weight:bold; }
  .card.empty { color:#c00; font-style:italic; font-size:13px; }
  .chars { font-size:10px; color:#aaa; margin-bottom:10px; }
  .regen { align-self:flex-start; background:#f0f0f0; border:1px solid #ddd; padding:6px 14px; font-size:11px; cursor:pointer; border-radius:4px; }
  .regen:hover { background:#e6e6e6; }
  .regen:disabled { opacity:0.5; cursor:wait; }
</style>
</head>
<body>
<h1>Research for Humans — Prompt Comparison</h1>
<p class="sub">Dev tool. Compares prompt versions side by side. Each submitted question becomes its own row. Rows accumulate — refresh to clear.</p>
<div id="bar">
  <input id="q" type="text" placeholder="Type a visitor question and press Enter">
  <button id="go" onclick="runCompare()">Compare</button>
</div>
<div id="status"></div>
<div id="rows"></div>
<script>
  var qEl = document.getElementById("q");
  var goEl = document.getElementById("go");
  var statusEl = document.getElementById("status");
  var rowsEl = document.getElementById("rows");

  var questions = []; // [{q, results: {version_name: result}}], newest first

  qEl.addEventListener("keydown", function(e){ if(e.key==="Enter"){ e.preventDefault(); runCompare(); } });

  function esc(t){ return (t||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
  function mdToHtml(t){
    var h = esc(t);
    h = h.replace(/\\*\\*([^*]+)\\*\\*/g, "<strong>$1</strong>");
    h = h.split("\\n\\n").join("<br><br>").split("\\n").join("<br>");
    return h;
  }

  function renderCell(r, rowIdx){
    var cell = document.createElement("div");
    cell.className = "col";
    cell.id = "cell-" + rowIdx + "-" + r.version_name;

    var tensionLine = "";
    if (r.tension_name) {
      var score = (r.tension_score !== null && r.tension_score !== undefined)
        ? " (" + Number(r.tension_score).toFixed(3) + ")" : "";
      tensionLine = '<div class="tension"><span class="tlabel">Tension' + score + '</span>' + esc(r.tension_name) + '</div>';
    }
    var stale = "";
    if (r.tensions_version && r.current_tensions_version && r.tensions_version !== r.current_tensions_version) {
      stale = '<div class="stale">Generated with an earlier tension set</div>';
    }
    var cardHtml = r.card
      ? '<div class="card">' + mdToHtml(r.card) + '</div>'
      : '<div class="card empty">No card generated (model unavailable or all attempts failed).</div>';

    cell.innerHTML =
      '<div class="vname">' + esc(r.version_name) + '</div>' +
      '<div class="vdesc">' + esc(r.description) + '</div>' +
      tensionLine + stale + cardHtml +
      '<div class="chars">' + (r.char_count || 0) + ' chars</div>' +
      '<button class="regen" onclick="regen(\\'' + r.version_name + '\\',' + rowIdx + ')">Regenerate</button>';
    return cell;
  }

  function renderRows(){
    rowsEl.innerHTML = "";
    questions.forEach(function(entry, rowIdx){
      var rowEl = document.createElement("div");
      rowEl.className = "qrow";
      rowEl.id = "qrow-" + rowIdx;

      var label = document.createElement("div");
      label.className = "qlabel";
      label.textContent = entry.q;
      rowEl.appendChild(label);

      var cellsEl = document.createElement("div");
      cellsEl.className = "cells";
      Object.keys(entry.results).forEach(function(vname){
        cellsEl.appendChild(renderCell(entry.results[vname], rowIdx));
      });
      rowEl.appendChild(cellsEl);
      rowsEl.appendChild(rowEl);
    });
  }

  function runCompare(){
    var q = qEl.value.trim();
    if (!q) return;
    goEl.disabled = true;
    statusEl.textContent = "Generating… (10–30s per uncached version)";
    fetch("/compare", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({question:q})
    }).then(function(res){ return res.json(); })
      .then(function(data){
        var results = {};
        (data.results||[]).forEach(function(r){ results[r.version_name] = r; });
        questions.unshift({q:q, results:results});
        renderRows();
        statusEl.textContent = "";
        goEl.disabled = false;
      }).catch(function(e){
        statusEl.textContent = "Error: " + e;
        goEl.disabled = false;
      });
  }

  function regen(version, rowIdx){
    var entry = questions[rowIdx];
    if (!entry) return;
    var cellId = "cell-" + rowIdx + "-" + version;
    var old = document.getElementById(cellId);
    var btn = old ? old.querySelector(".regen") : null;
    if (btn){ btn.disabled = true; btn.textContent = "Regenerating…"; }
    fetch("/compare/regenerate", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({question:entry.q, version:version})
    }).then(function(res){ return res.json(); })
      .then(function(data){
        var r = (data.results||[])[0];
        if (r){
          entry.results[version] = r;
          if (old){ old.replaceWith(renderCell(r, rowIdx)); }
        }
      }).catch(function(e){
        if (btn){ btn.disabled = false; btn.textContent = "Regenerate"; }
      });
  }
  qEl.focus();
</script>
</body>
</html>"""

# ── server (threaded so concurrent asks don't block each other) ───────────────
class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): pass

    def do_GET(self):
        if self.path == "/icon":
            try:
                with open(ICON_PATH, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-type", "image/png")
                self.end_headers()
                self.wfile.write(data)
            except Exception:
                self.send_response(404)
                self.end_headers()
            return

        if self.path.startswith("/qr?url="):
            url = self.path[8:]
            try:
                from urllib.parse import unquote
                url = unquote(url)
                qr = qrcode_lib.QRCode(
                    error_correction=qrcode_lib.constants.ERROR_CORRECT_M,
                    box_size=4, border=2
                )
                qr.add_data(url)
                qr.make(fit=True)
                qr_img = qr.make_image(fill_color="black", back_color="white")
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    tmp = f.name
                qr_img.save(tmp)
                with open(tmp, "rb") as f:
                    png = f.read()
                os.unlink(tmp)
                self.send_response(200)
                self.send_header("Content-type", "image/png")
                self.end_headers()
                self.wfile.write(png)
            except Exception as e:
                print(f"QR error: {e}")
                self.send_response(500)
                self.end_headers()
            return

        if self.path.startswith("/refs/"):
            uid = self.path[6:].split("?")[0]
            entry = _ref_cache.get(uid)
            if not entry:
                self.send_response(404)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"<p style='font-family:Arial;padding:24px'>References not found or session expired.</p>")
                return
            question   = entry.get("question", "")
            ref_papers = entry.get("ref_papers", [])
            card_text  = entry.get("card", "")
            total_retrieved = sum(len(p.get("chunks", [])) for p in ref_papers)
            corpus_count = col.count() if col else 74330

            # ── Section 1: Static method ──────────────────────────────────────
            static_html = f"""<div class="info-section">
  <div class="info-label">How retrieval works</div>
  <p>The corpus is {corpus_count:,} text chunks from 7,463 ISLS conference papers, stored locally on this device in ChromaDB. No internet connection is used for retrieval or generation.</p>
  <p>Your question was embedded using nomic-embed-text (running locally via Ollama) and matched against all {corpus_count:,} chunks by cosine similarity. The {N_REFS} closest chunks were retrieved. There is no clustering — the match is nearest-neighbor only.</p>
  <p>The {N_RESULTS} highest-scoring chunks were passed to the model as context. Up to {MAX_CITATIONS} distinct papers from those {N_RESULTS} chunks are cited. The model ({OLMO_MODEL}, running locally via Ollama) generated the card from the retrieved excerpts only — it has no access to the rest of the corpus.</p>
</div>"""

            # ── Section 2: What the selection determined ──────────────────────
            years = [int(p["year"]) for p in ref_papers if p.get("year","").isdigit()]
            year_span = f"{min(years)}–{max(years)}" if len(years) >= 2 else (str(years[0]) if years else "unknown")
            distinct_works = len(ref_papers)
            # Scores are cosine distances (lower = more similar); available but not labeled
            scores_available = any("best_score" in p for p in ref_papers)
            selection_html = f"""<div class="info-section">
  <div class="info-label">What the selection determined</div>
  <table class="sel-table">
    <tr><td class="sel-key">Question</td><td>&#8220;{question.replace(chr(34),"&quot;")}&#8221;</td></tr>
    <tr><td class="sel-key">Works in pool</td><td>{distinct_works} distinct papers from {total_retrieved} retrieved passages</td></tr>
    <tr><td class="sel-key">Year span</td><td>{year_span}</td></tr>
    <tr><td class="sel-key">Match scores</td><td>Cosine distance available for all {distinct_works} works; not labeled here</td></tr>
    <tr><td class="sel-key">Cited (model saw)</td><td>Top {N_RESULTS} passages, deduplicated to at most {MAX_CITATIONS} papers</td></tr>
  </table>
</div>"""

            # ── Section 3: What the model determined ─────────────────────────
            if card_text:
                # Render card markdown: **bold** → <strong>, \n\n → paragraph break
                import re as _re
                card_html_body = card_text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                card_html_body = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', card_html_body)
                card_html_body = card_html_body.replace("\n\n", "</p><p>")
                card_display = f'<p>{card_html_body}</p>'
                model_html = f"""<div class="info-section">
  <div class="info-label">What the model generated</div>
  <div class="card-output">{card_display}</div>
  <p class="card-note">This is the card as generated — the same text on the printed card. Not regenerated here.</p>
</div>"""
            else:
                model_html = """<div class="info-section">
  <div class="info-label">What the model generated</div>
  <p class="card-note placeholder">Card text not available — cards generated before this session are not persisted to the references page.</p>
</div>"""

            rows = ""
            for i, p in enumerate(ref_papers):
                authors_display = _format_ref_author(p.get("authors", ""))
                title_esc = p.get("title","").replace("&","&amp;").replace("<","&lt;")
                conf_esc  = p.get("conference","").replace("&","&amp;").replace("<","&lt;")
                url = p.get("url","")
                link_o = f'<a href="{url}" target="_blank">' if url else ""
                link_c = "</a>" if url else ""
                passages_html = ""
                for j, chunk in enumerate(p.get("chunks", [])):
                    chunk_esc = chunk.replace("&","&amp;").replace("<","&lt;")
                    label = "Passage retrieved" if len(p["chunks"]) == 1 else f"Passage {j+1} retrieved"
                    passages_html += f'<div class="passage-label">{label}</div><div class="passage">{chunk_esc}</div>'
                abstract_link = f'<a href="{url}" target="_blank" class="abstract-link">Read abstract on ISLS repository ↗</a>' if url else ""
                rows += f"""
<div class="paper">
  <div class="rank">{"Most relevant" if i == 0 else f"Source {i+1}"}</div>
  <div class="title">{link_o}{title_esc}{link_c}</div>
  <div class="meta">{authors_display} &middot; {p.get("year","")}{"&emsp;&middot;&emsp;" + conf_esc if conf_esc else ""}</div>
  {abstract_link}
  {passages_html}
</div>"""

            page = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Research for Humans — Sources</title>
<style>
  body {{ font-family:Arial,sans-serif; max-width:680px; margin:0 auto; padding:24px 16px; color:#333; line-height:1.5; }}
  h1 {{ font-size:17px; font-weight:bold; margin-bottom:2px; }}
  .subtitle {{ font-size:11px; color:#aaa; margin-bottom:20px; }}
  .q {{ font-size:13px; color:#444; margin-bottom:24px; font-style:italic; border-left:3px solid #C8FF00; padding:6px 10px; background:#fafafa; }}
  .info-section {{ margin-bottom:28px; border-top:2px solid #C8FF00; padding-top:14px; }}
  .info-label {{ font-size:10px; color:#bbb; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:10px; }}
  .info-section p {{ font-size:12px; color:#555; margin-bottom:8px; }}
  .sel-table {{ border-collapse:collapse; width:100%; font-size:12px; color:#555; margin-bottom:4px; }}
  .sel-table tr {{ border-bottom:1px solid #f0f0f0; }}
  .sel-key {{ color:#aaa; font-size:10px; text-transform:uppercase; letter-spacing:0.06em; padding:5px 12px 5px 0; white-space:nowrap; vertical-align:top; width:130px; }}
  .sel-table td:last-child {{ padding:5px 0; }}
  .card-output {{ background:#fafafa; border-left:3px solid #C8FF00; padding:12px 14px; margin:8px 0 6px; font-size:13px; color:#333; line-height:1.6; }}
  .card-output strong {{ font-weight:bold; }}
  .card-note {{ font-size:10px; color:#bbb; margin-top:4px; }}
  .card-note.placeholder {{ font-style:italic; }}
  .paper {{ margin-bottom:36px; border-top:1px solid #eee; padding-top:16px; }}
  .rank {{ font-size:10px; color:#bbb; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:5px; }}
  .title {{ font-size:14px; font-weight:bold; margin-bottom:3px; }}
  .title a {{ color:#000; text-decoration:none; }}
  .title a:hover {{ text-decoration:underline; }}
  .meta {{ font-size:11px; color:#999; margin-bottom:8px; }}
  .abstract-link {{ font-size:11px; color:#777; display:inline-block; margin-bottom:12px; }}
  .passage-label {{ font-size:10px; color:#bbb; text-transform:uppercase; letter-spacing:0.07em; margin-top:12px; margin-bottom:4px; }}
  .passage {{ font-size:12px; color:#555; line-height:1.6; background:#f9f9f9; padding:10px 12px; border-left:2px solid #eee; }}
  .works-header {{ font-size:10px; color:#bbb; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; }}
  footer {{ margin-top:48px; padding-top:16px; border-top:1px solid #eee; font-size:11px; color:#ccc; }}
</style>
</head><body>
<h1>Research for Humans</h1>
<p class="subtitle">ISLS 2026 &middot; Digital Futures Institute, Teachers College Columbia University</p>
{static_html}
{selection_html}
{model_html}
<p class="works-header">Works retrieved — {len(ref_papers)} papers, {total_retrieved} passages</p>
{rows}
<footer>Research for Humans &middot; researchforhumans.org &middot; <a href="https://github.com/emilyreardon/research-for-humans" style="color:#ccc">GitHub</a></footer>
</body></html>"""
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page.encode("utf-8"))
            return

        if self.path.startswith("/log"):
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            key = qs.get("key", [""])[0]
            if key != LOG_KEY:
                self.send_response(401)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"<p style='font-family:Arial;padding:24px'>Access denied. Add <code>?key=PASSWORD</code> to the URL.</p>")
                return
            try:
                with open(LOG_PATH, encoding="utf-8") as f:
                    raw = f.read()
            except FileNotFoundError:
                raw = ""
            entries = [e.strip() for e in raw.split("=" * 60) if e.strip()]
            entries.reverse()
            rows = ""
            for entry in entries:
                lines = entry.splitlines()
                meta, card_lines, paper_lines = [], [], []
                in_card, in_papers = False, False
                for line in lines:
                    if line.startswith("CARD:"):
                        in_card, in_papers = True, False
                        continue
                    if line.startswith("PAPERS:"):
                        in_card, in_papers = False, True
                        continue
                    if in_card:
                        card_lines.append(line)
                    elif in_papers:
                        paper_lines.append(line)
                    else:
                        meta.append(line)
                meta_html = "".join(f'<div class="meta-line">{l.replace("&","&amp;").replace("<","&lt;")}</div>' for l in meta if l)
                card_html = "\n".join(card_lines).replace("&","&amp;").replace("<","&lt;")
                papers_html = "".join(f'<div class="paper-line">{l.replace("&","&amp;").replace("<","&lt;")}</div>' for l in paper_lines if l)
                rows += f'<div class="entry"><div class="meta-block">{meta_html}</div><pre class="card">{card_html}</pre><div class="papers">{papers_html}</div></div>'
            count = len(entries)
            page = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RfH Log — {count} responses</title>
<style>
  body {{ font-family:Arial,sans-serif; max-width:800px; margin:0 auto; padding:24px 16px; color:#333; }}
  h1 {{ font-size:17px; margin-bottom:4px; }}
  .count {{ font-size:12px; color:#aaa; margin-bottom:28px; }}
  .entry {{ margin-bottom:32px; border-top:2px solid #C8FF00; padding-top:14px; }}
  .meta-block {{ margin-bottom:8px; }}
  .meta-line {{ font-size:11px; color:#888; font-family:monospace; }}
  pre.card {{ font-size:12px; color:#333; background:#f9f9f9; padding:12px; white-space:pre-wrap; word-wrap:break-word; margin:8px 0; border-left:3px solid #C8FF00; }}
  .papers {{ }}
  .paper-line {{ font-size:11px; color:#999; font-family:monospace; }}
  a {{ color:#555; font-size:11px; }}
</style>
</head><body>
<h1>Research for Humans — Question Log</h1>
<p class="count">{count} responses &middot; most recent first &middot; <a href="?key={LOG_KEY}">refresh</a></p>
{rows if rows else "<p style='color:#aaa'>No entries yet.</p>"}
</body></html>"""
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page.encode("utf-8"))
            return

        if self.path.startswith("/compare-ui"):
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(COMPARE_HTML.encode("utf-8"))
            return

        # ── admin speed/sources control endpoints ─────────────────────────────
        if self.path.startswith("/admin/speed/") or self.path.startswith("/admin/sources/"):
            from urllib.parse import urlparse
            p = urlparse(self.path).path.rstrip("/")
            parts = p.split("/")
            if len(parts) == 4 and parts[1] == "admin":
                control, value = parts[2], parts[3]
                if control == "speed" and value in _SPEED_ATTEMPTS:
                    global _RUNTIME_SPEED
                    _RUNTIME_SPEED = value
                    _save_runtime_config()
                    print(f"Admin: speed → {value}", flush=True)
                elif control == "sources" and value in ("standard", "focused"):
                    global _RUNTIME_SOURCES
                    _RUNTIME_SOURCES = value
                    _save_runtime_config()
                    print(f"Admin: sources → {value}", flush=True)
                # else: invalid value — ignored, no state change
            self.send_response(303)
            self.send_header("Location", "/admin")
            self.end_headers()
            return

        if self.path == "/admin" or self.path == "/admin/":
            host = self.headers.get("Host", f"localhost:{_ADMIN_PORT}")
            base = f"http://{host}"

            def _btn(control, value, label, desc):
                active = (_RUNTIME_SPEED == value if control == "speed"
                          else _RUNTIME_SOURCES == value)
                bg = "#c8ff00" if active else "#f0f0f0"
                mark = " &#10003;" if active else ""
                return (f'<a href="{base}/admin/{control}/{value}" style="display:block;'
                        f'padding:10px 14px;margin:4px 0;background:{bg};color:#1a1a1a;'
                        f'text-decoration:none;border-radius:4px;font-size:14px;">'
                        f'<strong>{label}{mark}</strong> &mdash; {desc}</a>')

            page = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Oracle Controls</title>
<style>
  body{{font-family:system-ui,sans-serif;max-width:640px;margin:32px auto;padding:0 16px;color:#1a1a1a;line-height:1.5;}}
  h1{{font-size:22px;margin-bottom:4px;}}
  h2{{font-size:16px;margin:24px 0 4px;}}
  p{{font-size:14px;color:#444;margin:4px 0 10px;}}
  .note{{font-size:13px;color:#666;margin-top:20px;border-top:1px solid #ddd;padding-top:12px;}}
</style>
</head><body>
<h1>Oracle Controls</h1>
<h2>Speed</h2>
<p>How hard the oracle works to fit the visitor&#8217;s own topic into the card. Higher settings put more of what was actually asked into the card but take longer to print. Step down one level only if cards are taking too long or a line is forming &#8212; you can move back up anytime.</p>
{_btn("speed","quality","Quality","tries hardest to put the visitor&#8217;s topic in the card. Slowest. We open here.")}
{_btn("speed","balanced","Balanced","a middle setting.")}
{_btn("speed","fast","Fast","fewer tries; quicker cards, more of them fall back to a general answer.")}
{_btn("speed","single","Single pass","the oracle writes one card and prints it. Fastest. Use this if cards must come quickly.")}
<h2>Sources</h2>
<p>Which papers the oracle reaches for. We set this before doors open, based on a check we run on the cards. Leave it where it&#8217;s set unless you see the sign below.</p>
{_btn("sources","standard","Standard","a few broad papers tend to come up for almost any question, so the same handful of titles may appear on cards for very different questions.")}
{_btn("sources","focused","Focused","trims those catch-all papers so the oracle reaches for ones that match what was actually asked; cards cite sources closer to the visitor&#8217;s question. Switch to Focused if you notice the same few paper titles in the sources of cards for unrelated questions. If, after switching, cards start citing papers that plainly don&#8217;t fit or read worse, switch back to Standard.")}
<p class="note">Both switches are instant.</p>
</body></html>"""
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page.encode("utf-8"))
            return

        if self.path == "/ambient":
            audio_file = os.path.join(_HERE, "ambient.mp3")
            try:
                with open(audio_file, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-type", "audio/mpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Accept-Ranges", "none")
                self.end_headers()
                self.wfile.write(data)
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(HTML.encode())

    def _sse(self, payload):
        try:
            self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode())
            self.wfile.flush()
        except Exception:
            pass

    def _sse_keepalive(self):
        try:
            self.wfile.write(b": keepalive\n\n")
            self.wfile.flush()
        except Exception:
            pass

    def do_POST(self):
        if self.path == "/refocus":
            subprocess.Popen(
                ["osascript", "-e", 'tell application "Google Chrome" to activate'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return

        length = int(self.headers["Content-Length"])
        data   = json.loads(self.rfile.read(length))

        if self.path == "/print":
            try:
                body    = data.get("body", "")
                ref_url = data.get("ref_url", "")
                if ref_url and ("localhost" in ref_url or "127.0.0.1" in ref_url):
                    ref_url = re.sub(r'https?://[^/]+', _qr_base_url(), ref_url)
                raw = data.get("citations", [])
                papers = [{"citation": c.get("cite",""), "url": c.get("url",""),
                           "authors": c.get("authors",""), "title": c.get("title",""),
                           "year": c.get("year","")} for c in raw if c.get("cite") or c.get("title")]
                job_ids = print_label(body, papers, qr_url=ref_url)

                def _wait_for_job(job_id, fallback_secs):
                    if job_id:
                        deadline = _time.time() + 45
                        while _time.time() < deadline:
                            r = subprocess.run(['lpstat', '-o'], capture_output=True, text=True)
                            if job_id not in (r.stdout or ''):
                                break
                            _time.sleep(1)
                        _time.sleep(5)
                    else:
                        _time.sleep(fallback_secs)

                def _cpx_off_after_label(job_id):
                    _wait_for_job(job_id, 12)
                    cpx_off()

                def _ding_after_last_label(job_id):
                    _wait_for_job(job_id, 24)
                    print("Audio: ding", flush=True)
                    subprocess.Popen(
                        ["afplay", "-v", "4", "/System/Library/Sounds/Glass.aiff"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )

                first_id = job_ids[0] if job_ids else None
                last_id  = job_ids[-1] if job_ids else None
                threading.Thread(target=_cpx_off_after_label, args=(first_id,), daemon=True).start()
                threading.Thread(target=_ding_after_last_label, args=(last_id,), daemon=True).start()
                result = {"ok": True}
            except Exception as e:
                print(f"Print error: {e}")
                result = {"ok": False}
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                self.wfile.write(json.dumps(result).encode())
            except Exception:
                pass
            return

        if self.path in ("/compare", "/compare/regenerate"):
            if LIVE_RELOAD:
                _reload_framework()
            cmp_question = data.get("question", "").strip()
            if self.path == "/compare/regenerate":
                version = data.get("version", "")
                versions = [version] if version in PROMPT_VERSIONS else []
                results = run_comparison(cmp_question, versions, force_regen=True)
            else:
                results = run_comparison(cmp_question, PROMPT_VERSIONS, force_regen=False)
            payload = json.dumps({"results": results}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except Exception:
                pass
            return

        # /ask — SSE streaming so the tunnel stays alive during generation
        if LIVE_RELOAD:
            _reload_framework()
        question = data.get("question", "").strip()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        if not question:
            self._sse({"done": True, "answer": None})
            return

        cpx_on()
        context, papers, ref_papers = get_context(question)
        if not context:
            self._sse({"done": True, "answer": None})
            return

        tension, tension_sim = find_nearest_tension(question)
        on_domain = is_on_domain(question)

        if on_domain and tension and tension_sim >= TENSION_RELEVANCE_THRESHOLD:
            path_label = "tension-grounded"
            global _closing_move_counter
            _closing_move = _CLOSING_MOVES[_closing_move_counter % 4]
            _closing_move_counter += 1
            _template = load_prompt_template(ACTIVE_PROMPT_VERSION)
            _template = _template.replace("{closing_move}", _closing_move)
            prompt = render_prompt(_template, question, _format_tension_block(tension), context)
            chunks_text = context
        elif on_domain and (not tension or tension_sim < TENSION_RELEVANCE_THRESHOLD):
            path_label = "weak-match"
            prompt = WEAK_MATCH_PROMPT.format(question=question)
            chunks_text = ""
        else:
            path_label = "off-domain"
            prompt = (
                "The visitor's question lies beyond what the oracle can see. Stay the oracle. "
                "Do not answer the question, and do not invent anything to answer it with.\n\n"
                "Never explain what you are, what you know, or what you cannot do. Never apologize. "
                "Do not mention access, information, or capability. Speak in plain statements and images, "
                "as the oracle does.\n\n"
                "Acknowledge in the oracle's way that this question lies past its horizon, then turn toward "
                "what it can see, the long study of how people learn, struggle, and come to know. "
                "If the question hides a wish to find a way or to know something, speak to that wish.\n\n"
                "Write one bold opening sentence. Then one to three sentences in the same voice with real weight. "
                "End with a single question that turns the visitor toward something within the oracle's sight.\n\n"
                "Never refer to the card, the research, the sources, or your own writing. "
                "Do not include examples. Only speak.\n\n"
                f"VISITOR QUESTION: {question}"
            )
            chunks_text = ""

        # Topical grounding gate setup (tension-grounded path only)
        grounded        = False
        grounded_reason = None
        candidate_set   = set()
        if path_label == "tension-grounded":
            tension_block_str = _format_tension_block(tension)
            candidate_set     = _topical_candidate_set(context, tension_block_str, question)
            if not candidate_set:
                grounded_reason = "no_clean_term"
                print("Grounding gate: candidate set empty — will accept first valid card", flush=True)

        # Speed cap and wall-clock budget
        speed_cap      = _SPEED_ATTEMPTS[_RUNTIME_SPEED]
        gate_enabled   = (path_label == "tension-grounded" and
                          _RUNTIME_SPEED != "single" and bool(candidate_set))
        _wall_deadline = _time.monotonic() + 45.0

        # Run LLM in a background thread; send SSE keepalives every 4s while
        # waiting for the first token so proxies don't drop idle connections.
        token_q = queue.Queue()

        prompt_chars = len(SYSTEM_PROMPT) + len(prompt)
        print(f"Sending to Ollama: ~{prompt_chars} chars (~{prompt_chars//4} tokens est) "
              f"speed={_RUNTIME_SPEED} sources={_RUNTIME_SOURCES}", flush=True)

        def _llm_worker():
            try:
                r = req.post(
                    OLMO_URL,
                    json={"model": OLMO_MODEL, "system": SYSTEM_PROMPT, "prompt": prompt, "stream": True},
                    stream=True, timeout=25
                )
                r.raise_for_status()
                token_count = 0
                for line in r.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        tok = chunk.get("response", "")
                        if tok:
                            token_q.put(("token", tok))
                            token_count += 1
                        if chunk.get("done"):
                            print(f"LLM done: {token_count} tokens")
                            token_q.put(("done", None))
                            return
            except Exception as e:
                print(f"LLM worker error: {e}")
                token_q.put(("error", str(e)))

        # Regenerate-until-valid loop. Cap from runtime speed setting; wall-clock ≤45s.
        best_card         = None   # (score, len, text, failures)
        best_grounded_card = None  # first card that passes validator + gate
        attempts          = 0
        is_fallback       = False
        fallback_failures = []
        _card_t0          = _time.monotonic()

        for attempt in range(1, speed_cap + 1):
            if _time.monotonic() > _wall_deadline:
                print(f"Wall-clock budget exceeded at attempt {attempt}, using best so far", flush=True)
                break
            attempts  = attempt
            _att_t0   = _time.monotonic()
            threading.Thread(target=_llm_worker, daemon=True).start()

            raw = ""
            while True:
                try:
                    kind, val = token_q.get(timeout=4)
                except queue.Empty:
                    self._sse_keepalive()
                    if _time.monotonic() > _wall_deadline:
                        print(f"Wall-clock hit mid-stream, aborting attempt {attempt}", flush=True)
                        raw = ""
                        break
                    continue
                if kind == "token":
                    raw += val
                elif kind == "done":
                    break
                else:
                    print(f"LLM error on attempt {attempt}: {val}")
                    raw = ""
                    break

            if not raw.strip():
                print(f"Attempt {attempt}: empty response, retrying")
                continue

            card = _enforce_format(raw)
            card = _cleanup_card(card)
            passes, failures, score = _validate_card(card, chunks_text)
            _att_s = _time.monotonic() - _att_t0
            print(f"Attempt {attempt}: len={len(card)} score={score} failures={failures} elapsed={_att_s:.1f}s")

            if passes:
                if gate_enabled:
                    card_terms = _content_tokens(card)
                    if card_terms & candidate_set:
                        grounded        = True
                        grounded_reason = "has_term"
                        best_grounded_card = (score, len(card), card, [])
                        break
                    # Passes validator but misses gate — keep as best, try more
                    if best_card is None or score > best_card[0]:
                        best_card = (score, len(card), card, [])
                else:
                    # Single speed, non-tension path, or empty candidate set
                    if path_label == "tension-grounded" and candidate_set:
                        card_terms = _content_tokens(card)
                        grounded   = bool(card_terms & candidate_set)
                        grounded_reason = "has_term" if grounded else "no_term_single"
                    best_card = (score, len(card), card, [])
                    break
            else:
                if best_card is None or score > best_card[0] or (score == best_card[0] and len(card) < best_card[1]):
                    best_card = (score, len(card), card, failures)

        # Resolve final card
        if best_grounded_card is not None:
            resolved_card = best_grounded_card
        elif gate_enabled and best_card is not None:
            # Cap exhausted without grounded card — use best valid (grounded=False)
            grounded_reason = "cap_exhausted"
            resolved_card   = best_card
            print(f"Grounding gate: cap_exhausted — using best ungrounded card", flush=True)
        elif gate_enabled and best_card is None:
            # No valid card at all — generate tension-only fallback via generate_card
            grounded_reason = "cap_exhausted"
            print("Grounding gate: no valid card — generating tension-only fallback", flush=True)
            try:
                _tmpl_v4     = load_prompt_template(ACTIVE_PROMPT_VERSION)
                _tmpl_v4     = _tmpl_v4.replace("{closing_move}", _closing_move)
                _fallback_p  = render_prompt(_tmpl_v4, question, _format_tension_block(tension), "")
                _fb_card, _, _ = generate_card(_fallback_p, "")
                resolved_card = (0, len(_fb_card or ""), _fb_card or "", ["fallback"])
            except Exception as _fbe:
                print(f"Fallback generation error: {_fbe}", flush=True)
                self._sse({"done": True, "answer": None})
                return
        else:
            resolved_card = best_card

        if resolved_card is None:
            self._sse({"done": True, "answer": None})
            return

        # ── Opener-diversity backstop ─────────────────────────────────────────
        # After the main loop picks a card, check its closing-question opener
        # (first 3 words, lowercased/punct-stripped) against the last 4 accepted
        # openers. If it collides, try up to 3 extra non-streaming generations.
        # After 3 extra attempts (or if no collision), accept and update the window.
        # Never blocks or falls back over an opener.
        _opener_regen_count = 0
        _opener_hit_cap     = False
        _card_opener = _extract_opener(resolved_card[2])
        with _opener_lock:
            _opener_collides = bool(_card_opener) and _card_opener in _RECENT_OPENERS
        if _opener_collides:
            print(f"Opener backstop: collision on {_card_opener!r}, trying up to 3 extra", flush=True)
            for _oa in range(1, 4):
                _opener_regen_count = _oa
                _ob_card, _, _ob_fail = generate_card(prompt, chunks_text)
                if _ob_card:
                    _ob_opener = _extract_opener(_ob_card)
                    with _opener_lock:
                        _ob_collides = bool(_ob_opener) and _ob_opener in _RECENT_OPENERS
                    print(f"Opener regen {_oa}: opener={_ob_opener!r} collides={_ob_collides}", flush=True)
                    if not _ob_collides:
                        resolved_card = (resolved_card[0], len(_ob_card), _ob_card, _ob_fail)
                        _card_opener  = _ob_opener
                        break
            else:
                _opener_hit_cap = True
                print(f"Opener backstop: cap hit after 3 — accepting {_card_opener!r}", flush=True)
        with _opener_lock:
            if _card_opener:
                _RECENT_OPENERS.append(_card_opener)
                if len(_RECENT_OPENERS) > 4:
                    _RECENT_OPENERS.pop(0)
        print(f"Opener backstop: accepted opener={_card_opener!r} "
              f"regen={_opener_regen_count} cap={_opener_hit_cap} "
              f"window={_RECENT_OPENERS}", flush=True)

        _, _, full_reply, fallback_failures = resolved_card
        is_fallback = len(fallback_failures) > 0
        _card_s = _time.monotonic() - _card_t0
        print(f"Card done: attempt {attempts}, grounded={grounded} ({grounded_reason}), "
              f"fallback={is_fallback}, total={_card_s:.1f}s", flush=True)

        # Relevance footer on every tension-grounded card (no authorship claim)
        if path_label == "tension-grounded":
            full_reply += "\n\n— (from the research this question reaches toward)"
        cit_payload = [
            {"cite": p["citation"], "url": p["url"], "authors": p["authors"],
             "title": p["title"], "year": p["year"]}
            for p in papers
        ]
        ref_uid = uuid.uuid4().hex[:12]
        body_only = full_reply.split("\n\n— ")[0] if "\n\n— " in full_reply else full_reply
        _save_ref_entry(ref_uid, {"question": question, "ref_papers": ref_papers, "card": body_only})
        # Append to questions.log
        try:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(LOG_PATH, "a", encoding="utf-8") as lf:
                lf.write(f"\n{'='*60}\n")
                lf.write(f"TIME:     {ts}\n")
                lf.write(f"VERSION:  {GIT_VERSION}  (github.com/emilyreardon/research-for-humans/commit/{GIT_VERSION})\n")
                lf.write(f"SPEED:    {_RUNTIME_SPEED}  SOURCES: {_RUNTIME_SOURCES}\n")
                lf.write(f"ATTEMPTS: {attempts}  GROUNDED: {grounded} ({grounded_reason})"
                         f"{'  FALLBACK ' + str(fallback_failures) if is_fallback else ''}\n")
                lf.write(f"QUESTION: {question}\n")
                lf.write(f"CARD:\n{body_only}\n")
                lf.write(f"PAPERS:\n")
                for p in papers:
                    first_author = p['authors'].split('|')[0].strip()
                    lf.write(f"  {first_author} ({p['year']}) {p['title']}\n")
                    if p.get('url'):
                        lf.write(f"  {p['url']}\n")
        except Exception as e:
            print(f"Log error: {e}")
        log_question({
            "timestamp":      datetime.datetime.now().isoformat(),
            "question":       question,
            "source":         "ask",
            "path":           path_label,
            "tension_id":     tension.get("id") if tension else None,
            "tension_name":   _tension_label(tension),
            "tension_score":  round(tension_sim, 4) if path_label == "tension-grounded" else None,
            "prompt_version": ACTIVE_PROMPT_VERSION,
            "speed":          _RUNTIME_SPEED,
            "sources":        _RUNTIME_SOURCES,
            "grounded":       grounded,
            "grounded_reason": grounded_reason,
            "card":           body_only if path_label == "tension-grounded" else None,
            "attempt_count":  attempts,
        })
        self._sse({"done": True, "answer": full_reply, "citations": cit_payload, "ref_uid": ref_uid})

def _autopull_worker():
    while True:
        threading.Event().wait(60)
        try:
            result = subprocess.run(
                ["git", "pull"], cwd=_HERE, capture_output=True, text=True
            )
            if "Already up to date." not in result.stdout:
                print(f"git pull: {result.stdout.strip()}")
        except Exception as e:
            print(f"git pull error: {e}")

if LIVE_RELOAD:
    threading.Thread(target=_autopull_worker, daemon=True).start()
    print("Auto-pull enabled (every 60s)")

threading.Thread(target=_cpx_connect, daemon=True).start()

def _audio_loop():
    audio_file = os.path.join(_HERE, "ambient.mp3")
    if not os.path.exists(audio_file):
        print("Audio: ambient.mp3 not found, skipping", flush=True)
        return
    print("Audio: starting ambient loop", flush=True)
    while True:
        subprocess.run(["afplay", "-v", "2", audio_file])

# Ambient audio is now served via /ambient and played client-side (see HTML)
# threading.Thread(target=_audio_loop, daemon=True).start()

_ADMIN_PORT = 5050
_ADMIN_BASE  = f"http://{LOCAL_IP}:{_ADMIN_PORT}"
print(f"Control dashboard: {_ADMIN_BASE}/admin", flush=True)
print(f"Running at http://127.0.0.1:{_ADMIN_PORT}", flush=True)

# Generate control-panel.html for quick access without a browser bookmark
try:
    _cp_links = "\n".join(
        f'<li><a href="{_ADMIN_BASE}/admin/{ctrl}/{val}" style="font-size:18px;">'
        f'{ctrl}: {val}</a></li>'
        for ctrl, values in [("speed", ["quality","balanced","fast","single"]),
                              ("sources", ["standard","focused"])]
        for val in values
    )
    _cp_html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><title>Oracle Control Panel</title>
<style>body{{font-family:system-ui;max-width:480px;margin:40px auto;padding:0 16px;}}
h1{{font-size:20px;}} li{{margin:8px 0;}} a{{color:#1a1a1a;}}</style>
</head><body>
<h1>Oracle Control Panel</h1>
<p><a href="{_ADMIN_BASE}/admin" style="font-size:20px;font-weight:bold;">Open Dashboard</a></p>
<hr>
<ul>{_cp_links}</ul>
<p style="font-size:12px;color:#999;">Links go to {_ADMIN_BASE}</p>
</body></html>"""
    with open(os.path.join(_HERE, "control-panel.html"), "w", encoding="utf-8") as _cpf:
        _cpf.write(_cp_html)
    print(f"Control panel: {os.path.join(_HERE, 'control-panel.html')}", flush=True)
except Exception as _cpe:
    print(f"control-panel.html: generation failed ({_cpe})", flush=True)

ThreadingHTTPServer(("0.0.0.0", _ADMIN_PORT), Handler).serve_forever()
