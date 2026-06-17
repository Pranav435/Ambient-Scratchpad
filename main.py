# Minimum blended score to consider notes as related
# Higher = fewer but more meaningful relations (0.45 is fairly strict)
RELATION_MIN_SCORE = 0.45
# Minimum *raw embedding* cosine similarity a candidate must clear before any
# category/entity boost is applied. This is the meaning gate: boosts can only
# re-rank notes that are already semantically similar, never manufacture a
# relation between notes that merely share a keyword or category.
RELATION_MIN_EMBED_SIM = 0.55
# Embedding similarity at/above which a pair is treated as a confident semantic
# match (skips the optional LLM verification band — see USE_LLM_RELATION_VERIFY).
RELATION_STRONG_EMBED_SIM = 0.80
# Maximum number of related notes to store per note
RELATED_MAX_PER_NOTE = 5
# main.py
"""
Ambient Scratchpad — rumps menubar + PyObjC scrollable windows for large text.

Requirements:
- python -m pip install rumps google-generativeai python-dotenv pyobjc
- Put GEMINI_API_KEY=... in .env

Behavior:
- Uses rumps for the menubar and quick capture.
- Uses AppKit (PyObjC) for a non-blocking, scrollable, resizable window to show All Notes + Actionable Summary.
- Falls back to rumps.alert (truncated) if PyOBJC isn't available.
"""
import os
import sys
import json
import time
import uuid
import threading
import sqlite3
import struct
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# suppress noisy pkg_resources deprecation warning
import warnings
warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated as an API.*", category=UserWarning)

# UI: rumps for menu bar
try:
    import rumps
except Exception:
    print("Install rumps: pip install rumps")
    raise

# dotenv
try:
    from dotenv import load_dotenv
except Exception:
    print("Install python-dotenv: pip install python-dotenv")
    raise

# Gemini client
try:
    from google import genai
    from google.genai import types
except Exception:
    print("Install google-genai: pip install google-genai")
    raise

# Attempt to import PyObjC / AppKit for native windows
PYOBJC_AVAILABLE = False
try:
    from AppKit import (
        NSApplication, NSWindow, NSScrollView, NSTextView, NSMakeRect,
        NSRunningApplication, NSApplicationActivateIgnoringOtherApps, NSBackingStoreBuffered,
        NSTableView, NSTableColumn, NSSearchField, NSSplitView, NSStackView, NSTextField,
        NSColor
    , NSVisualEffectView, NSAppearance, NSFont, NSColor, NSBox, NSEvent, NSEventMaskKeyDown)
    from Foundation import NSString, NSObject, NSAttributedString, NSNotificationCenter
    from PyObjCTools import AppHelper
    import objc
    PYOBJC_AVAILABLE = True
except Exception:
    PYOBJC_AVAILABLE = False

# Extra AppKit symbols used by the redesigned "All Notes" window. Imported
# separately with graceful fallbacks so the core app still loads on older
# systems where a constant might be missing.
if PYOBJC_AVAILABLE:
    try:
        from AppKit import (
            NSView, NSMutableParagraphStyle,
            NSFontAttributeName, NSForegroundColorAttributeName,
            NSParagraphStyleAttributeName,
            NSLinkAttributeName, NSUnderlineStyleAttributeName,
        )
        from Foundation import NSIndexSet
    except Exception:
        logging.warning("Some AppKit symbols unavailable; UI will use plain styling")
    # Autoresizing mask + layout constants (use numeric fallbacks if absent).
    try:
        from AppKit import (
            NSViewWidthSizable, NSViewHeightSizable, NSViewMinXMargin,
            NSViewMaxXMargin, NSViewMinYMargin, NSViewMaxYMargin,
        )
    except Exception:
        NSViewWidthSizable, NSViewHeightSizable = 2, 16
        NSViewMinXMargin, NSViewMaxXMargin = 1, 4
        NSViewMinYMargin, NSViewMaxYMargin = 8, 32
    # Controls used by the note context menu + editor.
    try:
        from AppKit import NSMenu, NSMenuItem, NSPopUpButton, NSAlert
    except Exception:
        NSMenu = NSMenuItem = NSPopUpButton = NSAlert = None
    # Controls + APIs for the two-view UI (list/bubble), settings, and a11y.
    try:
        from AppKit import NSSegmentedControl, NSSlider, NSStepper
    except Exception:
        NSSegmentedControl = NSSlider = NSStepper = None
    try:
        from Foundation import NSTimer
    except Exception:
        NSTimer = None
    try:
        from AppKit import (
            NSAccessibilityElement, NSAccessibilityCustomAction,
            NSAccessibilityPostNotification, NSAccessibilityLayoutChangedNotification,
        )
        _A11Y_OK = True
    except Exception:
        NSAccessibilityElement = NSAccessibilityCustomAction = None
        NSAccessibilityPostNotification = NSAccessibilityLayoutChangedNotification = None
        _A11Y_OK = False

# Category → glyph used throughout the redesigned UI for quick visual scanning.
CATEGORY_GLYPHS = {
    "Idea": "💡",
    "To-Do": "✅",
    "Code Snippet": "💻",
    "Link": "🔗",
    "Quote": "❝",
    "Contact": "👤",
    "Other": "📝",
}

def category_glyph(category: str) -> str:
    return CATEGORY_GLYPHS.get(category or "", "📝")

# ==== PATHWAY: primary streaming engine for real-time note ingestion
PATHWAY_IMPORTED = False
try:
    warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated as an API.*", category=UserWarning)
    import pathway as pw
    PATHWAY_IMPORTED = True
except Exception:
    logging.warning("Pathway not installed - real-time streaming disabled. Install with: pip install pathway")
    PATHWAY_IMPORTED = False

# optional scrapers / numpy
try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except Exception:
    SCRAPING_AVAILABLE = False

import math  # Always import math for fallback operations

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False

# ---------------- Config ----------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CLASSIFY_MODEL = os.getenv("GEMINI_CLASSIFY_MODEL", "models/gemini-flash-latest")
EMBED_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
USE_PATHWAY_STREAM = os.getenv("USE_PATHWAY_STREAM", "1") == "1"  # Pathway streaming enabled by default
# When enabled, candidate relations that pass the embedding gate but aren't a
# confident match are verified with a single batched Gemini call that decides
# whether they are genuinely related *in meaning* and produces a short reason.
# Off by default: it adds one LLM call per capture. Set to 1 for maximum precision.
USE_LLM_RELATION_VERIFY = os.getenv("USE_LLM_RELATION_VERIFY", "0") == "1"

# --- ABSOLUTE PATHS (replace the 3 relative constants with this) ---
BASE_DIR = os.path.abspath(".")
INPUT_STREAM = os.path.join(BASE_DIR, "input_stream.jsonl")
DB_FILE = os.path.join(BASE_DIR, "knowledge_base.db")
LOG_FILE = os.path.join(BASE_DIR, "ambient.log")


# Logging
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logging.getLogger().addHandler(console)
logging.info("Ambient Scratchpad startup")

# Gemini client configuration
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------- User settings (persisted) ----------------
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
# Non-tuning preferences (e.g. which view to open by default).
APP_SETTINGS = {"default_view": "list"}

# Settings the user can edit and what kind of value they hold.
_TUNABLE_FLOATS = ("RELATION_MIN_SCORE", "RELATION_MIN_EMBED_SIM")
_TUNABLE_INTS = ("RELATED_MAX_PER_NOTE",)
_TUNABLE_BOOLS = ("USE_LLM_RELATION_VERIFY",)

def current_settings() -> dict:
    """Snapshot of all user-editable settings (for the settings UI)."""
    return {
        "RELATION_MIN_SCORE": RELATION_MIN_SCORE,
        "RELATION_MIN_EMBED_SIM": RELATION_MIN_EMBED_SIM,
        "RELATED_MAX_PER_NOTE": RELATED_MAX_PER_NOTE,
        "USE_LLM_RELATION_VERIFY": USE_LLM_RELATION_VERIFY,
        "default_view": APP_SETTINGS.get("default_view", "list"),
    }

def apply_settings(data: dict):
    """Apply a settings dict to the live module globals (no persistence)."""
    global RELATION_MIN_SCORE, RELATION_MIN_EMBED_SIM, RELATED_MAX_PER_NOTE, USE_LLM_RELATION_VERIFY
    try:
        for k in _TUNABLE_FLOATS:
            if k in data:
                globals()[k] = max(0.0, min(1.0, float(data[k])))
        for k in _TUNABLE_INTS:
            if k in data:
                globals()[k] = max(1, int(data[k]))
        for k in _TUNABLE_BOOLS:
            if k in data:
                globals()[k] = bool(data[k])
        if data.get("default_view") in ("list", "bubbles"):
            APP_SETTINGS["default_view"] = data["default_view"]
    except Exception:
        logging.exception("apply_settings failed")

def load_settings():
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            apply_settings(json.load(f))
        logging.info("Loaded settings from %s", SETTINGS_FILE)
    except FileNotFoundError:
        pass
    except Exception:
        logging.exception("load_settings failed")

def save_settings(data: dict):
    apply_settings(data)
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(current_settings(), f, indent=2)
        logging.info("Saved settings to %s", SETTINGS_FILE)
        return True
    except Exception:
        logging.exception("save_settings failed")
        return False

load_settings()

# ---------------- DB init + migration ----------------
# Cache of the notes table's column names — the schema is stable for a run, so
# we read it once instead of issuing a PRAGMA on every single-note fetch.
# Invalidated automatically if the db connection is swapped (e.g. in tests).
_NOTES_COLUMNS_CACHE = None
_NOTES_COLUMNS_CONN = None

def _notes_columns(cursor):
    global _NOTES_COLUMNS_CACHE, _NOTES_COLUMNS_CONN
    if _NOTES_COLUMNS_CACHE is None or _NOTES_COLUMNS_CONN is not db_conn:
        cursor.execute("PRAGMA table_info(notes)")
        _NOTES_COLUMNS_CACHE = [r[1] for r in cursor.fetchall()]
        _NOTES_COLUMNS_CONN = db_conn
    return _NOTES_COLUMNS_CACHE

def init_db_and_migrate():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS notes (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            content TEXT,
            scraped_content TEXT,
            category TEXT,
            tags TEXT,
            entities TEXT,
            embedding BLOB,
            related TEXT,
            summary TEXT
        )"""
    )
    conn.commit()
    # Add summary column if missing (defensive)
    try:
        c.execute("PRAGMA table_info(notes)")
        cols = [r[1] for r in c.fetchall()]
        if "summary" not in cols:
            c.execute("ALTER TABLE notes ADD COLUMN summary TEXT")
            conn.commit()
            logging.info("DB migration: added 'summary' column.")
    except Exception as e:
        logging.warning("DB migration error: %s", e)
    return conn

db_conn = init_db_and_migrate()
db_lock = threading.Lock()

# ---------------- Helpers ----------------
def _bootstrap_backfill_if_needed(max_lines=5000):
    # If nothing is in SQLite yet, process existing raw captures once.
    with db_lock:
        c = db_conn.cursor()
        try:
            c.execute("SELECT COUNT(*) FROM notes")
            count = c.fetchone()[0]
        except Exception:
            count = 0
    if count > 0:
        return 0

    raw = read_raw_input_stream(limit=max_lines)
    done = 0
    for obj in raw:
        try:
            process_capture_object(obj, store=True)
            done += 1
        except Exception:
            logging.exception("Backfill process failed for a raw line")
    logging.info("Bootstrap backfill processed %d historical captures.", done)
    return done


def now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def append_to_input_stream(obj: dict):
    os.makedirs(os.path.dirname(INPUT_STREAM) or ".", exist_ok=True)
    line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

    # Write as bytes, then flush and fsync so Pathway's watcher sees it immediately
    f = open(INPUT_STREAM, "ab", buffering=0)
    try:
        f.write(line)
        try:
            f.flush()
        except Exception:
            pass
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    finally:
        f.close()

    logging.info("Appended capture %s", obj.get("id"))


def read_raw_input_stream(limit: int = 500) -> List[dict]:
    if not os.path.exists(INPUT_STREAM):
        return []
    try:
        with open(INPUT_STREAM, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        logging.exception("read_raw_input_stream failed")
        return []
    if not lines:
        return []
    lines = lines[-limit:]
    parsed = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except Exception:
            obj = {"id": str(uuid.uuid4()), "timestamp": now_iso(), "content": ln}
        parsed.append(obj)
    return parsed[::-1]

def is_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

def fetch_url_text(url: str) -> str:
    if not SCRAPING_AVAILABLE:
        return ""
    try:
        r = requests.get(url, timeout=6.0, headers={"User-Agent":"ambient-scratchpad/1.0"})
        if r.status_code != 200:
            return ""
        ct = r.headers.get("content-type", "")
        if "html" in ct:
            soup = BeautifulSoup(r.text, "html.parser")
            article = soup.find("article")
            text = article.get_text(separator="\n") if article else soup.get_text(separator="\n")
            title = soup.title.string.strip() if soup.title else ""
            return json.dumps({"title": title, "text": text})
        return ""
    except Exception as e:
        logging.debug("fetch_url_text error: %s", e)
        return ""

# ---------------- Gemini helpers ----------------
def derive_summary_from_text(text: str, max_len: int = 160) -> str:
    """Build a readable one-line summary from raw note text.

    Used whenever the model fails to return a usable summary. Takes the first
    sentence (or line), collapses whitespace, and truncates on a word boundary —
    far better than a blank summary or a raw content dump.
    """
    if not text:
        return "No summary"
    s = " ".join(text.split()).strip()
    if not s:
        return "No summary"
    # Prefer the first sentence / line as the gist.
    import re
    first = re.split(r"(?<=[.!?])\s+|\n", s, maxsplit=1)[0].strip()
    candidate = first if 0 < len(first) <= max_len else s
    return friendly_snippet(candidate, max_len)


def _clean_summary(summary: Any, fallback_text: str) -> str:
    """Normalize a model-produced summary, deriving one from text if it's unusable.

    Treats None, empty, whitespace-only, and the literal 'No summary' as missing —
    fixing cases where the model returns ``"summary": ""`` (which ``setdefault``
    silently kept) and notes ended up with no summary.
    """
    if isinstance(summary, str):
        cleaned = summary.strip()
        if cleaned and cleaned.lower() != "no summary":
            return cleaned
    return derive_summary_from_text(fallback_text)


def classify_and_tag_with_gemini(text: str) -> Dict[str, Any]:
    if not text:
        return {"category":"Other","tags":[],"entities":[],"summary":"No summary"}
    prompt = f"""You are an assistant that extracts structured metadata from a short note.
Return a JSON object only with these fields:
- category: one of ["Idea","To-Do","Code Snippet","Link","Quote","Contact","Other"]
- tags: list (max 6) of lowercase tags
- entities: list of named entities
- summary: one-sentence summary

Input:
\"\"\"{text}\"\"\"

Return ONLY the JSON object, no other text.
"""
    try:
        if not gemini_client:
            raise Exception("Gemini client not configured")
        response = gemini_client.models.generate_content(
            model=CLASSIFY_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1024,
            )
        )
        # Handle case where response.text is None (e.g., blocked content, empty response, or max tokens hit)
        if response.text is None:
            logging.warning("Gemini classify returned None response.text (finish_reason may be MAX_TOKENS), using fallback")
            # Try to extract partial response from candidates
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                partial_text = response.candidates[0].content.parts[0].text
                if partial_text:
                    logging.info("Extracted partial response from candidates: %s", partial_text[:100])
                    out = partial_text.strip()
                else:
                    return {"category":"Other","tags":[],"entities":[],"summary": derive_summary_from_text(text)}
            else:
                return {"category":"Other","tags":[],"entities":[],"summary": text[:200] or "No summary"}
        else:
            out = response.text.strip()
        # Remove markdown code blocks if present (```json ... ```)
        if out.startswith("```"):
            lines = out.split("\n")
            # Remove first line (```json) and last line (```)
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            out = "\n".join(lines)
        try:
            data = json.loads(out)
        except Exception:
            s = out.find("{"); e = out.rfind("}")
            if s != -1 and e != -1:
                data = json.loads(out[s:e+1])
            else:
                data = {"category":"Other","tags":[],"entities":[],"summary": derive_summary_from_text(text)}
        data.setdefault("category","Other")
        data.setdefault("tags",[])
        data.setdefault("entities",[])
        # Always normalize the summary: the model sometimes returns "" or
        # whitespace, which would otherwise leave the note with no summary.
        data["summary"] = _clean_summary(data.get("summary"), text)
        logging.debug("AI classify: %s", data)
        return data
    except Exception:
        logging.exception("Gemini classify error")
        return {"category":"Other","tags":[],"entities":[],"summary": derive_summary_from_text(text)}

def get_embedding_gemini(text: str) -> List[float]:
    """Get embedding vector for text using Gemini API.
    
    Falls back to a simple but meaningful text-based embedding if API fails.
    The fallback uses character n-gram frequencies to create a sparse vector
    that at least captures some textual similarity.
    """
    if not text:
        return []
    try:
        if not gemini_client:
            raise Exception("Gemini client not configured")
        result = gemini_client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
        )
        vec = result.embeddings[0].values
        return list(vec)
    except Exception:
        logging.exception("Gemini embedding error, using fallback")
        # Create a more meaningful fallback embedding using character n-grams
        # This produces a 768-dim vector (matching Gemini text-embedding-004) based on text features
        text_lower = text.lower()
        vec = [0.0] * 768
        # Use character bigrams and trigrams to populate the vector
        for i in range(len(text_lower)):
            # Unigrams
            idx = ord(text_lower[i]) % 256
            vec[idx] += 1.0
            # Bigrams
            if i + 1 < len(text_lower):
                idx = (ord(text_lower[i]) * 31 + ord(text_lower[i+1])) % 256 + 256
                vec[idx] += 1.0
            # Trigrams
            if i + 2 < len(text_lower):
                idx = (ord(text_lower[i]) * 31 * 31 + ord(text_lower[i+1]) * 31 + ord(text_lower[i+2])) % 256 + 512
                vec[idx] += 1.0
        # Normalize the vector
        norm = math.sqrt(sum(x*x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

# ---------------- Embedding persistence ----------------
def get_embedding_bytes(emb: List[float]) -> bytes:
    try:
        import numpy as _np
        return _np.array(emb, dtype=_np.float64).tobytes()
    except Exception:
        return b"".join(struct.pack("d", float(x)) for x in emb)

def embedding_bytes_to_list(blob) -> List[float]:
    if not blob:
        return []
    if isinstance(blob, (bytes, bytearray)):
        try:
            n = len(blob) // 8
            return [struct.unpack_from("d", blob, i*8)[0] for i in range(n)]
        except Exception:
            try:
                return json.loads(blob)
            except Exception:
                return []
    else:
        try:
            return json.loads(blob)
        except Exception:
            return []

def read_all_embeddings_from_db() -> List[Dict[str, Any]]:
    out = []
    with db_lock:
        c = db_conn.cursor()
        try:
            c.execute("SELECT id, embedding FROM notes")
            rows = c.fetchall()
        except Exception:
            return []
    for rid, blob in rows:
        arr = embedding_bytes_to_list(blob)
        if arr:
            out.append({"id": rid, "emb": arr})
    return out

# Target embedding dimension (text-embedding-004 produces 768 by default)
TARGET_EMBEDDING_DIM = 768

def normalize_embedding_dimension(emb: List[float], target_dim: int = TARGET_EMBEDDING_DIM) -> List[float]:
    """Normalize embedding to target dimension.
    
    If embedding is longer, truncate to target dimension (safe for MRL-trained models 
    where leading dimensions preserve most information).
    If embedding is shorter, pad with zeros (not ideal but better than failing).
    """
    if not emb:
        return []
    if len(emb) == target_dim:
        return emb
    if len(emb) > target_dim:
        # Truncate - MRL embeddings preserve info in leading dimensions
        return emb[:target_dim]
    else:
        # Pad with zeros - not ideal but allows comparison
        return emb + [0.0] * (target_dim - len(emb))

# ---------------- Similarity & storage ----------------
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors.
    
    Handles mismatched dimensions by normalizing to the same length.
    Returns 0.0 if vectors are empty or contain only zeros.
    """
    if not a or not b:
        return 0.0
    # Handle vectors of different lengths by normalizing to smaller dimension
    # This allows comparing embeddings from different models/versions
    if len(a) != len(b):
        logging.debug("Cosine similarity: normalizing vector lengths (%d vs %d)", len(a), len(b))
        min_dim = min(len(a), len(b))
        a = a[:min_dim]
        b = b[:min_dim]
    if NUMPY_AVAILABLE:
        a_np = np.array(a, dtype=float); b_np = np.array(b, dtype=float)
        la = np.linalg.norm(a_np); lb = np.linalg.norm(b_np)
        if la == 0 or lb == 0:
            return 0.0
        return float(np.dot(a_np, b_np) / (la*lb))
    else:
        lena = math.sqrt(sum(x*x for x in a))
        lenb = math.sqrt(sum(x*x for x in b))
        if lena == 0 or lenb == 0:
            return 0.0
        s = sum(ai*bi for ai,bi in zip(a,b))
        return s / (lena * lenb)

# Category affinity matrix - categories that are semantically related
# Higher values mean stronger implicit relationship
CATEGORY_AFFINITY = {
    ("To-Do", "Idea"): 0.15,        # Ideas often become to-dos
    ("Idea", "To-Do"): 0.15,
    ("To-Do", "Code Snippet"): 0.10, # Code snippets support to-dos
    ("Code Snippet", "To-Do"): 0.10,
    ("Idea", "Code Snippet"): 0.12,  # Ideas implemented as code
    ("Code Snippet", "Idea"): 0.12,
    ("Link", "Idea"): 0.08,          # Links inspire ideas
    ("Idea", "Link"): 0.08,
    ("Quote", "Idea"): 0.10,         # Quotes inspire ideas
    ("Idea", "Quote"): 0.10,
    ("Contact", "To-Do"): 0.12,      # People assigned to tasks
    ("To-Do", "Contact"): 0.12,
}

def _entity_tokens(name: str) -> set:
    """Split an entity/tag into meaningful whole-word tokens.

    Word-boundary tokens (rather than raw substrings) let us match
    "John Smith" with "John" without also matching "Sam" with "Samsung".
    Single-character and purely numeric tokens are dropped as noise.
    """
    import re
    toks = re.split(r"[^a-z0-9]+", name.lower().strip())
    return {t for t in toks if len(t) >= 2 and not t.isdigit()}


def compute_entity_overlap_score(entities1: List[str], entities2: List[str]) -> float:
    """Compute meaning-based overlap between two entity lists.

    Matching is on whole-word tokens, not raw substrings, so it captures
    genuinely shared people/projects/topics ("John Smith" ~ "John") while
    rejecting coincidental string containment ("Sam" in "Samsung").
    """
    if not entities1 or not entities2:
        return 0.0

    # Normalize entities for comparison
    norm1 = set(e.lower().strip() for e in entities1 if e)
    norm2 = set(e.lower().strip() for e in entities2 if e)

    if not norm1 or not norm2:
        return 0.0

    # Exact (whole-entity) matches are the strongest signal.
    exact_overlap = len(norm1 & norm2)

    # Token-level matches: an entity in one list shares a whole word with an
    # entity in the other (e.g. "John Smith" / "call John"). Counted once per
    # unmatched entity so a single shared token can't dominate the score.
    tokens1 = {e: _entity_tokens(e) for e in norm1}
    tokens2 = {e: _entity_tokens(e) for e in norm2}
    partial_overlap = 0.0
    for e1 in norm1 - norm2:
        for e2 in norm2 - norm1:
            if tokens1[e1] and tokens1[e1] & tokens2[e2]:
                partial_overlap += 0.5
                break  # at most one partial credit per entity

    total_overlap = exact_overlap + partial_overlap
    union_size = len(norm1 | norm2)

    return min(total_overlap / union_size, 1.0) if union_size > 0 else 0.0

def compute_tag_semantic_score(tags1: List[str], tags2: List[str]) -> float:
    """Compute overlap between two tag lists using whole-word tokens.

    Uses exact Jaccard overlap plus shared whole-word tokens (so "python"
    matches "python tips"). Deliberately avoids substring / shared-prefix
    matching, which links unrelated tags by spelling accident ("portal" /
    "porter") rather than meaning. Tags are a weak, lexical signal and are
    not used by the live relation pipeline; this kept consistent for callers
    that opt back in.
    """
    if not tags1 or not tags2:
        return 0.0

    norm1 = set(t.lower().strip() for t in tags1 if t)
    norm2 = set(t.lower().strip() for t in tags2 if t)

    if not norm1 or not norm2:
        return 0.0

    # Exact overlap (Jaccard)
    intersection = len(norm1 & norm2)
    union = len(norm1 | norm2)
    jaccard = intersection / union if union > 0 else 0.0

    # Shared whole-word tokens between non-identical tags.
    tokens1 = {t: _entity_tokens(t) for t in norm1}
    tokens2 = {t: _entity_tokens(t) for t in norm2}
    partial_score = 0.0
    for t1 in norm1 - norm2:
        for t2 in norm2 - norm1:
            if tokens1[t1] and tokens1[t1] & tokens2[t2]:
                partial_score += 0.3
                break

    partial_normalized = min(partial_score / max(len(norm1), len(norm2)), 0.5)

    return min(jaccard + partial_normalized, 1.0)

def compute_related_with_scores(
    embedding: List[float],
    top_k: int = 10,
    exclude_id: str | None = None,
    current_category: str | None = None,
    current_tags: List[str] | None = None,
    current_entities: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Find semantically related notes using a hybrid scoring approach.
    
    Combines multiple signals for implicit, meaning-based relations:
    1. Embedding similarity (semantic meaning from content)
    2. Entity overlap (shared people, projects, topics)
    3. Tag similarity (conceptual/topical overlap)
    4. Category affinity (logical relationships between note types)
    
    This produces relations that are based on meaning, not just similar words.
    """
    rows = read_all_embeddings_from_db()
    if not rows or not embedding:
        return []
    
    # Fetch full metadata for semantic comparison
    note_metadata = {}
    with db_lock:
        c = db_conn.cursor()
        try:
            c.execute("SELECT id, category, tags, entities, summary FROM notes")
            for nid, cat, tags_json, entities_json, summary in c.fetchall():
                tags = []
                entities = []
                if tags_json:
                    try:
                        tags = json.loads(tags_json)
                    except Exception:
                        pass
                if entities_json:
                    try:
                        entities = json.loads(entities_json)
                    except Exception:
                        pass
                note_metadata[nid] = {
                    "category": cat or "",
                    "tags": tags or [],
                    "entities": entities or [],
                    "summary": summary or ""
                }
        except Exception:
            logging.exception("Error fetching note metadata for relations")
    
    sims: List[Dict[str, Any]] = []
    min_score = globals().get("RELATION_MIN_SCORE", 0.35)
    
    for r in rows:
        try:
            rid = r.get("id")
            if exclude_id and rid == exclude_id:
                continue
            
            other_emb = r.get("emb") or r.get("embedding") or []
            if not other_emb:
                continue
            
            meta = note_metadata.get(rid, {})

            # 1. Embedding similarity (primary semantic score) - weight: 0.85
            # This is the TRUE semantic signal - captures meaning, not just words
            emb_sim = cosine_similarity(embedding, other_emb)

            # Meaning gate: a relation must be grounded in genuine semantic
            # similarity. If the embeddings aren't close enough, no amount of
            # shared category or entity keywords should connect the notes.
            embed_gate = globals().get("RELATION_MIN_EMBED_SIM", 0.55)
            if emb_sim < embed_gate:
                continue

            # 2. Category affinity - weight: 0.10
            # Logical relationships between note types (ideas become todos, etc.)
            cat_affinity = 0.0
            if current_category and meta.get("category"):
                cat_pair = (current_category, meta["category"])
                cat_affinity = CATEGORY_AFFINITY.get(cat_pair, 0.0)
                # Same category gets a small boost
                if current_category == meta["category"]:
                    cat_affinity = 0.08
            
            # 3. Entity overlap - weight: 0.05 (reduced - too word-based)
            entity_score = 0.0
            if current_entities:
                entity_score = compute_entity_overlap_score(
                    current_entities, 
                    meta.get("entities", [])
                )
            
            # Weighted combination - heavily favor semantic embeddings
            final_score = (
                0.85 * emb_sim +
                0.10 * cat_affinity +
                0.05 * entity_score
            )
            
            # Small bonus for very strong embedding similarity (truly about same topic)
            if emb_sim > 0.75:
                final_score += 0.05
            
            final_score = min(final_score, 1.0)
            
            if final_score >= min_score:
                sims.append({
                    "id": rid,
                    "score": float(final_score),
                    # Raw embedding similarity, kept for the optional LLM
                    # verification band (strong matches skip verification).
                    "emb_sim": round(float(emb_sim), 4),
                    # Include breakdown for debugging/transparency
                    "_debug": {
                        "emb": round(emb_sim, 3),
                        "entity": round(entity_score, 3),
                        "cat": round(cat_affinity, 3)
                    }
                })
        except Exception:
            logging.exception("Error computing similarity for note %s", r.get("id"))
            continue
    
    sims.sort(key=lambda x: x["score"], reverse=True)
    # Remove debug info before returning (it's just for development)
    result = sims[: min(top_k, globals().get("RELATED_MAX_PER_NOTE", 5))]
    for r in result:
        r.pop("_debug", None)
    return result


def verify_and_explain_relations(
    source_summary: str,
    source_content: str,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Confirm and explain candidate relations using the LLM's understanding.

    The candidates have already passed the embedding meaning-gate, so this is a
    precision pass: a single batched Gemini call judges whether each borderline
    candidate is *genuinely* about the same thing/person/topic as the source
    note (not merely nearby in vector space) and returns a short reason.

    - Strong matches (emb_sim >= RELATION_STRONG_EMBED_SIM) are accepted without
      a call and get a generic reason.
    - On any failure, or when verification is disabled, candidates pass through
      unchanged so relations degrade gracefully to the embedding-only result.

    Returns the kept candidates, each annotated with a ``reason`` string.
    """
    if not candidates:
        return candidates
    if not (USE_LLM_RELATION_VERIFY and gemini_client):
        return candidates

    strong_bar = globals().get("RELATION_STRONG_EMBED_SIM", 0.80)

    # Split into confident matches (no call needed) and a verification band.
    band: List[Dict[str, Any]] = []
    band_summaries: List[str] = []
    for cand in candidates:
        if cand.get("emb_sim", 0.0) >= strong_bar:
            cand.setdefault("reason", "Strong semantic overlap")
            continue
        rn = fetch_note_by_id_safe(cand.get("id"))
        summ = (rn.get("summary") if rn else "") or (
            friendly_snippet(rn.get("content", ""), 160) if rn else ""
        )
        if not summ:
            # Nothing to reason about; keep it on embedding evidence alone.
            cand.setdefault("reason", "Semantic similarity")
            continue
        band.append(cand)
        band_summaries.append(summ)

    if not band:
        return candidates

    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(band_summaries))
    prompt = f"""You decide whether short notes are genuinely related in MEANING
(same topic, project, person, goal, or a clear cause/effect or follow-up link) —
not merely similar wording.

SOURCE NOTE:
\"\"\"{(source_summary or source_content or '').strip()[:600]}\"\"\"

CANDIDATE NOTES:
{numbered}

For each candidate, decide if it is genuinely related to the SOURCE NOTE.
Return ONLY a JSON array, one object per candidate, in order:
[{{"index": 0, "related": true, "reason": "<=8 word reason"}}, ...]
Be strict: if the link is only superficial keyword overlap, set related to false."""

    try:
        response = gemini_client.models.generate_content(
            model=CLASSIFY_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=1024),
        )
        out = (response.text or "").strip()
        if out.startswith("```"):
            lines = out.split("\n")
            lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            out = "\n".join(lines)
        try:
            verdicts = json.loads(out)
        except Exception:
            s = out.find("["); e = out.rfind("]")
            verdicts = json.loads(out[s:e + 1]) if (s != -1 and e != -1) else []
        by_index = {int(v.get("index")): v for v in verdicts if isinstance(v, dict) and "index" in v}
    except Exception:
        logging.exception("LLM relation verification failed; keeping embedding candidates")
        for cand in band:
            cand.setdefault("reason", "Semantic similarity")
        return candidates

    # Apply verdicts: drop candidates the model judged unrelated.
    kept_band_ids = set()
    for i, cand in enumerate(band):
        v = by_index.get(i)
        if v is None:
            # No verdict returned for this one: keep on embedding evidence.
            cand.setdefault("reason", "Semantic similarity")
            kept_band_ids.add(cand.get("id"))
        elif v.get("related"):
            cand["reason"] = (str(v.get("reason") or "Related in meaning")).strip()[:80]
            kept_band_ids.add(cand.get("id"))
        # else: judged unrelated -> dropped below

    band_id_set = {c.get("id") for c in band}
    result = [
        c for c in candidates
        if c.get("id") not in band_id_set or c.get("id") in kept_band_ids
    ]
    dropped = len(candidates) - len(result)
    if dropped:
        logging.info("LLM relation verify dropped %d superficial candidate(s)", dropped)
    return result


def store_enriched_note(note: dict):
    with db_lock:
        c = db_conn.cursor()
        try:
            emb_bytes = get_embedding_bytes(note.get("embedding", []))
        except Exception:
            emb_bytes = b""
        related_json = json.dumps(note.get("related", []), ensure_ascii=False)
        try:
            c.execute(
                """INSERT OR REPLACE INTO notes
                   (id,timestamp,content,scraped_content,category,tags,entities,summary,embedding,related)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    note["id"],
                    note["timestamp"],
                    note.get("content",""),
                    note.get("scraped_content",""),
                    note.get("category",""),
                    json.dumps(note.get("tags",[]), ensure_ascii=False),
                    json.dumps(note.get("entities",[]), ensure_ascii=False),
                    note.get("summary","No summary"),
                    emb_bytes,
                    related_json
                )
            )
            db_conn.commit()
            logging.info("Stored note %s category=%s related=%d",
                         note.get("id"),
                         note.get("category"),
                         len(note.get("related", [])))
            try:
                NSNotificationCenter.defaultCenter().postNotificationName_object_userInfo_(
                    "NotesDatabaseDidChange", None, None
                )
            except Exception:
                pass

        except Exception:
            logging.exception("DB insert error for %s", note.get("id"))

def fetch_note_by_id_safe(nid: str):
    with db_lock:
        c = db_conn.cursor()
        try:
            cols = _notes_columns(c)
            if not cols:
                return None
            sel_cols = ",".join(cols)
            c.execute(f"SELECT {sel_cols} FROM notes WHERE id=?", (nid,))
            row = c.fetchone()
        except Exception:
            logging.exception("DB fetch error")
            return None
    if not row:
        return None
    data = {}
    for i, col in enumerate(cols):
        data[col] = row[i]
    for k in ("tags","entities","related"):
        if k in data and k in ("tags","entities","related") and data[k]:
            try:
                data[k] = json.loads(data[k])
            except Exception:
                data[k] = []
        else:
            data[k] = []
    if "summary" not in data or data["summary"] is None:
        data["summary"] = "No summary"
    if "embedding" in data:
        data["embedding"] = embedding_bytes_to_list(data["embedding"])
    else:
        data["embedding"] = []
    return data

# ---------------- Processing core ----------------
def process_capture_object(obj: dict, *, store: bool = True) -> dict:
    """
    Enrich a single capture. When store=True (default), persists to SQLite.
    Pathway pipeline calls with store=True as well; idempotency is handled by INSERT OR REPLACE.
    """
    nid = obj.get("id") or str(uuid.uuid4())
    timestamp = obj.get("timestamp", now_iso())
    content = obj.get("content", "") or ""
    scraped = ""
    if is_url(content):
        scraped = fetch_url_text(content)
    effective = scraped or content or ""
    
    # First classify to get semantic understanding (summary, category, etc.)
    meta = classify_and_tag_with_gemini(effective)
    
    # Create embedding from SUMMARY + CONTENT for richer semantic meaning
    # The summary captures the AI's understanding of the note's meaning
    summary = meta.get("summary", "") or ""
    semantic_text = f"{summary} {effective}".strip()
    emb = get_embedding_gemini(semantic_text)
    
    # Pass category and entities for relation matching (tags removed - too word-based)
    candidates = compute_related_with_scores(
        emb, 
        top_k=10, 
        exclude_id=obj.get("id"),
        current_category=meta.get("category"),
        current_tags=[],  # Disabled tags - they're too word-based
        current_entities=meta.get("entities", [])
    )
    filtered = [r for r in candidates if r["id"] != nid]
    # Precision pass: confirm borderline candidates are genuinely related in
    # meaning (and attach a human-readable reason). No-op unless enabled.
    filtered = verify_and_explain_relations(summary, effective, filtered)
    top_related = filtered[:RELATED_MAX_PER_NOTE]
    # Keep stored relations compact: drop the internal embedding-similarity field.
    for r in top_related:
        r.pop("emb_sim", None)
    enriched = {
        "id": nid,
        "timestamp": timestamp,
        "content": content,
        "scraped_content": scraped,
        "category": meta.get("category","Other"),
        "tags": meta.get("tags", []),
        "entities": meta.get("entities", []),
        # Safety net: guarantee a usable summary even if classification slipped one through blank.
        "summary": _clean_summary(meta.get("summary"), effective),
        "embedding": emb,
        "related": top_related
    }
    if store:
        store_enriched_note(enriched)
    return enriched


def fetch_all_notes_dicts():
    """Return a list of notes as dicts, newest first."""
    with db_lock:
        c = db_conn.cursor()
        try:
            c.execute("SELECT * FROM notes ORDER BY timestamp DESC")
            rows_all = c.fetchall()
            cols = [r[1] for r in c.execute("PRAGMA table_info(notes)").fetchall()]
        except Exception:
            rows_all = []
            cols = []
    notes = []
    for row in rows_all:
        data = {}
        for i, col in enumerate(cols):
            data[col] = row[i]
        # normalize JSON columns
        for k in ("tags","entities","related"):
            if k in data and data[k]:
                try:
                    data[k] = json.loads(data[k])
                except Exception:
                    data[k] = []
            else:
                data[k] = []
        if "summary" not in data or data["summary"] is None:
            data["summary"] = "No summary"
        notes.append(data)
    return notes




# ---------------- Pipeline: reliable polling (default) ----------------
class PipelineProcessor(threading.Thread):
    def __init__(self, input_file: str):
        super().__init__(daemon=True)
        self.input_file = input_file
        open(self.input_file, "a").close()
        self._stop = threading.Event()
        self.using_pathway_stream = False
        self._pathway_runtime: Optional[pw.Runtime] = None  # ==== PATHWAY: keep a handle

    def stop(self):
        self._stop.set()
        # ==== PATHWAY: cooperative shutdown if runtime exists
        if PATHWAY_IMPORTED and self._pathway_runtime is not None:
            try:
                self._pathway_runtime.stop()
            except Exception:
                pass

    def run_polling(self):
        logging.info("Starting polling pipeline.")
        last_pos = 0
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                f.seek(0, os.SEEK_END)
                last_pos = f.tell()
        except Exception:
            last_pos = 0
        while not self._stop.is_set():
            try:
                with open(self.input_file, "r", encoding="utf-8") as f:
                    f.seek(last_pos)
                    lines = f.readlines()
                    last_pos = f.tell()
                for ln in lines:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        obj = json.loads(ln)
                    except Exception:
                        obj = {"id": str(uuid.uuid4()), "timestamp": now_iso(), "content": ln}
                    try:
                        process_capture_object(obj, store=True)
                    except Exception:
                        logging.exception("Error processing raw line")
            except Exception:
                logging.exception("Polling loop error")
            time.sleep(1.0)
        logging.info("Polling stopped.")

    # ==== PATHWAY: strongly-typed schema for safer streaming
    def _build_pathway_schema(self):
        class RawSchema(pw.Schema):
            id: str
            timestamp: str
            content: str
        return RawSchema

    # ==== PATHWAY: streaming pipeline (opt-in)
    def run_pathway_stream(self):
        if not PATHWAY_IMPORTED or not USE_PATHWAY_STREAM:
            logging.info("Pathway stream not used.")
            return False
        try:
            RawSchema = self._build_pathway_schema()

            # Always hand Pathway the absolute path used by our writer
            path = INPUT_STREAM

            try:
                src = pw.io.jsonlines.read(
                    path=path,
                    schema=RawSchema,
                    mode="streaming",
                    autocommit_duration_ms=200,
                )
            except Exception:
                src = pw.io.fs.read(
                    path,
                    format="json",
                    mode="streaming",
                    schema=RawSchema,
                )

            def _ts_key(ts: str) -> str:
                return ts or ""

            normalized = src.select(
                note_id=pw.this.id,
                timestamp=pw.this.timestamp,
                content=pw.this.content,
                ts_key=pw.apply(_ts_key, pw.this.timestamp),
            )

            latest_ts = (
                normalized
                .groupby(pw.this.note_id)
                .reduce(
                    note_id=pw.this.note_id,
                    max_ts=pw.reducers.max(pw.this.ts_key),
                )
            )

            jr = latest_ts.join(
                normalized,
                latest_ts.note_id == normalized.note_id
            )

            latest_rows = (
                jr
                .filter(pw.left.max_ts == pw.right.ts_key)
                .select(
                    note_id=pw.left.note_id,
                    timestamp=pw.left.max_ts,
                    content=pw.right.content,
                )
            )

            enriched = latest_rows.select(
                enriched=pw.apply(
                    lambda _nid, _ts, _c: process_capture_object(
                        {"id": _nid, "timestamp": _ts, "content": _c},
                        store=True
                    ),
                    pw.this.note_id, pw.this.timestamp, pw.this.content
                )
            )

            class _Heartbeat:
                def __init__(self):
                    self.count = 0
                def on_change(self, _table, change):
                    added = getattr(change, "added_count", 1)
                    try:
                        if hasattr(change, "data"):
                            added = len(change.data)
                        elif hasattr(change, "added"):
                            added = len(change.added)
                    except Exception:
                        pass
                    self.count += added if isinstance(added, int) else 1
                    if self.count % 20 == 0:
                        logging.info("Pathway heartbeat: processed %d rows", self.count)

            try:
                hb = _Heartbeat()
                pw.io.python.write(enriched, hb)
            except Exception:
                pass

            try:
                _bootstrap_backfill_if_needed(max_lines=5000)
            except Exception:
                logging.exception("Bootstrap backfill failed")

            logging.info("Starting Pathway streaming (opt-in).")
            self.using_pathway_stream = True
            try:
                self._pathway_runtime = pw.run()
            except TypeError:
                self._pathway_runtime = pw.run(enriched)

            return True

        except Exception:
            logging.exception("Pathway streaming failed")
            self.using_pathway_stream = False
            return False


    def run(self):
        if PATHWAY_IMPORTED and USE_PATHWAY_STREAM:
            ok = self.run_pathway_stream()
            if ok:
                return
        self.run_polling()

# ---------------- Repair utilities ----------------
def enrich_missing_notes(batch_sleep=0.1):
    with db_lock:
        c = db_conn.cursor()
        try:
            c.execute("SELECT id FROM notes")
            rows = c.fetchall()
        except Exception:
            rows = []
    updated = 0
    for (nid,) in rows:
        note = fetch_note_by_id_safe(nid)
        if not note:
            continue
        need = False
        emb = note.get("embedding")
        if not emb:
            need = True
        # Also check for embedding dimension mismatch (e.g., old 1536 vs new 768)
        elif isinstance(emb, list) and len(emb) != TARGET_EMBEDDING_DIM:
            logging.info("Note %s has embedding dim %d, expected %d - re-embedding",
                        nid, len(emb), TARGET_EMBEDDING_DIM)
            need = True
        # Missing or placeholder summary
        summ = (note.get("summary") or "").strip()
        if not summ or summ.lower() == "no summary":
            need = True
        # Missing classification (a fully enriched note always has a category)
        if not (note.get("category") or "").strip():
            need = True
        if need:
            process_capture_object({"id": nid, "timestamp": note.get("timestamp", now_iso()), "content": note.get("content","")}, store=True)
            updated += 1
            time.sleep(batch_sleep)
    logging.info("Enrich missing done: %d updated", updated)
    return updated

def update_note_fields(nid: str, fields: dict) -> bool:
    """Apply user edits to a note and refresh its derived data.

    Manual edits to content/summary/category/tags/entities are preserved (we do
    NOT re-run AI classification, which would overwrite them). Only the embedding
    and the related-notes list are recomputed, so connections stay meaningful.
    Persisting posts NotesDatabaseDidChange, so open windows refresh themselves.
    """
    note = fetch_note_by_id_safe(nid)
    if not note:
        return False
    content = fields.get("content", note.get("content", "")) or ""
    summary = (fields.get("summary") or "").strip() or derive_summary_from_text(content)
    category = (fields.get("category") or note.get("category") or "Other").strip() or "Other"
    tags = fields.get("tags", note.get("tags", []) or [])
    entities = fields.get("entities", note.get("entities", []) or [])

    emb = get_embedding_gemini(f"{summary} {content}".strip())
    candidates = compute_related_with_scores(
        emb, top_k=10, exclude_id=nid,
        current_category=category, current_entities=entities,
    )
    candidates = [r for r in candidates if r.get("id") != nid]
    candidates = verify_and_explain_relations(summary, content, candidates)[:RELATED_MAX_PER_NOTE]
    for r in candidates:
        r.pop("emb_sim", None)

    store_enriched_note({
        "id": nid,
        "timestamp": note.get("timestamp", now_iso()),
        "content": content,
        "scraped_content": note.get("scraped_content", "") or "",
        "category": category,
        "tags": tags,
        "entities": entities,
        "summary": summary,
        "embedding": emb,
        "related": candidates,
    })
    return True


def delete_note(nid: str) -> bool:
    """Delete a note and prune references to it from other notes' related lists."""
    with db_lock:
        c = db_conn.cursor()
        try:
            c.execute("DELETE FROM notes WHERE id=?", (nid,))
            c.execute("SELECT id, related FROM notes")
            for oid, rel_json in c.fetchall():
                if not rel_json:
                    continue
                try:
                    rels = json.loads(rel_json)
                except Exception:
                    continue
                pruned = [r for r in rels if r.get("id") != nid]
                if len(pruned) != len(rels):
                    c.execute("UPDATE notes SET related=? WHERE id=?",
                              (json.dumps(pruned, ensure_ascii=False), oid))
            db_conn.commit()
        except Exception:
            logging.exception("delete_note failed for %s", nid)
            return False
    try:
        NSNotificationCenter.defaultCenter().postNotificationName_object_userInfo_(
            "NotesDatabaseDidChange", None, None
        )
    except Exception:
        pass
    return True


def recompute_links():
    with db_lock:
        c = db_conn.cursor()
        try:
            c.execute("SELECT id FROM notes")
            ids = [r[0] for r in c.fetchall()]
        except Exception:
            ids = []
    # ensure embeddings
    for nid in ids:
        note = fetch_note_by_id_safe(nid)
        if not note:
            continue
        if not note.get("embedding"):
            process_capture_object({"id": nid, "timestamp": note.get("timestamp", now_iso()), "content": note.get("content","")}, store=True)
            time.sleep(0.05)
    all_embs = read_all_embeddings_from_db()
    emb_map = {r["id"]: r["emb"] for r in all_embs}
    count = 0
    for nid in ids:
        emb = emb_map.get(nid)
        if not emb:
            continue
        # Use the same hybrid, meaning-based scorer as the live capture path so
        # "Recompute Links" produces identical, semantically-grounded relations
        # (embedding gate + category/entity signals) rather than raw cosine.
        note = fetch_note_by_id_safe(nid)
        top_related = compute_related_with_scores(
            emb,
            top_k=RELATED_MAX_PER_NOTE,
            exclude_id=nid,
            current_category=(note or {}).get("category"),
            current_entities=(note or {}).get("entities", []),
        )
        top_related = verify_and_explain_relations(
            (note or {}).get("summary", ""),
            (note or {}).get("content", ""),
            top_related,
        )[:RELATED_MAX_PER_NOTE]
        for r in top_related:
            r.pop("emb_sim", None)
        with db_lock:
            c = db_conn.cursor()
            try:
                c.execute("UPDATE notes SET related=? WHERE id=?", (json.dumps(top_related, ensure_ascii=False), nid))
                db_conn.commit()
                count += 1
            except Exception:
                logging.exception("Failed to update related for %s", nid)
    logging.info("Recompute links done: %d updated", count)
    return count

# ---------------- UI display helpers (scrollable windows) ----------------
NOTES_PAGE_SIZE = 12  # show this many notes per page

# keep windows referenced so they do not get GC'd
GLOBAL_WINDOWS = []

def show_text_window(title: str, text: str, width: int = 800, height: int = 600):
    """
    Show a native macOS scrollable, resizable, non-blocking window via AppKit (PyObjC).
    Falls back to rumps.alert if PyOBJC isn't available.
    """
    if not PYOBJC_AVAILABLE:
        # fallback: truncated rumps alert
        rumps.alert(title, text[:4000])
        return
    try:
        rect = NSMakeRect(160.0, 160.0, float(width), float(height))
        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, 15, NSBackingStoreBuffered, False
        )
        win.setTitle_(title or "")
        win.setReleasedWhenClosed_(False)
        tv = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, float(width), float(height)))
        tv.setEditable_(False)
        tv.setSelectable_(True)
        tv.setRichText_(False)
        try:
            tv.setFont_(NSFont.systemFontOfSize_(13.0))
            tv.setTextColor_(NSColor.labelColor())
            tv.setTextContainerInset_((16.0, 14.0))
            tv.setAccessibilityLabel_(title or "Text")
        except Exception:
            pass
        tv.setString_(text or "")
        scroll = NSScrollView.alloc().initWithFrame_(win.contentView().bounds())
        scroll.setDocumentView_(tv)
        scroll.setHasVerticalScroller_(True)
        scroll.setAutohidesScrollers_(True)
        try:
            scroll.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
            win.contentView().setAutoresizesSubviews_(True)
        except Exception:
            pass
        win.contentView().addSubview_(scroll)
        try:
            _set_policy_regular_temporarily()
            NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        except Exception:
            pass
        win.makeKeyAndOrderFront_(None)
        # keep a reference so the window isn't garbage-collected
        try:
            GLOBAL_WINDOWS.append(win)
        except Exception:
            pass
    except Exception:
        logging.exception("show_text_window failed")


# ---------- Native "Show All Notes" UI (macOS) ----------
if PYOBJC_AVAILABLE:

    # --- App activation policy helpers ---
    try:
        from AppKit import NSApplication, NSApp
    except Exception:
        NSApplication = None
        NSApp = None
    try:
        from AppKit import NSApplicationActivationPolicyRegular as _AP_REGULAR
        from AppKit import NSApplicationActivationPolicyAccessory as _AP_ACCESSORY
    except Exception:
        _AP_REGULAR, _AP_ACCESSORY = 0, 1  # numeric fallbacks

    def _set_policy_regular_temporarily():
        try:
            app = NSApplication.sharedApplication()
            try:
                _orig = int(app.activationPolicy())
            except Exception:
                _orig = _AP_ACCESSORY
            if _orig != _AP_REGULAR:
                app.setActivationPolicy_(_AP_REGULAR)
            try:
                app.activateIgnoringOtherApps_(True)
            except Exception:
                pass
            return _orig
        except Exception:
            return None

    def _restore_policy(policy_value):
        try:
            if policy_value is None:
                return
            app = NSApplication.sharedApplication()
            app.setActivationPolicy_(policy_value)
        except Exception:
            pass

    # Style mask compatibility (AppKit constants vary by macOS / PyObjC)
    try:
        from AppKit import NSWindowStyleMaskTitled as _STYLE_TITLED
        from AppKit import NSWindowStyleMaskClosable as _STYLE_CLOSABLE
    except Exception:
        try:
            from AppKit import NSTitledWindowMask as _STYLE_TITLED
            from AppKit import NSClosableWindowMask as _STYLE_CLOSABLE
        except Exception:
            _STYLE_TITLED, _STYLE_CLOSABLE = (1 << 0), (1 << 1)  # sensible defaults

    # Common AppKit classes used by QuickCapture panel
    try:
        from AppKit import NSButton, NSApplication
    except Exception:
        pass

    try:
        from AppKit import NSBezierPath
    except Exception:
        NSBezierPath = None

    class NoteGraphView(NSView):
        """A clickable 'web' of the selected note and its related notes.

        The selected note sits at the center; related notes orbit around it,
        connected by edges whose thickness/opacity reflect relation strength.
        Clicking a node re-centers the web on that note. (Screen-reader users
        get the same connectivity via the detail pane's linked list.)
        """
        def initWithFrame_(self, frame):
            self = objc.super(NoteGraphView, self).initWithFrame_(frame)
            if self is None:
                return None
            self._center_note = None
            self._related = []      # list of {id, title, glyph, score}
            self._on_select = None  # python callable(note_id)
            self._hit = []          # list of (x0, y0, x1, y1, note_id)
            return self

        def isFlipped(self):
            return True

        @objc.python_method
        def configure(self, center_note, related, on_select):
            self._center_note = center_note
            self._related = related or []
            self._on_select = on_select
            self._updateA11y()
            try:
                self.setNeedsDisplay_(True)
            except Exception:
                pass

        @objc.python_method
        def _updateA11y(self):
            try:
                if self._center_note is None:
                    self.setAccessibilityLabel_("Connection map. No note selected.")
                    return
                names = ", ".join(r.get("title", "") for r in self._related[:6])
                lbl = f"Connection map. Selected: {self._center_note.get('title','')}. "
                lbl += (f"Connected to {len(self._related)} notes: {names}." if self._related else "No connections yet.")
                self.setAccessibilityLabel_(lbl)
            except Exception:
                pass

        def drawRect_(self, dirty):
            try:
                self._draw()
            except Exception:
                logging.exception("Graph draw failed")

        @objc.python_method
        def _accent(self):
            try:
                return NSColor.controlAccentColor()
            except Exception:
                return NSColor.systemBlueColor()

        @objc.python_method
        def _draw(self):
            import math as _m
            b = self.bounds()
            W, Hh = b.size.width, b.size.height
            cx, cy = W / 2.0, Hh / 2.0
            self._hit = []
            accent = self._accent()

            if self._center_note is None:
                self._text_(NSMakeRect(10, cy - 10, W - 20, 20), "Select a note to see its connections", center=True, dim=True)
                return

            # Cap nodes so the web stays readable; ellipse layout fills the box
            # width/height so nodes spread out and don't overlap each other or
            # the center node.
            rel = self._related[:6]
            n = len(rel)
            node_w, node_h = 130.0, 34.0
            pad = 14.0
            Rx = max(120.0, W / 2.0 - node_w / 2.0 - pad)
            Ry = max(70.0, Hh / 2.0 - node_h / 2.0 - pad)

            # Edges first (under the nodes). Start at the top, go clockwise.
            positions = []
            for i in range(n):
                ang = -_m.pi / 2 + (2 * _m.pi * i / n)
                nx, ny = cx + Rx * _m.cos(ang), cy + Ry * _m.sin(ang)
                positions.append((nx, ny))
                self._edge_(cx, cy, nx, ny, float(rel[i].get("score", 0.0) or 0.0), accent)

            # Related nodes
            for i, r in enumerate(rel):
                nx, ny = positions[i]
                rectN = NSMakeRect(nx - node_w / 2, ny - node_h / 2, node_w, node_h)
                self._node_(rectN, r.get("glyph", "📝"), r.get("title", ""), r.get("score"), False)
                self._hit.append((rectN.origin.x, rectN.origin.y, rectN.origin.x + node_w, rectN.origin.y + node_h, r.get("id")))

            # Center node on top
            cw, ch = 140.0, 40.0
            rectC = NSMakeRect(cx - cw / 2, cy - ch / 2, cw, ch)
            self._node_(rectC, category_glyph(self._center_note.get("category", "")), self._center_note.get("title", ""), None, True)

            if not rel:
                self._text_(NSMakeRect(10, Hh - 28, W - 20, 20), "No connections yet", center=True, dim=True)

        @objc.python_method
        def _edge_(self, x0, y0, x1, y1, score, color):
            if NSBezierPath is None:
                return
            s = max(0.0, min(1.0, score))
            path = NSBezierPath.bezierPath()
            path.moveToPoint_((x0, y0))
            path.lineToPoint_((x1, y1))
            path.setLineWidth_(1.0 + 4.0 * s)
            try:
                color.colorWithAlphaComponent_(0.18 + 0.55 * s).set()
            except Exception:
                color.set()
            path.stroke()
            # score label ~2/3 toward the related node (keeps it clear of the center node)
            try:
                pct = int(round(s * 100))
                lx = x0 + 0.62 * (x1 - x0)
                ly = y0 + 0.62 * (y1 - y0)
                self._text_(NSMakeRect(lx - 20, ly - 8, 40, 16), f"{pct}%", center=True, dim=True, size=9.5)
            except Exception:
                pass

        @objc.python_method
        def _node_(self, rect, glyph, title, score, is_center):
            if NSBezierPath is None:
                return
            radius = rect.size.height / 2.0
            path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(rect, radius, radius)
            try:
                fill = self._accent() if is_center else NSColor.controlBackgroundColor()
            except Exception:
                fill = NSColor.windowBackgroundColor()
            fill.set()
            path.fill()
            try:
                path.setLineWidth_(1.5 if is_center else 1.0)
                (NSColor.whiteColor() if is_center else self._accent()).colorWithAlphaComponent_(0.55).set()
                path.stroke()
            except Exception:
                pass
            color = NSColor.whiteColor() if is_center else NSColor.labelColor()
            label = f"{glyph}  {friendly_snippet(title or 'Untitled', 16)}"
            self._text_(NSMakeRect(rect.origin.x + 8, rect.origin.y + (rect.size.height - 18) / 2, rect.size.width - 16, 18),
                        label, center=True, color=color, bold=is_center)

        @objc.python_method
        def _text_(self, rect, text, center=False, dim=False, color=None, bold=False, size=11.5):
            try:
                para = NSMutableParagraphStyle.alloc().init()
                para.setAlignment_(2 if center else 0)  # 2 = center
                para.setLineBreakMode_(4)                # truncate tail
                if color is None:
                    color = NSColor.tertiaryLabelColor() if dim else NSColor.labelColor()
                font = NSFont.boldSystemFontOfSize_(size + 0.5) if bold else NSFont.systemFontOfSize_(size)
                attrs = {
                    NSFontAttributeName: font,
                    NSForegroundColorAttributeName: color,
                    NSParagraphStyleAttributeName: para,
                }
                NSAttributedString.alloc().initWithString_attributes_(text or "", attrs).drawInRect_(rect)
            except Exception:
                pass

        def mouseDown_(self, event):
            try:
                p = self.convertPoint_fromView_(event.locationInWindow(), None)
                px, py = p.x, p.y
                for (x0, y0, x1, y1, nid) in self._hit:
                    if x0 <= px <= x1 and y0 <= py <= y1:
                        if self._on_select and nid:
                            self._on_select(nid)
                        return
            except Exception:
                logging.exception("Graph click failed")

    class NoteBubbleView(NSView):
        """The 'bubble' view: every note is a speech bubble, scattered at first
        and pulled together by a force simulation so related notes cluster and
        connect — an interconnected web of the whole notebook.

        Accessibility: the canvas exposes one accessibility element per bubble
        (a button) whose label describes the note and its connection count.
        Each element carries custom actions — one per relationship — so a
        VoiceOver user can open the actions menu on a bubble and jump straight
        to any connected note, *traversing the graph by relationship* rather
        than by screen position. A final 'Open in list view' action dives into
        the full details. Sighted keyboard users can also use arrow keys to move
        between bubbles and number keys 1–9 to follow a connection.
        """
        def initWithFrame_(self, frame):
            self = objc.super(NoteBubbleView, self).initWithFrame_(frame)
            if self is None:
                return None
            self._nodes = []      # dicts: id,title,glyph,cat,x,y,vx,vy,rel,deg
            self._edges = []      # (i, j, score)
            self._idindex = {}    # note id -> node index
            self._on_open = None  # callable(nid) -> open in list view
            self._selected = None
            self._timer = None
            self._steps = 0
            self._a11y = []
            self._bw, self._bh = 152.0, 42.0
            return self

        def isFlipped(self):
            return True

        def acceptsFirstResponder(self):
            return True

        @objc.python_method
        def configure(self, notes, on_open):
            self._on_open = on_open
            import random
            b = self.bounds()
            W = b.size.width or 800.0
            H = b.size.height or 500.0
            self._nodes = []
            self._idindex = {}
            for k, n in enumerate(notes or []):
                nid = n.get("id")
                self._idindex[nid] = k
                self._nodes.append({
                    "id": nid,
                    "title": n.get("summary") or n.get("content", "") or "Untitled",
                    "glyph": category_glyph(n.get("category", "")),
                    "cat": n.get("category", "") or "Other",
                    "x": random.uniform(0.12 * W, 0.88 * W),
                    "y": random.uniform(0.12 * H, 0.88 * H),
                    "vx": 0.0, "vy": 0.0,
                    "rel": [{"id": r.get("id"),
                             "score": float(r.get("score", 0.0) or 0.0),
                             "reason": r.get("reason")} for r in (n.get("related") or [])],
                })
            seen = set()
            self._edges = []
            for k, nd in enumerate(self._nodes):
                for r in nd["rel"]:
                    j = self._idindex.get(r["id"])
                    if j is None:
                        continue
                    key = (min(k, j), max(k, j))
                    if key in seen:
                        continue
                    seen.add(key)
                    self._edges.append((k, j, r["score"]))
            if self._selected not in self._idindex:
                self._selected = self._nodes[0]["id"] if self._nodes else None
            self._startSim()

        @objc.python_method
        def _startSim(self):
            self._steps = 0
            try:
                if self._timer is not None:
                    self._timer.invalidate()
            except Exception:
                pass
            if NSTimer is not None:
                try:
                    self._timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                        0.02, self, "stepSim:", None, True)
                except Exception:
                    self._timer = None
                    self._settleImmediately()
            else:
                self._settleImmediately()
            self.setNeedsDisplay_(True)

        @objc.python_method
        def _settleImmediately(self):
            for _ in range(140):
                self._simStep()
            self._rebuildA11y()
            self.setNeedsDisplay_(True)

        def stepSim_(self, timer):
            try:
                self._simStep()
                self._steps += 1
                self.setNeedsDisplay_(True)
                if self._steps > 150:
                    timer.invalidate()
                    self._timer = None
                    self._rebuildA11y()
            except Exception:
                try: timer.invalidate()
                except Exception: pass
                self._timer = None

        @objc.python_method
        def _simStep(self):
            nodes = self._nodes
            n = len(nodes)
            if n == 0:
                return
            b = self.bounds()
            W = b.size.width or 800.0
            H = b.size.height or 500.0
            cx, cy = W / 2.0, H / 2.0
            k_rep, k_spring, k_center, damp = 12000.0, 0.018, 0.004, 0.86
            for i in range(n):
                fx = fy = 0.0
                xi, yi = nodes[i]["x"], nodes[i]["y"]
                for j in range(n):
                    if i == j:
                        continue
                    dx = xi - nodes[j]["x"]
                    dy = yi - nodes[j]["y"]
                    d2 = dx * dx + dy * dy + 0.01
                    inv = 1.0 / (d2 ** 0.5)
                    f = k_rep / d2
                    fx += f * dx * inv
                    fy += f * dy * inv
                fx += (cx - xi) * k_center
                fy += (cy - yi) * k_center
                nodes[i]["_fx"] = fx
                nodes[i]["_fy"] = fy
            for (a, bb, sc) in self._edges:
                dx = nodes[bb]["x"] - nodes[a]["x"]
                dy = nodes[bb]["y"] - nodes[a]["y"]
                dist = (dx * dx + dy * dy) ** 0.5 or 1.0
                rest = 150.0 - 70.0 * max(0.0, min(1.0, sc))  # stronger link → closer
                f = k_spring * (dist - rest)
                ux, uy = dx / dist, dy / dist
                nodes[a]["_fx"] += f * ux
                nodes[a]["_fy"] += f * uy
                nodes[bb]["_fx"] -= f * ux
                nodes[bb]["_fy"] -= f * uy
            for nd in nodes:
                nd["vx"] = (nd["vx"] + nd.get("_fx", 0.0)) * damp
                nd["vy"] = (nd["vy"] + nd.get("_fy", 0.0)) * damp
                nd["x"] += max(-14.0, min(14.0, nd["vx"]))
                nd["y"] += max(-14.0, min(14.0, nd["vy"]))
                nd["x"] = max(self._bw / 2 + 6, min(W - self._bw / 2 - 6, nd["x"]))
                nd["y"] = max(self._bh / 2 + 18, min(H - self._bh / 2 - 6, nd["y"]))

        # ---- drawing ----
        def drawRect_(self, dirty):
            try:
                self._draw()
            except Exception:
                logging.exception("Bubble draw failed")

        @objc.python_method
        def _accent(self):
            try:
                return NSColor.controlAccentColor()
            except Exception:
                return NSColor.systemBlueColor()

        @objc.python_method
        def _draw(self):
            nodes = self._nodes
            if not nodes:
                self._text_(NSMakeRect(20, (self.bounds().size.height or 100) / 2 - 10, (self.bounds().size.width or 200) - 40, 20),
                            "No notes yet — capture a thought to see it here.", center=True, dim=True)
                return
            accent = self._accent()
            # edges
            if NSBezierPath is not None:
                for (a, bb, sc) in self._edges:
                    s = max(0.0, min(1.0, sc))
                    p = NSBezierPath.bezierPath()
                    p.moveToPoint_((nodes[a]["x"], nodes[a]["y"]))
                    p.lineToPoint_((nodes[bb]["x"], nodes[bb]["y"]))
                    p.setLineWidth_(1.0 + 3.5 * s)
                    try:
                        accent.colorWithAlphaComponent_(0.15 + 0.5 * s).set()
                    except Exception:
                        accent.set()
                    p.stroke()
            # bubbles
            for nd in nodes:
                self._bubble_(nd, nd["id"] == self._selected)

        @objc.python_method
        def _bubble_(self, nd, selected):
            if NSBezierPath is None:
                return
            w, h = self._bw, self._bh
            x, y = nd["x"] - w / 2, nd["y"] - h / 2
            rect = NSMakeRect(x, y, w, h)
            radius = 12.0
            body = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(rect, radius, radius)
            # speech-bubble tail (small triangle at bottom-left)
            tail = NSBezierPath.bezierPath()
            tail.moveToPoint_((x + 22, y + h - 1))
            tail.lineToPoint_((x + 14, y + h + 9))
            tail.lineToPoint_((x + 34, y + h - 1))
            tail.closePath()
            try:
                fill = self._accent() if selected else NSColor.controlBackgroundColor()
            except Exception:
                fill = NSColor.windowBackgroundColor()
            fill.set()
            body.fill()
            tail.fill()
            try:
                (NSColor.whiteColor() if selected else self._accent()).colorWithAlphaComponent_(0.55).set()
                body.setLineWidth_(1.5 if selected else 1.0)
                body.stroke()
            except Exception:
                pass
            color = NSColor.whiteColor() if selected else NSColor.labelColor()
            label = f"{nd['glyph']}  {friendly_snippet(nd['title'], 17)}"
            self._text_(NSMakeRect(x + 8, y + (h - 18) / 2, w - 16, 18), label, center=True, color=color, bold=selected)

        @objc.python_method
        def _text_(self, rect, text, center=False, dim=False, color=None, bold=False, size=11.5):
            try:
                para = NSMutableParagraphStyle.alloc().init()
                para.setAlignment_(2 if center else 0)
                para.setLineBreakMode_(4)
                if color is None:
                    color = NSColor.tertiaryLabelColor() if dim else NSColor.labelColor()
                font = NSFont.boldSystemFontOfSize_(size + 0.5) if bold else NSFont.systemFontOfSize_(size)
                attrs = {NSFontAttributeName: font, NSForegroundColorAttributeName: color, NSParagraphStyleAttributeName: para}
                NSAttributedString.alloc().initWithString_attributes_(text or "", attrs).drawInRect_(rect)
            except Exception:
                pass

        # ---- mouse + keyboard selection ----
        @objc.python_method
        def _nodeAt_(self, px, py):
            w, h = self._bw, self._bh
            for nd in self._nodes:
                if (nd["x"] - w / 2) <= px <= (nd["x"] + w / 2) and (nd["y"] - h / 2) <= py <= (nd["y"] + h / 2):
                    return nd
            return None

        def mouseDown_(self, event):
            try:
                p = self.convertPoint_fromView_(event.locationInWindow(), None)
                nd = self._nodeAt_(p.x, p.y)
                if nd is None:
                    return
                self._selected = nd["id"]
                self.setNeedsDisplay_(True)
                if int(event.clickCount()) >= 2 and self._on_open:
                    self._on_open(nd["id"])
            except Exception:
                logging.exception("Bubble click failed")

        def keyDown_(self, event):
            try:
                chars = str(event.charactersIgnoringModifiers() or "")
            except Exception:
                chars = ""
            # Enter opens the selected note in the list view
            if chars in ("\r", "\n", "\x03") and self._on_open and self._selected:
                self._on_open(self._selected)
                return
            # Number keys 1-9 follow the Nth connection of the selected note
            if chars.isdigit() and chars != "0":
                self._followConnection_(int(chars) - 1)
                return
            # Arrow keys / tab cycle through bubbles
            if chars in ("\t", " ") or (event.keyCode() in (123, 124, 125, 126)):
                self._cycleSelection_(forward=event.keyCode() not in (123, 126))
                return
            objc.super(NoteBubbleView, self).keyDown_(event)

        @objc.python_method
        def _cycleSelection_(self, forward=True):
            if not self._nodes:
                return
            idx = self._idindex.get(self._selected, 0)
            idx = (idx + (1 if forward else -1)) % len(self._nodes)
            self._selected = self._nodes[idx]["id"]
            self.setNeedsDisplay_(True)
            self._announceSelection_()

        @objc.python_method
        def _followConnection_(self, n):
            idx = self._idindex.get(self._selected)
            if idx is None:
                return
            rels = self._nodes[idx]["rel"]
            if 0 <= n < len(rels):
                rid = rels[n]["id"]
                if rid in self._idindex:
                    self._selected = rid
                    self.setNeedsDisplay_(True)
                    self._announceSelection_()

        @objc.python_method
        def _titleFor_(self, nid):
            j = self._idindex.get(nid)
            return self._nodes[j]["title"] if j is not None else "a note"

        @objc.python_method
        def _connDescription_(self, rel):
            """A full, descriptive sentence for a relationship (for VoiceOver):
            the connected note's category + full summary + strength + reason."""
            j = self._idindex.get(rel.get("id"))
            if j is None:
                return None
            target = self._nodes[j]
            pct = int(round(max(0.0, min(1.0, rel.get("score", 0.0))) * 100))
            desc = f"{target['cat']}: {target['title']}"
            if pct:
                desc += f". {pct}% match"
            reason = rel.get("reason")
            if reason:
                desc += f". Because: {reason}"
            return desc

        @objc.python_method
        def _announceSelection_(self):
            try:
                idx = self._idindex.get(self._selected)
                if idx is None:
                    return
                nd = self._nodes[idx]
                conns = len(nd["rel"])
                # Concise: name the note + connection count. The connections
                # themselves are reachable through the actions menu, not read out.
                msg = f"{nd['cat']}: {nd['title']}. {conns} connection{'s' if conns != 1 else ''}."
                if NSAccessibilityPostNotification is not None and NSAccessibilityLayoutChangedNotification is not None:
                    NSAccessibilityPostNotification(self, NSAccessibilityLayoutChangedNotification)
                self.setAccessibilityLabel_(msg)
            except Exception:
                pass

        # ---- accessibility: one element per bubble, connections as custom actions ----
        @objc.python_method
        def _screenRectFor_(self, nd):
            w, h = self._bw, self._bh
            r = NSMakeRect(nd["x"] - w / 2, nd["y"] - h / 2, w, h)
            try:
                rw = self.convertRect_toView_(r, None)
                win = self.window()
                return win.convertRectToScreen_(rw) if win is not None else rw
            except Exception:
                return r

        @objc.python_method
        def _makeActions_(self, nd):
            actions = []
            if NSAccessibilityCustomAction is None:
                return actions
            try:
                # One action per relationship — jump straight to that connected
                # note. The action name carries the full description so VoiceOver
                # reads the whole connected note, its strength, and why it relates.
                rels = nd["rel"][:8]
                total = len(rels)
                for i, rel in enumerate(rels):
                    rid = rel.get("id")
                    if rid not in self._idindex:
                        continue
                    desc = self._connDescription_(rel) or self._titleFor_(rid)
                    name = f"Go to connection {i + 1} of {total}: {desc}"

                    def _handler(target=rid):
                        try:
                            self._selected = target
                            self.setNeedsDisplay_(True)
                            self._announceSelection_()
                        except Exception:
                            pass
                        return True
                    actions.append(NSAccessibilityCustomAction.alloc().initWithName_handler_(name, _handler))
                # Final action: dive into full details in the list view.
                nid = nd["id"]

                def _open(target=nid):
                    try:
                        if self._on_open:
                            self._on_open(target)
                    except Exception:
                        pass
                    return True
                actions.append(NSAccessibilityCustomAction.alloc().initWithName_handler_("Open full details in list view", _open))
            except Exception:
                logging.exception("Building a11y actions failed")
            return actions

        @objc.python_method
        def _rebuildA11y(self):
            if NSAccessibilityElement is None:
                self._a11y = []
                return
            kids = []
            try:
                for nd in self._nodes:
                    conns = len(nd["rel"])
                    # Keep the label concise — the connections are exposed only
                    # through the element's custom actions, not read inline.
                    suffix = (f"{conns} connection{'s' if conns != 1 else ''}." if conns else "No connections.")
                    label = f"{nd['cat']} note. {nd['title']}. {suffix}"
                    el = NSAccessibilityElement.accessibilityElementWithRole_frame_label_parent_(
                        "AXButton", self._screenRectFor_(nd), label, self)
                    try:
                        el.setAccessibilityHelp_("Use the actions menu to jump to a connected note or open full details.")
                        el.setAccessibilityCustomActions_(self._makeActions_(nd))
                    except Exception:
                        pass
                    kids.append(el)
            except Exception:
                logging.exception("Rebuilding a11y elements failed")
            self._a11y = kids
            try:
                if NSAccessibilityPostNotification is not None and NSAccessibilityLayoutChangedNotification is not None:
                    NSAccessibilityPostNotification(self, NSAccessibilityLayoutChangedNotification)
            except Exception:
                pass

        def accessibilityChildren(self):
            return self._a11y or []

        def accessibilityRole(self):
            return "AXGroup"

        def accessibilityLabel(self):
            n = len(self._nodes)
            e = len(self._edges)
            return f"Bubble graph: {n} notes, {e} connections. Navigate bubbles, then use each bubble's actions to follow its connections."

    class SettingsWindowController(NSObject):
        """A small settings window for the most useful knobs."""
        def init(self):
            self = objc.super(SettingsWindowController, self).init()
            if self is None:
                return None
            self.window = None
            self._on_saved = None
            self._score = None
            self._embed = None
            self._maxrel = None
            self._verify = None
            self._view = None
            return self

        @objc.python_method
        def configure(self, on_saved):
            self._on_saved = on_saved

        @objc.python_method
        def _label(self, text, frame, bold=False, dim=True):
            lbl = NSTextField.alloc().initWithFrame_(frame)
            lbl.setBezeled_(False); lbl.setDrawsBackground_(False)
            lbl.setEditable_(False); lbl.setSelectable_(False)
            lbl.setStringValue_(text)
            try:
                lbl.setFont_(NSFont.boldSystemFontOfSize_(13.0 if bold else 11.0))
                lbl.setTextColor_(NSColor.secondaryLabelColor() if dim else NSColor.labelColor())
            except Exception:
                pass
            return lbl

        def build(self):
            Wd, H = 460.0, 340.0
            self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(200, 200, Wd, H), 15, NSBackingStoreBuffered, False)
            self.window.setTitle_("Settings")
            self.window.setReleasedWhenClosed_(False)
            c = self.window.contentView()
            s = current_settings()

            t = self._label("⚙︎  Settings", NSMakeRect(20, H - 42, 420, 24), bold=True, dim=False)
            try: t.setFont_(NSFont.boldSystemFontOfSize_(16.0))
            except Exception: pass
            c.addSubview_(t)

            def field(value, y, a11y):
                c.addSubview_(self._label(a11y, NSMakeRect(20, y + 2, 280, 16)))
                f = NSTextField.alloc().initWithFrame_(NSMakeRect(300, y, 120, 24))
                f.setStringValue_(str(value))
                try: f.setAccessibilityLabel_(a11y)
                except Exception: pass
                c.addSubview_(f)
                return f

            self._score = field(round(s["RELATION_MIN_SCORE"], 3), H - 80, "Min relation score (0–1)")
            self._embed = field(round(s["RELATION_MIN_EMBED_SIM"], 3), H - 114, "Min embedding similarity (0–1)")
            self._maxrel = field(s["RELATED_MAX_PER_NOTE"], H - 148, "Max related notes per note")

            self._verify = NSButton.alloc().initWithFrame_(NSMakeRect(20, H - 190, 400, 22))
            try: self._verify.setButtonType_(3)  # switch/checkbox
            except Exception: pass
            self._verify.setTitle_("Verify relationships with the LLM (more precise, slower)")
            self._verify.setState_(1 if s["USE_LLM_RELATION_VERIFY"] else 0)
            try: self._verify.setAccessibilityLabel_("Verify relationships with the LLM")
            except Exception: pass
            c.addSubview_(self._verify)

            c.addSubview_(self._label("Default view", NSMakeRect(20, H - 226, 120, 16)))
            self._view = NSPopUpButton.alloc().initWithFrame_pullsDown_(NSMakeRect(140, H - 230, 200, 26), False)
            self._view.addItemsWithTitles_(["List", "Bubbles"])
            self._view.selectItemWithTitle_("Bubbles" if s["default_view"] == "bubbles" else "List")
            try: self._view.setAccessibilityLabel_("Default view")
            except Exception: pass
            c.addSubview_(self._view)

            save = NSButton.alloc().initWithFrame_(NSMakeRect(Wd - 100, 20, 84, 32)); save.setTitle_("Save")
            cancel = NSButton.alloc().initWithFrame_(NSMakeRect(Wd - 196, 20, 84, 32)); cancel.setTitle_("Cancel")
            try:
                save.setBezelStyle_(1); cancel.setBezelStyle_(1)
                self.window.setDefaultButtonCell_(save.cell())
            except Exception:
                pass
            save.setTarget_(self); save.setAction_("saveClicked:")
            cancel.setTarget_(self); cancel.setAction_("cancelClicked:")
            c.addSubview_(save); c.addSubview_(cancel)

        @objc.python_method
        def present(self):
            if self.window is None:
                self.build()
            try:
                _set_policy_regular_temporarily()
                NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
            except Exception:
                pass
            self.window.makeKeyAndOrderFront_(None)

        @objc.python_method
        def _num(self, field, default):
            try:
                return float(str(field.stringValue()).strip())
            except Exception:
                return default

        def saveClicked_(self, sender):
            cur = current_settings()
            data = {
                "RELATION_MIN_SCORE": self._num(self._score, cur["RELATION_MIN_SCORE"]),
                "RELATION_MIN_EMBED_SIM": self._num(self._embed, cur["RELATION_MIN_EMBED_SIM"]),
                "RELATED_MAX_PER_NOTE": int(self._num(self._maxrel, cur["RELATED_MAX_PER_NOTE"])),
                "USE_LLM_RELATION_VERIFY": bool(int(self._verify.state())),
                "default_view": "bubbles" if str(self._view.titleOfSelectedItem()) == "Bubbles" else "list",
            }
            save_settings(data)
            try: self.window.orderOut_(None)
            except Exception: pass
            if self._on_saved:
                self._on_saved()

        def cancelClicked_(self, sender):
            try: self.window.orderOut_(None)
            except Exception: pass

    class NoteEditorWindowController(NSObject):
        """A simple editor for a note's full details: text, summary, category,
        tags, and entities. Saving preserves these manual edits and only
        recomputes the embedding + related notes (see update_note_fields)."""
        def init(self):
            self = objc.super(NoteEditorWindowController, self).init()
            if self is None:
                return None
            self.window = None
            self._note = {}
            self._nid = None
            self._on_save = None
            self._content_tv = None
            self._summary_fld = None
            self._category_pop = None
            self._tags_fld = None
            self._entities_fld = None
            return self

        @objc.python_method
        def configure(self, note, on_save):
            self._note = note or {}
            self._nid = self._note.get("id")
            self._on_save = on_save

        @objc.python_method
        def _mklabel(self, text, frame):
            lbl = NSTextField.alloc().initWithFrame_(frame)
            lbl.setBezeled_(False); lbl.setDrawsBackground_(False)
            lbl.setEditable_(False); lbl.setSelectable_(False)
            lbl.setStringValue_(text)
            try:
                lbl.setFont_(NSFont.boldSystemFontOfSize_(11.0))
                lbl.setTextColor_(NSColor.secondaryLabelColor())
            except Exception:
                pass
            return lbl

        @objc.python_method
        def _field(self, value, frame, a11y):
            f = NSTextField.alloc().initWithFrame_(frame)
            f.setStringValue_(value or "")
            try: f.setAccessibilityLabel_(a11y)
            except Exception: pass
            return f

        def build(self):
            Wd, H = 520.0, 560.0
            self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(180, 160, Wd, H), 15, NSBackingStoreBuffered, False
            )
            self.window.setTitle_("Edit Note")
            self.window.setReleasedWhenClosed_(False)
            content = self.window.contentView()
            note = self._note

            titleLbl = self._mklabel("✏️  Edit Note", NSMakeRect(20, 524, 480, 24))
            try: titleLbl.setFont_(NSFont.boldSystemFontOfSize_(16.0)); titleLbl.setTextColor_(NSColor.labelColor())
            except Exception: pass
            content.addSubview_(titleLbl)

            content.addSubview_(self._mklabel("Note text", NSMakeRect(20, 500, 200, 16)))
            self._content_tv = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, 480, 196))
            self._content_tv.setRichText_(False)
            self._content_tv.setEditable_(True)
            try:
                self._content_tv.setFont_(NSFont.systemFontOfSize_(13.0))
                self._content_tv.setString_(note.get("content", "") or "")
                self._content_tv.setAccessibilityLabel_("Note text")
            except Exception:
                pass
            cscroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(20, 300, 480, 196))
            cscroll.setDocumentView_(self._content_tv)
            cscroll.setHasVerticalScroller_(True)
            try: cscroll.setBorderType_(1)
            except Exception: pass
            content.addSubview_(cscroll)

            content.addSubview_(self._mklabel("Summary", NSMakeRect(20, 276, 200, 16)))
            self._summary_fld = self._field(note.get("summary", ""), NSMakeRect(20, 250, 480, 24), "Summary")
            content.addSubview_(self._summary_fld)

            content.addSubview_(self._mklabel("Category", NSMakeRect(20, 222, 200, 16)))
            cats = ["Idea", "To-Do", "Code Snippet", "Link", "Quote", "Contact", "Other"]
            self._category_pop = NSPopUpButton.alloc().initWithFrame_pullsDown_(NSMakeRect(20, 192, 240, 26), False)
            self._category_pop.addItemsWithTitles_(cats)
            cur = note.get("category") or "Other"
            if cur in cats:
                self._category_pop.selectItemWithTitle_(cur)
            try: self._category_pop.setAccessibilityLabel_("Category")
            except Exception: pass
            content.addSubview_(self._category_pop)

            content.addSubview_(self._mklabel("Tags (comma-separated)", NSMakeRect(20, 166, 300, 16)))
            self._tags_fld = self._field(", ".join(note.get("tags", []) or []), NSMakeRect(20, 140, 480, 24), "Tags")
            content.addSubview_(self._tags_fld)

            content.addSubview_(self._mklabel("Entities (comma-separated)", NSMakeRect(20, 112, 300, 16)))
            self._entities_fld = self._field(", ".join(note.get("entities", []) or []), NSMakeRect(20, 86, 480, 24), "Entities")
            content.addSubview_(self._entities_fld)

            saveBtn = NSButton.alloc().initWithFrame_(NSMakeRect(416, 20, 84, 32))
            saveBtn.setTitle_("Save")
            cancelBtn = NSButton.alloc().initWithFrame_(NSMakeRect(322, 20, 84, 32))
            cancelBtn.setTitle_("Cancel")
            try:
                saveBtn.setBezelStyle_(1); cancelBtn.setBezelStyle_(1)
                self.window.setDefaultButtonCell_(saveBtn.cell())
            except Exception:
                pass
            saveBtn.setTarget_(self); saveBtn.setAction_("saveClicked:")
            cancelBtn.setTarget_(self); cancelBtn.setAction_("cancelClicked:")
            content.addSubview_(saveBtn)
            content.addSubview_(cancelBtn)

        @objc.python_method
        def present(self):
            if self.window is None:
                self.build()
            try:
                _set_policy_regular_temporarily()
                NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
            except Exception:
                pass
            self.window.makeKeyAndOrderFront_(None)

        @objc.python_method
        def _split_csv(self, s):
            return [x.strip() for x in (s or "").split(",") if x.strip()]

        def saveClicked_(self, sender):
            fields = None
            try:
                fields = {
                    "content": str(self._content_tv.string() or ""),
                    "summary": str(self._summary_fld.stringValue() or ""),
                    "category": str(self._category_pop.titleOfSelectedItem() or "Other"),
                    "tags": self._split_csv(str(self._tags_fld.stringValue() or "")),
                    "entities": self._split_csv(str(self._entities_fld.stringValue() or "")),
                }
            except Exception:
                logging.exception("Gathering edited fields failed")
            try: self.window.orderOut_(None)
            except Exception: pass
            if fields is not None and self._on_save:
                self._on_save(self._nid, fields)

        def cancelClicked_(self, sender):
            try: self.window.orderOut_(None)
            except Exception: pass

    # --- QuickCapture panel controller (focus-safe) ---
    class QuickCapturePanelController(NSObject):
        def init(self):
            self = objc.super(QuickCapturePanelController, self).init()
            if self is None:
                return None
            self.window = None
            self.textField = None
            self.resultText = None
            return self

        def build(self):
            rect = NSMakeRect(0, 0, 520, 120)
            style = _STYLE_TITLED | _STYLE_CLOSABLE
            self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(rect, style, NSBackingStoreBuffered, False)
            self.window.setTitle_("Quick Capture")
            try:
                self.window.center()
            except Exception:
                pass
            content = self.window.contentView()
            label = NSTextField.alloc().initWithFrame_(NSMakeRect(16, 78, 488, 18))
            label.setBezeled_(False)
            label.setDrawsBackground_(False)
            label.setEditable_(False)
            label.setSelectable_(False)
            label.setStringValue_("Type your thought and press Return to save")
            self.textField = NSTextField.alloc().initWithFrame_(NSMakeRect(16, 46, 488, 24))
            try:
                self.textField.setPlaceholderString_("What's on your mind?")
            except Exception:
                pass
            saveBtn = NSButton.alloc().initWithFrame_(NSMakeRect(330, 12, 80, 28))
            saveBtn.setTitle_("Save")
            cancelBtn = NSButton.alloc().initWithFrame_(NSMakeRect(420, 12, 80, 28))
            cancelBtn.setTitle_("Cancel")
            try:
                saveBtn.setBezelStyle_(1)
                cancelBtn.setBezelStyle_(1)
            except Exception:
                pass
            try:
                saveBtn.setTarget_(self)
                saveBtn.setAction_(objc.selector(self.saveClicked_, signature=b'v@:@'))
                cancelBtn.setTarget_(self)
                cancelBtn.setAction_(objc.selector(self.cancelClicked_, signature=b'v@:@'))
                self.window.setDefaultButtonCell_(saveBtn.cell())
            except Exception:
                pass
            try:
                content.addSubview_(label)
            except Exception:
                pass
            content.addSubview_(self.textField)
            content.addSubview_(saveBtn)
            content.addSubview_(cancelBtn)

        def runModalAndGetText(self):
            _orig_policy = _set_policy_regular_temporarily()
            try:
                NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
            except Exception:
                pass
            if self.window is None:
                self.build()
            self.window.makeKeyAndOrderFront_(None)
            try:
                self.window.makeKeyWindow()
            except Exception:
                pass
            try:
                self.window.makeFirstResponder_(self.textField)
            except Exception:
                pass
            try:
                NSApplication.sharedApplication().runModalForWindow_(self.window)
            except Exception:
                pass
            res = self.resultText or ""
            _restore_policy(_orig_policy)
            return res

        def saveClicked_(self, sender):
            try:
                self.resultText = str(self.textField.stringValue() or "").strip()
            except Exception:
                self.resultText = ""
            try:
                self.window.orderOut_(None)
            except Exception:
                pass
            try:
                NSApplication.sharedApplication().stopModalWithCode_(1)
            except Exception:
                pass

        def cancelClicked_(self, sender):
            self.resultText = ""
            try:
                self.window.orderOut_(None)
            except Exception:
                pass
            try:
                NSApplication.sharedApplication().stopModalWithCode_(0)
            except Exception:
                pass
    class ShowAllNotesWindowController(NSObject):
        """
        A split-view native window:
          - Left: list of notes (title + time + tags count)
          - Right: details with sections (Title, Summary, Tags, Related, Full Note).
          - Bottom expandable section for "Actionable Summary" (computed in background).
        """
        window = objc.IBOutlet()
        table = objc.IBOutlet()
        detailText = objc.IBOutlet()
        summaryText = objc.IBOutlet()
        searchField = objc.IBOutlet()
        statusField = objc.IBOutlet()

        def init(self):
            self = objc.super(ShowAllNotesWindowController, self).init()
            if self is None:
                return None
            self._notes = []
            self._filtered = []
            self._selected_index = 0
            self._window = None
            self._table = None
            self._detail = None
            self._summary = None
            self._one_line = None
            self._status = None
            self._search = None
            self._graph = None
            self._busy = False
            self._bubble = None
            self._listSplit = None
            self._segmented = None
            self._settingsWC = None
            self._miList = None
            self._miBubbles = None
            return self
            # --- sleek styling ---
            try:
                # Transparent titlebar + unified look
                if self._window is not None:
                    self._window.setTitleVisibility_(1)  # NSWindowTitleHidden
                    self._window.setTitlebarAppearsTransparent_(True)
                    self._window.setMovableByWindowBackground_(True)

                # Vibrant background
                if self._window is not None:
                    vev = NSVisualEffectView.alloc().initWithFrame_(self._window.contentView().bounds())
                    vev.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
                    try:
                        # Sidebar material keeps things subtle in both light/dark
                        vev.setMaterial_(14)  # NSVisualEffectMaterialSidebar (value stable across 11+)
                    except Exception:
                        pass
                    vev.setBlendingMode_(0)  # behind window
                    self._window.setContentView_(vev)

                    # Re-add content root into the visual effect view (if we created a root already elsewhere)
                    # We'll lazily create stack root if missing
                    if getattr(self, "_rootStack", None) is None:
                        self._rootStack = NSStackView.alloc().init()
                        self._rootStack.setOrientation_(1)  # vertical
                        self._rootStack.setSpacing_(8)
                        self._rootStack.setEdgeInsets_((12, 12, 12, 12))
                        vev.addSubview_(self._rootStack)

                # Text styles
                def _apply_text_style(tv, size=13.0, mono=False, dim=False):
                    try:
                        font = NSFont.monospacedSystemFontOfSize_weight_(size, 0) if mono else NSFont.systemFontOfSize_(size)
                        tv.setFont_(font)
                    except Exception:
                        pass
                    try:
                        color = NSColor.secondaryLabelColor() if dim else NSColor.labelColor()
                        tv.setTextColor_(color)
                    except Exception:
                        pass
                    try:
                        tv.setDrawsBackground_(False)
                        tv.enclosingScrollView().setBorderType_(0)  # no border
                        tv.textContainer().setLineFragmentPadding_(4.0)
                        tv.setTextContainerInset_((4.0, 2.0))
                    except Exception:
                        pass

                if self._detail is not None:
                    _apply_text_style(self._detail, size=13.0, mono=False, dim=False)
                if self._summary is not None:
                    _apply_text_style(self._summary, size=13.0, mono=False, dim=False)
            except Exception:
                pass
            # --- end sleek styling ---


        # ---- data helpers ----
        def loadNotes_(self, notes):
            self._notes = list(notes or [])
            self._filtered = self._notes[:]
            self.reloadTable()
            if self._filtered:
                self.selectRow_(0)
            # Keep the bubble view in sync when it's the active view.
            try:
                if self._bubble is not None and not self._bubble.isHidden():
                    self._bubble.configure(self._notes, self._openNoteByID_)
            except Exception:
                logging.exception("Refreshing bubble view failed")

        def reloadTable(self):
            if self._table is not None:
                self._table.reloadData()
                self.updateStatus()

        def updateStatus(self):
            total = len(self._notes)
            shown = len(self._filtered)
            msg = f"{shown} of {total} notes" if shown != total else f"{total} notes"
            self._setStatusText_(msg)

        @objc.python_method
        def _setStatusText_(self, text):
            # Prefer the native window subtitle (macOS 11+); fall back to a label.
            try:
                if self._window is not None:
                    self._window.setSubtitle_(text or "")
            except Exception:
                pass
            try:
                if self._status is not None:
                    self._status.setStringValue_(text or "")
            except Exception:
                pass

        def filterNotes_(self, query):
            q = (query or "").strip().lower()
            if not q:
                self._filtered = self._notes[:]
            else:
                def hit(n):
                    blob = " ".join([
                        n.get("title","") or "",
                        n.get("summary","") or "",
                        n.get("content","") or "",
                        " ".join(n.get("tags",[]) or []),
                        n.get("category","") or ""
                    ]).lower()
                    return q in blob
                self._filtered = [n for n in self._notes if hit(n)]
            self.reloadTable()
            if self._filtered:
                self.selectRow_(0)
            else:
                self._detail.setString_("No notes match your search.")
                if getattr(self, "_graph", None) is not None:
                    try:
                        self._graph.configure(None, [], None)
                    except Exception:
                        pass

        def searchChanged_(self, sender):
            try:
                q = self._search.stringValue()
            except Exception:
                q = ""
            self.filterNotes_(q)

        # ---- NSTableView data source / delegate ----
        def numberOfRowsInTableView_(self, tableView):
            return len(self._filtered)

        # View-based rows: each note renders as a rich "card" instead of a
        # spreadsheet row, giving the list real structure and scannability.
        def tableView_heightOfRow_(self, tableView, row):
            return 72.0

        def tableView_viewForTableColumn_row_(self, tableView, column, row):
            try:
                note = self._filtered[row]
            except Exception:
                return None
            try:
                # Reuse recycled row views (snappy scrolling, fewer allocations).
                view = tableView.makeViewWithIdentifier_owner_("NoteCard", self)
                if view is None:
                    view = self._buildCard_()
                    view.setIdentifier_("NoteCard")
                self._fillCard_(view, note)
                return view
            except Exception:
                logging.exception("Note row build failed; using plain label")
                lbl = NSTextField.alloc().initWithFrame_(NSMakeRect(8, 8, 280, 20))
                lbl.setBezeled_(False); lbl.setDrawsBackground_(False)
                lbl.setEditable_(False); lbl.setSelectable_(False)
                lbl.setStringValue_(note.get("summary") or note.get("content","") or "Untitled")
                return lbl

        @objc.python_method
        def _label_(self, text, frame, size=13.0, bold=False, color=None, align=0):
            lbl = NSTextField.alloc().initWithFrame_(frame)
            lbl.setBezeled_(False); lbl.setDrawsBackground_(False)
            lbl.setEditable_(False); lbl.setSelectable_(False)
            try:
                lbl.setFont_(NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size))
            except Exception:
                pass
            if color is not None:
                try: lbl.setTextColor_(color)
                except Exception: pass
            try: lbl.setAlignment_(align)
            except Exception: pass
            lbl.setStringValue_(text or "")
            try: lbl.cell().setLineBreakMode_(4)  # truncate tail
            except Exception: pass
            return lbl

        @objc.python_method
        def _section_(self, title, contentView, frame):
            """A clean, borderless section: a small uppercase header above its
            content — the modern macOS look, instead of a dated group box."""
            w, h = frame.size.width, frame.size.height
            container = NSView.alloc().initWithFrame_(frame)
            try: container.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
            except Exception: pass
            hdr = self._label_(title, NSMakeRect(2.0, h - 20.0, w - 4.0, 16.0),
                               size=11.0, bold=True, color=NSColor.secondaryLabelColor())
            try: hdr.setAutoresizingMask_(NSViewWidthSizable | NSViewMinYMargin)
            except Exception: pass
            container.addSubview_(hdr)
            try:
                contentView.setFrame_(NSMakeRect(0.0, 0.0, w, h - 26.0))
                contentView.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
            except Exception:
                pass
            container.addSubview_(contentView)
            return container

        # Tags identify the labels inside a reusable card so we can repopulate
        # without rebuilding the view hierarchy on every scroll.
        _T_ICON, _T_TITLE, _T_SUMMARY, _T_META, _T_BADGE = 104, 101, 102, 103, 105

        @objc.python_method
        def _buildCard_(self):
            W, H = 320, 72
            view = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, W, H))
            try: view.setAutoresizingMask_(NSViewWidthSizable)
            except Exception: pass

            icon = self._label_("", NSMakeRect(12, 26, 26, 26), size=18)
            icon.setTag_(self._T_ICON)
            view.addSubview_(icon)

            titleLbl = self._label_("", NSMakeRect(46, 46, W - 118, 20), size=13, bold=True, color=NSColor.labelColor())
            titleLbl.setTag_(self._T_TITLE)
            try: titleLbl.setAutoresizingMask_(NSViewWidthSizable)
            except Exception: pass
            view.addSubview_(titleLbl)

            # Width reserves room on the right for the 🔗 link badge (avoids overlap).
            summLbl = self._label_("", NSMakeRect(46, 26, W - 118, 18), size=11, color=NSColor.secondaryLabelColor())
            summLbl.setTag_(self._T_SUMMARY)
            try: summLbl.setAutoresizingMask_(NSViewWidthSizable)
            except Exception: pass
            view.addSubview_(summLbl)

            metaLbl = self._label_("", NSMakeRect(46, 7, W - 118, 14), size=10, color=NSColor.tertiaryLabelColor())
            metaLbl.setTag_(self._T_META)
            try: metaLbl.setAutoresizingMask_(NSViewWidthSizable)
            except Exception: pass
            view.addSubview_(metaLbl)

            badge = self._label_("", NSMakeRect(W - 58, 28, 50, 16), size=11, color=NSColor.secondaryLabelColor(), align=1)
            badge.setTag_(self._T_BADGE)
            try: badge.setAutoresizingMask_(NSViewMinXMargin)
            except Exception: pass
            view.addSubview_(badge)
            return view

        @objc.python_method
        def _fillCard_(self, view, note):
            cat = note.get("category", "") or "Other"
            title = note.get("title") or friendly_snippet(note.get("summary") or note.get("content", ""), 70) or "Untitled"
            summ = friendly_snippet(note.get("summary") or note.get("content", ""), 90)
            ts = (note.get("timestamp", "") or "")[:16].replace("T", " ")
            tags = note.get("tags", []) or []
            tagstr = " ".join(f"#{t}" for t in tags[:3])
            meta = f"{cat} · {ts}" + (f" · {tagstr}" if tagstr else "")
            nrel = len(note.get("related", []) or [])

            def setv(tag, s):
                v = view.viewWithTag_(tag)
                if v is not None:
                    v.setStringValue_(s)

            setv(self._T_ICON, category_glyph(cat))
            setv(self._T_TITLE, title)
            setv(self._T_SUMMARY, summ)
            setv(self._T_META, meta)
            badge = view.viewWithTag_(self._T_BADGE)
            if badge is not None:
                badge.setStringValue_(f"🔗 {nrel}" if nrel else "")
                try: badge.setHidden_(nrel == 0)
                except Exception: pass
            try:
                view.setAccessibilityLabel_(
                    f"{cat} note. {title}. {summ}." + (f" {nrel} linked notes." if nrel else " No linked notes.")
                )
            except Exception:
                pass

        def tableViewSelectionDidChange_(self, notification):
            row = self._table.selectedRow()
            if row >= 0:
                self.selectRow_(int(row))

        def selectRow_(self, idx):
            if idx < 0 or idx >= len(self._filtered):
                return
            self._selected_index = idx
            note = self._filtered[idx]
            self.renderDetail_(note)
            self._updateGraph_(note)

        @objc.python_method
        def _updateGraph_(self, note):
            if getattr(self, "_graph", None) is None:
                return
            rels = []
            for rel in (note.get("related") or [])[:8]:
                rn = fetch_note_by_id_safe(rel.get("id"))
                if not rn:
                    continue
                rels.append({
                    "id": rel.get("id"),
                    "title": rn.get("summary") or rn.get("content", "") or "Untitled",
                    "glyph": category_glyph(rn.get("category", "")),
                    "score": rel.get("score", 0.0),
                })
            center = {
                "title": note.get("summary") or note.get("content", "") or "Untitled",
                "category": note.get("category", ""),
            }
            try:
                self._graph.configure(center, rels, self._selectByID_)
            except Exception:
                logging.exception("Graph update failed")

        @objc.python_method
        def _selectByID_(self, nid):
            for i, n in enumerate(self._filtered):
                if n.get("id") == nid:
                    self._selectAndReveal_(i)
                    return
            # Not in current filter: clear it and jump to the note.
            for i, n in enumerate(self._notes):
                if n.get("id") == nid:
                    try:
                        self._search.setStringValue_("")
                    except Exception:
                        pass
                    self._filtered = self._notes[:]
                    self.reloadTable()
                    self._selectAndReveal_(i)
                    return

        def renderDetail_(self, note):
            # Prefer a beautiful, structured rich-text detail; fall back to plain.
            try:
                self._renderDetailRich_(note)
            except Exception:
                logging.exception("Rich detail render failed; using plain text")
                self._renderDetailPlain_(note)

        @objc.python_method
        def _para_(self, spacing=2.0, space_before=0.0, space_after=0.0, leading=0.0):
            p = NSMutableParagraphStyle.alloc().init()
            try:
                p.setLineSpacing_(spacing)
                p.setParagraphSpacing_(space_after)
                p.setParagraphSpacingBefore_(space_before)
                if leading:
                    p.setHeadIndent_(leading)
                p.setLineBreakMode_(0)  # word wrap
            except Exception:
                pass
            return p

        @objc.python_method
        def _append_(self, mstr, text, font, color, para=None, link=None, underline=False):
            attrs = {}
            try:
                if font is not None: attrs[NSFontAttributeName] = font
                if color is not None: attrs[NSForegroundColorAttributeName] = color
                if para is not None: attrs[NSParagraphStyleAttributeName] = para
                if link is not None: attrs[NSLinkAttributeName] = NSString.stringWithString_(link)
                if underline: attrs[NSUnderlineStyleAttributeName] = 1
            except Exception:
                pass
            piece = NSAttributedString.alloc().initWithString_attributes_(text or "", attrs)
            mstr.appendAttributedString_(piece)

        @objc.python_method
        def _renderDetailRich_(self, note):
            from Foundation import NSMutableAttributedString
            mstr = NSMutableAttributedString.alloc().init()
            label = NSColor.labelColor()
            sec = NSColor.secondaryLabelColor()
            ter = NSColor.tertiaryLabelColor()
            try:
                accent = NSColor.controlAccentColor()
            except Exception:
                accent = NSColor.systemBlueColor()

            cat = note.get("category", "") or "Other"
            glyph = category_glyph(cat)
            title = note.get("title") or friendly_snippet(note.get("summary") or note.get("content", ""), 80) or "Untitled"

            # Title
            self._append_(mstr, f"{glyph}  {title}\n", NSFont.boldSystemFontOfSize_(22), label, self._para_(space_after=6.0))
            # Category + time
            ts = (note.get("timestamp", "") or "").replace("T", " ")[:19]
            self._append_(mstr, f"{cat}   ·   {ts}\n", NSFont.systemFontOfSize_(11), ter, self._para_(space_after=8.0))
            # Tags
            tags = note.get("tags", []) or []
            if tags:
                for t in tags[:8]:
                    self._append_(mstr, f"#{t}", NSFont.systemFontOfSize_(11), accent)
                    self._append_(mstr, "   ", NSFont.systemFontOfSize_(11), sec)
                self._append_(mstr, "\n", NSFont.systemFontOfSize_(11), sec, self._para_(space_after=10.0))

            # Summary
            self._append_(mstr, "SUMMARY\n", NSFont.boldSystemFontOfSize_(11), sec, self._para_(space_before=10.0, space_after=4.0))
            self._append_(mstr, (note.get("summary") or "No summary") + "\n", NSFont.systemFontOfSize_(14), label, self._para_(spacing=3.0, space_after=12.0))

            # Connected notes — the inter-note connectivity, with reasons + clickable links
            rels = note.get("related", []) or []
            self._append_(mstr, f"CONNECTED NOTES ({len(rels)})\n", NSFont.boldSystemFontOfSize_(11), sec, self._para_(space_before=10.0, space_after=6.0))
            if rels:
                for rel in rels[:10]:
                    rid = rel.get("id")
                    score = rel.get("score", 0.0)
                    reason = rel.get("reason")
                    rn = fetch_note_by_id_safe(rid)
                    if not rn:
                        continue
                    rglyph = category_glyph(rn.get("category", ""))
                    snippet = friendly_snippet(rn.get("summary") or rn.get("content", ""), 110)
                    pct = int(round(max(0.0, min(1.0, float(score))) * 100))
                    self._append_(mstr, f"{rglyph}  ", NSFont.systemFontOfSize_(13), label, self._para_(leading=22.0))
                    self._append_(mstr, snippet, NSFont.systemFontOfSize_(13), accent, self._para_(leading=22.0), link=f"note://{rid}", underline=True)
                    self._append_(mstr, f"   ({pct}% match)\n", NSFont.systemFontOfSize_(11), ter, self._para_(space_after=(1.0 if reason else 8.0), leading=22.0))
                    if reason:
                        self._append_(mstr, f"↳ {reason}\n", NSFont.systemFontOfSize_(11), sec, self._para_(space_after=8.0, leading=22.0))
            else:
                self._append_(mstr, "No connections yet — related notes will appear here.\n", NSFont.systemFontOfSize_(13), ter, self._para_(space_after=12.0))

            # Full note
            self._append_(mstr, "FULL NOTE\n", NSFont.boldSystemFontOfSize_(11), sec, self._para_(space_before=10.0, space_after=4.0))
            self._append_(mstr, (note.get("content", "") or "") + "\n", NSFont.systemFontOfSize_(13), label, self._para_(spacing=3.0))

            try:
                self._detail.textStorage().setAttributedString_(mstr)
            except Exception:
                self._detail.setString_(note.get("content", "") or "")

        @objc.python_method
        def _renderDetailPlain_(self, note):
            lines = [note.get("title") or "Untitled", "",
                     f"Time: {note.get('timestamp','')}    Category: {note.get('category','')}",
                     f"Tags: {', '.join(note.get('tags',[]) or []) or '-'}", "",
                     "Summary:", note.get("summary") or "No summary", "", "Related:"]
            rels = note.get("related", []) or []
            if rels:
                for rel in rels[:10]:
                    rn = fetch_note_by_id_safe(rel.get("id"))
                    if rn:
                        lines.append(f"• {friendly_snippet(rn.get('summary') or rn.get('content',''), 100)}  [{rel.get('score',0.0):.2f}]")
                        if rel.get("reason"):
                            lines.append(f"    ↳ {rel.get('reason')}")
            else:
                lines.append("— None —")
            lines += ["", "Full Note:", note.get("content", "")]
            self._detail.setString_("\n".join(lines))

        # Click-through on a connected-note link selects that note (accessible navigation).
        def textView_clickedOnLink_atIndex_(self, textView, link, charIndex):
            try:
                s = str(link)
            except Exception:
                return False
            if not s.startswith("note://"):
                return False
            nid = s[len("note://"):]
            for i, n in enumerate(self._filtered):
                if n.get("id") == nid:
                    self._selectAndReveal_(i)
                    return True
            # Not in the current filter — clear it and jump to the note.
            for i, n in enumerate(self._notes):
                if n.get("id") == nid:
                    try:
                        self._search.setStringValue_("")
                    except Exception:
                        pass
                    self._filtered = self._notes[:]
                    self.reloadTable()
                    self._selectAndReveal_(i)
                    return True
            return False

        @objc.python_method
        def _selectAndReveal_(self, idx):
            try:
                self._table.selectRowIndexes_byExtendingSelection_(NSIndexSet.indexSetWithIndex_(idx), False)
                self._table.scrollRowToVisible_(idx)
            except Exception:
                pass
            self.selectRow_(idx)

        # ---- Window construction ----
        def show_(self, sender):
            rect = NSMakeRect(120.0, 120.0, 1080.0, 740.0)
            style_mask = 15  # titled, closable, resizable, miniaturizable
            self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                rect, style_mask, NSBackingStoreBuffered, False
            )
            self._window.setTitle_("Ambient Notes")
            self._window.setReleasedWhenClosed_(False)
            # Native polish: unified translucent titlebar, remembered size/position,
            # and a sensible minimum size.
            try:
                self._window.setTitlebarAppearsTransparent_(True)
                self._window.setMinSize_((760.0, 520.0))
                self._window.setFrameAutosaveName_("AmbientNotesWindow")
            except Exception:
                pass

            content = self._window.contentView()
            try:
                content.setAutoresizesSubviews_(True)
            except Exception:
                pass
            cb = content.bounds()
            W = cb.size.width
            Hc = cb.size.height
            M, GAP, HDR, HI = 14.0, 8.0, 34.0, 96.0

            # ---- Title ----
            titleLbl = self._label_("📓  All Notes", NSMakeRect(0, 0, 220, HDR), size=20, bold=True, color=NSColor.labelColor())

            # ---- Status (count) ----
            self._status = self._label_("", NSMakeRect(0, 0, 180, HDR), size=11, color=NSColor.secondaryLabelColor(), align=2)
            try: self._status.setAccessibilityLabel_("Note count")
            except Exception: pass

            # The view switcher now lives in the menu bar's "View" menu
            # (installed in _installViewMenu_), not in the window header.
            self._segmented = None

            # ---- Toolbar buttons ----
            refreshBtn = self._button_("↻  Refresh Details", "refreshDetailsClicked:")
            summaryBtn = self._button_("✦  Summary", "showSummaryClicked:")
            settingsBtn = self._button_("⚙︎  Settings", "settingsClicked:")

            # ---- Search ----
            self._search = NSSearchField.alloc().initWithFrame_(NSMakeRect(0, 0, 200, 24))
            self._search.setPlaceholderString_("Search notes…")
            try: self._search.setAccessibilityLabel_("Search notes")
            except Exception: pass

            # ---- Header (title · switcher · buttons · search) ----
            header = NSStackView.alloc().initWithFrame_(NSMakeRect(M, Hc - M - HDR, W - 2*M, HDR))
            header.setOrientation_(0)
            try:
                header.setAlignment_(10)  # center-Y
                header.setSpacing_(10.0)
                header.setDistribution_(1)
            except Exception:
                pass
            header.addView_inGravity_(titleLbl, 1)       # leading
            header.addView_inGravity_(settingsBtn, 3)    # trailing
            header.addView_inGravity_(refreshBtn, 3)
            header.addView_inGravity_(summaryBtn, 3)
            header.addView_inGravity_(self._search, 3)
            try: header.setAutoresizingMask_(NSViewWidthSizable | NSViewMinYMargin)
            except Exception: pass

            # ---- Highlights (multiline, in a titled box) ----
            self._one_line = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, W - 2*M - 24, HI - 28))
            self._one_line.setEditable_(False)
            self._one_line.setSelectable_(True)
            self._one_line.setRichText_(False)
            try:
                self._one_line.setDrawsBackground_(False)
                self._one_line.setFont_(NSFont.systemFontOfSize_(12.5))
                self._one_line.setTextColor_(NSColor.labelColor())
                self._one_line.setTextContainerInset_((4.0, 4.0))
                self._one_line.setHorizontallyResizable_(False)
                self._one_line.textContainer().setWidthTracksTextView_(True)
                self._one_line.setAccessibilityLabel_("Highlights across your notes")
            except Exception:
                pass
            hi_scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 0, W - 2*M, HI - 8))
            hi_scroll.setDocumentView_(self._one_line)
            hi_scroll.setHasVerticalScroller_(True)
            hi_scroll.setAutohidesScrollers_(True)
            hi_scroll.setDrawsBackground_(False)
            hi_box = self._section_("✨  HIGHLIGHTS", hi_scroll, NSMakeRect(M, Hc - M - HDR - GAP - HI, W - 2*M, HI))
            try:
                hi_box.setAutoresizingMask_(NSViewWidthSizable | NSViewMinYMargin)
            except Exception:
                pass

            # ---- Left list (rich cards, no spreadsheet header) ----
            self._table = NSTableView.alloc().initWithFrame_(NSMakeRect(0, 0, 320, 600))
            col = NSTableColumn.alloc().initWithIdentifier_("card")
            col.setWidth_(320)
            try: col.setResizingMask_(1)  # autoresize with table
            except Exception: pass
            self._table.addTableColumn_(col)
            try:
                self._table.setColumnAutoresizingStyle_(1)  # uniform: column fills width
                self._table.setHeaderView_(None)  # no Excel-style header
                self._table.setRowSizeStyle_(3)   # custom
                self._table.setIntercellSpacing_((0.0, 6.0))
                self._table.setUsesAlternatingRowBackgroundColors_(False)
                self._table.setBackgroundColor_(NSColor.clearColor())
                self._table.setSelectionHighlightStyle_(-1)  # source-list (modern rounded selection)
                self._table.setAccessibilityLabel_("Notes list")
            except Exception:
                pass
            self._table.setAllowsMultipleSelection_(False)
            self._table.setDelegate_(self)
            self._table.setDataSource_(self)

            # Right-click context menu: edit details or delete the note under the cursor.
            try:
                if NSMenu is not None:
                    menu = NSMenu.alloc().init()
                    for title, sel in (("Edit Details…", "editNoteClicked:"),
                                       ("Delete Note", "deleteNoteClicked:")):
                        item = NSMenuItem.alloc().init()
                        item.setTitle_(title)
                        item.setAction_(sel)
                        item.setTarget_(self)
                        menu.addItem_(item)
                    self._table.setMenu_(menu)
            except Exception:
                logging.exception("Context menu setup failed")

            left_scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 0, 320, 600))
            left_scroll.setDocumentView_(self._table)
            left_scroll.setHasVerticalScroller_(True)
            left_scroll.setAutohidesScrollers_(True)
            try: left_scroll.setDrawsBackground_(False)
            except Exception: pass

            # ---- Right detail (rich text, clickable connections) ----
            self._detail = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, 640, 600))
            self._detail.setEditable_(False)
            self._detail.setSelectable_(True)
            self._detail.setRichText_(True)
            try:
                self._detail.setDrawsBackground_(True)
                self._detail.setBackgroundColor_(NSColor.textBackgroundColor())
                self._detail.setTextContainerInset_((16.0, 14.0))
                self._detail.setDelegate_(self)
                self._detail.setAccessibilityLabel_("Note detail")
            except Exception:
                pass
            self._detail.setUsesFontPanel_(False)
            self._detail.setAutomaticQuoteSubstitutionEnabled_(False)
            self._detail.setAutomaticDataDetectionEnabled_(True)
            self._detail.setAllowsUndo_(False)

            right_scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 0, 640, 320))
            right_scroll.setDocumentView_(self._detail)
            right_scroll.setHasVerticalScroller_(True)
            right_scroll.setAutohidesScrollers_(True)
            try: right_scroll.setBorderType_(0)
            except Exception: pass
            detail_box = self._section_("📄  DETAILS", right_scroll, NSMakeRect(0, 0, 660, 320))

            # ---- Connection web (top of right pane) ----
            self._graph = NoteGraphView.alloc().initWithFrame_(NSMakeRect(0, 0, 640, 300))
            try:
                self._graph.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
            except Exception:
                pass
            graph_box = self._section_("🕸  CONNECTIONS", self._graph, NSMakeRect(0, 0, 660, 300))

            # right split: connection web (top) over note details (bottom)
            right_split = NSSplitView.alloc().initWithFrame_(NSMakeRect(0, 0, 660, 600))
            right_split.setDividerStyle_(2)
            right_split.setVertical_(False)
            right_split.setAutosaveName_("ShowAllNotesRightSplitV3")
            right_split.setDelegate_(self)
            right_split.addSubview_(graph_box)
            right_split.addSubview_(detail_box)
            try:
                right_split.setPosition_ofDividerAtIndex_(310.0, 0)
            except Exception:
                pass

            # Translucent sidebar material behind the notes list (native sidebar feel).
            sidebar = left_scroll
            try:
                vev = NSVisualEffectView.alloc().initWithFrame_(NSMakeRect(0, 0, 330, 600))
                try: vev.setMaterial_(14)       # NSVisualEffectMaterialSidebar
                except Exception: pass
                try: vev.setBlendingMode_(0)     # behind window
                except Exception: pass
                try: vev.setState_(1)            # active
                except Exception: pass
                left_scroll.setFrame_(vev.bounds())
                left_scroll.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
                vev.addSubview_(left_scroll)
                sidebar = vev
            except Exception:
                logging.exception("Sidebar vibrancy unavailable")

            # main split: list (left) | connections+details (right)
            split = NSSplitView.alloc().initWithFrame_(NSMakeRect(M, M, W - 2*M, Hc - M - HDR - GAP - HI - GAP - M))
            split.setDividerStyle_(2)
            split.setVertical_(True)
            split.setAutosaveName_("ShowAllNotesMainSplitV2")
            split.setDelegate_(self)
            split.addSubview_(sidebar)
            split.addSubview_(right_split)
            try:
                split.setPosition_ofDividerAtIndex_(330.0, 0)
                split.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
            except Exception:
                pass

            self._listSplit = split

            # ---- Bubble view (occupies the same area as the list split) ----
            body_frame = NSMakeRect(M, M, W - 2*M, Hc - M - HDR - GAP - HI - GAP - M)
            self._bubble = NoteBubbleView.alloc().initWithFrame_(body_frame)
            try:
                self._bubble.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
            except Exception:
                pass

            content.addSubview_(header)
            content.addSubview_(hi_box)
            content.addSubview_(split)
            content.addSubview_(self._bubble)

            # Install the View menu first so the initial view's checkmark is set.
            self._installViewMenu_()
            self._applyViewMode_("bubbles" if APP_SETTINGS.get("default_view") == "bubbles" else "list")

            # wire up search field change (Cocoa target–action)
            self._search.setTarget_(self)
            try:
                self._search.setSendsWholeSearchString_(False)
                self._search.setContinuous_(True)
            except Exception:
                pass
            self._search.setAction_("searchChanged:")

            try:
                NSNotificationCenter.defaultCenter().addObserver_selector_name_object_(
                    self,
                    objc.selector(self._notesDidChange_, signature=b'v@:@'),
                    'NotesDatabaseDidChange',
                    None
                )
            except Exception:
                pass
            self._window.makeKeyAndOrderFront_(None)

        # ---------- Safe setters & public API (inside class) ----------
        @objc.python_method
        def _setTextViewString(self, tv, string):
            # Apply an explicit label color + font. A bare attributed string has
            # no color attribute and renders black — invisible in dark mode.
            try:
                attrs = {
                    NSFontAttributeName: NSFont.systemFontOfSize_(12.5),
                    NSForegroundColorAttributeName: NSColor.labelColor(),
                }
                tv.textStorage().setAttributedString_(
                    NSAttributedString.alloc().initWithString_attributes_(string or "", attrs))
            except Exception:
                try:
                    tv.setString_(string or "")
                    tv.setTextColor_(NSColor.labelColor())
                except Exception:
                    pass

        def _updateSummary_(self, s):
            try:
                logging.info("UI: updating actionable summary (len=%d)", len(s or ""))
            except Exception:
                pass
            if self._summary is not None:
                self._setTextViewString(self._summary, (s or "(no actionable summary)"))

        @objc.python_method
        def _setFieldString(self, field, string):
            try:
                if field is not None:
                    field.setStringValue_(string or "")
            except Exception:
                pass

        def _updateOneLine_(self, s):
            try:
                logging.info("UI: updating highlights to: %r", s)
            except Exception:
                pass
            # Highlights is now a multiline text view, not a single-line field.
            if self._one_line is not None:
                self._setTextViewString(self._one_line, (s or "Highlights will appear here once your notes are analyzed."))

        # public API
        def setActionableSummary_(self, text):
            # Set from any thread (dispatch to main via AppHelper)
            try:
                AppHelper.callAfter(self._updateSummary_, text)
            except Exception:
                # last resort: set directly (may warn if not main thread)
                self._updateSummary_(text)

        def setOneLineSummary_(self, text):
            # Set from any thread (dispatch to main via AppHelper)
            try:
                AppHelper.callAfter(self._updateOneLine_, text)
            except Exception:
                self._updateOneLine_(text)

        def setDetailText_(self, text):
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                objc.selector(self._setTextViewString, signature=b'v@:@@'),
                (self._detail, text), False
            )

        # ---------- Toolbar / refresh / live updates ----------
        @objc.python_method
        def _button_(self, title, action_sel):
            btn = NSButton.alloc().initWithFrame_(NSMakeRect(0, 0, 150, 26))
            btn.setTitle_(title)
            try:
                btn.setBezelStyle_(1)   # rounded
                btn.setButtonType_(0)   # momentary push
                btn.setFont_(NSFont.systemFontOfSize_(12.0))
            except Exception:
                pass
            btn.setTarget_(self)
            btn.setAction_(action_sel)
            try:
                btn.setAccessibilityLabel_(title.replace("↻", "").replace("✦", "").strip())
            except Exception:
                pass
            return btn

        @objc.python_method
        def _loadNotesFromDB(self):
            notes = []
            with db_lock:
                c = db_conn.cursor()
                try:
                    c.execute("SELECT * FROM notes ORDER BY timestamp DESC")
                    rows = c.fetchall()
                    cols = [r[1] for r in c.execute("PRAGMA table_info(notes)").fetchall()]
                except Exception:
                    rows, cols = [], []
            for row in rows:
                d = {}
                for i, col in enumerate(cols):
                    d[col] = row[i]
                for k in ("tags", "entities", "related"):
                    if d.get(k):
                        try:
                            d[k] = json.loads(d[k])
                        except Exception:
                            d[k] = []
                    else:
                        d[k] = []
                if not d.get("summary"):
                    d["summary"] = "No summary"
                notes.append(d)
            return notes

        @objc.python_method
        def _reloadFromDB(self):
            sel_id = None
            try:
                if 0 <= self._selected_index < len(self._filtered):
                    sel_id = self._filtered[self._selected_index].get("id")
            except Exception:
                pass
            notes = self._loadNotesFromDB()
            self.loadNotes_(notes)
            if sel_id:
                for i, n in enumerate(self._filtered):
                    if n.get("id") == sel_id:
                        self._selectAndReveal_(i)
                        break

        # Live-update the window whenever the database changes (notes stored/enriched).
        def _notesDidChange_(self, note):
            # Skip during a batch refresh — it issues one consolidated reload at the end.
            if getattr(self, "_busy", False):
                return
            try:
                AppHelper.callAfter(self._reloadFromDB)
            except Exception:
                self._reloadFromDB()

        # "Refresh Details": find notes missing details and generate them.
        def refreshDetailsClicked_(self, sender):
            if getattr(self, "_busy", False):
                return
            self._busy = True
            try: sender.setEnabled_(False)
            except Exception: pass

            def job():
                try:
                    AppHelper.callAfter(self._setStatusText_, "Generating missing details…")
                except Exception:
                    pass
                try:
                    n = enrich_missing_notes()
                except Exception:
                    logging.exception("Refresh Details failed")
                    n = 0
                def done():
                    self._reloadFromDB()
                    self._setStatusText_(f"Generated details for {n} note(s)" if n else "All notes already complete")
                    self._busy = False
                    try: sender.setEnabled_(True)
                    except Exception: pass
                try:
                    AppHelper.callAfter(done)
                except Exception:
                    done()
            threading.Thread(target=job, daemon=True).start()

        # "Summary": compute the cross-note actionable summary in its own window.
        def showSummaryClicked_(self, sender):
            self._setStatusText_("Generating actionable summary…")
            notes = list(self._notes)
            def job():
                try:
                    txt = build_full_notes_summary_with_gemini(notes)
                except Exception:
                    logging.exception("Summary build failed")
                    txt = "Could not compute summary right now."
                if not txt:
                    txt = "No notes to summarize yet."
                # Single-arg callAfter is the reliable form; present on main thread.
                try:
                    AppHelper.callAfter(self.presentSummaryText_, txt)
                except Exception:
                    self.presentSummaryText_(txt)
            threading.Thread(target=job, daemon=True).start()

        def presentSummaryText_(self, txt):
            try:
                self._setStatusText_("")
            except Exception:
                pass
            show_text_window("Actionable Summary", str(txt or ""), 760, 540)

        # ---------- Context-menu actions (edit / delete) ----------
        @objc.python_method
        def _clickedNote(self):
            # The row under the cursor on right-click; fall back to the selection.
            try:
                row = int(self._table.clickedRow())
            except Exception:
                row = -1
            if not (0 <= row < len(self._filtered)):
                row = self._selected_index if 0 <= self._selected_index < len(self._filtered) else -1
            if row < 0:
                return None
            self._selectAndReveal_(row)
            return self._filtered[row]

        def editNoteClicked_(self, sender):
            note = self._clickedNote()
            if not note:
                return
            try:
                editor = NoteEditorWindowController.alloc().init()
                editor.configure(note, self._onNoteEdited_)
                editor.present()
                GLOBAL_WINDOWS.append(editor)  # keep a reference so it isn't GC'd
            except Exception:
                logging.exception("Opening note editor failed")

        @objc.python_method
        def _onNoteEdited_(self, nid, fields):
            self._setStatusText_("Saving changes…")
            def job():
                try:
                    update_note_fields(nid, fields)
                except Exception:
                    logging.exception("Saving note edits failed")
            threading.Thread(target=job, daemon=True).start()

        def deleteNoteClicked_(self, sender):
            note = self._clickedNote()
            if not note:
                logging.info("Delete: no note under cursor")
                return
            nid = note.get("id")
            logging.info("Delete requested for note %s", nid)
            # Bring the (accessory) app forward so the confirmation alert is
            # actually visible and focused — without this it appears behind
            # everything and the click seems to do nothing.
            orig_policy = None
            try:
                orig_policy = _set_policy_regular_temporarily()
                NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
            except Exception:
                pass
            proceed = True
            try:
                if NSAlert is not None:
                    alert = NSAlert.alloc().init()
                    alert.setMessageText_("Delete this note?")
                    alert.setInformativeText_(friendly_snippet(note.get("summary") or note.get("content", ""), 140))
                    alert.addButtonWithTitle_("Delete")
                    alert.addButtonWithTitle_("Cancel")
                    if self._window is not None:
                        try: alert.window().setLevel_(5)  # float above
                        except Exception: pass
                    proceed = (int(alert.runModal()) == 1000)  # NSAlertFirstButtonReturn
            except Exception:
                logging.exception("Delete confirmation failed")
            finally:
                try:
                    if orig_policy is not None:
                        _restore_policy(orig_policy)
                except Exception:
                    pass
            if not proceed:
                logging.info("Delete cancelled for note %s", nid)
                return
            # Deletion is fast (no network) — do it now and refresh immediately.
            try:
                delete_note(nid)
            except Exception:
                logging.exception("Deleting note failed")
            self._setStatusText_("Note deleted")
            self._reloadFromDB()

        # ---------- View switching (list / bubbles) + settings ----------
        @objc.python_method
        def _installViewMenu_(self):
            """Add a 'View' menu to the app's menu bar with List / Bubbles
            (⌘1 / ⌘2). The window is the responder context, so the menu acts on
            this controller while the All Notes window is active."""
            if NSMenu is None or NSApplication is None:
                return
            try:
                app = NSApplication.sharedApplication()
                mainMenu = app.mainMenu()
                if mainMenu is None:
                    mainMenu = NSMenu.alloc().init()
                    app.setMainMenu_(mainMenu)
                # Avoid duplicates if the window is reopened.
                existing = mainMenu.itemWithTitle_("View")
                if existing is not None:
                    mainMenu.removeItem_(existing)
                viewItem = NSMenuItem.alloc().init()
                viewItem.setTitle_("View")
                viewMenu = NSMenu.alloc().initWithTitle_("View")
                try: viewMenu.setAutoenablesItems_(False)
                except Exception: pass

                self._miList = NSMenuItem.alloc().init()
                self._miList.setTitle_("List")
                self._miList.setAction_("viewListClicked:")
                self._miList.setTarget_(self)
                self._miList.setKeyEquivalent_("1")

                self._miBubbles = NSMenuItem.alloc().init()
                self._miBubbles.setTitle_("Bubbles")
                self._miBubbles.setAction_("viewBubblesClicked:")
                self._miBubbles.setTarget_(self)
                self._miBubbles.setKeyEquivalent_("2")

                viewMenu.addItem_(self._miList)
                viewMenu.addItem_(self._miBubbles)
                viewItem.setSubmenu_(viewMenu)
                mainMenu.addItem_(viewItem)
            except Exception:
                logging.exception("Installing View menu failed")

        @objc.python_method
        def _applyViewMode_(self, mode):
            is_bubbles = (mode == "bubbles")
            try:
                if self._listSplit is not None:
                    self._listSplit.setHidden_(is_bubbles)
                if self._bubble is not None:
                    self._bubble.setHidden_(not is_bubbles)
                    if is_bubbles:
                        # (Re)configure with the current notes and focus the canvas.
                        self._bubble.configure(self._notes, self._openNoteByID_)
                        try: self._window.makeFirstResponder_(self._bubble)
                        except Exception: pass
                # Reflect the active view as a checkmark in the View menu.
                try:
                    if getattr(self, "_miList", None) is not None:
                        self._miList.setState_(0 if is_bubbles else 1)
                    if getattr(self, "_miBubbles", None) is not None:
                        self._miBubbles.setState_(1 if is_bubbles else 0)
                except Exception:
                    pass
            except Exception:
                logging.exception("Applying view mode failed")

        def viewListClicked_(self, sender):
            self._applyViewMode_("list")

        def viewBubblesClicked_(self, sender):
            self._applyViewMode_("bubbles")

        @objc.python_method
        def _openNoteByID_(self, nid):
            # From a bubble: jump to the list view with the note selected so its
            # full details + connection map show.
            self._applyViewMode_("list")
            self._selectByID_(nid)

        def settingsClicked_(self, sender):
            try:
                self._settingsWC = SettingsWindowController.alloc().init()
                self._settingsWC.configure(self._onSettingsSaved_)
                self._settingsWC.present()
                GLOBAL_WINDOWS.append(self._settingsWC)
            except Exception:
                logging.exception("Opening settings failed")

        @objc.python_method
        def _onSettingsSaved_(self):
            # Honor a changed default view immediately and refresh.
            self._setStatusText_("Settings saved")
            self._reloadFromDB()

# small formatting helpers
def friendly_snippet(text: str, length: int = 120) -> str:
    """Create a truncated snippet that respects word boundaries."""
    if not text:
        return ""
    s = text.replace("\n", " ").strip()
    if len(s) <= length:
        return s
    # Find the last space before the length limit to avoid cutting words
    truncated = s[:length]
    last_space = truncated.rfind(" ")
    if last_space > length // 2:  # Only use word boundary if it's reasonable
        truncated = truncated[:last_space]
    return truncated.rstrip() + "..."

def readable_related_list(related: List[Dict[str, Any]], max_items=3) -> List[str]:
    out = []
    for r in (related or [])[:max_items]:
        rid = r.get("id")
        score = r.get("score", 0.0)
        reason = r.get("reason")
        rn = fetch_note_by_id_safe(rid)
        if rn:
            summ = rn.get("summary", "") or friendly_snippet(rn.get("content",""), 60)
            cat = rn.get("category","")
            why = f" — {reason}" if reason else ""
            out.append(f"{summ} ({cat}) [{score:.2f}]{why}")
        else:
            out.append(f"<missing> [{score:.2f}]")
    return out

def format_note_brief(note: dict) -> str:
    summary = note.get("summary") or "No summary"
    snippet = friendly_snippet(note.get("content",""), 120)
    rel_display = "; ".join(readable_related_list(note.get("related", []), max_items=3)) or "None"
    tags = ", ".join(note.get("tags", [])[:3]) if note.get("tags") else ""
    ts = note.get("timestamp", "")[:19].replace("T", " ")
    return f"[{ts}] {note.get('category')} | {summary}\n{snippet}\nTags: {tags}\nRelated: {rel_display}\n"


# ---------------- One-line summary helper ----------------
def build_one_line_summary(notes: List[dict]) -> str:
    if not notes:
        return "No notes."
    try:
        recent = sorted(notes, key=lambda n: n.get("timestamp",""), reverse=True)[:5]
        categories = {}
        for n in notes:
            c = n.get("category") or "Note"
            categories[c] = categories.get(c, 0) + 1
        top_cat = sorted(categories.items(), key=lambda kv: kv[1], reverse=True)[0][0]
        key_phrases = []
        for n in recent:
            if n.get("summary"):
                # Use full summary since it's already a concise one-sentence description
                key_phrases.append(n["summary"].strip())
            else:
                key_phrases.append(friendly_snippet(n.get("content",""), 80))
        key = "; ".join([k for k in key_phrases if k][:3])
        return f"{len(notes)} notes — focus on {top_cat}. Highlights: {key}"
    except Exception:
        return f"{len(notes)} notes captured."

# ---------------- Gemini-driven cross-note highlights ----------------
def _recurring_threads(notes: List[dict], top_n: int = 6) -> List[str]:
    """Find entities/tags that recur across multiple notes (the connective tissue).

    Only items appearing in 2+ notes count — these are the threads that tie the
    notebook together, which is what the highlights should be built around.
    """
    from collections import Counter
    counter: Counter = Counter()
    for n in notes:
        seen = set()
        for item in list(n.get("entities", []) or []) + list(n.get("tags", []) or []):
            key = str(item).strip().lower()
            if key and key not in seen:
                seen.add(key)
                counter[key] += 1
    return [item for item, cnt in counter.most_common(top_n) if cnt >= 2]


def build_highlights_with_gemini(notes: List[dict], max_notes: int = 80) -> str:
    """Produce highlights that reflect understanding *across* notes.

    Unlike ``build_one_line_summary`` (which just echoes the latest summaries),
    this asks the model to synthesize recurring themes, the projects/people that
    connect notes, and the single most important thing to act on. Recurring
    entities/tags are passed as explicit hints so the model anchors on genuine
    cross-note threads. Falls back to the local one-liner on any failure.
    """
    if not notes:
        return "No notes."
    if not gemini_client:
        return build_one_line_summary(notes)

    threads = _recurring_threads(notes)
    recent = sorted(notes, key=lambda n: n.get("timestamp", ""), reverse=True)
    note_lines = []
    for n in recent[:max_notes]:
        cat = n.get("category", "")
        summ = (n.get("summary") or friendly_snippet(n.get("content", ""), 120)).strip()
        note_lines.append(f"- ({cat}) {summ}")
    joined = "\n".join(note_lines)
    threads_hint = ", ".join(threads) if threads else "none detected"

    prompt = f"""You are reviewing someone's full notebook of {len(notes)} short notes.
Recurring threads (people/topics that appear across multiple notes): {threads_hint}

Notes (newest first):
{joined}

Write 2–4 TOP HIGHLIGHTS, each on its OWN LINE, that capture what's going on
ACROSS the notes — recurring projects/people, themes that connect multiple notes,
and the single most important thing to act on. Reference specific names/projects.
Each highlight ≤ 14 words. Do NOT just restate the most recent notes, and avoid
generic phrases like "various topics". No bullets, no numbering, no preamble —
just one highlight per line."""

    try:
        response = gemini_client.models.generate_content(
            model=CLASSIFY_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=512),
        )
        text = (response.text or "").strip()
        if not text and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            text = (response.candidates[0].content.parts[0].text or "").strip()
        # Split into clean highlight lines and render as bullets.
        lines = []
        for ln in (text or "").splitlines():
            ln = ln.strip().lstrip("-•*0123456789. ").strip()
            if ln:
                lines.append(ln)
        if not lines:
            raise Exception("Empty highlights response")
        return "\n".join(f"•  {ln}" for ln in lines[:4])
    except Exception:
        logging.exception("Gemini highlights generation failed; using local one-liner")
        return build_one_line_summary(notes)


# ---------------- Gemini-driven full summary ----------------
def build_full_notes_summary_with_gemini(notes: List[dict]) -> str:
    if not notes:
        return "No notes available."
    note_texts = []
    for n in notes:
        ts = n.get("timestamp","")
        cat = n.get("category","")
        tags = ",".join(n.get("tags",[])) if n.get("tags") else ""
        summ = n.get("summary","")
        cont = n.get("content","")
        note_texts.append(f"- [{ts}] ({cat}) tags:{tags} summary:{summ} content:{friendly_snippet(cont, 250)}")
    joined = "\n".join(note_texts[:200])  # protect length
    prompt = f"""
You are an assistant that reads a user's short note exports and returns an actionable, concise summary.
Input notes:
{joined}

Return a plain-text structured summary with these sections:
1) Top actionable To-Dos (each item short and actionable; include estimated priority: low/medium/high)
2) Project or multi-step ideas (list briefly, with next 1-2 steps)
3) Quick ideas & inspirations to remember
4) Follow-ups / people / links to check
5) Overall summary (1-2 sentences that specifically mention key people, projects, or themes from the notes — avoid generic phrases like "various topics" or "multiple items")

Make bullet lists and keep items short (no more than 12 words each). If nothing fits a section write "None".
"""
    try:
        if not gemini_client:
            raise Exception("Gemini client not configured")
        response = gemini_client.models.generate_content(
            model=CLASSIFY_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=2048,
            )
        )
        # Handle case where response.text is None (e.g., blocked content, empty response, or max tokens hit)
        if response.text is None:
            # Try to extract partial response from candidates
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                partial_text = response.candidates[0].content.parts[0].text
                if partial_text:
                    logging.info("Extracted partial summary from candidates")
                    return partial_text.strip()
            logging.warning("Gemini summary returned None response.text, using fallback")
            raise Exception("Empty response from Gemini")
        return response.text.strip()
    except Exception:
        logging.exception("Gemini summary generation failed")
        # fallback simple aggregator
        todos = []
        ideas = []
        for n in notes:
            if n.get("category") == "To-Do":
                todos.append(n.get("summary") or friendly_snippet(n.get("content",""),60))
            elif n.get("category") == "Idea":
                ideas.append(n.get("summary") or friendly_snippet(n.get("content",""),60))
        parts = []
        parts.append("Top actionable To-Dos:\n" + ("\n".join(f"- {t}" for t in todos) if todos else "None"))
        parts.append("Project or multi-step ideas:\n" + ("\n".join(f"- {i}" for i in ideas) if ideas else "None"))
        parts.append("Quick ideas & inspirations:\nNone")
        parts.append("Follow-ups / people / links:\nNone")
        parts.append("Overall summary:\nNotes summarized locally.")
        return "\n\n".join(parts)

# ---------------- UI (rumps menu) ----------------
APP_NAME = "AmbientScratchpad"

class AppGUI(rumps.App):
    def __init__(self):
        super().__init__("✍︎", quit_button=None)
        self.menu = [

            rumps.MenuItem("Quick Capture"),
            rumps.MenuItem("Open DB Folder"),
            rumps.MenuItem("Show Latest Note"),
            rumps.MenuItem("Show All Notes"),
            rumps.MenuItem("Process Raw Now"),
            rumps.MenuItem("Enrich Missing"),
            rumps.MenuItem("Recompute Links"),
            rumps.MenuItem("Pipeline Status"),
            rumps.MenuItem("View Log"),
            rumps.MenuItem("Quit")
    
        ]
        self.processor = PipelineProcessor(INPUT_STREAM)
        self.processor.start()
    
        # Make app a foreground app so text fields can become first responder
        try:
            if PYOBJC_AVAILABLE and NSApplication is not None:
                NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyRegular)
        except Exception:
            pass
# Install global Option+Space hotkey for Quick Capture
        self._install_global_hotkey()

    def _install_global_hotkey(self):
        """
        Global hotkey: Option + Space opens Quick Capture from anywhere.
        """
        if not PYOBJC_AVAILABLE:
            return
        try:
            # Modern modifier flag; fallback for older macOS naming
            from AppKit import NSEventModifierFlagOption as _OPT_FLAG
        except Exception:
            try:
                from AppKit import NSAlternateKeyMask as _OPT_FLAG  # legacy
            except Exception:
                _OPT_FLAG = 0x00080000  # best-effort fallback

        def _handle_global(event):
            try:
                flags = int(event.modifierFlags())
                try:
                    from AppKit import NSEventModifierFlagOption as _OPT_FLAG
                except Exception:
                    try:
                        from AppKit import NSAlternateKeyMask as _OPT_FLAG
                    except Exception:
                        _OPT_FLAG = 0x00080000
                is_opt = bool(flags & _OPT_FLAG)
                keycode = int(getattr(event, "keyCode", lambda: 0)())
                chars = str(getattr(event, "charactersIgnoringModifiers", lambda: "")())
                if is_opt and (keycode == 49 or chars == " "):  # 49 == spacebar
                    # ALWAYS dispatch to the main thread for UI
                    try:
                        AppHelper.callAfter(self.quick_capture, None)
                    except Exception:
                        pass
            except Exception:
                pass
            return event

        # Register monitor (can't stop event propagation, but good enough to trigger UI)
        try:
            self._hotkey_global = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(NSEventMaskKeyDown, _handle_global)
        except Exception:
            self._hotkey_global = None


    def _refresh_all_notes_ui(self):
        """Reload notes from DB and push into the Show All Notes window, if it's open."""
        if not PYOBJC_AVAILABLE:
            return
        wc = getattr(self, "_allNotesWC", None)
        if not wc:
            return
        # Pull latest notes
        with db_lock:
            c = db_conn.cursor()
            try:
                c.execute("SELECT * FROM notes ORDER BY timestamp DESC")
                rows_all = c.fetchall()
                cols = [r[1] for r in c.execute("PRAGMA table_info(notes)").fetchall()]
            except Exception:
                rows_all, cols = [], []
        notes = []
        for row in rows_all:
            data = {}
            for i, col in enumerate(cols):
                data[col] = row[i]
            try:
                data['tags'] = json.loads(data.get('tags') or "[]")
                data['entities'] = json.loads(data.get('entities') or "[]")
                data['related'] = json.loads(data.get('related') or "[]")
                data['actions'] = json.loads(data.get('actions') or "[]")
            except Exception:
                pass
            notes.append(data)
        # Dispatch loadNotes_ on main thread to be safe
        try:
            AppHelper.callAfter(wc.loadNotes_, notes)
        except Exception:
            wc.loadNotes_(notes)
        # Ensure status line (counts) refreshes too
        try:
            AppHelper.callAfter(wc.updateStatus)
        except Exception:
            wc.updateStatus()

        # Update the top highlight (one-line summary) immediately
        try:
            one_line = build_one_line_summary(notes)
        except Exception:
            one_line = f"{len(notes)} notes."
        try:
            AppHelper.callAfter(wc.setOneLineSummary_, one_line)
        except Exception:
            wc.setOneLineSummary_(one_line)

        # Refresh the cross-note highlights asynchronously so the UI stays
        # responsive (the instant local one-liner is already shown above).
        # The heavier actionable summary is now computed on demand via the
        # "Summary" button, keeping the main window uncluttered.
        def _recompute_highlights():
            try:
                highlights = build_highlights_with_gemini(notes)
            except Exception:
                highlights = None
            if highlights:
                try:
                    AppHelper.callAfter(wc.setOneLineSummary_, highlights)
                except Exception:
                    wc.setOneLineSummary_(highlights)
        threading.Thread(target=_recompute_highlights, daemon=True).start()



    @rumps.clicked("Quick Capture")
    def quick_capture(self, _):

        # Ensure we're on the main thread; if not, bounce to main
        try:
            import threading
            if not threading.current_thread() is threading.main_thread():
                try:
                    from PyObjCTools import AppHelper as _AH
                    _AH.callAfter(self.quick_capture, None)
                    return
                except Exception:
                    pass
        except Exception:
            pass
        logging.info("QuickCapture: invoked")
        text = ""
        if PYOBJC_AVAILABLE:
            try:
                controller = QuickCapturePanelController.alloc().init()
                try:
                    if PYOBJC_AVAILABLE and NSApplication is not None:
                        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
                except Exception:
                    pass
                text = controller.runModalAndGetText()
            except Exception:
                logging.exception("QuickCapture: panel failed; fallback to rumps.Window")

        if not text:
            try:
                win = rumps.Window(title="Quick Capture", message="Type your thought:", default_text="", ok="Save", cancel="Cancel")
                resp = win.run()
                if int(getattr(resp, "clicked", 0) or 0) == 1:
                    text = (getattr(resp, "text", "") or "").strip()
            except Exception:
                logging.exception("QuickCapture: rumps fallback failed")

        if not text:
            rumps.notification(APP_NAME, "Capture not saved", "Empty or cancelled")
            return

        obj = {"id": str(uuid.uuid4()), "timestamp": now_iso(), "content": text}
        try:
            append_to_input_stream(obj)
            logging.info("QuickCapture: appended %s", obj["id"])
        except Exception:
            logging.exception("QuickCapture: append failed")

        def _save_and_refresh():
            try:
                logging.info("QuickCapture: processing/store start")
                process_capture_object(obj, store=True)
                logging.info("QuickCapture: processed/stored; refreshing UI")
                self._refresh_all_notes_ui()
            except Exception:
                logging.exception("QuickCapture: process/store failed")

        t = threading.Thread(target=_save_and_refresh, daemon=True)
        t.start()
        rumps.notification(APP_NAME, "Saved", text[:200])

    @rumps.clicked("Process Raw Now")
    def process_raw_now(self, _):
        def job():
            raw = read_raw_input_stream(limit=2000)
            if not raw:
                rumps.notification(APP_NAME, "Process Raw Now", "No raw captures found.")
                return
            count = 0
            for obj in raw:
                try:
                    process_capture_object(obj, store=True)
                    count += 1
                except Exception:
                    logging.exception("Failed during Process Raw Now")
            rumps.notification(APP_NAME, "Process Raw Now", f"Processed {count} raw captures.")
            logging.info("Process Raw Now processed %d captures", count)
        threading.Thread(target=job, daemon=True).start()

    @rumps.clicked("Show Latest Note")
    def show_latest(self, _):
        with db_lock:
            c = db_conn.cursor()
            try:
                c.execute("SELECT id FROM notes ORDER BY timestamp DESC LIMIT 1")
                row = c.fetchone()
            except Exception:
                row = None
        if not row:
            rumps.notification(APP_NAME, "Latest Note", "No processed notes yet.")
            return
        nid = row[0]
        note = fetch_note_by_id_safe(nid)
        if not note:
            rumps.notification(APP_NAME, "Latest Note", "Note not found.")
            return
        lines = [
            f"Time: {note.get('timestamp')}",
            f"Category: {note.get('category')}",
            f"Tags: {', '.join(note.get('tags',[]))}",
            f"Entities: {', '.join(note.get('entities',[]))}",
            f"Summary: {note.get('summary') or 'No summary'}",
            "",
            "Content:",
            note.get("content",""),
            "",
            "Related:"
        ]
        for rel in note.get("related", []):
            rid = rel.get("id"); score = rel.get("score", 0.0)
            rn = fetch_note_by_id_safe(rid)
            if rn:
                snippet = friendly_snippet(rn.get("summary") or rn.get("content",""), 120)
                lines.append(f"- {snippet} ({rn.get('category')}) [{score:.2f}]")
            else:
                lines.append(f"- <missing> [{score:.2f}]")
        body = "\n".join(lines)
        show_text_window("Latest Note", body, width=700, height=420)

    @rumps.clicked("Show All Notes")
    def show_all(self, _):
        # Load notes for UI
        with db_lock:
            c = db_conn.cursor()
            try:
                c.execute("SELECT * FROM notes ORDER BY timestamp DESC")
                rows_all = c.fetchall()
                cols = [r[1] for r in c.execute("PRAGMA table_info(notes)").fetchall()]
            except Exception:
                rows_all = []
                cols = []
        notes = []
        for row in rows_all:
            data = {}
            for i, col in enumerate(cols):
                data[col] = row[i]
            for k in ("tags","entities","related"):
                if k in data and data[k]:
                    try:
                        data[k] = json.loads(data[k])
                    except Exception:
                        data[k] = []
                else:
                    data[k] = []
            if "summary" not in data or data["summary"] is None:
                data["summary"] = "No summary"
            notes.append(data)

        if PYOBJC_AVAILABLE:
            try:
                self._allNotesWC = ShowAllNotesWindowController.alloc().init()
                self._allNotesWC.show_(None)
                self._allNotesWC.loadNotes_(notes)
            except Exception as e:
                # fallback to text window if something goes wrong
                combined = "\n\n".join([format_note_brief(n) for n in notes[:200]])
                show_text_window("All Notes", combined, width=920, height=700)
        else:
            combined = "\n\n".join([format_note_brief(n) for n in notes[:200]])
            show_text_window("All Notes", combined, width=920, height=700)

        # Compute the cross-note highlights in the background (the actionable
        # summary is now on demand via the window's "Summary" button).
        def highlights_thread():
            try:
                one_line = build_highlights_with_gemini(notes)
            except Exception:
                logging.exception("Highlights computation failed")
                one_line = build_one_line_summary(notes)
            if PYOBJC_AVAILABLE and getattr(self, "_allNotesWC", None):
                self._allNotesWC.setOneLineSummary_(one_line or "(no highlights)")

        threading.Thread(target=highlights_thread, daemon=True).start()

    @rumps.clicked("Enrich Missing")
    def enrich_missing_menu(self, _):
        def job():
            rumps.notification(APP_NAME, "Enrich Missing", "Starting enrichment...")
            updated = enrich_missing_notes(batch_sleep=0.12)
            logging.info("Enrich Missing updated %d notes", updated)
            rumps.notification(APP_NAME, "Enrich Missing", f"Completed. Updated {updated} notes.")
        threading.Thread(target=job, daemon=True).start()

    @rumps.clicked("Recompute Links")
    def recompute_links_menu(self, _):
        def job():
            rumps.notification(APP_NAME, "Recompute Links", "Starting recompute...")
            count = recompute_links()
            logging.info("Recompute Links updated %d notes", count)
            rumps.notification(APP_NAME, "Recompute Links", f"Completed. Updated {count} notes.")
        threading.Thread(target=job, daemon=True).start()

    @rumps.clicked("Pipeline Status")
    def pipeline_status(self, _):
        using = getattr(self.processor, "using_pathway_stream", False)
        alive = self.processor.is_alive()
        with db_lock:
            c = db_conn.cursor()
            try:
                c.execute("SELECT COUNT(*) FROM notes")
                count = c.fetchone()[0]
            except Exception:
                count = 0
        raw_count = 0
        if os.path.exists(INPUT_STREAM):
            try:
                raw_count = sum(1 for _ in open(INPUT_STREAM, "r", encoding="utf-8"))
            except Exception:
                raw_count = 0
        rumps.notification(APP_NAME, "Pipeline Status", f"Alive: {alive}\nUsing Pathway: {using}\nProcessed: {count}\nRaw captures: {raw_count}")

    @rumps.clicked("View Log")
    def view_log(self, _):
        if os.path.exists(LOG_FILE):
            if sys.platform == "darwin":
                os.system(f'open "{os.path.abspath(LOG_FILE)}"')
            else:
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    content = f.read()[-4000:]
                rumps.alert("Log (tail)", content[:2000])
        else:
            rumps.notification(APP_NAME, "View Log", "No log file yet.")

    @rumps.clicked("Open DB Folder")
    def open_folder(self, _):
        folder = os.path.abspath(".")
        if sys.platform == "darwin":
            os.system(f'open "{folder}"')
        else:
            rumps.notification(APP_NAME, "Open DB Folder", folder)

    @rumps.clicked("Quit")
    def quit_app(self, _):
        try:
            self.processor.stop()
        except Exception:
            pass
        time.sleep(0.25)
        rumps.quit_application()

# ---------------- Run ----------------
if __name__ == "__main__":
    logging.info("Ambient Scratchpad started. Input: %s DB: %s", os.path.abspath(INPUT_STREAM), os.path.abspath(DB_FILE))
    logging.info("PyObjC available: %s  Pathway imported: %s  USE_PATHWAY_STREAM: %s", PYOBJC_AVAILABLE, PATHWAY_IMPORTED, USE_PATHWAY_STREAM)
    AppGUI().run()
