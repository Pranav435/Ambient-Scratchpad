# main.py
"""
Ambient Scratchpad — rumps menubar + PyObjC scrollable windows for large text.

Requirements:
- python -m pip install rumps openai python-dotenv pyobjc
- Put OPENAI_API_KEY=sk-... in .env

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

# new openai client (v1)
try:
    from openai import OpenAI
except Exception:
    print("Install openai>=1.0: pip install --upgrade openai")
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

# ==== PATHWAY: optional import (we make it meaningful but optional)
PATHWAY_IMPORTED = False
try:
    warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated as an API.*", category=UserWarning)
    import pathway as pw
    PATHWAY_IMPORTED = True
except Exception:
    PATHWAY_IMPORTED = False

# optional scrapers / numpy
try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except Exception:
    SCRAPING_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False
    import math

# ---------------- Config ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CLASSIFY_MODEL = os.getenv("OPENAI_CLASSIFY_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
USE_PATHWAY_STREAM = os.getenv("USE_PATHWAY_STREAM", "0") == "1"

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

# OpenAI client
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        openai_client = OpenAI()
else:
    openai_client = OpenAI()  # will raise on calls if not configured

# ---------------- DB init + migration ----------------
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
    return datetime.utcnow().isoformat() + "Z"

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

# ---------------- OpenAI (v1) helpers ----------------
def classify_and_tag_with_openai(text: str) -> Dict[str, Any]:
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
"""
    try:
        resp = openai_client.chat.completions.create(
            model=CLASSIFY_MODEL,
            messages=[
                {"role":"system","content":"You extract metadata."},
                {"role":"user","content":prompt}
            ],
            temperature=0.0,
            max_tokens=300,
        )
        out = ""
        try:
            out = resp.choices[0].message["content"]
        except Exception:
            out = getattr(resp.choices[0].message, "content", "") or str(resp)
        out = (out or "").strip()
        try:
            data = json.loads(out)
        except Exception:
            s = out.find("{"); e = out.rfind("}")
            if s != -1 and e != -1:
                data = json.loads(out[s:e+1])
            else:
                data = {"category":"Other","tags":[],"entities":[],"summary": text[:200]}
        data.setdefault("category","Other")
        data.setdefault("tags",[])
        data.setdefault("entities",[])
        data.setdefault("summary", data.get("summary") or (text[:200] or "No summary"))
        if data["summary"] is None:
            data["summary"] = "No summary"
        logging.debug("AI classify: %s", data)
        return data
    except Exception:
        logging.exception("OpenAI classify error")
        return {"category":"Other","tags":[],"entities":[],"summary":"No summary"}

def get_embedding_openai(text: str) -> List[float]:
    if not text:
        return []
    try:
        resp = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
        vec = resp.data[0].embedding
        return vec
    except Exception:
        logging.exception("OpenAI embedding error")
        h = abs(hash(text)) % (10**8)
        return [((h >> (i*8)) & 255)/255.0 for i in range(128)]

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

# ---------------- Similarity & storage ----------------
def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    if NUMPY_AVAILABLE:
        a_np = np.array(a, dtype=float); b_np = np.array(b, dtype=float)
        la = np.linalg.norm(a_np); lb = np.linalg.norm(b_np)
        if la == 0 or lb == 0:
            return 0.0
        return float(np.dot(a_np, b_np) / (la*lb))
    else:
        lena = math.sqrt(sum(x*x for x in a)) or 1.0
        lenb = math.sqrt(sum(x*x for x in b)) or 1.0
        s = sum(ai*bi for ai,bi in zip(a,b))
        return s / (lena * lenb)

def compute_related_with_scores(embedding: List[float], top_k=5) -> List[Dict[str, Any]]:
    rows = read_all_embeddings_from_db()
    if not rows:
        return []
    sims = []
    for r in rows:
        try:
            sc = cosine_similarity(embedding, r["emb"])
            if sc >= RELATED_MIN_SCORE:
                sims.append({"id": r["id"], "score": float(sc)})
        except Exception:
            continue
    sims.sort(key=lambda x: x["score"], reverse=True)
    return sims[: min(top_k, RELATED_MAX_PER_NOTE)]


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
            c.execute("PRAGMA table_info(notes)")
            cols = [r[1] for r in c.fetchall()]
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
    meta = classify_and_tag_with_openai(effective)
    emb = get_embedding_openai(effective)
    candidates = compute_related_with_scores(emb, top_k=10)
    filtered = [r for r in candidates if r["id"] != nid]
    top_related = filtered[:5]
    enriched = {
        "id": nid,
        "timestamp": timestamp,
        "content": content,
        "scraped_content": scraped,
        "category": meta.get("category","Other"),
        "tags": meta.get("tags", []),
        "entities": meta.get("entities", []),
        "summary": meta.get("summary","No summary"),
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
        if not note.get("embedding"):
            need = True
        if not note.get("summary") or note.get("summary","").strip()=="":
            need = True
        if need:
            process_capture_object({"id": nid, "timestamp": note.get("timestamp", now_iso()), "content": note.get("content","")}, store=True)
            updated += 1
            time.sleep(batch_sleep)
    logging.info("Enrich missing done: %d updated", updated)
    return updated

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
        sims = []
        for other_id, other_emb in emb_map.items():
            if other_id == nid:
                continue
            try:
                score = cosine_similarity(emb, other_emb)
                sims.append({"id": other_id, "score": score})
            except Exception:
                continue
        sims = [s for s in sims if s.get("score", 0.0) >= RELATED_MIN_SCORE]
        sims.sort(key=lambda x: x["score"], reverse=True)
        top_related = sims[:RELATED_MAX_PER_NOTE]
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


# ---------- Native "Show All Notes" UI (macOS) ----------
if PYOBJC_AVAILABLE:
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

        def reloadTable(self):
            if self._table is not None:
                self._table.reloadData()
                self.updateStatus()

        def updateStatus(self):
            total = len(self._notes)
            shown = len(self._filtered)
            if self._status is not None:
                self._status.setStringValue_(f"Showing {shown} of {total} notes")

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

        def searchChanged_(self, sender):
            try:
                q = self._search.stringValue()
            except Exception:
                q = ""
            self.filterNotes_(q)

        # ---- NSTableView data source / delegate ----
        def numberOfRowsInTableView_(self, tableView):
            return len(self._filtered)

        def tableView_objectValueForTableColumn_row_(self, tableView, column, row):
            try:
                n = self._filtered[row]
            except Exception:
                return ""
            ident = column.identifier()
            if ident == "title":
                title = n.get("title") or (n.get("content","")[:60])
                return title or "Untitled"
            if ident == "time":
                return n.get("timestamp","")
            if ident == "tags":
                return str(len(n.get("tags",[]) or []))
            return ""

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

        def renderDetail_(self, note):
            # Build a richly structured, readable detail string
            lines = []
            title = note.get("title") or "Untitled"
            lines.append(f"{title}")
            lines.append("")
            lines.append(f"Time: {note.get('timestamp','')}    Category: {note.get('category','')}")
            tags = ", ".join(note.get("tags",[]) or [])
            lines.append(f"Tags: {tags if tags else '-'}")
            lines.append("")
            lines.append("Summary:")
            lines.append(note.get("summary") or "No summary")
            lines.append("")
            lines.append("Related:")
            rels = note.get("related",[]) or []
            if rels:
                for rel in rels[:10]:
                    rid = rel.get("id")
                    score = rel.get("score",0.0)
                    rn = fetch_note_by_id_safe(rid)
                    if rn:
                        snippet = friendly_snippet(rn.get("summary") or rn.get("content",""), 100)
                        lines.append(f"• {snippet}  [{score:.2f}]")
            else:
                lines.append("— None —")
            lines.append("")
            lines.append("Full Note:")
            lines.append(note.get("content",""))
            detail_text = "\n".join(lines)
            self._detail.setString_(detail_text)

        # ---- Window construction ----
        def show_(self, sender):
            rect = NSMakeRect(120.0, 120.0, 980.0, 700.0)
            style_mask = 15  # titled, closable, resizable, miniaturizable
            self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                rect, style_mask, NSBackingStoreBuffered, False
            )
            self._window.setTitle_("All Notes")
            self._window.setReleasedWhenClosed_(False)

            # top search field
            self._search = NSSearchField.alloc().initWithFrame_(NSMakeRect(0,0,200,22))
            self._search.setPlaceholderString_("Filter…")

            # status field
            self._status = NSTextField.alloc().initWithFrame_(NSMakeRect(0,0,200,22))
            self._status.setBezeled_(False)
            self._status.setDrawsBackground_(False)
            self._status.setEditable_(False)
            self._status.setSelectable_(False)

            # one-line summary field
            self._one_line = NSTextField.alloc().initWithFrame_(NSMakeRect(0,0,400,22))
            self._one_line.setBezeled_(False)
            self._one_line.setDrawsBackground_(False)
            self._one_line.setEditable_(False)
            self._one_line.setSelectable_(False)

            # left table
            self._table = NSTableView.alloc().initWithFrame_(NSMakeRect(0,0,300,600))
            col_titles = [("title","Title", 260), ("time","Time", 150), ("tags","Tags",60)]
            for ident, title, width in col_titles:
                col = NSTableColumn.alloc().initWithIdentifier_(ident)
                col.headerCell().setStringValue_(title)
                col.setWidth_(width)
                self._table.addTableColumn_(col)
            self._table.setAllowsMultipleSelection_(False)
            self._table.setDelegate_(self)
            self._table.setDataSource_(self)

            left_scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0,0,320,600))
            left_scroll.setDocumentView_(self._table)
            left_scroll.setHasVerticalScroller_(True)
            left_scroll.setAutohidesScrollers_(True)

            # right detail text view
            self._detail = NSTextView.alloc().initWithFrame_(NSMakeRect(0,0,640,600))
            self._detail.setEditable_(False)
            self._detail.setRichText_(False)
            try:
                self._detail.setDrawsBackground_(True)
                self._detail.setBackgroundColor_(NSColor.textBackgroundColor())
                self._detail.setTextColor_(NSColor.textColor())
            except Exception:
                pass
            self._detail.setUsesFontPanel_(False)
            self._detail.setAutomaticQuoteSubstitutionEnabled_(False)
            self._detail.setAutomaticDataDetectionEnabled_(True)
            self._detail.setAllowsUndo_(False)
            try:
                self._detail.setTextColor_(NSColor.labelColor())
            except Exception:
                pass

            right_scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0,0,640,600))
            right_scroll.setDocumentView_(self._detail)
            right_scroll.setHasVerticalScroller_(True)
            right_scroll.setAutohidesScrollers_(True)

            # summary (bottom) text view
            self._summary = NSTextView.alloc().initWithFrame_(NSMakeRect(0,0,940,140))
            self._summary.setEditable_(False)
            self._summary.setRichText_(False)
            try:
                self._summary.setDrawsBackground_(True)
                self._summary.setBackgroundColor_(NSColor.textBackgroundColor())
                self._summary.setTextColor_(NSColor.whiteColor())
            except Exception:
                pass
            try:
                self._summary.setTextColor_(NSColor.whiteColor())
            except Exception:
                pass
            sum_scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0,0,940,140))
            sum_scroll.setDocumentView_(self._summary)
            sum_scroll.setHasVerticalScroller_(True)
            sum_scroll.setAutohidesScrollers_(True)

            # layout with split view on right (detail + summary)
            right_split = NSSplitView.alloc().initWithFrame_(NSMakeRect(0,0,660,600))
            right_split.setDividerStyle_(1)
            right_split.setVertical_(False)  # horizontal divider: top = detail, bottom = summary
            right_split.setAutosaveName_("ShowAllNotesRightSplit")
            right_split.setDelegate_(self)
            right_split.addSubview_(right_scroll)
            right_split.addSubview_(sum_scroll)
            try:
                frame = right_split.frame()
                # Give the bottom summary more space by default
                right_split.setPosition_ofDividerAtIndex_(frame.size.height - 280.0, 0)
            except Exception:
                pass

            # split view: left list, right details
            split = NSSplitView.alloc().initWithFrame_(NSMakeRect(0,0,960,640))
            split.setDividerStyle_(1)
            split.setVertical_(True)
            split.setAutosaveName_("ShowAllNotesMainSplit")
            split.setDelegate_(self)
            split.addSubview_(left_scroll)
            split.addSubview_(right_split)
            split.setPosition_ofDividerAtIndex_(320.0, 0)

            # top bar container (search + status)
            top_stack = NSStackView.alloc().initWithFrame_(NSMakeRect(0,0,960,30))
            top_stack.setOrientation_(0)  # horizontal
            top_stack.setAlignment_(1)    # leading
            top_stack.setDistribution_(1) # fill
            top_stack.setSpacing_(8.0)
            try:
                # add a slight inset so labels aren't flush left
                top_stack.setEdgeInsets_((4.0, 12.0, 4.0, 12.0))
            except Exception:
                pass
            top_stack.addView_inGravity_(self._search, 1)
            try:
                pass
            except Exception:
                pass
            top_stack.addView_inGravity_(self._one_line, 1)
            top_stack.addView_inGravity_(self._status, 1)

            # root vertical stack
            root = NSStackView.alloc().initWithFrame_(NSMakeRect(0,0,960,680))
            root.setOrientation_(1)
            root.setAlignment_(1)
            root.setDistribution_(1)
            root.setSpacing_(8.0)
            root.addView_inGravity_(top_stack, 1)
            root.addView_inGravity_(split, 1)

            content = self._window.contentView()
            content.addSubview_(root)

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
            try:
                tv.textStorage().setAttributedString_(NSAttributedString.alloc().initWithString_(string))
            except Exception:
                tv.setString_(string)

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
                logging.info("UI: updating one-line summary to: %r", s)
            except Exception:
                pass
            self._setFieldString(self._one_line, (s or "(no TL;DR)"))

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

# small formatting helpers
def friendly_snippet(text: str, length: int = 120) -> str:
    if not text:
        return ""
    s = text.replace("\n", " ")
    return (s[:length] + ("..." if len(s) > length else ""))

def readable_related_list(related: List[Dict[str, Any]], max_items=3) -> List[str]:
    out = []
    for r in (related or [])[:max_items]:
        rid = r.get("id")
        score = r.get("score", 0.0)
        rn = fetch_note_by_id_safe(rid)
        if rn:
            summ = rn.get("summary", "") or friendly_snippet(rn.get("content",""), 60)
            cat = rn.get("category","")
            out.append(f"{summ} ({cat}) [{score:.2f}]")
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
                key_phrases.append(n["summary"][:50])
            else:
                key_phrases.append((n.get("content","")[:50]).strip())
        key = "; ".join([k for k in key_phrases if k][:3])
        return f"{len(notes)} notes — focus on {top_cat}. Highlights: {key}"
    except Exception:
        return f"{len(notes)} notes captured."

# ---------------- OpenAI-driven full summary ----------------
def build_full_notes_summary_with_openai(notes: List[dict]) -> str:
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
5) One-sentence overall summary of the note collection

Make bullet lists and keep items short (no more than 12 words each). If nothing fits a section write "None".
"""
    try:
        resp = openai_client.chat.completions.create(
            model=CLASSIFY_MODEL,
            messages=[
                {"role":"system","content":"You synthesize notes into actionable lists."},
                {"role":"user","content":prompt}
            ],
            temperature=0.0,
            max_tokens=600,
        )
        out = ""
        try:
            out = resp.choices[0].message["content"]
        except Exception:
            out = getattr(resp.choices[0].message, "content", "") or str(resp)
        return (out or "").strip()
    except Exception:
        logging.exception("OpenAI summary generation failed")
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
                is_opt = bool(flags & _OPT_FLAG)
                keycode = int(getattr(event, "keyCode", lambda: 0)())
                chars = str(getattr(event, "charactersIgnoringModifiers", lambda: "")())
                if is_opt and (keycode == 49 or chars == " "):  # 49 == spacebar
                    try:
                        AppHelper.callAfter(self.quick_capture, None)
                    except Exception:
                        self.quick_capture(None)
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

        # Optionally refresh the actionable summary asynchronously (heavier)
        def _recompute_actionable():
            try:
                summary_text = build_full_notes_summary_with_openai(notes)
            except Exception:
                summary_text = None
            if summary_text:
                try:
                    AppHelper.callAfter(wc.setActionableSummary_, summary_text)
                except Exception:
                    wc.setActionableSummary_(summary_text)
        threading.Thread(target=_recompute_actionable, daemon=True).start()



    @rumps.clicked("Quick Capture")
    def quick_capture(self, _):
        win = rumps.Window(title="Quick Capture", message="Type your thought:", default_text="", ok="Save", cancel="Cancel")
        resp = win.run()
        if resp.clicked:
            text = (resp.text or "").strip()
            if not text:
                rumps.notification(APP_NAME, "Capture not saved", "Empty input")
                return
            obj = {"id": str(uuid.uuid4()), "timestamp": now_iso(), "content": text}
            append_to_input_stream(obj)

            # --- Immediate enrich patch ---
            threading.Thread(
                target=lambda: (process_capture_object(obj, store=True), self._refresh_all_notes_ui()),
                daemon=True
            ).start()
            # --- end patch ---

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

        # compute actionable summary in background and push into the bottom pane
        def summary_thread():
            try:
                logging.info("Computing actionable summary for %d notes", len(notes))
                summary_text = build_full_notes_summary_with_openai(notes)
            except Exception:
                logging.exception("Summary computation failed")
                summary_text = "Could not compute summary right now."
            try:
                one_line = build_one_line_summary(notes)
            except Exception:
                logging.exception("One-line summary computation failed")
                one_line = f"{len(notes)} notes."
            logging.info("Pushing summaries to UI: one_line=%r len(summary)=%d", one_line, len(summary_text or ""))
            if PYOBJC_AVAILABLE and getattr(self, "_allNotesWC", None):
                self._allNotesWC.setActionableSummary_(summary_text or "(no actionable summary)")
                self._allNotesWC.setOneLineSummary_(one_line or "(no TL;DR)")
            else:
                # fallback: show a second window
                show_text_window("Actionable Summary", summary_text, width=800, height=400)

        threading.Thread(target=summary_thread, daemon=True).start()

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
