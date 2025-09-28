# main.py
"""
Ambient Scratchpad — rumps menubar + PyObjC scrollable windows for large text.

Requirements:
- python -m pip install rumps openai python-dotenv pyobjc
- Put OPENAI_API_KEY=sk-... in .env

Behavior:
- Uses rumps for the menubar and quick capture.
- Uses AppKit (PyObjC) for a non-blocking, scrollable, resizable window to show All Notes + Actionable Summary.
- Falls back to rumps.alert (truncated) if PyObjC isn't available.
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
        NSRunningApplication, NSApplicationActivateIgnoringOtherApps, NSBackingStoreBuffered
    )
    from Foundation import NSString
    from PyObjCTools import AppHelper
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
            sims.append({"id": r["id"], "score": sc})
        except Exception:
            continue
    sims.sort(key=lambda x: x["score"], reverse=True)
    return sims[:top_k]

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
            logging.info("Stored note %s category=%s related=%d", note["id"], note.get("category"), len(note.get("related", [])))
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
        if k in data and data[k]:
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

            # Prefer jsonlines reader; fall back to fs.read
            RawSchema = self._build_pathway_schema()

            # Always hand Pathway the absolute path used by our writer
            path = INPUT_STREAM  # absolute because of Patch 1

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


            # Normalize columns and avoid reserved 'id'
            def _ts_key(ts: str) -> str:
                return ts or ""

            normalized = src.select(
                note_id=pw.this.id,
                timestamp=pw.this.timestamp,
                content=pw.this.content,
                ts_key=pw.apply(_ts_key, pw.this.timestamp),
            )

            # Get latest timestamp per note_id
            latest_ts = (
                normalized
                .groupby(pw.this.note_id)
                .reduce(
                    note_id=pw.this.note_id,
                    max_ts=pw.reducers.max(pw.this.ts_key),
                )
            )

            # Join back to fetch content (and any other fields) for that max_ts
            # Join back to fetch the row matching the max timestamp per note_id
            jr = latest_ts.join(
                normalized,
                latest_ts.note_id == normalized.note_id
            )

            # Use pw.left / pw.right instead of *_left / *_right
            latest_rows = (
                jr
                .filter(pw.left.max_ts == pw.right.ts_key)
                .select(
                    note_id=pw.left.note_id,
                    timestamp=pw.left.max_ts,
                    content=pw.right.content,
                )
            )


            # Enrich each latest row by calling your existing function (writes to SQLite)
            enriched = latest_rows.select(
                enriched=pw.apply(
                    lambda _nid, _ts, _c: process_capture_object(
                        {"id": _nid, "timestamp": _ts, "content": _c},
                        store=True
                    ),
                    pw.this.note_id, pw.this.timestamp, pw.this.content
                )
            )

            # Lightweight sink for a health heartbeat
            # --- version-agnostic heartbeat sink ---
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
                # optional: fallback to debug print if heartbeat fails
                # pw.debug.print(enriched)
                pass


            # Backfill any existing raw captures into SQLite once at startup
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
        sims.sort(key=lambda x: x["score"], reverse=True)
        top_related = sims[:5]
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
    Falls back to rumps.alert if PyObjC isn't available.
    """
    if not PYOBJC_AVAILABLE:
        # fallback: truncated rumps alert
        rumps.alert(title, text[:4000])
        return

    def _create():
        rect = NSMakeRect(200.0, 200.0, float(width), float(height))
        style_mask = 15  # titled, closable, resizable, miniaturizable
        window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style_mask, NSBackingStoreBuffered, False
        )
        window.setTitle_(title)

        # create scroll view
        scroll = NSScrollView.alloc().initWithFrame_(rect)
        scroll.setHasVerticalScroller_(True)
        scroll.setHasHorizontalScroller_(False)
        # autoresize with window
        try:
            scroll.setAutoresizingMask_(18)  # width + height sizable
        except Exception:
            pass

        # create text view
        text_view = NSTextView.alloc().initWithFrame_(rect)
        text_view.setEditable_(False)
        text_view.setSelectable_(True)
        text_view.setVerticallyResizable_(True)
        text_view.setHorizontallyResizable_(False)
        # allow width to be fixed and vertical to expand
        try:
            text_view.setAutoresizingMask_(18)
        except Exception:
            pass

        # configure text container so it can grow vertically and allow scrolling
        container = text_view.textContainer()
        # very tall container height
        try:
            container.setContainerSize_((float(width), 1e7))
            # don't force width-tracking (so vertical scrolling works properly)
            container.setWidthTracksTextView_(False)
        except Exception:
            pass

        # set text (NSString ensures PyObjC passes correctly)
        ns_text = NSString.stringWithString_(text)
        text_view.setString_(ns_text)

        scroll.setDocumentView_(text_view)
        window.setContentView_(scroll)
        window.center()
        window.makeKeyAndOrderFront_(None)

        # bring to front (without stealing focus too badly)
        try:
            NSRunningApplication.currentApplication().activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
        except Exception:
            pass

        # retain window so it doesn't get GC'd
        GLOBAL_WINDOWS.append(window)

    # schedule creation on Cocoa main loop
    AppHelper.callAfter(_create)

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
        super().__init__(APP_NAME, quit_button=None)
        self.menu = [
            "Quick Capture",
            "Open DB Folder",
            "Show Latest Note",
            "Show All Notes",
            "Process Raw Now",
            "Enrich Missing",
            "Recompute Links",
            "Pipeline Status",
            "View Log",
            "Quit"
        ]
        self.processor = PipelineProcessor(INPUT_STREAM)
        self.processor.start()

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
            target=lambda: process_capture_object(obj, store=True),
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
        with db_lock:
            c = db_conn.cursor()
            try:
                c.execute("SELECT id FROM notes ORDER BY timestamp DESC")
                rows = c.fetchall()
            except Exception:
                rows = []
        if not rows:
            raw = read_raw_input_stream(limit=200)
            if not raw:
                rumps.notification(APP_NAME, "All Notes", "No notes or raw captures found.")
                return
            lines = [f"Raw captures ({len(raw)}):"]
            for r in raw[:NOTES_PAGE_SIZE]:
                ts = r.get("timestamp"); content = r.get("content","")
                lines.append(f"[{ts}] {content[:140].replace('\\n',' ')}")
            body = "\n".join(lines)
            show_text_window("All Notes (raw)", body, width=700, height=420)
            return

        ids = [r[0] for r in rows]
        total = len(ids)
        pages = [ids[i:i+NOTES_PAGE_SIZE] for i in range(0, total, NOTES_PAGE_SIZE)]
        # compile combined big body for scrolling: show all pages in one scrollable window
        combined_lines = []
        for pi, page in enumerate(pages, start=1):
            combined_lines.append(f"All Notes (page {pi}/{len(pages)}):\n")
            for nid in page:
                note = fetch_note_by_id_safe(nid)
                if not note:
                    continue
                combined_lines.append(format_note_brief(note))
            combined_lines.append("\n" + ("-" * 40) + "\n")
        combined_body = "\n".join(combined_lines)

        # compute actionable summary and append into same window (in background so UI stays snappy)
        def summary_thread():
            notes = []
            with db_lock:
                c = db_conn.cursor()
                try:
                    c.execute("SELECT * FROM notes ORDER BY timestamp DESC")
                    rows_all = c.fetchall()
                    cols = [r[1] for r in c.execute("PRAGMA table_info(notes)").fetchall()]
                except Exception:
                    rows_all = []
                    cols = []
            for row in rows_all:
                data = {}
                for i, col in enumerate(cols):
                    data[col] = row[i]
                for k in ("tags","entities","related"):
                    if k in data and k in data and data[k]:
                        try:
                            data[k] = json.loads(data[k])
                        except Exception:
                            data[k] = []
                    else:
                        data[k] = []
                if "embedding" in data:
                    data["embedding"] = embedding_bytes_to_list(data["embedding"])
                notes.append(data)

            summary_text = build_full_notes_summary_with_openai(notes)
            final_body = combined_body + "\n\n==== Actionable Summary ====\n\n" + summary_text
            show_text_window(f"All Notes (total {total})", final_body, width=920, height=700)

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
