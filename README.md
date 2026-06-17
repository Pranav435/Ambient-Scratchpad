# Ambient Scratchpad

_A lightweight, AI-augmented note-taking app for macOS that lives in your menubar._

Ambient Scratchpad is built for fast capture and intelligent recall. Every note is more than just text: it's enriched with **AI classification, tags, summaries, embeddings, and semantic connections**.

---

## ✨ Why Ambient Scratchpad?

- 🖊️ **Quick capture** from anywhere in macOS, right from the menubar (`⌥+Space`)
- 🧠 **AI enrichment**:
  - Classifies each note (Idea, To-Do, Code Snippet, Link, Quote, Contact, Other)
  - Extracts tags and named entities
  - Generates a concise one-line summary
  - Creates semantic embeddings for meaning-based search
- 🔗 **Semantic linking** — notes are connected by *meaning*, not just keywords
- 📊 **Actionable summaries** — automatically structured into to-dos, projects, inspirations, and follow-ups
- 🍎 **Native macOS feel** via [rumps](https://github.com/jaredks/rumps) + [PyObjC](https://pypi.org/project/pyobjc/)

**The key insight:**  
👉 It's not just "note storage". It's an **ambient intelligence layer** — your thoughts get **structured, connected, and actionable automatically**.

---

## 🧩 How it Works

### Data Flow

1. **Capture**  
   Notes are captured via the menubar Quick Capture (`⌥+Space`) or appended to `input_stream.jsonl`.

2. **Enrichment**  
   Each note flows through `process_capture_object`:
   - **Classification** — Gemini categorizes the note and extracts tags/entities
   - **Summarization** — AI generates a one-sentence summary capturing the note's meaning
   - **Embedding** — The summary + content is embedded into a 1536-dimensional vector space
   - **Relation computation** — Semantic similarity finds related notes

3. **Storage**  
   Notes are persisted in `knowledge_base.db` (SQLite).  
   Embeddings are stored as binary blobs for efficient retrieval.  
   Writes are **idempotent** (`INSERT OR REPLACE`), so duplicates never corrupt the DB.

4. **Summarization**  
   When you open "Show All Notes", the app builds a **structured actionable summary**:
   - Top actionable To-Dos (with priority)
   - Project ideas (with next steps)
   - Quick inspirations to remember
   - Follow-ups / people / links to check
   - Specific overall summary mentioning key themes

---

## 🔗 How Note Relations Work

Ambient Scratchpad uses a **semantic similarity system** to find related notes based on *meaning*, not just shared words.

### The Algorithm

Relations are computed using a **weighted hybrid scoring** approach:

| Signal | Weight | Description |
|--------|--------|-------------|
| **Embedding Similarity** | 85% | Cosine similarity between Gemini embeddings — captures deep semantic meaning |
| **Category Affinity** | 10% | Logical relationships between note types (see below) |
| **Entity Overlap** | 5% | Shared named entities (people, projects, etc.) |

### Why Embedding-First?

Gemini's `text-embedding-004` model understands *meaning*:
- "Buy groceries" relates to "Shopping list for dinner" (same intent)
- "Meeting with John about the project" relates to "Project deadline next week" (same context)
- But "IIT Ropar has pizza" won't relate to "Pizzaaaaa!" unless they share deeper meaning

The embedding is created from **summary + content**, so the AI's understanding of the note's meaning is captured in the vector.

### The Meaning Gate

Category affinity and entity overlap can only ever *re-rank or strengthen* a
relation — they can never create one. Before any boost is applied, a candidate
must clear a minimum **raw embedding cosine similarity** (`RELATION_MIN_EMBED_SIM`,
default `0.55`). This guarantees every relation is grounded in genuine semantic
similarity, so two notes that merely share a category or a keyword will **not**
be linked unless their meanings are actually close.

### Entity & Tag Matching is Token-Based, Not Substring-Based

Entity and tag signals match on **whole words**, not raw substrings or shared
prefixes. This removes spurious links created by spelling accidents:
- ❌ `"Sam"` is no longer matched to `"Samsung"`
- ❌ `"portal"` is no longer matched to `"porter"` (old 4-char-prefix hack)
- ✅ `"John Smith"` still matches `"call John"` (shared whole word)

### Optional: LLM Relationship Verification

With `USE_LLM_RELATION_VERIFY=1`, borderline candidates that pass the meaning
gate go through a single batched Gemini call that decides whether they are
*genuinely related in meaning* (same topic/project/person, or a clear
cause-effect / follow-up link) and returns a short reason. Confident matches
(`emb_sim ≥ 0.80`) skip the call. The reason is stored and shown in the note
detail view (`↳ same project deadline`). If the call fails, relations fall back
gracefully to the embedding-only result.

### Category Affinity Matrix

Certain note types have logical relationships:

| Category Pair | Affinity | Reason |
|--------------|----------|--------|
| To-Do ↔ Idea | 15% | Ideas often become tasks |
| Code Snippet ↔ Idea | 12% | Ideas get implemented as code |
| Contact ↔ To-Do | 12% | People are assigned to tasks |
| Link/Quote ↔ Idea | 8-10% | External sources inspire ideas |

### Thresholds

- **Minimum embedding similarity (the meaning gate)**: 0.55 — a relation must be grounded in semantic similarity before any boost applies
- **Minimum blended score**: 0.45 (fairly strict — only truly related notes connect)
- **Strong-match threshold**: 0.80 (skips optional LLM verification)
- **Max relations per note**: 5 (quality over quantity)

---

## ⚙️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Streaming Engine** | [Pathway](https://pathway.com/) | Real-time data ingestion, deduplication, and model updates |
| UI Framework | [rumps](https://github.com/jaredks/rumps) | macOS menubar app |
| Native Windows | [PyObjC / AppKit](https://pypi.org/project/pyobjc/) | Scrollable, resizable windows |
| AI | [Google Gemini API](https://ai.google.dev/) | Classification, embeddings, summarization |
| Database | SQLite | Persistent note storage |
| Config | python-dotenv | Environment variables |
| Web Scraping | requests + BeautifulSoup | Link content extraction |
| Math | NumPy | Fast vector operations |

### 🌊 Pathway: Real-Time Streaming

Pathway powers the **real-time note injection pipeline**:
- Watches `input_stream.jsonl` for new notes
- Provides **incremental processing** — only new/changed notes are processed
- Handles **deduplication** with built-in idempotency
- Enables **live model updates** as notes stream in

This is the default and recommended mode. Set `USE_PATHWAY_STREAM=0` only for debugging.

---

## 🚀 Setup & Run

### 1. Clone

```bash
git clone https://github.com/yourusername/ambient-scratchpad.git
cd ambient-scratchpad
```

### 2. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure `.env`

```ini
GEMINI_API_KEY=...yourkey...
GEMINI_CLASSIFY_MODEL=gemini-1.5-flash
GEMINI_EMBEDDING_MODEL=models/text-embedding-004
USE_PATHWAY_STREAM=1        # Pathway streaming enabled by default (set to 0 only for debugging)
USE_LLM_RELATION_VERIFY=0   # Set to 1 to verify/explain borderline relations with an LLM (adds one call per capture)
```

### 4. Run

```bash
python main.py
```

The app appears in your **macOS menubar**.

---

## 📂 Project Structure

```
Ambient-Scratchpad/
├── main.py                    # Core app logic & processing pipeline
├── test_ambient_scratchpad.py # Test suite (71 tests)
├── knowledge_base.db          # SQLite DB (auto-created)
├── input_stream.jsonl         # Raw captures (append-only)
├── ambient.log                # Application logs
├── requirements.txt           # Python dependencies
└── README.md
```

---

## 🔍 Example: Note Lifecycle

1. **Capture** a note:
   ```
   "Discuss project timeline with Sarah on Monday"
   ```

2. **AI enriches** it:
   ```json
   {
     "category": "To-Do",
     "tags": ["meeting", "project", "planning"],
     "entities": ["Sarah", "Monday"],
     "summary": "Schedule a meeting with Sarah to discuss project timeline."
   }
   ```

3. **Embedding** is created from summary + content, capturing the semantic meaning (meetings, planning, people).

4. **Relations** are computed:
   - Finds other notes about Sarah (entity overlap boost)
   - Finds other project-related notes (embedding similarity)
   - Connects to related Ideas (category affinity: To-Do ↔ Idea)

5. **In "Show All Notes"**, the actionable summary appears with this note properly contextualized.

---

## 🧪 Testing

Run the test suite:

```bash
python -m pytest test_ambient_scratchpad.py -v
```

The test suite covers:
- Cosine similarity calculations
- Database operations
- Embedding generation (including fallback)
- Relation computation with semantic scoring
- Summary generation

---

## 🛠 Configuration

### Tuning Relations

In `main.py`, adjust these constants:

```python
RELATION_MIN_SCORE = 0.45        # Higher = fewer, stricter blended-score relations
RELATION_MIN_EMBED_SIM = 0.55    # The meaning gate: min raw embedding cosine before any boost
RELATION_STRONG_EMBED_SIM = 0.80 # At/above this, a pair is a confident match (skips LLM verify)
RELATED_MAX_PER_NOTE = 5         # Maximum related notes per note
```

### Category Affinity

Modify `CATEGORY_AFFINITY` dict to change how note types relate:

```python
CATEGORY_AFFINITY = {
    ("To-Do", "Idea"): 0.15,
    ("Code Snippet", "Idea"): 0.12,
    # Add your own pairs...
}
```

---

## 📜 License

MIT License © 2025 Pranav Bhaven Savla & Dhiraj Dinesh Deva
