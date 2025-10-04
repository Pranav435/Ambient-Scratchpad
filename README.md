# Ambient Scratchpad

_A lightweight, AI-augmented note-taking app for macOS that lives in your menubar._

Ambient Scratchpad is built for **fast capture and intelligent recall**. Every note is more than just text: it’s enriched with **AI classification, tags, summaries, embeddings, and related connections**.  
What makes this unique is the **real-time streaming pipeline** — powered by [Pathway](https://pathway.com/) and [OpenAI](https://platform.openai.com/) — combined with a **native macOS experience**.

---

## ✨ Why Ambient Scratchpad?

- 🖊️ **Quick capture** from anywhere in macOS, right from the menubar (`⌥+Space`).
- 🧠 **AI enrichment**:
  - Classifies each note (Idea, To-Do, Snippet, etc.)
  - Adds tags and entities
  - Generates a one-line summary
  - Embeds notes into vector space for similarity search
- 🔗 **Smart linking** between notes using cosine similarity on embeddings.
- 📊 **Actionable full summaries**: automatically structured into to-dos, projects, inspirations, and follow-ups.
- ⚡ **Pathway streaming engine**: real-time deduplication, ordering, and consistent enrichment pipeline.
- 🍎 **Native macOS feel**:
  - Menubar app via [rumps](https://github.com/jaredks/rumps)
  - Scrollable, resizable windows via [PyObjC / AppKit](https://pypi.org/project/pyobjc/)

Ambient Scratchpad’s USP:  
👉 It’s not just “note storage”. It’s an **ambient intelligence layer** — your thoughts get **structured, connected, and actionable automatically**.

---

## 🧩 How it Works

### Data Flow
1. **Capture**  
   - Notes are captured via the menubar Quick Capture (`⌥+Space`) or appended to `input_stream.jsonl`.

2. **Streaming**  
   - **Preferred:** Pathway watches the JSONL file as a **live table** with schema `(id, timestamp, content)`.
   - It groups by note `id`, picks the **latest timestamp**, and ensures **only the newest version** of each note is processed.
   - **Fallback:** If Pathway is not enabled, a simple polling loop tails the file and processes new lines.

3. **Enrichment**  
   Each latest note flows through `process_capture_object`:
   - OpenAI `chat.completions` → classification, tags, entities, one-line summary
   - OpenAI `embeddings` → vector representation for similarity search
   - Related notes computed via cosine similarity

4. **Storage**  
   Notes are persisted in `knowledge_base.db` (SQLite).  
   Embeddings are stored as binary blobs, ensuring durability and reusability.  
   Writes are **idempotent** (`INSERT OR REPLACE`), so replays/duplicates never corrupt the DB.

5. **Connections**  
   Each note is linked to its most relevant neighbors (using a configurable similarity threshold).

6. **Summarization**  
   When you open “Show All Notes”, the app can build a **structured actionable summary**:
   - To-Dos
   - Project ideas
   - Inspirations
   - Follow-ups
   - One-line overview

7. **Presentation**  
   - Menubar menus for quick actions  
   - Native PyObjC windows for large summaries, scrollable & resizable  

---

## ⚙️ Tech Stack

- **Python 3.10+**
- **[rumps](https://github.com/jaredks/rumps)** — lightweight macOS menubar apps
- **[PyObjC / AppKit](https://pypi.org/project/pyobjc/)** — native scrollable windows
- **[OpenAI](https://platform.openai.com/)** — classification, embeddings, summarization
- **[Pathway](https://pathway.com/)** — real-time streaming pipeline (dedupe + consistency)
- **SQLite** — persistent storage
- **dotenv** — environment configuration
- **requests + BeautifulSoup** — lightweight web scraping for links

---

## 🚀 Setup & Run

### 1. Clone
```bash
git clone https://github.com/yourusername/ambient-scratchpad.git
cd ambient-scratchpad
````

### 2. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Or manually:

```bash
pip install rumps openai python-dotenv pyobjc pathway requests beautifulsoup4 numpy
```

### 3. Configure `.env`

```ini
OPENAI_API_KEY=sk-...yourkey...
OPENAI_CLASSIFY_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
USE_PATHWAY_STREAM=1   # 1 = use Pathway (recommended), 0 = fallback polling
```

### 4. Run

```bash
python main.py
```

The app appears in your **macOS menubar**.

---

## 📦 Packaging as a macOS `.app`

1. Install py2app:

   ```bash
   pip install py2app
   ```
2. Add a `setup.py` (example in repo).
3. Build:

   ```bash
   python setup.py py2app
   ```
4. Find the app in `dist/AmbientScratchpad.app`.

---

## 📂 Project Structure

```
AmbientScratchpad/
│
├── main.py              # app logic & Pathway pipeline
├── knowledge_base.db    # SQLite DB (auto-created)
├── input_stream.jsonl   # raw captures (append-only)
├── ambient.log          # log file
├── requirements.txt
└── README.md
```

---

## 🔍 Example: Note Lifecycle

1. Capture a note:

   ```
   “Work on pitch deck by Friday”
   ```
2. Pathway ensures only the **latest version** of this note is processed.
3. OpenAI enriches it:

   ```json
   {
     "category": "To-Do",
     "tags": ["work", "deadline"],
     "entities": ["Friday"],
     "summary": "Task: finish pitch deck by Friday"
   }
   ```
4. Embedding links it to related notes.
5. In “Show All Notes”, the full actionable summary appears.

---

## 🛠 Development Notes

* Quick Capture is always **non-blocking** (runs in threads).
* Pathway streaming is **preferred**:

  * Deduplicates notes by `id`
  * Processes only the latest version
  * Provides crash-safe, idempotent state
* Polling mode is kept as a fallback for environments without Pathway.

---

## 🗺️ Roadmap

* [ ] Export summaries to Markdown/HTML
* [ ] Advanced search & filter by tags/entities
* [ ] Sync across devices
* [ ] Collaborative editing

---

## 📜 License

MIT License © 2025 Pranav Bhaven Savla & Dhiraj Dinesh Deva
