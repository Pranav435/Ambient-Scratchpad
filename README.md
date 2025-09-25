# Ambient Scratchpad

_A lightweight, AI-augmented note-taking app for macOS that lives in your menubar._

Ambient Scratchpad is built for **fast capture and intelligent recall**. Instead of just storing text, it enriches your notes automatically with **AI classification, tags, summaries, and related connections**. A unique selling point is its **real-time AI pipeline** — powered by [Pathway](https://pathway.com/) and [OpenAI](https://platform.openai.com/) — combined with a **native macOS experience**.

---

## ✨ Why Ambient Scratchpad?

- 🖊️ **Quick capture** from anywhere in macOS, right from the menubar.
- 🧠 **AI enrichment**:
  - Classifies each note (Idea, To-Do, Snippet, etc.)
  - Adds tags and entities
  - Generates a one-line summary
- 🔗 **Smart linking** between notes using embeddings + cosine similarity.
- 📊 **Actionable full summaries** — prioritized to-dos, projects, inspirations, and follow-ups in one place.
- ⚡ **Pathway integration** for optional real-time streaming.
- 🍎 **Native macOS feel**:
  - Menubar app via [rumps](https://github.com/jaredks/rumps)
  - Scrollable, resizable windows via [PyObjC / AppKit](https://pypi.org/project/pyobjc/)

Ambient Scratchpad’s USP:  
👉 It’s not just “note storage”. It’s an **ambient intelligence layer** — your thoughts get **structured, connected, and actionable automatically**.

---

## 🧩 How it Works

### Data Flow
1. **Capture**  
   Notes are captured either via the menubar “Quick Capture” or by appending to an input stream file (`input_stream.jsonl`).

2. **Enrichment**  
   Each note is processed by OpenAI:
   - `chat.completions` → classification, tags, entities, and summary
   - `embeddings` → vector representation for similarity search

3. **Storage**  
   Enriched notes are stored in an SQLite database (`knowledge_base.db`) with embeddings, metadata, and related links.

4. **Connections**  
   Cosine similarity finds related notes, stored alongside each entry.

5. **Summarization**  
   When viewing all notes, the app queries OpenAI again to build a **structured summary**:
   - To-Dos (with priority)
   - Project ideas
   - Inspirations
   - Follow-ups
   - Overall one-line summary

6. **Presentation**  
   - Menubar menus for quick actions  
   - Scrollable native windows for long text views (notes + summaries)  

---

## ⚙️ Tech Stack

- **Python 3.10+**
- **[rumps](https://github.com/jaredks/rumps)** — lightweight menubar apps
- **[PyObjC / AppKit](https://pypi.org/project/pyobjc/)** — native macOS scrollable windows
- **[OpenAI](https://platform.openai.com/)** — classification, embeddings, and summarization
- **[Pathway](https://pathway.com/)** — (optional) real-time streaming pipeline
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

Create a `.env` file:

```ini
OPENAI_API_KEY=sk-...yourkey...
OPENAI_CLASSIFY_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
USE_PATHWAY_STREAM=0
```

### 4. Run

```bash
python main.py
```

The app appears in your **macOS menubar**.

---

## 📦 Packaging as a macOS `.app`

To build a standalone `.app` bundle:

1. Install py2app:

   ```bash
   pip install py2app
   ```
2. Add a `setup.py` (example provided in repo).
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
├── main.py              # main app logic
├── knowledge_base.db    # SQLite DB (auto-created)
├── input_stream.jsonl   # raw captures (append-only)
├── ambient.log          # log file
├── requirements.txt
└── README.md
```

---

## 🔍 Example: Note Lifecycle

1. You capture a note:

   ```
   “Work on pitch deck by Friday”
   ```
2. OpenAI enriches it:

   ```json
   {
     "category": "To-Do",
     "tags": ["work", "deadline"],
     "entities": ["Friday"],
     "summary": "Task: finish pitch deck by Friday"
   }
   ```
3. It’s embedded and linked to related notes about “work” and “projects”.
4. In “Show All Notes” view, the full summary shows:

   ```
   ==== Actionable Summary ====

   1) Top actionable To-Dos
   - Finish pitch deck by Friday (high)

   2) Project ideas
   None

   3) Quick inspirations
   None

   4) Follow-ups / people / links
   None

   5) Overall
   Notes suggest urgent work deliverables this week.
   ```

---

## 🛠 Development Notes

* Menubar menu interactions are **non-blocking** (threads for background AI).
* Large note views use **PyObjC AppKit windows**, so:

  * They are scrollable & resizable
  * They do **not gray out** the menubar
* Pathway is **optional**: polling is used by default for safety.

---

## 🗺️ Roadmap

* [ ] Export summaries to text/Markdown files
* [ ] Search & filter notes by tags/entities
* [ ] Sync across devices
* [ ] Collaborative editing mode

---

## 📜 License

MIT License © 2025 Pranav Bhaven Savla & Dhiraj Dinesh Deva