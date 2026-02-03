"""
Comprehensive Test Suite for Ambient Scratchpad

Tests all core functionality including:
- Database operations (CRUD, migrations)
- Helper functions (timestamps, URL detection, embedding utilities)
- Note processing and enrichment
- Similarity calculations
- Related notes computation
- Input stream operations
- Pipeline processing
- Summary generation
"""

import os
import sys
import json
import uuid
import time
import struct
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timezone
from typing import List, Dict, Any

# Add parent dir to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock macOS-specific modules before importing main
sys.modules['rumps'] = MagicMock()
sys.modules['AppKit'] = MagicMock()
sys.modules['Foundation'] = MagicMock()
sys.modules['PyObjCTools'] = MagicMock()
sys.modules['PyObjCTools.AppHelper'] = MagicMock()
sys.modules['objc'] = MagicMock()

# Mock pathway as it may not be available
mock_pathway = MagicMock()
mock_pathway.Schema = type('Schema', (), {})
sys.modules['pathway'] = mock_pathway

# Import the module to test
import main

# Ensure PYOBJC_AVAILABLE is False for tests (we mocked it)
main.PYOBJC_AVAILABLE = False


class TestHelperFunctions(unittest.TestCase):
    """Tests for utility/helper functions."""
    
    def test_now_iso_format(self):
        """Test that now_iso() returns properly formatted ISO timestamp."""
        result = main.now_iso()
        self.assertIsInstance(result, str)
        self.assertTrue(result.endswith("Z"), f"Expected Z suffix, got: {result}")
        # Should be parseable
        try:
            datetime.fromisoformat(result.replace("Z", "+00:00"))
        except ValueError:
            self.fail(f"Invalid ISO format: {result}")
    
    def test_is_url_valid_http(self):
        """Test URL detection for http:// URLs."""
        self.assertTrue(main.is_url("http://example.com"))
        self.assertTrue(main.is_url("http://example.com/path?query=1"))
    
    def test_is_url_valid_https(self):
        """Test URL detection for https:// URLs."""
        self.assertTrue(main.is_url("https://example.com"))
        self.assertTrue(main.is_url("https://sub.example.com/path"))
    
    def test_is_url_invalid(self):
        """Test URL detection for non-URLs."""
        self.assertFalse(main.is_url("not a url"))
        self.assertFalse(main.is_url("ftp://example.com"))
        self.assertFalse(main.is_url(""))
        self.assertFalse(main.is_url(None))
        self.assertFalse(main.is_url(123))
    
    def test_friendly_snippet_short_text(self):
        """Test snippet generation for short text."""
        text = "Short text"
        result = main.friendly_snippet(text, 120)
        self.assertEqual(result, "Short text")
    
    def test_friendly_snippet_long_text(self):
        """Test snippet generation truncates long text."""
        text = "A" * 200
        result = main.friendly_snippet(text, 50)
        self.assertEqual(len(result), 53)  # 50 chars + "..."
        self.assertTrue(result.endswith("..."))
    
    def test_friendly_snippet_newlines(self):
        """Test that snippet removes newlines."""
        text = "Line 1\nLine 2\nLine 3"
        result = main.friendly_snippet(text, 100)
        self.assertNotIn("\n", result)
    
    def test_friendly_snippet_empty(self):
        """Test snippet with empty input."""
        self.assertEqual(main.friendly_snippet("", 50), "")
        self.assertEqual(main.friendly_snippet(None, 50), "")


class TestEmbeddingUtilities(unittest.TestCase):
    """Tests for embedding conversion utilities."""
    
    def test_get_embedding_bytes_basic(self):
        """Test converting embedding list to bytes."""
        emb = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = main.get_embedding_bytes(emb)
        self.assertIsInstance(result, bytes)
        # Each float64 is 8 bytes
        self.assertEqual(len(result), len(emb) * 8)
    
    def test_embedding_bytes_to_list_basic(self):
        """Test converting bytes back to embedding list."""
        original = [0.1, 0.2, 0.3, 0.4, 0.5]
        as_bytes = main.get_embedding_bytes(original)
        result = main.embedding_bytes_to_list(as_bytes)
        self.assertEqual(len(result), len(original))
        for i, (r, o) in enumerate(zip(result, original)):
            self.assertAlmostEqual(r, o, places=10, msg=f"Mismatch at index {i}")
    
    def test_embedding_bytes_to_list_empty(self):
        """Test handling of empty/None input."""
        self.assertEqual(main.embedding_bytes_to_list(None), [])
        self.assertEqual(main.embedding_bytes_to_list(b""), [])
    
    def test_embedding_roundtrip(self):
        """Test that embeddings survive roundtrip conversion."""
        original = [float(i) / 100.0 for i in range(128)]
        as_bytes = main.get_embedding_bytes(original)
        recovered = main.embedding_bytes_to_list(as_bytes)
        self.assertEqual(len(recovered), len(original))
        for o, r in zip(original, recovered):
            self.assertAlmostEqual(o, r, places=10)


class TestCosineSimilarity(unittest.TestCase):
    """Tests for cosine similarity calculations."""
    
    def test_identical_vectors(self):
        """Test similarity of identical vectors is 1.0."""
        vec = [1.0, 2.0, 3.0]
        result = main.cosine_similarity(vec, vec)
        self.assertAlmostEqual(result, 1.0, places=5)
    
    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors is 0.0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        result = main.cosine_similarity(a, b)
        self.assertAlmostEqual(result, 0.0, places=5)
    
    def test_opposite_vectors(self):
        """Test similarity of opposite vectors is -1.0."""
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        result = main.cosine_similarity(a, b)
        self.assertAlmostEqual(result, -1.0, places=5)
    
    def test_empty_vectors(self):
        """Test handling of empty vectors."""
        self.assertEqual(main.cosine_similarity([], [1.0, 2.0]), 0.0)
        self.assertEqual(main.cosine_similarity([1.0, 2.0], []), 0.0)
        self.assertEqual(main.cosine_similarity([], []), 0.0)
    
    def test_different_length_vectors(self):
        """Test handling of vectors with different lengths.
        
        Now truncates to the smaller dimension instead of returning 0,
        allowing comparison of embeddings from different models.
        """
        a = [1.0, 2.0, 3.0, 4.0]
        b = [1.0, 2.0]
        result = main.cosine_similarity(a, b)
        # Should truncate to min length and compute similarity
        # The first 2 elements are identical, so similarity should be 1.0
        self.assertAlmostEqual(result, 1.0, places=5)
        
        # Different vectors truncated
        a2 = [1.0, 0.0, 3.0, 4.0]  
        b2 = [0.0, 1.0]  # Orthogonal to first 2 elements of a2
        result2 = main.cosine_similarity(a2, b2)
        self.assertAlmostEqual(result2, 0.0, places=5)
    
    def test_zero_vectors(self):
        """Test handling of zero vectors."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        result = main.cosine_similarity(a, b)
        self.assertEqual(result, 0.0)
    
    def test_similar_vectors(self):
        """Test similarity of similar but not identical vectors."""
        a = [1.0, 2.0, 3.0]
        b = [1.1, 2.1, 3.1]
        result = main.cosine_similarity(a, b)
        self.assertGreater(result, 0.99)
        self.assertLess(result, 1.0)


class TestDatabaseOperations(unittest.TestCase):
    """Tests for database initialization and operations."""
    
    def setUp(self):
        """Create a temporary database for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_kb.db")
        # Patch DB_FILE temporarily
        self._original_db = main.DB_FILE
        main.DB_FILE = self.test_db
        # Create fresh connection
        self.conn = sqlite3.connect(self.test_db)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
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
            )
        """)
        self.conn.commit()
    
    def tearDown(self):
        """Clean up temporary database."""
        main.DB_FILE = self._original_db
        self.conn.close()
        try:
            os.remove(self.test_db)
            os.rmdir(self.temp_dir)
        except:
            pass
    
    def test_store_and_fetch_note(self):
        """Test storing and fetching a note."""
        note = {
            "id": str(uuid.uuid4()),
            "timestamp": main.now_iso(),
            "content": "Test content",
            "scraped_content": "",
            "category": "Idea",
            "tags": ["test", "note"],
            "entities": ["Test Entity"],
            "summary": "Test summary",
            "embedding": [0.1] * 128,
            "related": []
        }
        
        # Store using the actual function
        with patch.object(main, 'db_conn', self.conn):
            main.store_enriched_note(note)
        
        # Verify it was stored
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM notes WHERE id=?", (note["id"],))
        row = cursor.fetchone()
        
        self.assertIsNotNone(row)
        # id, timestamp, content, scraped_content, category, tags, entities, embedding, related, summary
        self.assertEqual(row[0], note["id"])
        self.assertEqual(row[2], note["content"])
        self.assertEqual(row[4], note["category"])
    
    def test_store_note_idempotent(self):
        """Test that storing same note twice is idempotent (INSERT OR REPLACE)."""
        note_id = str(uuid.uuid4())
        note = {
            "id": note_id,
            "timestamp": main.now_iso(),
            "content": "Original content",
            "category": "Idea",
            "tags": [],
            "entities": [],
            "summary": "Original summary",
            "embedding": [0.1] * 10,
            "related": []
        }
        
        with patch.object(main, 'db_conn', self.conn):
            main.store_enriched_note(note)
            # Store again with modified content
            note["content"] = "Updated content"
            main.store_enriched_note(note)
        
        # Should only have one row with updated content
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*), content FROM notes WHERE id=?", (note_id,))
        count, content = cursor.fetchone()
        self.assertEqual(count, 1)
        self.assertEqual(content, "Updated content")


class TestInputStreamOperations(unittest.TestCase):
    """Tests for input stream (JSONL) operations."""
    
    def setUp(self):
        """Create temporary input stream file."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_stream = os.path.join(self.temp_dir, "test_stream.jsonl")
        self._original_stream = main.INPUT_STREAM
        main.INPUT_STREAM = self.test_stream
    
    def tearDown(self):
        """Clean up temporary files."""
        main.INPUT_STREAM = self._original_stream
        try:
            os.remove(self.test_stream)
            os.rmdir(self.temp_dir)
        except:
            pass
    
    def test_append_to_input_stream(self):
        """Test appending a capture to the input stream."""
        obj = {
            "id": str(uuid.uuid4()),
            "timestamp": main.now_iso(),
            "content": "Test note content"
        }
        
        main.append_to_input_stream(obj)
        
        # Verify file exists and contains the JSON
        self.assertTrue(os.path.exists(self.test_stream))
        with open(self.test_stream, "r") as f:
            line = f.readline().strip()
            parsed = json.loads(line)
            self.assertEqual(parsed["id"], obj["id"])
            self.assertEqual(parsed["content"], obj["content"])
    
    def test_append_multiple_captures(self):
        """Test appending multiple captures."""
        for i in range(5):
            obj = {
                "id": str(uuid.uuid4()),
                "timestamp": main.now_iso(),
                "content": f"Note {i}"
            }
            main.append_to_input_stream(obj)
        
        with open(self.test_stream, "r") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 5)
    
    def test_read_raw_input_stream(self):
        """Test reading raw input stream."""
        # Write some test data
        test_objs = [
            {"id": str(uuid.uuid4()), "timestamp": main.now_iso(), "content": f"Note {i}"}
            for i in range(3)
        ]
        with open(self.test_stream, "w") as f:
            for obj in test_objs:
                f.write(json.dumps(obj) + "\n")
        
        result = main.read_raw_input_stream(limit=10)
        
        self.assertEqual(len(result), 3)
        # Result should be reversed (newest first)
        self.assertEqual(result[0]["content"], "Note 2")
    
    def test_read_raw_input_stream_limit(self):
        """Test that limit parameter works."""
        with open(self.test_stream, "w") as f:
            for i in range(100):
                obj = {"id": str(i), "content": f"Note {i}"}
                f.write(json.dumps(obj) + "\n")
        
        result = main.read_raw_input_stream(limit=10)
        self.assertEqual(len(result), 10)
    
    def test_read_raw_input_stream_nonexistent(self):
        """Test reading from nonexistent file."""
        main.INPUT_STREAM = "/nonexistent/path/file.jsonl"
        result = main.read_raw_input_stream()
        self.assertEqual(result, [])


class TestClassifyAndTag(unittest.TestCase):
    """Tests for Gemini classification (mocked)."""
    
    @patch.object(main, 'gemini_client')
    def test_classify_returns_valid_structure(self, mock_client):
        """Test that classify_and_tag returns proper structure."""
        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "category": "Idea",
            "tags": ["test", "mock"],
            "entities": ["Test Entity"],
            "summary": "A test summary"
        })
        mock_client.models.generate_content.return_value = mock_response
        
        result = main.classify_and_tag_with_gemini("Test text")
        
        self.assertIn("category", result)
        self.assertIn("tags", result)
        self.assertIn("entities", result)
        self.assertIn("summary", result)
        self.assertEqual(result["category"], "Idea")
    
    @patch.object(main, 'gemini_client')
    def test_classify_handles_error(self, mock_client):
        """Test graceful handling of API errors."""
        mock_client.models.generate_content.side_effect = Exception("API Error")
        
        result = main.classify_and_tag_with_gemini("Test text")
        
        # Should return defaults on error
        self.assertEqual(result["category"], "Other")
        self.assertEqual(result["tags"], [])
        self.assertEqual(result["entities"], [])
    
    def test_classify_empty_text(self):
        """Test classification with empty text."""
        result = main.classify_and_tag_with_gemini("")
        self.assertEqual(result["category"], "Other")
        self.assertEqual(result["summary"], "No summary")


class TestGetEmbedding(unittest.TestCase):
    """Tests for embedding generation (mocked)."""
    
    @patch.object(main, 'gemini_client')
    def test_get_embedding_returns_list(self, mock_client):
        """Test that get_embedding returns a list of floats."""
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 768
        mock_result = MagicMock()
        mock_result.embeddings = [mock_embedding]
        mock_client.models.embed_content.return_value = mock_result
        
        result = main.get_embedding_gemini("Test text")
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 768)
        self.assertTrue(all(isinstance(x, float) for x in result))
    
    @patch.object(main, 'gemini_client')
    def test_get_embedding_handles_error(self, mock_client):
        """Test fallback on API error."""
        mock_client.models.embed_content.side_effect = Exception("API Error")
        
        result = main.get_embedding_gemini("Test text")
        
        # Should return n-gram based fallback with 768 dimensions (matching Gemini)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 768)
    
    def test_get_embedding_empty_text(self):
        """Test embedding with empty text."""
        result = main.get_embedding_gemini("")
        self.assertEqual(result, [])


class TestProcessCaptureObject(unittest.TestCase):
    """Tests for the main processing pipeline."""
    
    @patch.object(main, 'store_enriched_note')
    @patch.object(main, 'compute_related_with_scores')
    @patch.object(main, 'get_embedding_gemini')
    @patch.object(main, 'classify_and_tag_with_gemini')
    @patch.object(main, 'fetch_url_text')
    def test_process_basic_note(self, mock_fetch, mock_classify, mock_embed, mock_related, mock_store):
        """Test processing a basic note."""
        mock_classify.return_value = {
            "category": "Idea",
            "tags": ["test"],
            "entities": [],
            "summary": "Test summary"
        }
        mock_embed.return_value = [0.1] * 128
        mock_related.return_value = []
        
        obj = {
            "id": "test-id",
            "timestamp": main.now_iso(),
            "content": "Test content"
        }
        
        result = main.process_capture_object(obj, store=True)
        
        self.assertEqual(result["id"], "test-id")
        self.assertEqual(result["category"], "Idea")
        self.assertEqual(result["summary"], "Test summary")
        mock_store.assert_called_once()
    
    @patch.object(main, 'store_enriched_note')
    @patch.object(main, 'compute_related_with_scores')
    @patch.object(main, 'get_embedding_gemini')
    @patch.object(main, 'classify_and_tag_with_gemini')
    @patch.object(main, 'fetch_url_text')
    def test_process_url_note(self, mock_fetch, mock_classify, mock_embed, mock_related, mock_store):
        """Test processing a URL note fetches content."""
        mock_fetch.return_value = '{"title": "Test Page", "text": "Page content"}'
        mock_classify.return_value = {
            "category": "Link",
            "tags": [],
            "entities": [],
            "summary": "A webpage"
        }
        mock_embed.return_value = [0.1] * 128
        mock_related.return_value = []
        
        obj = {
            "id": "url-note",
            "timestamp": main.now_iso(),
            "content": "https://example.com"
        }
        
        result = main.process_capture_object(obj, store=True)
        
        mock_fetch.assert_called_once_with("https://example.com")
        self.assertEqual(result["category"], "Link")
    
    @patch.object(main, 'store_enriched_note')
    @patch.object(main, 'compute_related_with_scores')
    @patch.object(main, 'get_embedding_gemini')
    @patch.object(main, 'classify_and_tag_with_gemini')
    def test_process_generates_id_if_missing(self, mock_classify, mock_embed, mock_related, mock_store):
        """Test that missing ID is auto-generated."""
        mock_classify.return_value = {
            "category": "Other",
            "tags": [],
            "entities": [],
            "summary": "Test"
        }
        mock_embed.return_value = []
        mock_related.return_value = []
        
        obj = {"content": "No ID provided"}
        
        result = main.process_capture_object(obj, store=False)
        
        self.assertIsNotNone(result["id"])
        self.assertTrue(len(result["id"]) > 0)


class TestComputeRelated(unittest.TestCase):
    """Tests for computing related notes."""
    
    @patch.object(main, 'read_all_embeddings_from_db')
    def test_compute_related_empty_db(self, mock_read):
        """Test with empty database."""
        mock_read.return_value = []
        
        result = main.compute_related_with_scores([0.1] * 128, top_k=5)
        
        self.assertEqual(result, [])
    
    @patch.object(main, 'read_all_embeddings_from_db')
    def test_compute_related_excludes_self(self, mock_read):
        """Test that note is excluded from its own related list."""
        # The function looks for "emb" key
        mock_read.return_value = [
            {"id": "self", "emb": [1.0] * 128},
            {"id": "other", "emb": [0.99] * 128},
        ]
        
        result = main.compute_related_with_scores(
            [1.0] * 128,
            top_k=5,
            exclude_id="self"
        )
        
        ids = [r["id"] for r in result]
        self.assertNotIn("self", ids)
    
    @patch.object(main, 'read_all_embeddings_from_db')
    def test_compute_related_respects_threshold(self, mock_read):
        """Test that low-similarity notes are excluded."""
        # Use vectors that will have high/low similarity when normalized
        # Similar: same direction as query
        # Dissimilar: opposite direction
        mock_read.return_value = [
            {"id": "similar", "emb": [1.0] * 128},  # Same direction, high similarity
            {"id": "dissimilar", "emb": [-1.0] * 128},  # Opposite direction, negative similarity
        ]
        
        result = main.compute_related_with_scores([1.0] * 128, top_k=10)
        
        ids = [r["id"] for r in result]
        # "similar" should be included (cosine similarity = 1.0)
        self.assertIn("similar", ids)
        # "dissimilar" should be filtered out (cosine similarity = -1.0, below threshold)
        self.assertNotIn("dissimilar", ids)


class TestBuildOnLineSummary(unittest.TestCase):
    """Tests for one-line summary generation."""
    
    def test_empty_notes(self):
        """Test with no notes."""
        result = main.build_one_line_summary([])
        self.assertEqual(result, "No notes.")
    
    def test_single_note(self):
        """Test with a single note."""
        notes = [{
            "timestamp": main.now_iso(),
            "category": "Idea",
            "summary": "A great idea"
        }]
        result = main.build_one_line_summary(notes)
        self.assertIn("1 notes", result)
        self.assertIn("Idea", result)
    
    def test_multiple_notes(self):
        """Test with multiple notes of different categories."""
        notes = [
            {"timestamp": "2024-01-03T00:00:00Z", "category": "Idea", "summary": "Idea 1"},
            {"timestamp": "2024-01-02T00:00:00Z", "category": "To-Do", "summary": "Task 1"},
            {"timestamp": "2024-01-01T00:00:00Z", "category": "Idea", "summary": "Idea 2"},
        ]
        result = main.build_one_line_summary(notes)
        self.assertIn("3 notes", result)
        # Should focus on most common category
        self.assertIn("Idea", result)


class TestReadableRelatedList(unittest.TestCase):
    """Tests for readable related list formatting."""
    
    @patch.object(main, 'fetch_note_by_id_safe')
    def test_formats_related_with_scores(self, mock_fetch):
        """Test formatting of related notes with scores."""
        mock_fetch.return_value = {
            "summary": "Related note summary",
            "category": "Idea",
            "content": "Full content"
        }
        
        related = [
            {"id": "note-1", "score": 0.85},
            {"id": "note-2", "score": 0.72},
        ]
        
        result = main.readable_related_list(related, max_items=2)
        
        self.assertEqual(len(result), 2)
        self.assertIn("0.85", result[0])
        self.assertIn("Idea", result[0])
    
    @patch.object(main, 'fetch_note_by_id_safe')
    def test_handles_missing_notes(self, mock_fetch):
        """Test handling when related note is missing."""
        mock_fetch.return_value = None
        
        related = [{"id": "missing", "score": 0.5}]
        
        result = main.readable_related_list(related)
        
        self.assertIn("<missing>", result[0])


class TestPipelineProcessor(unittest.TestCase):
    """Tests for the pipeline processor thread."""
    
    def setUp(self):
        """Create temporary input stream."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_stream = os.path.join(self.temp_dir, "test_stream.jsonl")
    
    def tearDown(self):
        """Clean up."""
        try:
            os.remove(self.test_stream)
            os.rmdir(self.temp_dir)
        except:
            pass
    
    def test_processor_creates_file_if_missing(self):
        """Test that processor creates input file if it doesn't exist."""
        processor = main.PipelineProcessor(self.test_stream)
        self.assertTrue(os.path.exists(self.test_stream))
    
    @patch.object(main, 'USE_PATHWAY_STREAM', False)
    @patch.object(main, 'PATHWAY_IMPORTED', False)
    def test_processor_can_start_and_stop(self):
        """Test processor thread lifecycle."""
        processor = main.PipelineProcessor(self.test_stream)
        # Ensure pathway is disabled so it uses polling
        processor.using_pathway_stream = False
        processor.start()
        time.sleep(0.2)  # Give it time to start
        self.assertTrue(processor.is_alive())
        processor.stop()
        processor.join(timeout=3.0)
        self.assertFalse(processor.is_alive())


class TestFormatNoteBrief(unittest.TestCase):
    """Tests for note brief formatting."""
    
    @patch.object(main, 'fetch_note_by_id_safe')
    def test_formats_complete_note(self, mock_fetch):
        """Test formatting a complete note."""
        mock_fetch.return_value = {"summary": "Related summary", "category": "Idea"}
        
        note = {
            "timestamp": "2024-01-15T10:30:00Z",
            "category": "To-Do",
            "summary": "Complete the task",
            "content": "Full content of the note that might be longer",
            "tags": ["urgent", "work", "project"],
            "related": [{"id": "rel-1", "score": 0.8}]
        }
        
        result = main.format_note_brief(note)
        
        self.assertIn("2024-01-15 10:30:00", result)
        self.assertIn("To-Do", result)
        self.assertIn("Complete the task", result)
        self.assertIn("urgent", result)


class TestFetchUrlText(unittest.TestCase):
    """Tests for URL content fetching."""
    
    @patch.object(main, 'SCRAPING_AVAILABLE', False)
    def test_returns_empty_when_scraping_unavailable(self):
        """Test that empty string is returned when scraping is disabled."""
        result = main.fetch_url_text("https://example.com")
        self.assertEqual(result, "")
    
    @patch.object(main, 'SCRAPING_AVAILABLE', True)
    @patch('main.requests')
    def test_fetches_html_content(self, mock_requests):
        """Test fetching and parsing HTML."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = "<html><head><title>Test</title></head><body><p>Content</p></body></html>"
        mock_requests.get.return_value = mock_response
        
        result = main.fetch_url_text("https://example.com")
        
        self.assertIn("Test", result)
        self.assertIn("Content", result)
    
    @patch.object(main, 'SCRAPING_AVAILABLE', True)
    @patch('main.requests')
    def test_handles_non_200_status(self, mock_requests):
        """Test handling of non-200 HTTP status."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_requests.get.return_value = mock_response
        
        result = main.fetch_url_text("https://example.com/notfound")
        
        self.assertEqual(result, "")


class TestBuildFullNotesSummary(unittest.TestCase):
    """Tests for full notes summary generation."""
    
    def test_empty_notes(self):
        """Test with no notes."""
        result = main.build_full_notes_summary_with_gemini([])
        self.assertEqual(result, "No notes available.")
    
    @patch.object(main, 'gemini_client')
    def test_generates_structured_summary(self, mock_client):
        """Test that summary is generated with Gemini."""
        summary_text = """
1) Top actionable To-Dos
- Task 1 (high)
- Task 2 (medium)

2) Project ideas
- Project A

3) Quick ideas
- Idea 1

4) Follow-ups
- Person X

5) Overall summary
A collection of notes.
"""
        # Mock the Gemini response
        mock_response = MagicMock()
        mock_response.text = summary_text
        mock_client.models.generate_content.return_value = mock_response
        
        notes = [
            {"timestamp": main.now_iso(), "category": "To-Do", "tags": [], "summary": "Do this", "content": "Task"},
        ]
        
        result = main.build_full_notes_summary_with_gemini(notes)
        
        self.assertIn("To-Dos", result)
        mock_client.models.generate_content.assert_called_once()


class TestDatabaseMigration(unittest.TestCase):
    """Tests for database migration functionality."""
    
    def test_init_creates_table_and_columns(self):
        """Test that init creates table with all columns."""
        temp_dir = tempfile.mkdtemp()
        test_db = os.path.join(temp_dir, "test_migration.db")
        
        original = main.DB_FILE
        main.DB_FILE = test_db
        
        try:
            conn = main.init_db_and_migrate()
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(notes)")
            cols = [row[1] for row in cursor.fetchall()]
            
            expected_cols = ["id", "timestamp", "content", "scraped_content", 
                           "category", "tags", "entities", "embedding", 
                           "related", "summary"]
            for col in expected_cols:
                self.assertIn(col, cols, f"Missing column: {col}")
            
            conn.close()
        finally:
            main.DB_FILE = original
            try:
                os.remove(test_db)
                os.rmdir(temp_dir)
            except:
                pass


class TestEnrichMissingNotes(unittest.TestCase):
    """Tests for the enrich_missing_notes utility."""
    
    @patch.object(main, 'process_capture_object')
    @patch.object(main, 'fetch_note_by_id_safe')
    def test_enriches_notes_missing_embeddings(self, mock_fetch, mock_process):
        """Test that notes without embeddings are enriched."""
        mock_fetch.return_value = {
            "id": "note-1",
            "timestamp": main.now_iso(),
            "content": "Test",
            "embedding": None,
            "summary": "Summary"
        }
        
        # Mock db to return one note ID
        temp_dir = tempfile.mkdtemp()
        test_db = os.path.join(temp_dir, "test.db")
        conn = sqlite3.connect(test_db)
        conn.execute("""
            CREATE TABLE notes (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                content TEXT,
                embedding BLOB,
                summary TEXT
            )
        """)
        conn.execute("INSERT INTO notes (id, timestamp, content) VALUES (?, ?, ?)",
                    ("note-1", main.now_iso(), "Test"))
        conn.commit()
        
        original_conn = main.db_conn
        original_lock = main.db_lock
        main.db_conn = conn
        main.db_lock = threading.Lock()
        
        try:
            result = main.enrich_missing_notes(batch_sleep=0)
            # Should have processed the note
            self.assertEqual(result, 1)
            mock_process.assert_called_once()
        finally:
            main.db_conn = original_conn
            main.db_lock = original_lock
            conn.close()
            os.remove(test_db)
            os.rmdir(temp_dir)


class TestFetchNoteByIdSafe(unittest.TestCase):
    """Tests for fetch_note_by_id_safe."""
    
    def setUp(self):
        """Create test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test.db")
        self.conn = sqlite3.connect(self.test_db)
        self.conn.execute("""
            CREATE TABLE notes (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                content TEXT,
                category TEXT,
                tags TEXT,
                entities TEXT,
                related TEXT,
                summary TEXT,
                embedding BLOB
            )
        """)
        # Insert test note
        self.conn.execute("""
            INSERT INTO notes (id, timestamp, content, category, tags, entities, related, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("test-id", "2024-01-01T00:00:00Z", "Content", "Idea", 
              '["tag1", "tag2"]', '["Entity1"]', '[]', "Summary"))
        self.conn.commit()
        
        self._original_conn = main.db_conn
        self._original_lock = main.db_lock
        main.db_conn = self.conn
        main.db_lock = threading.Lock()
    
    def tearDown(self):
        """Clean up."""
        main.db_conn = self._original_conn
        main.db_lock = self._original_lock
        self.conn.close()
        os.remove(self.test_db)
        os.rmdir(self.temp_dir)
    
    def test_fetch_existing_note(self):
        """Test fetching an existing note."""
        result = main.fetch_note_by_id_safe("test-id")
        
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "test-id")
        self.assertEqual(result["category"], "Idea")
        self.assertEqual(result["tags"], ["tag1", "tag2"])
        self.assertEqual(result["entities"], ["Entity1"])
    
    def test_fetch_nonexistent_note(self):
        """Test fetching a non-existent note."""
        result = main.fetch_note_by_id_safe("nonexistent")
        self.assertIsNone(result)
    
    def test_fetch_note_default_summary(self):
        """Test that missing summary defaults to 'No summary'."""
        self.conn.execute("""
            INSERT INTO notes (id, timestamp, content, category, tags, entities, related, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("no-summary", "2024-01-01T00:00:00Z", "Content", "Other", '[]', '[]', '[]', None))
        self.conn.commit()
        
        result = main.fetch_note_by_id_safe("no-summary")
        self.assertEqual(result["summary"], "No summary")


class TestFetchAllNotesDicts(unittest.TestCase):
    """Tests for fetch_all_notes_dicts."""
    
    def setUp(self):
        """Create test database with multiple notes."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test.db")
        self.conn = sqlite3.connect(self.test_db)
        self.conn.execute("""
            CREATE TABLE notes (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                content TEXT,
                category TEXT,
                tags TEXT,
                entities TEXT,
                related TEXT,
                summary TEXT,
                embedding BLOB
            )
        """)
        # Insert test notes
        for i in range(3):
            self.conn.execute("""
                INSERT INTO notes (id, timestamp, content, category, tags, entities, related, summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (f"note-{i}", f"2024-01-0{i+1}T00:00:00Z", f"Content {i}", "Idea", 
                  '[]', '[]', '[]', f"Summary {i}"))
        self.conn.commit()
        
        self._original_conn = main.db_conn
        self._original_lock = main.db_lock
        main.db_conn = self.conn
        main.db_lock = threading.Lock()
    
    def tearDown(self):
        """Clean up."""
        main.db_conn = self._original_conn
        main.db_lock = self._original_lock
        self.conn.close()
        os.remove(self.test_db)
        os.rmdir(self.temp_dir)
    
    def test_fetches_all_notes(self):
        """Test fetching all notes."""
        result = main.fetch_all_notes_dicts()
        
        self.assertEqual(len(result), 3)
    
    def test_notes_sorted_newest_first(self):
        """Test that notes are sorted by timestamp descending."""
        result = main.fetch_all_notes_dicts()
        
        timestamps = [n["timestamp"] for n in result]
        self.assertEqual(timestamps, sorted(timestamps, reverse=True))


class TestConstants(unittest.TestCase):
    """Tests for module constants."""
    
    def test_relation_min_score_is_float(self):
        """Test RELATION_MIN_SCORE is a valid float."""
        self.assertIsInstance(main.RELATION_MIN_SCORE, (int, float))
        self.assertGreater(main.RELATION_MIN_SCORE, 0)
        self.assertLess(main.RELATION_MIN_SCORE, 1)
    
    def test_notes_page_size_defined(self):
        """Test NOTES_PAGE_SIZE is defined."""
        self.assertTrue(hasattr(main, 'NOTES_PAGE_SIZE'))
        self.assertGreater(main.NOTES_PAGE_SIZE, 0)


class TestDuplicateFunctionDefinitions(unittest.TestCase):
    """Test to identify duplicate function definitions in the code."""
    
    def test_compute_related_with_scores_consistency(self):
        """Test that compute_related_with_scores works correctly.
        
        This tests the function behavior regardless of which definition is used.
        """
        # The function should work with embeddings
        with patch.object(main, 'read_all_embeddings_from_db') as mock_read:
            mock_read.return_value = [
                {"id": "note-1", "emb": [0.5] * 10},
                {"id": "note-2", "emb": [0.9] * 10},
            ]
            
            result = main.compute_related_with_scores(
                embedding=[1.0] * 10,
                top_k=5,
                exclude_id=None
            )
            
            # Should return list of dicts with id and score
            self.assertIsInstance(result, list)
            if result:
                self.assertIn("id", result[0])
                self.assertIn("score", result[0])


class TestRecomputeLinks(unittest.TestCase):
    """Tests for recompute_links functionality."""
    
    def setUp(self):
        """Create test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test.db")
        self.conn = sqlite3.connect(self.test_db)
        self.conn.execute("""
            CREATE TABLE notes (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                content TEXT,
                category TEXT,
                tags TEXT,
                entities TEXT,
                related TEXT,
                summary TEXT,
                embedding BLOB
            )
        """)
        
        self._original_conn = main.db_conn
        self._original_lock = main.db_lock
        main.db_conn = self.conn
        main.db_lock = threading.Lock()
    
    def tearDown(self):
        """Clean up."""
        main.db_conn = self._original_conn
        main.db_lock = self._original_lock
        self.conn.close()
        os.remove(self.test_db)
        os.rmdir(self.temp_dir)
    
    @patch.object(main, 'process_capture_object')
    def test_recompute_with_no_notes(self, mock_process):
        """Test recompute_links with empty database."""
        result = main.recompute_links()
        self.assertEqual(result, 0)
        mock_process.assert_not_called()


class TestComputeRelatedWithScores(unittest.TestCase):
    """Tests for the compute_related_with_scores function."""
    
    def setUp(self):
        """Create a temporary database for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_kb.db")
        self._original_db = main.DB_FILE
        self._original_conn = main.db_conn
        self._original_lock = main.db_lock
        main.DB_FILE = self.test_db
        main.db_conn = sqlite3.connect(self.test_db, check_same_thread=False)
        main.db_lock = threading.Lock()
        
        # Create the notes table
        c = main.db_conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS notes (
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
            )
        """)
        main.db_conn.commit()
    
    def tearDown(self):
        """Clean up."""
        main.db_conn = self._original_conn
        main.db_lock = self._original_lock
        main.DB_FILE = self._original_db
        try:
            os.remove(self.test_db)
            os.rmdir(self.temp_dir)
        except Exception:
            pass
    
    def test_empty_database(self):
        """Test compute_related with no notes in database."""
        embedding = [0.1] * 1536
        result = main.compute_related_with_scores(embedding)
        self.assertEqual(result, [])
    
    def test_empty_embedding(self):
        """Test compute_related with empty embedding."""
        result = main.compute_related_with_scores([])
        self.assertEqual(result, [])
    
    def test_excludes_self(self):
        """Test that the current note is excluded from results."""
        # Insert a test note
        emb = [0.5] * 1536
        emb_bytes = main.get_embedding_bytes(emb)
        c = main.db_conn.cursor()
        c.execute(
            "INSERT INTO notes (id, timestamp, content, category, tags, embedding) VALUES (?, ?, ?, ?, ?, ?)",
            ("note1", "2025-01-01T00:00:00Z", "Test", "Other", "[]", emb_bytes)
        )
        main.db_conn.commit()
        
        # Query with same embedding but exclude_id set
        result = main.compute_related_with_scores(emb, exclude_id="note1")
        self.assertEqual(len(result), 0)
    
    def test_finds_similar_notes(self):
        """Test that similar notes are found based on embedding similarity."""
        # Insert two notes with similar embeddings
        emb1 = [0.5] * 1536
        emb2 = [0.5] * 1536  # Identical
        emb2[0] = 0.51  # Slightly different
        
        c = main.db_conn.cursor()
        c.execute(
            "INSERT INTO notes (id, timestamp, content, category, tags, embedding) VALUES (?, ?, ?, ?, ?, ?)",
            ("note1", "2025-01-01T00:00:00Z", "Test 1", "Other", "[]", main.get_embedding_bytes(emb1))
        )
        c.execute(
            "INSERT INTO notes (id, timestamp, content, category, tags, embedding) VALUES (?, ?, ?, ?, ?, ?)",
            ("note2", "2025-01-02T00:00:00Z", "Test 2", "Other", "[]", main.get_embedding_bytes(emb2))
        )
        main.db_conn.commit()
        
        # Query for related notes
        result = main.compute_related_with_scores(emb1, exclude_id="note1")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "note2")
        # Hybrid scoring: embedding contributes 60%, so ~0.6 for very similar embeddings
        self.assertGreater(result[0]["score"], 0.55)
    
    def test_category_affinity_boost(self):
        """Test that related categories (To-Do and Idea) get affinity boost."""
        # Create base embedding
        base_emb = [0.5] * 1536
        
        # Create two slightly different embeddings
        emb1 = base_emb.copy()
        emb1[0] = 0.52
        emb2 = base_emb.copy()
        emb2[0] = 0.52
        
        c = main.db_conn.cursor()
        c.execute(
            "INSERT INTO notes (id, timestamp, content, category, tags, entities, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("note1", "2025-01-01T00:00:00Z", "Test 1", "To-Do", "[]", "[]", main.get_embedding_bytes(emb1))
        )
        c.execute(
            "INSERT INTO notes (id, timestamp, content, category, tags, entities, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("note2", "2025-01-02T00:00:00Z", "Test 2", "Idea", "[]", "[]", main.get_embedding_bytes(emb2))
        )
        main.db_conn.commit()
        
        # Query with To-Do category - should get affinity boost with Idea
        result = main.compute_related_with_scores(
            base_emb, 
            exclude_id=None,
            current_category="To-Do"
        )
        
        # Both notes should be found
        scores = {r["id"]: r["score"] for r in result}
        self.assertIn("note1", scores)
        self.assertIn("note2", scores)
        # note2 (Idea) should get affinity boost from To-Do query
        # note1 (To-Do) gets same-category boost (0.05)
        # But To-Do -> Idea affinity is 0.15, which is higher
        # So note2 should have higher score
        self.assertGreater(scores["note2"], scores["note1"])
    
    def test_entity_overlap_boost(self):
        """Test that shared entities boost relation scores."""
        base_emb = [0.5] * 1536
        emb1 = base_emb.copy()
        emb1[0] = 0.52
        
        c = main.db_conn.cursor()
        # Note with entity "John"
        c.execute(
            "INSERT INTO notes (id, timestamp, content, category, tags, entities, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("note1", "2025-01-01T00:00:00Z", "Meeting with John", "To-Do", "[]", '["John"]', main.get_embedding_bytes(emb1))
        )
        main.db_conn.commit()
        
        # Query with same entity
        result = main.compute_related_with_scores(
            base_emb, 
            exclude_id=None,
            current_entities=["John"]
        )
        
        self.assertEqual(len(result), 1)
        # Should get entity overlap boost
        self.assertGreater(result[0]["score"], 0.6)


class TestFallbackEmbedding(unittest.TestCase):
    """Tests for the fallback embedding generation."""
    
    @patch.object(main, 'gemini_client')
    def test_fallback_produces_consistent_embeddings(self, mock_client):
        """Test that same text produces same embedding."""
        mock_client.models.embed_content.side_effect = Exception("API Error")
        
        emb1 = main.get_embedding_gemini("Hello world")
        emb2 = main.get_embedding_gemini("Hello world")
        
        self.assertEqual(emb1, emb2)
    
    @patch.object(main, 'gemini_client')
    def test_fallback_similar_text_similar_embedding(self, mock_client):
        """Test that similar text produces similar embeddings."""
        mock_client.models.embed_content.side_effect = Exception("API Error")
        
        emb1 = main.get_embedding_gemini("Buy groceries from store")
        emb2 = main.get_embedding_gemini("Get groceries from supermarket")
        emb3 = main.get_embedding_gemini("Write Python code")
        
        sim_similar = main.cosine_similarity(emb1, emb2)
        sim_different = main.cosine_similarity(emb1, emb3)
        
        # Similar text should have higher similarity than different text
        self.assertGreater(sim_similar, sim_different)
    
    @patch.object(main, 'gemini_client')
    def test_fallback_embedding_is_normalized(self, mock_client):
        """Test that fallback embedding is normalized."""
        mock_client.models.embed_content.side_effect = Exception("API Error")
        
        emb = main.get_embedding_gemini("Test text")
        
        # Check magnitude is approximately 1.0
        import math
        magnitude = math.sqrt(sum(x*x for x in emb))
        self.assertAlmostEqual(magnitude, 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
