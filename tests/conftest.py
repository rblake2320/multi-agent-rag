"""Test configuration and fixtures."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file for testing."""
    text_file = temp_dir / "sample.txt"
    text_file.write_text("This is a sample document for testing.")
    return text_file


@pytest.fixture
def mock_llm():
    """Mock LLM for testing without requiring actual model files."""
    mock = Mock()
    mock.return_value = "Mocked LLM response"
    return mock


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing."""
    mock = Mock()
    mock.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock


@pytest.fixture
def mock_chroma():
    """Mock ChromaDB for testing."""
    mock = Mock()
    mock.from_documents.return_value = mock
    mock.as_retriever.return_value = Mock()
    return mock


@pytest.fixture(autouse=True)
def set_test_env():
    """Set environment variables for testing."""
    os.environ["LLAMA_MODEL_PATH"] = "/non/existent/path/for/testing"
    yield
    # Cleanup
    if "LLAMA_MODEL_PATH" in os.environ:
        del os.environ["LLAMA_MODEL_PATH"]