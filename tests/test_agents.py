"""Tests for the agents module."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from agents.base_agent import BaseAgent
from agents.research_agent import ResearchAgent


class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""
    
    def run(self, *args, **kwargs):
        return "test_run_result"


class TestBaseAgent:
    """Test cases for BaseAgent class."""

    @patch('agents.base_agent.LlamaCpp')
    def test_base_agent_initialization(self, mock_llama):
        """Test BaseAgent initialization."""
        mock_llm = Mock()
        mock_llama.return_value = mock_llm
        
        agent = ConcreteAgent("test_model_path", n_ctx=2048)
        
        assert agent.llm == mock_llm
        mock_llama.assert_called_once_with(
            model_path="test_model_path", 
            n_ctx=2048
        )

    @patch('agents.base_agent.LlamaCpp')
    def test_base_agent_prompt_method(self, mock_llama):
        """Test BaseAgent _prompt method."""
        mock_llm = Mock()
        mock_llm.return_value = "LLM response"
        mock_llama.return_value = mock_llm
        
        agent = ConcreteAgent("test_model_path")
        response = agent._prompt("test prompt")
        
        assert response == "LLM response"
        mock_llm.assert_called_once_with("test prompt")

    def test_base_agent_run_abstract(self):
        """Test that BaseAgent is abstract and can't be instantiated directly."""
        with patch('agents.base_agent.LlamaCpp'):
            with pytest.raises(TypeError):
                BaseAgent("test_model_path")


class TestResearchAgent:
    """Test cases for ResearchAgent class."""

    @patch('langchain_community.embeddings.HuggingFaceEmbeddings')
    @patch('agents.research_agent.BaseAgent.__init__')
    def test_research_agent_initialization(self, mock_base_init, mock_embeddings):
        """Test ResearchAgent initialization."""
        mock_base_init.return_value = None
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        agent = ResearchAgent("test_model_path", vector_base="custom_base")
        
        assert agent.vector_base == Path("custom_base")
        assert agent.embedder == mock_embeddings_instance
        mock_embeddings.assert_called_once_with(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        mock_base_init.assert_called_once_with("test_model_path")

    @patch('langchain_community.embeddings.HuggingFaceEmbeddings')
    @patch('agents.research_agent.BaseAgent.__init__')
    def test_research_agent_default_vector_base(self, mock_base_init, mock_embeddings):
        """Test ResearchAgent with default vector base."""
        mock_base_init.return_value = None
        
        agent = ResearchAgent("test_model_path")
        
        assert agent.vector_base == Path("vector_stores")

    @patch('langchain_chroma.Chroma')
    @patch('langchain.text_splitter.RecursiveCharacterTextSplitter')
    @patch('langchain_community.embeddings.HuggingFaceEmbeddings')
    @patch('agents.research_agent.BaseAgent.__init__')
    def test_ingest_folder(self, mock_base_init, mock_embeddings, mock_splitter, mock_chroma, temp_dir):
        """Test the ingest_folder method."""
        mock_base_init.return_value = None
        
        # Create test files
        (temp_dir / "file1.txt").write_text("Content 1")
        (temp_dir / "file2.txt").write_text("Content 2")
        
        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_splitter_instance = Mock()
        mock_chunks = [Mock(), Mock()]
        mock_splitter_instance.create_documents.return_value = mock_chunks
        mock_splitter.return_value = mock_splitter_instance
        
        mock_chroma_instance = Mock()
        mock_chroma.from_documents.return_value = mock_chroma_instance
        
        agent = ResearchAgent("test_model_path")
        result = agent.ingest_folder("test_domain", temp_dir)
        
        # Verify the result
        assert "Ingested 2 chunks" in result
        assert "test_domain_chroma" in result
        
        # Verify method calls
        mock_splitter.assert_called_once_with(chunk_size=800, chunk_overlap=100)
        mock_splitter_instance.create_documents.assert_called_once()
        mock_chroma.from_documents.assert_called_once_with(
            mock_chunks, 
            mock_embeddings_instance, 
            persist_directory=str(agent.vector_base / "test_domain_chroma")
        )

    @patch('langchain_community.embeddings.HuggingFaceEmbeddings')
    @patch('agents.research_agent.BaseAgent.__init__')
    def test_run_with_ingest_command(self, mock_base_init, mock_embeddings, temp_dir):
        """Test the run method with ingest command."""
        mock_base_init.return_value = None
        
        with patch.object(ResearchAgent, 'ingest_folder') as mock_ingest:
            mock_ingest.return_value = "Ingestion successful"
            
            agent = ResearchAgent("test_model_path")
            result = agent.run("ingest", "test_domain", str(temp_dir))
            
            assert result == "Ingestion successful"
            mock_ingest.assert_called_once_with("test_domain", Path(str(temp_dir)))

    @patch('langchain_community.embeddings.HuggingFaceEmbeddings')
    @patch('agents.research_agent.BaseAgent.__init__')
    def test_run_with_unsupported_command(self, mock_base_init, mock_embeddings):
        """Test the run method with unsupported command."""
        mock_base_init.return_value = None
        
        agent = ResearchAgent("test_model_path")
        
        with pytest.raises(ValueError, match="Unsupported command"):
            agent.run("unsupported_command", "arg1", "arg2")

    @patch('langchain_chroma.Chroma')
    @patch('langchain.text_splitter.RecursiveCharacterTextSplitter')
    @patch('langchain_community.embeddings.HuggingFaceEmbeddings')
    @patch('agents.research_agent.BaseAgent.__init__')
    def test_ingest_folder_with_no_files(self, mock_base_init, mock_embeddings, mock_splitter, mock_chroma, temp_dir):
        """Test ingest_folder with empty directory."""
        mock_base_init.return_value = None
        
        mock_splitter_instance = Mock()
        mock_splitter_instance.create_documents.return_value = []
        mock_splitter.return_value = mock_splitter_instance
        
        agent = ResearchAgent("test_model_path")
        result = agent.ingest_folder("test_domain", temp_dir)
        
        assert "Ingested 0 chunks" in result