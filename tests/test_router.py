"""Tests for the router module."""

import pytest
import os
from unittest.mock import patch, Mock
import router


class TestRouter:
    """Test cases for query routing functionality."""

    def test_initialize_llm_with_missing_model(self):
        """Test LLM initialization with missing model file."""
        with patch('os.path.exists', return_value=False):
            llm = router.initialize_llm()
            assert llm is None

    @patch('router.initialize_llm')
    def test_initialize_llm_with_existing_model(self, mock_init_llm):
        """Test LLM initialization with existing model file."""
        mock_llm_instance = Mock()
        mock_init_llm.return_value = mock_llm_instance
        
        llm = router.initialize_llm()
        
        assert llm == mock_llm_instance

    def test_initialize_llm_with_exception(self):
        """Test LLM initialization when LlamaCpp raises exception."""
        with patch('os.path.exists', return_value=True), \
             patch('langchain_community.llms.LlamaCpp', side_effect=Exception("Model error")), \
             patch('builtins.print') as mock_print:
            
            llm = router.initialize_llm()
            
            assert llm is None
            mock_print.assert_called()

    def test_initialize_router_with_no_llm(self):
        """Test router initialization when LLM is None."""
        with patch('router.LLM', None):
            router_instance = router.initialize_router()
            assert router_instance is None

    def test_initialize_router_with_llm(self):
        """Test router initialization with valid LLM."""
        mock_llm = Mock()
        with patch('router.LLM', mock_llm), \
             patch('router.RouterChain') as mock_router_chain:
            mock_router_instance = Mock()
            mock_router_chain.from_defaults.return_value = mock_router_instance
            
            router_instance = router.initialize_router()
            
            assert router_instance == mock_router_instance
            mock_router_chain.from_defaults.assert_called_once()

    @patch('router.build_retriever')
    def test_build_retriever(self, mock_build_retriever):
        """Test building retriever for a domain."""
        mock_retriever = Mock()
        mock_build_retriever.return_value = mock_retriever
        
        retriever = router.build_retriever("legal")
        
        assert retriever == mock_retriever

    def test_answer_with_no_llm_or_router(self):
        """Test answer function when LLM or router is None."""
        with patch('router.LLM', None), \
             patch('router.router', None):
            
            response, domain = router.answer("test question")
            
            assert "Error" in response
            assert domain == "error"

    def test_answer_with_valid_setup(self):
        """Test answer function with valid LLM and router setup."""
        mock_llm = Mock()
        mock_router = Mock()
        mock_router.run.return_value = "legal"
        
        with patch('router.LLM', mock_llm), \
             patch('router.router', mock_router), \
             patch('router.RetrievalQA') as mock_qa, \
             patch('router.retrievers') as mock_retrievers:
            
            mock_qa_instance = Mock()
            mock_qa_instance.run.return_value = "Test response"
            mock_qa.from_chain_type.return_value = mock_qa_instance
            
            mock_retriever = Mock()
            mock_retrievers.__getitem__.return_value = mock_retriever
            
            response, domain = router.answer("test question")
            
            assert response == "Test response"
            assert domain == "legal"
            mock_router.run.assert_called_once_with("test question")
            mock_qa.from_chain_type.assert_called_once_with(
                llm=mock_llm, 
                retriever=mock_retriever
            )

    def test_domains_configuration(self):
        """Test that domains are properly configured."""
        expected_domains = {"legal", "code", "finance"}
        assert set(router.DOMAINS.keys()) == expected_domains
        
        # Ensure all domain values end with '_chroma'
        for domain_key, domain_value in router.DOMAINS.items():
            assert domain_value.endswith("_chroma")

    def test_model_path_environment_variable(self):
        """Test that model path respects environment variable."""
        # Test default path
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            importlib.reload(router)
            assert router.MODEL_PATH == router.DEFAULT_MODEL_PATH
        
        # Test custom path
        custom_path = "/custom/model/path.gguf"
        with patch.dict(os.environ, {"LLAMA_MODEL_PATH": custom_path}):
            importlib.reload(router)
            assert router.MODEL_PATH == custom_path

    def test_vector_base_constant(self):
        """Test that vector base constant is properly defined."""
        assert router.VECTOR_BASE == "vector_stores"