from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .base_agent import BaseAgent


class ResearchAgent(BaseAgent):
    """Ingests documents into domain-specific Chroma stores for other agents."""

    def __init__(self, model_path: str, vector_base: str = "vector_stores", **llm_kwargs: Any) -> None:
        """Initialize the research agent.
        
        Args:
            model_path: Path to the LLM model file
            vector_base: Base directory for vector stores
            **llm_kwargs: Additional arguments for LLM initialization
        """
        super().__init__(model_path, **llm_kwargs)
        self.vector_base = Path(vector_base)
        self.embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def ingest_folder(self, domain: str, folder: Path) -> str:
        """Ingest documents from a folder into a domain-specific vector store.
        
        Args:
            domain: Domain name for the vector store
            folder: Path to folder containing documents
            
        Returns:
            Status message indicating ingestion result
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = [Path(f).read_text(errors="ignore") for f in folder.rglob('*.*')]
        chunks = splitter.create_documents(docs)
        vs_path = self.vector_base / f"{domain}_chroma"
        vs_path.mkdir(parents=True, exist_ok=True)
        Chroma.from_documents(chunks, self.embedder, persist_directory=str(vs_path))
        return f"Ingested {len(chunks)} chunks → {vs_path}"

    def run(self, command: str, *args: Any) -> str:
        """Execute the agent with a specific command.
        
        Args:
            command: Command to execute
            *args: Arguments for the command
            
        Returns:
            Result of command execution
            
        Raises:
            ValueError: If command is not supported
        """
        if command == 'ingest':
            domain, folder = args
            return self.ingest_folder(domain, Path(folder))
        raise ValueError('Unsupported command')
