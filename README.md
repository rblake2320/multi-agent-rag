# Multi-Agent RAG System

A Python-based multi-agent Retrieval-Augmented Generation (RAG) system built with LangChain, ChromaDB, and Llama models. This system intelligently routes queries to domain-specific knowledge bases and provides accurate, context-aware responses.

## Features

- **Multi-Domain Support**: Separate vector stores for legal, code, and finance domains
- **Intelligent Routing**: Automatically determines the appropriate domain for each query
- **Document Ingestion**: Supports multiple file formats (PDF, DOCX, TXT, CSV)
- **Configurable Models**: Flexible LLM configuration via environment variables
- **Agent Architecture**: Modular agent design for extensibility

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rblake2320/multi-agent-rag.git
cd multi-agent-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your model path (optional):
```bash
export LLAMA_MODEL_PATH="/path/to/your/model.gguf"
```

## Quick Start

Try the example script to see the system in action:

```bash
python example.py
```

This will create sample documents and demonstrate the ingestion and querying workflow (note: requires a model file for full functionality).

## Usage

### Document Ingestion

Ingest documents into domain-specific vector stores:

```bash
python ingest.py --domain legal_docs --path /path/to/legal/documents
python ingest.py --domain code --path /path/to/code/files
python ingest.py --domain finance --path /path/to/financial/docs
```

### Query Routing

Ask questions that will be automatically routed to the appropriate domain:

```bash
python router.py "What are the tax implications of stock options?"
python router.py "How do I implement a binary search in Python?"
python router.py "What is the statute of limitations for contract disputes?"
```

### Using as a Library

```python
from agents import ResearchAgent
from router import answer

# Initialize research agent
agent = ResearchAgent(model_path="path/to/model.gguf")

# Ingest documents
result = agent.run("ingest", "legal_docs", "/path/to/documents")

# Query the system
response, domain = answer("Your question here")
print(f"[{domain}] {response}")
```

## Project Structure

```
multi-agent-rag/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py      # Abstract base agent class
│   └── research_agent.py  # Document ingestion agent
├── data/                  # Sample data directory
├── vector_stores/         # ChromaDB vector stores (generated)
├── ingest.py             # Document ingestion script
├── router.py             # Query routing and answering
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Configuration

### Environment Variables

- `LLAMA_MODEL_PATH`: Path to your GGUF model file (default: `models/llama-3.1-70b-q2_k.gguf`)

### Supported File Formats

- PDF files (`.pdf`)
- Word documents (`.docx`)
- Text files (`.txt`)
- CSV files (`.csv`)

## Requirements

- Python 3.8+
- LLM model in GGUF format (e.g., Llama models)
- Sufficient disk space for vector stores

## Dependencies

- `langchain` - LLM framework
- `langchain-community` - Community integrations
- `langchain-chroma` - ChromaDB integration
- `chromadb` - Vector database
- `llama-cpp-python` - Llama model inference
- `sentence-transformers` - Text embeddings
- `unstructured` - Document parsing
- `python-dotenv` - Environment variable management

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Troubleshooting

### Model Not Found Error

If you see "Model file not found" errors:

1. Download a compatible GGUF model (e.g., from Hugging Face)
2. Set the `LLAMA_MODEL_PATH` environment variable
3. Or place your model at `models/llama-3.1-70b-q2_k.gguf`

### Import Errors

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

For development dependencies:
```bash
pip install -r requirements-dev.txt
```

### Network/Offline Usage

If you encounter network issues when downloading models:
- Download embedding models manually and set up offline mode
- Use local model files instead of downloading from Hugging Face
- Check firewall settings if downloads fail

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=term-missing

# Run specific test file
pytest tests/test_ingest.py -v
```

## Roadmap

- [ ] Add more document loaders (HTML, Markdown, etc.)
- [ ] Implement caching for better performance
- [ ] Add web interface for easier interaction
- [ ] Support for multiple embedding models
- [ ] Enhanced error handling and logging
- [ ] Docker support for easy deployment