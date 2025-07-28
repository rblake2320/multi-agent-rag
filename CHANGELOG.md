# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive test suite with pytest
- GitHub Actions CI/CD pipeline
- Type hints throughout codebase
- Contributor guidelines (CONTRIBUTING.md)
- Security policy (SECURITY.md)
- Example usage script
- Configuration file support
- Professional project structure with pyproject.toml

### Changed
- Updated all LangChain imports to use new module structure
- Made model path configurable via environment variables
- Improved error handling and graceful degradation
- Enhanced documentation with detailed README

### Fixed
- Deprecated LangChain import warnings
- Hard-coded model path issues
- Missing package initialization files
- Base64 encoded placeholder content

## [0.1.0] - 2024-01-XX

### Added
- Initial multi-agent RAG system
- Document ingestion with multiple format support
- Intelligent query routing between domains
- BaseAgent and ResearchAgent implementations
- ChromaDB vector store integration
- LLama model support via llama-cpp-python

### Features
- Support for PDF, DOCX, TXT, and CSV documents
- Domain-specific vector stores (legal, code, finance)
- Configurable embedding models
- Command-line interfaces for ingestion and querying

## Development Notes

### Dependencies Updated
- langchain -> langchain + langchain-community + langchain-chroma
- Added comprehensive development dependencies
- Pinned versions for stability

### Code Quality Improvements
- Added type hints for better IDE support
- Improved docstrings and documentation
- Consistent error handling patterns
- Professional project structure

### Testing Infrastructure
- Unit tests with mocking for external dependencies
- Coverage reporting
- Continuous integration with multiple Python versions
- Security scanning and dependency checks