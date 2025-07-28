# Contributing to Multi-Agent RAG

Thank you for your interest in contributing to the Multi-Agent RAG project! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- A compatible LLM model in GGUF format (optional for development)

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/multi-agent-rag.git
   cd multi-agent-rag
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

5. Set up pre-commit hooks (optional but recommended):
   ```bash
   pre-commit install
   ```

## Development Workflow

### Creating a Branch

Create a new branch for your feature or bugfix:
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

### Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run these tools before committing:
```bash
# Format code
black .
isort .

# Check for issues
flake8 .
mypy . --ignore-missing-imports
```

### Testing

We use pytest for testing. Run tests with:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=term-missing

# Run specific test file
pytest tests/test_ingest.py
```

### Writing Tests

- Write tests for all new features and bug fixes
- Place tests in the `tests/` directory
- Use descriptive test names that explain what is being tested
- Mock external dependencies (LLMs, vector stores, etc.)
- Ensure tests are deterministic and don't require network access

Example test structure:
```python
def test_function_name_should_do_something():
    """Test that function_name does something specific."""
    # Arrange
    input_data = "test input"
    expected_output = "expected result"
    
    # Act
    result = function_name(input_data)
    
    # Assert
    assert result == expected_output
```

## Contribution Guidelines

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix existing issues
- **Features**: Add new functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code
- **Refactoring**: Improve code structure without changing functionality

### Reporting Issues

When reporting bugs or requesting features:

1. Check if the issue already exists
2. Use a clear, descriptive title
3. Provide detailed steps to reproduce (for bugs)
4. Include your environment details (Python version, OS, etc.)
5. Add relevant error messages or logs

### Pull Request Process

1. Ensure your code follows the style guidelines
2. Add or update tests for your changes
3. Update documentation if needed
4. Ensure all tests pass
5. Write a clear commit message describing your changes
6. Create a pull request with:
   - Clear title and description
   - Reference to any related issues
   - Screenshots if applicable (for UI changes)

### Commit Message Format

Use clear, concise commit messages:
```
feat: add support for PDF document ingestion
fix: handle missing model file gracefully
docs: update README with installation instructions
test: add tests for router functionality
```

## Code Organization

### Project Structure

```
multi-agent-rag/
├── agents/             # Agent implementations
│   ├── __init__.py
│   ├── base_agent.py   # Abstract base class
│   └── research_agent.py
├── tests/              # Test files
├── data/               # Sample data
├── vector_stores/      # Generated vector stores
├── ingest.py           # Document ingestion
├── router.py           # Query routing
└── requirements.txt    # Dependencies
```

### Adding New Agents

When adding new agent types:

1. Inherit from `BaseAgent`
2. Implement the `run` method
3. Add comprehensive tests
4. Update documentation
5. Add usage examples

### Adding New Document Loaders

To support new file formats:

1. Add the loader to `LOADER_MAP` in `ingest.py`
2. Ensure the loader is imported
3. Add tests for the new format
4. Update documentation

## Documentation

### Code Documentation

- Use clear, descriptive docstrings for all functions and classes
- Follow Google-style docstrings
- Include parameter and return type information
- Provide usage examples for complex functions

Example:
```python
def ingest_documents(domain: str, path: Path) -> str:
    """Ingest documents into a domain-specific vector store.
    
    Args:
        domain: The domain name for the vector store
        path: Path to directory containing documents
        
    Returns:
        Status message indicating number of chunks ingested
        
    Raises:
        ValueError: If domain name is invalid
        FileNotFoundError: If path doesn't exist
        
    Example:
        >>> ingest_documents("legal", Path("./legal_docs"))
        "Ingested 150 chunks into legal_chroma"
    """
```

### README Updates

Keep the README current with:
- Installation instructions
- Usage examples
- Configuration options
- Troubleshooting information

## Release Process

(For maintainers)

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release notes
4. Tag the release
5. Create GitHub release

## Getting Help

- Check existing issues and discussions
- Review the documentation
- Ask questions in issue comments
- Reach out to maintainers

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers learn
- Focus on the technical aspects
- Follow GitHub's community guidelines

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- GitHub contributors page

Thank you for contributing to Multi-Agent RAG! 🚀