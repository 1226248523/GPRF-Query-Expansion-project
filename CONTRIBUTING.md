# Contributing to GPRF Query Expansion

Thank you for your interest in contributing to GPRF Query Expansion! We welcome contributions from everyone.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- PyTorch 1.12.0 or higher (with CUDA support recommended)

### Installation Steps

1. **Fork the repository**

   - Click the "Fork" button on GitHub
   - Clone your fork locally:

   ```bash
   git clone https://github.com/your-username/gprf-query-expansion.git
   cd gprf-query-expansion
   ```
2. **Create a virtual environment**

   ```bash
   python -m venv venv

   # Windows:
   venv\Scripts\activate

   # Linux/Mac:
   source venv/bin/activate
   ```
3. **Install development dependencies**

   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```
4. **Install pre-commit hooks** (optional but recommended)

   ```bash
   pre-commit install
   ```
5. **Verify installation**

   ```bash
   python -c "import gprf; print('Installation successful!')"
   ```

## Development Workflow

### 1. Choose an Issue

- Check [open issues](https://github.com/your-username/gprf-query-expansion/issues) for tasks
- Comment on the issue to indicate you're working on it
- Create a new issue if you have a feature request or bug report

### 2. Create a Feature Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# For bug fixes:
git checkout -b bugfix/issue-number-description

# For documentation:
git checkout -b docs/update-readme
```

### 3. Make Your Changes

- Write clear, concise commit messages
- Follow the existing code style
- Add tests for new functionality
- Update documentation if needed

### 4. Run Quality Checks

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/unit/test_generators.py

# Check code formatting
black --check src/
isort --check-only src/

# Run linting
flake8 src/

# Run type checking
mypy src/
```

### 5. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with a clear message
git commit -m "feat: add new retriever model

- Implement DenseRetriever class
- Add configuration support
- Include unit tests
- Update documentation

Closes #123"
```

### 6. Push and Create Pull Request

```bash
# Push your branch
git push origin feature/your-feature-name

# Create a Pull Request on GitHub
# - Go to your fork on GitHub
# - Click "Compare & pull request"
# - Fill out the PR template
# - Request review from maintainers
```

## Code Style Guidelines

### Python Code Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting (88 character line length)
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use type hints for function parameters and return values

### Example:

```python
from typing import Dict, List, Optional
import torch
from torch import nn

def create_model(config: Dict[str, any]) -> nn.Module:
    """Create a neural network model.

    Args:
        config: Model configuration dictionary

    Returns:
        Configured neural network model
    """
    # Implementation here
    pass
```

### Commit Message Format

Follow [Conventional Commits](https://conventionalcommits.org/) format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```
feat: add support for DPR retriever
fix: resolve memory leak in BART generator
docs: update installation instructions
test: add integration tests for evaluation
```

## Testing Guidelines

### Unit Tests

- Place test files in `tests/unit/`
- Name files as `test_*.py`
- Use descriptive test names
- Aim for >80% code coverage

```python
import pytest
from gprf.core.generators import BartQueryGenerator

class TestBartQueryGenerator:
    def test_initialization(self):
        """Test generator initializes correctly."""
        config = {"model_name": "facebook/bart-base"}
        generator = BartQueryGenerator(config)
        assert generator is not None

    def test_format_input(self):
        """Test input formatting for model."""
        # Test implementation
        pass
```

### Integration Tests

- Place in `tests/integration/`
- Test component interactions
- May require test data or mocked services

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gprf --cov-report=html

# Run specific test file
pytest tests/unit/test_generators.py

# Run tests matching pattern
pytest -k "test_format"
```

## Documentation

### Code Documentation

- Use docstrings for all public functions, classes, and modules
- Follow Google-style docstrings:

```python
def evaluate_model(model: nn.Module, test_data: DataLoader) -> Dict[str, float]:
    """Evaluate model performance on test data.

    Args:
        model: Trained neural network model
        test_data: DataLoader containing test samples

    Returns:
        Dictionary with evaluation metrics (accuracy, loss, etc.)

    Raises:
        ValueError: If model or data is invalid

    Example:
        >>> metrics = evaluate_model(model, test_loader)
        >>> print(f"Accuracy: {metrics['accuracy']:.2f}")
    """
```

### README and Docs

- Keep README.md up to date
- Add examples for new features
- Update API documentation in `docs/`

## Pull Request Process

### Before Submitting

- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

### PR Template

Please fill out the PR template with:

1. **Description**: What changes were made and why
2. **Type of change**: Bug fix, feature, documentation, etc.
3. **Checklist**: Confirmation that requirements are met
4. **Testing**: How the changes were tested

### Review Process

1. Automated checks run (tests, linting, etc.)
2. Code review by maintainers
3. Changes requested or approval
4. Merge to main branch

## Community Guidelines

- Be respectful and inclusive
- Help newcomers learn and contribute
- Focus on constructive feedback
- Follow our [Code of Conduct](CODE_OF_CONDUCT.md)

## Getting Help

- Check [existing issues](https://github.com/your-username/gprf-query-expansion/issues) first
- Open a [new issue](https://github.com/your-username/gprf-query-expansion/issues/new) for questions
- Join our [discussions](https://github.com/your-username/gprf-query-expansion/discussions) for general chat

## Recognition

Contributors will be recognized in:

- CHANGELOG.md for significant contributions
- Repository contributors list
- Project acknowledgments

Thank you for contributing to GPRF Query Expansion! 🚀
