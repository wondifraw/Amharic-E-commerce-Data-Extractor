# Contributing to Ethiopian Telegram NER System

Thank you for your interest in contributing to the Ethiopian Telegram NER System! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Telegram API credentials (for data ingestion)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/ethiopian-ner-system.git
   cd ethiopian-ner-system
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   make install-dev
   # or
   pip install -r requirements.txt
   pip install pytest pytest-cov black flake8 mypy pre-commit
   ```

4. **Setup Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

5. **Create Data Directories**
   ```bash
   make setup-data
   ```

## 🛠️ Development Workflow

### Code Style
We use several tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pytest**: Testing

### Before Committing
Run the following commands to ensure your code meets our standards:

```bash
# Format code
make format

# Run linting
make lint

# Run type checking
make type-check

# Run tests
make test
```

Or run all checks at once:
```bash
make ci-test
```

### Testing Guidelines

#### Unit Tests
- Write unit tests for all new functions and classes
- Place tests in `tests/unit/`
- Use descriptive test names
- Aim for high test coverage

```bash
# Run unit tests
make test-unit
```

#### Integration Tests
- Write integration tests for complete workflows
- Place tests in `tests/integration/`
- Mark slow tests with `@pytest.mark.slow`

```bash
# Run integration tests
make test-integration
```

### Code Organization

```
src/
├── data_ingestion/     # Telegram scraping and data collection
├── preprocessing/      # Text processing and CoNLL labeling
├── ner/               # Model training and inference
├── evaluation/        # Model comparison and metrics
├── interpretability/   # SHAP and LIME explanations
├── vendor_analytics/   # Scorecard and lending analysis
└── dashboard/         # Streamlit interface
```

## 📝 Contribution Types

### 🐛 Bug Reports
When reporting bugs, please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)
- Error messages and stack traces

### ✨ Feature Requests
For new features, please provide:
- Clear description of the feature
- Use case and motivation
- Proposed implementation approach
- Any breaking changes

### 🔧 Code Contributions

#### Pull Request Process
1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow coding standards
   - Add tests for new functionality
   - Update documentation if needed

3. **Test Your Changes**
   ```bash
   make ci-test
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

#### Commit Message Format
We follow conventional commits:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test additions/modifications
- `refactor:` Code refactoring
- `style:` Code style changes
- `chore:` Maintenance tasks

## 🌍 Amharic Language Contributions

### Text Processing
- Ensure proper Unicode handling for Amharic text
- Test with various Amharic fonts and encodings
- Consider regional variations and dialects

### Entity Recognition
- Add new entity patterns for Amharic text
- Improve tokenization for Amharic-English mixed text
- Validate BIO tagging consistency

### Dataset Contributions
- Provide high-quality labeled examples
- Ensure diverse representation of Ethiopian e-commerce terms
- Follow CoNLL format standards

## 📊 Data and Model Contributions

### Dataset Guidelines
- Ensure data privacy and anonymization
- Provide clear licensing information
- Document data collection methodology
- Include data validation scripts

### Model Improvements
- Document model architecture changes
- Provide performance comparisons
- Include interpretability analysis
- Test on diverse Ethiopian channels

## 🔒 Security Guidelines

### Data Handling
- Never commit API keys or credentials
- Use environment variables for sensitive data
- Anonymize personal information in datasets
- Follow GDPR and data protection guidelines

### Code Security
- Validate all user inputs
- Use secure communication protocols
- Regular dependency updates
- Security scanning with bandit

## 📚 Documentation

### Code Documentation
- Use Google-style docstrings
- Include type hints
- Document complex algorithms
- Provide usage examples

### User Documentation
- Update README for new features
- Add configuration examples
- Include troubleshooting guides
- Maintain API documentation

## 🤝 Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Welcome newcomers and questions
- Provide constructive feedback
- Focus on the project's goals

### Communication
- Use GitHub Issues for bug reports and feature requests
- Use GitHub Discussions for general questions
- Be clear and concise in communications
- Provide context and examples

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- Project documentation

## 📞 Getting Help

If you need help:
1. Check existing documentation
2. Search GitHub Issues
3. Create a new issue with detailed information
4. Join our community discussions

## 🔄 Release Process

### Version Numbering
We follow Semantic Versioning (SemVer):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version number bumped
- [ ] Changelog updated
- [ ] Security scan completed

Thank you for contributing to the Ethiopian Telegram NER System! 🇪🇹