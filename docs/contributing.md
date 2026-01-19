# Contributing to EZMPI

Thank you for your interest in contributing to EZMPI! We welcome contributions from the community.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request, please open an issue on our [GitHub repository](https://github.com/minaskar/ezmpi/issues). Include:

- Clear description of the problem or feature
- Steps to reproduce (for bugs)
- Expected behavior
- Environment details (Python version, MPI implementation, OS)

### Contributing Code

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ezmpi.git
   cd ezmpi
   ```

3. **Create a new branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make your changes** and ensure:
   - Code follows the existing style (black formatting)
   - Tests pass: `pytest tests/`
   - MPI tests pass: `mpiexec -n 4 pytest --with-mpi tests/test_integration.py`
   - Documentation builds: `mkdocs build`

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** on GitHub with:
   - Clear description of changes
   - Reference to any related issues
   - Test results

### Development Setup

For development, install with test dependencies:

```bash
pip install -e ".[test,dill,docs]"
```

### Code Style

- Format code with `black src/ tests/`
- Sort imports with `isort src/ tests/`
- Lint with `ruff check src/ tests/`
- All checks run automatically in CI

### Testing

Run different test suites:

```bash
# Unit tests (mocked MPI)
pytest tests/test_unit.py

# Integration tests (requires real MPI)
mpiexec --oversubscribe -n 4 pytest --with-mpi tests/test_integration.py

# Performance tests
mpiexec --oversubscribe -n 4 pytest --with-mpi tests/test_performance.py

# All tests
pytest tests/
```

### Documentation

Update documentation when adding features:

```bash
# Edit docs in docs/ directory
vim docs/getting_started.md

# Build and preview
cd docs && mkdocs serve
```

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming environment for all contributors.

## Questions?

Feel free to open an issue for questions or reach out to the maintainers.

Thank you for contributing to EZMPI!
