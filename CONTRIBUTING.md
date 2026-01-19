# Contributing to EZMPI

Thank you for your interest in contributing to EZMPI! This document provides guidelines and instructions for contributing.

## Types of Contributions

We welcome many types of contributions:

- **Bug reports**: Report issues via [GitHub Issues](https://github.com/minaskar/ezmpi/issues)
- **Bug fixes**: Submit PRs with fixes and tests
- **Features**: Add new functionality with tests and docs
- **Documentation**: Improve docstrings, examples, or guides
- **Tests**: Add tests for edge cases or improve coverage
- **Performance**: Optimize existing code
- **Examples**: Add real-world usage examples

## Development Setup

### Prerequisites

- Python 3.8+
- MPI implementation (OpenMPI, MPICH, or similar)
- Git

### Setup Steps

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/ezmpi.git
   cd ezmpi
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[dev,test,docs]"
   ```

4. Verify installation:
   ```bash
   python -c "from ezmpi import MPIPool; print('OK')"
   ```

5. Run tests:
   ```bash
   # Unit tests (no MPI)
   pytest tests/test_unit.py -v
   
   # Integration tests (requires MPI)
   mpiexec -n 4 pytest tests/test_integration.py -v
   ```

## Code Style Guidelines

### Python Code

- **Formatting**: Use black (`black src/ tests/`)
- **Sorting**: Use isort (`isort src/ tests/`)
- **Linting**: Use ruff (`ruff check src/ tests/`)
- **Type hints**: Use mypy for type checking (`mypy src/ezmpi`)

Pre-commit hook (recommended):
```bash
#!/bin/sh
# .git/hooks/pre-commit
black src/ tests/
isort src/ tests/
ruff check src/ tests/
pytest tests/test_unit.py -q
```

### Docstrings

- Use **numpydoc** style for all public methods
- Include Parameters, Returns, Raises, Examples sections
- All examples should be runnable

Example:
```python
def map(self, worker, tasks):
    """Execute a worker function on each task in parallel.
    
    Parameters
    ----------
    worker : callable
        A function that takes a single argument
    tasks : iterable
        Tasks to distribute to workers
    
    Returns
    -------
    list
        Results in the same order as input tasks
    
    Examples
    --------
    >>> with MPIPool() as pool:
    ...     results = pool.map(lambda x: x*2, [1, 2, 3])
    ...     print(results)
    [2, 4, 6]
    """
```

## Testing Requirements

### All PRs Must Include

1. **Unit tests**: Test logic with mocked MPI (see `tests/mpi_mocks.py`)
2. **Integration tests**: Test with real MPI if adding features
3. **Documentation**: Update docstrings and user docs
4. **Changelog**: Add entry to CHANGELOG.md

### Test Guidelines

- **Unit tests**: Use `MockMPIEnvironment` fixture
- **Integration tests**: Use `@pytest.mark.skipif(not IS_MPI, ...)`
- **Descriptive names**: `test_<component>_<behavior>_<expected>`
- **Edge cases**: Test error conditions, boundary values
- **Coverage**: Aim for >80% coverage on new code

Example test structure:
```python
def test_map_preserves_order(self):
    """Test that map returns results in input order."""
    with MockMPIEnvironment(rank=0, size=4) as env:
        pool = env['pool_class'](use_dill=False)
        # ... test logic
        assert results == expected
```

## Pull Request Process

### Before Submitting

1. **Update your branch**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run full test suite**:
   ```bash
   # Unit tests
   pytest tests/test_unit.py -v
   
   # Integration tests if you have MPI
   mpiexec -n 4 pytest tests/test_integration.py -v
   
   # Check coverage
   pytest --cov=ezmpi tests/
   ```

3. **Check documentation builds**:
   ```bash
   cd docs
   mkdocs serve
   # Open http://localhost:8000 and verify
   ```

### PR Requirements

PR description must include:
- **What**: Clear description of changes
- **Why**: Motivation and problem solved
- **How**: Implementation approach
- **Testing**: Summary of test results
- **Breaking changes**: Any API changes

Example PR template:
```markdown
## Summary
Brief description of changes

## Motivation
Why this change is needed

## Implementation
High-level approach

## Test Results
- Unit tests: X/Y pass
- Integration tests: X/Y pass (if applicable)
- New tests added: List them

## Breaking Changes
- None / List any API changes
```

### PR Review Process

1. **Automated checks must pass**:
   - Code style (black, isort, ruff)
   - Unit tests (pytest)
   - Type checking (mypy)

2. **Manual review**:
   - Code correctness
   - Test coverage
   - Documentation quality
   - Performance impact (if relevant)

3. **Approval**:
   - At least 1 maintainer approval required
   - All review comments addressed

## Release Process (Maintainers Only)

1. **Update version** in `src/ezmpi/__init__.py`
2. **Update CHANGELOG.md** with release notes
3. **Create release PR** with version bump
4. **After merge, create GitHub release** with tag
5. **Build and publish to PyPI**:
   ```bash
   python -m build
   twine upload dist/*
   ```

## Questions?

- **GitHub Issues**: Bug reports, feature requests
- **Discussions**: General questions, usage help
- **Email**: Contact maintainers directly

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers get started
- Assume good intentions

## License

By contributing, you agree that your contributions will be licensed under 
the MIT License.
