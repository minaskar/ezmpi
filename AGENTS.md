# EZMPI Agent Guidelines

## Build, Lint, and Test Commands

### Running Tests
**No test framework is currently configured** - this repository lacks:
- Test files (no `test_*.py` or `*_test.py` files found)
- Test runner configuration (pytest, unittest, etc.)
- CI/CD configuration

**Recommended setup:**
```bash
# Install testing dependencies
pip install pytest pytest-mpi pytest-cov

# Run all tests (once tests are created)
pytest -v

# Run a single test file
pytest tests/test_parallel.py -v

# Run a specific test function
pytest tests/test_parallel.py::test_mpi_pool -v

# Run with MPI support
mpiexec -n 4 pytest -v

# Run with coverage
pytest --cov=ezmpi --cov-report=html
```

### Code Quality Commands
**No linting tools are currently configured.** The codebase uses basic Python syntax.

**Recommended setup:**
```bash
# Install linting tools
pip install ruff black isort mypy

# Lint code
ruff check .

# Format code
ruff format .

# Type checking
mypy src/ezmpi

# Import sorting
isort src/
```

### Building and Installation
```bash
# Install in development mode
pip install -e .

# Install with optional dependencies
pip install -e ".[dev,test]"
```

## Code Style Guidelines

### Imports and Dependencies
- **mpi4py**: Core dependency for MPI functionality
- **dill**: Optional dependency for improved pickling (set `use_dill=True` for complex objects)
- Import structure: Standard library first, then third-party imports
- Use `import` statements at module level, not inside functions when possible

### Formatting and Style
Since no formatting tools are configured:
- Follow PEP 8 style guidelines
- Use 4 spaces for indentation
- Line length: aim for 79-88 characters
- Use descriptive variable names (e.g., `worker`, `tasklist`, not `w`, `tl`)

### Type Annotations
**No type checking currently implemented.** The codebase lacks type hints.

**Recommended approach:**
```python
from typing import Callable, Iterable, Any, Optional

def map(self, worker: Callable[[Any], Any], tasks: Iterable[Any]) -> list[Any]:
    ...
```

### Naming Conventions
- Classes: `PascalCase` (e.g., `MPIPool`)
- Functions/methods: `snake_case` (e.g., `is_master`, `wait`)
- Variables: `snake_case` (e.g., `tasklist`, `workerset`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MPI`, `DEFAULT_COMM`)
- Private methods: Leading underscore (e.g., `_import_mpi`)

### Error Handling
- Use specific exceptions with descriptive messages
- Import errors: Raise `ImportError` with installation instructions
- Validate runtime conditions (e.g., minimum MPI processes)
- Use atexit for cleanup (already implemented)

### Documentation
- Use reStructuredText format in docstrings (numpydoc style)
- Document all public methods and classes
- Include Parameters, Returns, Notes sections
- Reference external modules/libraries where relevant

### MPI-Specific Patterns
- **Process roles**: Master (rank 0) coordinates, workers (rank > 0) execute
- **Communication**: Use `self.comm.send()`/`recv()` with tags for task tracking
- **Synchronization**: Use `ssend()` for synchronous sends to workers
- **Cleanup**: Register cleanup functions with `atexit`
- **Error handling**: Check MPI process count at initialization

### Logging
**No logging currently implemented.** Consider adding:
- Debug logging for task distribution
- Info logging for initialization/shutdown
- Warning for edge cases

Example:
```python
import logging
logger = logging.getLogger(__name__)
```

## Cursor/Copilot Rules
**No existing Cursor or Copilot configuration found.**

---

**Note**: This is a minimal codebase with only one module (`parallel.py`) and no tests, linting, or automation configured. The guidelines above represent best practices that should be implemented as the project grows.