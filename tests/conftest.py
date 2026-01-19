"""Pytest configuration for EZMPI test suite."""

import pytest

# Import our mock factory
from .mpi_mocks import MockMPIEnvironment


def pytest_configure(config):
    """Register pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests with mocked MPI")
    config.addinivalue_line(
        "markers", "integration: Integration tests requiring real MPI"
    )
    config.addinivalue_line("markers", "performance: Performance and benchmark tests")
    config.addinivalue_line(
        "markers", "dill: Test requiring dill for complex object pickling"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on available features."""
    skip_no_dill = pytest.mark.skip(reason="dill not available")

    try:
        import importlib

        dill_available = importlib.util.find_spec("dill") is not None
    except ImportError:
        dill_available = False

    for item in items:
        # Handle dill tests - skip only when dill specifically needed but not available
        if "dill" in item.keywords and not dill_available:
            # Skip tests that specifically test dill functionality
            if (
                "import_dill" in item.name
                or "test_integration_complex_objects" in item.name
            ):
                item.add_marker(skip_no_dill)


@pytest.fixture
def sample_tasks():
    """Provide sample tasks for testing."""
    return [1, 2, 3, 4, 5]


@pytest.fixture
def sample_worker():
    """Provide a sample worker function."""

    def square(x):
        return x * x

    return square


@pytest.fixture
def complex_worker():
    """Provide a worker that requires dill."""
    multiplier = 3

    def multiply(x):
        return x * multiplier

    return multiply


@pytest.fixture
def mock_mpi_env():
    """Provide a complete mock MPI environment for unit tests.

    This fixture properly mocks mpi4py.MPI at the module level before
    importing ezmpi, avoiding issues with mpi4py's read-only C attributes.
    """
    with MockMPIEnvironment(rank=0, size=4, use_dill=True) as env:
        yield env


@pytest.fixture
def mock_mpi_env_worker():
    """Provide a mock MPI environment simulating a worker process (rank 1)."""
    with MockMPIEnvironment(rank=1, size=4, use_dill=True) as env:
        yield env


@pytest.fixture
def mock_mpi_env_single_process():
    """Provide a mock MPI environment with only 1 process."""
    with MockMPIEnvironment(rank=0, size=1, use_dill=True) as env:
        yield env


@pytest.fixture
def mock_mpi_env_without_dill():
    """Provide a mock MPI environment without dill."""
    with MockMPIEnvironment(rank=0, size=4, use_dill=False) as env:
        yield env
