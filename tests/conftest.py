"""Pytest configuration for EZMPI test suite."""

import os
import sys
import pytest


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
    # Note: We're NOT registering 'mpi' marker to avoid conflicts with pytest-mpi


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on available features."""
    # Check if we're running in an MPI environment
    is_mpi_environment = (
        "OMPI_COMM_WORLD_SIZE" in os.environ
        or "PMI_SIZE" in os.environ
        or "MPICH_COMM_WORLD_SIZE" in os.environ
        or "SLURM_NTASKS" in os.environ
    )

    # Force enable MPI tests if we're actually in an MPI environment
    actually_running_mpi = False
    if is_mpi_environment:
        try:
            from ezmpi import MPIPool

            actually_running_mpi = MPIPool().comm.Get_size() > 1
        except:
            actually_running_mpi = False

    skip_no_mpi = pytest.mark.skip(reason="Run with: mpiexec -n 4 pytest tests/")
    skip_no_dill = pytest.mark.skip(reason="dill not available")

    try:
        import mpi4py

        mpi_available = True
    except ImportError:
        mpi_available = False

    try:
        import dill

        dill_available = True
    except ImportError:
        dill_available = False

    for item in items:
        # Handle integration tests - enable them if we're actually running under MPI
        has_integration_marker = "integration" in item.keywords

        if has_integration_marker:
            if actually_running_mpi:
                # Remove any skip markers added by pytest-mpi
                item.own_markers = [
                    m for m in item.own_markers if "skip" not in m.name.lower()
                ]
            elif not mpi_available:
                item.add_marker(skip_no_mpi)

        # Handle dill tests - skip only when dill specifically needed but not available
        if "dill" in item.keywords and not dill_available:
            # Check if the test name suggests it's testing dill itself
            if "import_dill" in item.name or "dill" in item.nodeid.lower():
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
def mock_mpi_comm(mocker):
    """Create a mock MPI communicator."""
    mock_comm = mocker.MagicMock()
    mock_comm.Get_rank.return_value = 0
    mock_comm.Get_size.return_value = 4
    mock_comm.COMM_WORLD = mock_comm
    return mock_comm


@pytest.fixture
def mock_mpi(mocker, mock_mpi_comm):
    """Mock the entire MPI module."""
    mock_mpi_module = mocker.MagicMock()
    mock_mpi_module.COMM_WORLD = mock_mpi_comm
    mock_mpi_module.Status.return_value = mocker.MagicMock()
    mock_mpi_module.ANY_SOURCE = mock_mpi_comm.ANY_SOURCE
    mock_mpi_module.ANY_TAG = mock_mpi_comm.ANY_TAG

    # Patch the mpi4py import
    mocker.patch.dict("sys.modules", {"mpi4py": mocker.MagicMock()})
    mocker.patch.dict("sys.modules", {"mpi4py.MPI": mock_mpi_module})

    return mock_mpi_module
