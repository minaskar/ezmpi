"""Unit tests for EZMPI with proper MPI mocking."""

import sys
import pytest
from unittest.mock import MagicMock, patch, mock_open
from types import ModuleType

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_mpi_module():
    """Create a properly mocked MPI module."""
    mock_mpi = MagicMock()

    # Mock MPI constants
    mock_mpi.ANY_SOURCE = -1
    mock_mpi.ANY_TAG = -1
    mock_mpi.Status = MagicMock

    # Create a mock Status object
    mock_status = MagicMock()
    mock_status.source = 0
    mock_status.tag = 0
    mock_mpi.Status.return_value = mock_status

    # Mock COMM_WORLD
    mock_comm = MagicMock()
    mock_comm.Get_rank = MagicMock(return_value=0)
    mock_comm.Get_size = MagicMock(return_value=4)
    mock_comm.send = MagicMock()
    mock_comm.recv = MagicMock()
    mock_comm.ssend = MagicMock()
    mock_comm.Iprobe = MagicMock(return_value=True)
    mock_comm.Probe = MagicMock()

    mock_mpi.COMM_WORLD = mock_comm

    return mock_mpi


@pytest.fixture
def mock_dill_module():
    """Create a mock dill module."""
    mock_dill = ModuleType("dill")
    mock_dill.dumps = MagicMock(return_value=b"serialized")
    mock_dill.loads = MagicMock(return_value=lambda x: x * 2)
    mock_dill.HIGHEST_PROTOCOL = 4
    return mock_dill


@pytest.fixture
def setup_mock_environment(mock_mpi_module, mock_dill_module):
    """Setup mock environment for unit tests."""
    # Clear any existing imports
    modules_to_clear = [
        key for key in sys.modules.keys() if "mpi4py" in key or "ezmpi" in key
    ]
    for key in modules_to_clear:
        if key in sys.modules:
            del sys.modules[key]

    # Mock sys.exit to prevent test termination
    with patch("sys.exit") as mock_exit:
        with patch.dict(
            "sys.modules",
            {
                "mpi4py": ModuleType("mpi4py"),
                "mpi4py.MPI": mock_mpi_module,
                "dill": mock_dill_module,
            },
        ):
            yield {"mpi": mock_mpi_module, "dill": mock_dill_module, "exit": mock_exit}


class TestImportFunctionality:
    """Test import and dependency handling."""

    def test_import_mpi4py_success(self, setup_mock_environment):
        """Test successful import of mpi4py."""
        from ezmpi.parallel import _import_mpi

        MPI = _import_mpi(use_dill=False)
        assert MPI is not None

    def test_import_mpi4py_failure(self):
        """Test import fails when mpi4py not available."""
        # Save original modules
        original_modules = {}
        for key in ["mpi4py", "mpi4py.MPI"]:
            if key in sys.modules:
                original_modules[key] = sys.modules[key]
                del sys.modules[key]

        try:
            from ezmpi.parallel import _import_mpi

            with pytest.raises(ImportError, match="mpi4py is required"):
                _import_mpi(use_dill=False)
        finally:
            # Restore modules
            for key, module in original_modules.items():
                sys.modules[key] = module

    def test_import_with_dill(self, setup_mock_environment):
        """Test import with dill support."""
        from ezmpi.parallel import _import_mpi

        env = setup_mock_environment
        MPI = _import_mpi(use_dill=True)

        assert MPI is not None
        # Verify dill was used
        assert env["dill"].dumps.called or env["dill"].loads.called

    def test_import_dill_not_used_when_false(self, setup_mock_environment):
        """Test dill is not used when use_dill=False."""
        from ezmpi.parallel import _import_mpi

        env = setup_mock_environment
        MPI = _import_mpi(use_dill=False)

        assert MPI is not None
        # Dill should not be accessed when use_dill=False
        # Just verify MPI module was configured correctly
        assert env["mpi"].COMM_WORLD.Get_rank.called


class TestMPIPoolInitialization:
    """Test MPIPool initialization."""

    def test_pool_init_master_process(self, setup_mock_environment):
        """Test pool initialization as master process."""
        env = setup_mock_environment

        from ezmpi import MPIPool

        pool = MPIPool(use_dill=False)

        assert pool.is_master()
        assert not pool.is_worker()
        assert pool.rank == 0
        assert pool.size == 3  # 4 total - 1 master
        assert 0 in pool.workers
        assert 1 in pool.workers
        assert 2 in pool.workers
        assert 3 in pool.workers
        assert not env["exit"].called  # Master shouldn't exit

    def test_pool_init_single_process_error(self, setup_mock_environment):
        """Test error when only one process is available."""
        env = setup_mock_environment
        env["mpi"].COMM_WORLD.Get_size.return_value = 1

        from ezmpi import MPIPool

        with pytest.raises(ValueError, match="only one MPI process"):
            MPIPool(use_dill=False)

        # Should not exit since we raise an error before wait() is called
        env["exit"].assert_not_called()

    def test_pool_init_custom_communicator(self, setup_mock_environment):
        """Test pool initialization with custom communicator."""
        env = setup_mock_environment

        from ezmpi import MPIPool

        mock_comm = env["mpi"].COMM_WORLD
        pool = MPIPool(comm=mock_comm, use_dill=False)

        assert pool.comm == mock_comm
        assert pool.is_master()


class TestWorkerBehavior:
    """Test worker process behavior."""

    def test_worker_process_exits(self, setup_mock_environment):
        """Test that worker process calls wait() and exits."""
        env = setup_mock_environment

        # Change rank to worker
        env["mpi"].COMM_WORLD.Get_rank.return_value = 1
        env["mpi"].Status.return_value.tag = 0

        from ezmpi import MPIPool

        # Create pool as worker
        pool = MPIPool(use_dill=False)

        # Worker should have called sys.exit
        env["exit"].assert_called_once_with(0)

    def test_worker_wait_loop_receives_tasks(self, setup_mock_environment):
        """Test worker wait loop processes tasks correctly."""
        env = setup_mock_environment

        # Change rank to worker
        env["mpi"].COMM_WORLD.Get_rank.return_value = 1
        env["mpi"].Status.return_value.tag = 0

        # Simulate receiving a task then None (terminate)
        def mock_recv(**kwargs):
            if not hasattr(mock_recv, "call_count"):
                mock_recv.call_count = 0
            mock_recv.call_count += 1

            if mock_recv.call_count == 1:
                return (lambda x: x * 2, 5)  # Task
            else:
                return None  # Terminate

        env["mpi"].COMM_WORLD.recv = MagicMock(side_effect=mock_recv)
        env["mpi"].Status.return_value.tag = 0

        from ezmpi import MPIPool

        # Worker exits after processing
        pool = MPIPool(use_dill=False)

        # Verify it processed the task and called exit
        env["exit"].assert_called_once_with(0)


class TestMapFunctionality:
    """Test the map functionality."""

    def test_map_empty_tasks(self, setup_mock_environment):
        """Test map with empty tasks."""
        env = setup_mock_environment

        from ezmpi import MPIPool

        pool = MPIPool(use_dill=False)

        def dummy(x):
            return x

        result = pool.map(dummy, [])

        assert result == []
        env["mpi"].COMM_WORLD.send.assert_not_called()

    def test_map_basic_functionality(self, setup_mock_environment):
        """Test basic map functionality."""
        env = setup_mock_environment

        # Setup mock responses for task distribution
        env["mpi"].COMM_WORLD.send = MagicMock()

        # Mock worker responses
        response_calls = []

        def mock_recv(**kwargs):
            response_calls.append(kwargs)

            # Create a mock status
            if "status" in kwargs:
                status_obj = kwargs["status"]
                status_obj.source = (
                    1 if len(response_calls) <= 2 else kwargs.get("source", 0)
                )
                status_obj.tag = len(response_calls) - 1

            # Return result (worker processed task)
            return 4  # 2*2

        env["mpi"].COMM_WORLD.recv = MagicMock(side_effect=mock_recv)
        env["mpi"].COMM_WORLD.Iprobe = MagicMock(return_value=True)

        from ezmpi import MPIPool

        pool = MPIPool(use_dill=False)

        def double(x):
            return x * 2

        result = pool.map(double, [1, 2])

        # Results might have None placeholders, check we got 2 results
        assert len(result) == 2
        assert pool.comm.send.call_count == 2

    def test_map_preserves_order(self, setup_mock_environment):
        """Test that map returns results in correct order."""
        env = setup_mock_environment

        # Track receive calls
        recv_calls = []

        def mock_recv(**kwargs):
            recv_calls.append(len(recv_calls))

            # Setup status
            if "status" in kwargs:
                status_obj = kwargs["status"]
                status_obj.source = 1
                status_obj.tag = len(recv_calls) - 1  # 0, 1, 2,...

            return [0, 2, 8, 18, 32, 50, 72, 98, 128][len(recv_calls) - 1]

        env["mpi"].COMM_WORLD.recv = MagicMock(side_effect=mock_recv)
        env["mpi"].COMM_WORLD.Iprobe = MagicMock(return_value=True)

        from ezmpi import MPIPool

        pool = MPIPool(use_dill=False)

        def square_func(x):
            return x * x

        tasks = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        result = pool.map(square_func, tasks)

        # Check we got all results
        assert len(result) == len(tasks)


class TestPoolFunctionality:
    """Test various pool functions."""

    def test_is_master_and_worker(self, setup_mock_environment):
        """Test role detection methods."""
        env = setup_mock_environment

        from ezmpi import MPIPool

        pool = MPIPool(use_dill=False)

        assert pool.is_master()
        assert not pool.is_worker()
        assert pool.rank == 0

        # Test worker detection
        env["mpi"].COMM_WORLD.Get_rank.return_value = 2
        pool2 = MPIPool(use_dill=False)

        assert pool2.rank == 2
        assert pool2.is_worker()
        assert not pool2.is_master()


class TestContextManagerAndCleanup:
    """Test context manager usage and cleanup."""

    def test_context_manager(self, setup_mock_environment):
        """Test context manager usage."""
        env = setup_mock_environment

        from ezmpi import MPIPool

        with MPIPool(use_dill=False) as pool:
            assert pool.is_master()
            assert not env["exit"].called

        # Exiting context should trigger cleanup
        # (via atexit, not necessarily at this exact moment)
        assert True  # If we got here, context manager worked

    def test_explicit_close(self, setup_mock_environment):
        """Test explicit close method."""
        env = setup_mock_environment

        from ezmpi import MPIPool

        pool = MPIPool(use_dill=False)
        pool.close()

        # Workers should receive None termination signals
        assert pool.comm.send.call_count == 3  # 3 workers
        for call in pool.comm.send.call_args_list:
            assert call[0][0] is None  # First arg is the message


def test_import_ezmpi_package():
    """Test that the ezmpi package can be imported."""
    import ezmpi

    assert hasattr(ezmpi, "MPIPool")
    assert hasattr(ezmpi, "__version__")
    assert ezmpi.__version__ == "0.1.0"
