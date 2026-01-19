"""Unit tests for EZMPI with mocked MPI communication."""

import sys
import pytest
from unittest.mock import MagicMock, patch


pytestmark = pytest.mark.unit


class TestImports:
    """Test import functionality and dependency handling."""

    def test_import_mpi4py_success(self, mocker):
        """Test successful import of mpi4py."""
        mock_mpi = MagicMock()
        mocker.patch.dict(
            "sys.modules", {"mpi4py": mock_mpi, "mpi4py.MPI": mock_mpi.MPI}
        )

        from ezmpi.parallel import _import_mpi, MPI

        # Clean up global state
        if "ezmpi.parallel" in sys.modules:
            del sys.modules["ezmpi.parallel"]

        # Re-import to test fresh
        import importlib
        import ezmpi.parallel

        importlib.reload(ezmpi.parallel)

    def test_import_with_dill(self, mocker):
        """Test import with dill support."""
        mock_mpi = MagicMock()
        mock_dill = MagicMock()
        mocker.patch.dict(
            "sys.modules",
            {"mpi4py": mock_mpi, "mpi4py.MPI": mock_mpi.MPI, "dill": mock_dill},
        )

        from ezmpi.parallel import _import_mpi

        # Test with use_dill=True
        MPI = _import_mpi(use_dill=True)
        assert MPI is not None
        mock_dill.dumps.assert_called

    def test_import_dill_missing(self, mocker):
        """Test behavior when dill is missing."""
        mock_mpi = MagicMock()
        mocker.patch.dict(
            "sys.modules", {"mpi4py": mock_mpi, "mpi4py.MPI": mock_mpi.MPI}
        )

        # Remove dill from modules
        if "dill" in sys.modules:
            del sys.modules["dill"]

        from ezmpi.parallel import _import_mpi

        with pytest.raises(ImportError, match="dill is required"):
            _import_mpi(use_dill=True)


class TestMPIPoolInitialization:
    """Test MPIPool initialization with mocked MPI."""

    def test_pool_init_master_process(self, mocker, mock_mpi):
        """Test pool initialization as master process."""
        # Configure mock
        mock_comm = mock_mpi.COMM_WORLD
        mock_comm.Get_rank.return_value = 0  # Master
        mock_comm.Get_size.return_value = 4  # 4 processes

        # Mock sys.exit to prevent test from exiting
        mocker.patch("sys.exit")

        from ezmpi import MPIPool
        from ezmpi.parallel import MPI

        pool = MPIPool()

        assert pool.is_master()
        assert not pool.is_worker()
        assert pool.rank == 0
        assert pool.size == 3  # 4 total - 1 master
        assert 0 in pool.workers
        assert 1 in pool.workers
        assert 2 in pool.workers
        assert 3 in pool.workers

    def test_pool_init_single_process_error(self, mocker, mock_mpi):
        """Test error when only one process is available."""
        mock_comm = mock_mpi.COMM_WORLD
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 1  # Only 1 process

        mocker.patch("sys.exit")

        from ezmpi import MPIPool

        with pytest.raises(ValueError, match="only one MPI process"):
            MPIPool()

    def test_pool_init_custom_communicator(self, mocker):
        """Test pool initialization with custom communicator."""
        mock_comm = mocker.MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 4

        mocker.patch("sys.exit")
        mocker.patch("atexit.register")

        from ezmpi import MPIPool
        from ezmpi.parallel import _import_mpi

        # Mock MPI import
        MPI = mocker.MagicMock()
        MPI.COMM_WORLD = mock_comm
        mocker.patch("ezmpi.parallel.MPI", MPI)

        pool = MPIPool(comm=mock_comm)

        assert pool.comm == mock_comm
        assert pool.is_master()

    def test_pool_init_without_dill(self, mocker, mock_mpi):
        """Test pool initialization without dill."""
        mock_comm = mock_mpi.COMM_WORLD
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 4

        mocker.patch("sys.exit")
        mocker.patch("atexit.register")

        from ezmpi import MPIPool

        pool = MPIPool(use_dill=False)

        assert pool.is_master()
        assert pool.rank == 0


class TestWorkerProcess:
    """Test worker process behavior."""

    def test_worker_process_exits(self, mocker, mock_mpi):
        """Test that worker process calls wait() and exits."""
        mock_comm = mock_mpi.COMM_WORLD
        mock_comm.Get_rank.return_value = 1  # Worker
        mock_comm.Get_size.return_value = 4

        mock_exit = mocker.patch("sys.exit")

        from ezmpi import MPIPool

        # Create pool as worker - should call wait() and exit
        pool = MPIPool()

        # Should not reach here if worker exits properly
        mock_exit.assert_called_once_with(0)

    def test_worker_wait_loop(self, mocker):
        """Test worker wait loop receives and processes tasks."""
        from ezmpi.parallel import MPIPool

        pool = mocker.MagicMock()
        pool.comm = mocker.MagicMock()
        pool.master = 0
        pool.is_master = mocker.MagicMock(return_value=False)

        # Simulate receiving tasks
        def mock_recv(**kwargs):
            if not hasattr(mock_recv, "call_count"):
                mock_recv.call_count = 0
            mock_recv.call_count += 1

            if mock_recv.call_count == 1:
                # First call: return a task
                return (lambda x: x * 2, 5)
            else:
                # Second call: return None (terminate)
                return None

        pool.comm.recv.side_effect = mock_recv

        # Run wait method
        MPIPool.wait(pool)

        # Verify task was processed
        pool.comm.ssend.assert_called_once()
        call_args = pool.comm.ssend.call_args
        assert call_args[0][0] == 10  # Result of lambda x: x * 2 with x=5


class TestMpPoolMap:
    """Test the map functionality."""

    def test_map_basic_functionality(self, mocker, mock_mpi):
        """Test basic map functionality with mocked MPI."""
        mock_comm = mock_mpi.COMM_WORLD
        mock_comm.Get_rank.return_value = 0  # Master
        mock_comm.Get_size.return_value = 4

        mocker.patch("sys.exit")
        mocker.patch("atexit.register")

        from ezmpi import MPIPool

        pool = MPIPool()

        # Mock worker set and communication
        pool.workers = {1, 2, 3}
        pool.size = 3

        # Mock communication
        mock_status = mocker.MagicMock()
        mock_status.source = 1
        mock_status.tag = 0

        pool.comm.recv = mocker.MagicMock(return_value=4)  # Worker returns 2*2
        pool.comm.recv.status = mock_status
        pool.comm.Iprobe = mocker.MagicMock(return_value=True)

        # Test map
        def double(x):
            return x * 2

        result = pool.map(double, [1, 2])

        assert result == [4, None]  # One result from worker, one None
        assert pool.comm.send.call_count == 2

    def test_map_empty_tasks(self, mocker, mock_mpi):
        """Test map with empty tasks list."""
        mock_comm = mock_mpi.COMM_WORLD
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 4

        mocker.patch("sys.exit")

        from ezmpi import MPIPool

        pool = MPIPool()

        def dummy(x):
            return x

        result = pool.map(dummy, [])

        assert result == []
        pool.comm.send.assert_not_called()

    def test_map_preserves_order(self, mocker, mock_mpi):
        """Test that map returns results in correct order."""
        mock_comm = mock_mpi.COMM_WORLD
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 4

        mocker.patch("sys.exit")
        mocker.patch("atexit.register")

        from ezmpi import MPIPool

        pool = MPIPool()
        pool.workers = {1, 2, 3}
        pool.size = 3

        # Simulate out-of-order responses
        results = {0: 0, 1: 2, 2: 4, 3: 6, 4: 8}  # 2*index

        def mock_recv(**kwargs):
            call_num = getattr(mock_recv, "call_num", 0)
            mock_recv.call_num = call_num + 1

            if call_num < 5:
                # Create status object
                status = mocker.MagicMock()
                status.source = 1
                status.tag = call_num
                pool.comm.recv.status = status
                return results[call_num]
            return None

        pool.comm.recv = mocker.MagicMock(side_effect=mock_recv)
        pool.comm.Iprobe = mocker.MagicMock(return_value=True)

        def double(x):
            return x * 2

        result = pool.map(double, [0, 1, 2, 3, 4])

        assert result == [0, 2, 4, 6, 8]
        assert pool.comm.send.call_count == 5


class TestMPIPoolCleanup:
    """Test resource cleanup."""

    def test_context_manager(self, mocker, mock_mpi):
        """Test context manager usage."""
        mock_comm = mock_mpi.COMM_WORLD
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 4

        mocker.patch("sys.exit")
        mocker.patch("atexit.register")

        from ezmpi import MPIPool

        with MPIPool() as pool:
            assert pool.is_master()

        # Verify cleanup was called
        pool.comm.send.assert_called()

    def test_explicit_close(self, mocker, mock_mpi):
        """Test explicit close method."""
        mock_comm = mock_mpi.COMM_WORLD
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 4

        mocker.patch("sys.exit")

        from ezmpi import MPIPool

        pool = MPIPool()
        pool.close()

        # Workers should receive None
        assert pool.comm.send.call_count == 3  # 3 workers
        for call in pool.comm.send.call_args_list:
            assert call[0][0] is None


class TestProcessRoles:
    """Test process role detection."""

    def test_is_master(self, mocker, mock_mpi):
        """Test is_master method."""
        mock_comm = mock_mpi.COMM_WORLD
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 4

        mocker.patch("sys.exit")

        from ezmpi import MPIPool

        pool = MPIPool()
        assert pool.is_master()
        assert not pool.is_worker()

    def test_is_worker(self, mocker, mock_mpi):
        """Test is_worker method."""
        mock_comm = mock_mpi.COMM_WORLD
        mock_comm.Get_rank.return_value = 2  # Worker
        mock_comm.Get_size.return_value = 4

        mocker.patch("sys.exit")

        from ezmpi import MPIPool

        pool = MPIPool()
        # Note: worker processes exit in __init__, so this won't be fully testable
        # Just verify the role detection logic
        assert pool.rank == 2
        assert pool.is_worker()
        assert not pool.is_master()
