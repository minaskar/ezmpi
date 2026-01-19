"""Complete mock MPI module factory for unit testing.

This module provides factories to create complete mock MPI environments
that mimic real mpi4py behavior for unit testing without requiring MPI.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock


def create_mpi_communicator(rank=0, size=4):
    """Create a mock MPI communicator with configurable rank and size.

    Args:
        rank (int): The rank of this process
        size (int): Total number of processes

    Returns:
        MagicMock: A fully configured mock communicator
    """
    comm = MagicMock()
    comm.Get_rank = MagicMock(return_value=rank)
    comm.Get_size = MagicMock(return_value=size)

    # mpi4py also has .rank and .size properties
    # We need to set these as attributes, not methods
    comm.rank = rank
    comm.size = size

    comm.send = MagicMock()
    comm.ssend = MagicMock()
    comm.Probe = MagicMock()
    comm.Iprobe = MagicMock(return_value=True)

    # recv will be configured per-test, but provide a default
    def default_recv(*args, **kwargs):
        """Default recv behavior - can be overridden per test."""
        if "status" in kwargs:
            status = kwargs["status"]
            status.source = 1 if size > 1 else 0
            status.tag = 0
        # Return None by default (simulates no task/termination)
        return None

    comm.recv = MagicMock(side_effect=default_recv)

    return comm


def create_mpi_module(rank=0, size=4):
    """Create a complete mock mpi4py.MPI module.

    Args:
        rank (int): The rank of this process in COMM_WORLD
        size (int): Total number of processes in COMM_WORLD

    Returns:
        ModuleType: A mock MPI module with COMM_WORLD and constants
    """
    mpi = ModuleType("mpi")

    # MPI constants
    mpi.ANY_SOURCE = -1
    mpi.ANY_TAG = -1

    # Status class
    status = MagicMock()
    status.source = 0
    status.tag = 0
    mpi.Status = MagicMock(return_value=status)

    # COMM_WORLD communicator
    mpi.COMM_WORLD = create_mpi_communicator(rank, size)

    # Add pickle attribute for dill integration
    # Use a simple class with __init__ instead of MagicMock
    class PickleMock:
        def __init__(self):
            self.__init__ = MagicMock()

    mpi.pickle = PickleMock()

    return mpi


def create_dill_module():
    """Create a mock dill module.

    Returns:
        ModuleType: A mock dill module
    """
    dill = ModuleType("dill")
    dill.dumps = MagicMock(return_value=b"serialized_data")
    dill.loads = MagicMock(return_value=lambda x: x * 2)  # Default: double function
    dill.HIGHEST_PROTOCOL = 4
    return dill


def clear_ezmpi_modules():
    """Remove ezmpi modules from sys.modules to force reimport.

    This is critical for unit tests to ensure each test gets a fresh
    import of ezmpi with the current mock configuration.
    """
    modules_to_clear = [key for key in sys.modules.keys() if "ezmpi" in key]
    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]


class MockMPIEnvironment:
    """Context manager for setting up a complete mock MPI environment."""

    def __init__(self, rank=0, size=4, use_dill=False):
        """Initialize mock environment.

        Args:
            rank (int): Process rank (0=master)
            size (int): Total number of processes
            use_dill (bool): Whether to include dill module
        """
        self.rank = rank
        self.size = size
        self.use_dill = use_dill
        self.original_modules = {}
        self.mock_exit = None

    def __enter__(self):
        """Enter the mock environment."""
        # Save original modules
        self.original_modules = {}
        for key in ["mpi4py", "mpi4py.MPI", "dill"]:
            if key in sys.modules:
                self.original_modules[key] = sys.modules[key]

        # Clear ezmpi to force reimport
        clear_ezmpi_modules()

        # Create mocks
        mock_mpi = create_mpi_module(self.rank, self.size)

        # Mock sys.exit to prevent termination
        self.mock_exit = MagicMock()
        import unittest.mock

        self.exit_patcher = unittest.mock.patch("sys.exit", self.mock_exit)
        self.exit_patcher.start()

        # Build module dictionary
        modules = {
            "mpi4py": ModuleType("mpi4py"),
            "mpi4py.MPI": mock_mpi,
        }

        if self.use_dill:
            modules["dill"] = create_dill_module()

        # Apply mocks
        self.module_patcher = unittest.mock.patch.dict("sys.modules", modules)
        self.module_patcher.start()

        # Now import ezmpi with mocks in place
        from ezmpi import MPIPool

        result = {
            "pool_class": MPIPool,
            "mpi": mock_mpi,
            "exit": self.mock_exit,
        }

        # Add dill if it was created
        if self.use_dill:
            result["dill"] = modules["dill"]

        return result

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the mock environment and restore original state."""
        # Restore modules
        self.module_patcher.stop()
        self.exit_patcher.stop()

        # Restore original modules
        for key, module in self.original_modules.items():
            sys.modules[key] = module

        # Clear any newly added modules
        for key in ["mpi4py", "mpi4py.MPI", "dill"]:
            if key in sys.modules and key not in self.original_modules:
                del sys.modules[key]

        return False  # Don't suppress exceptions
