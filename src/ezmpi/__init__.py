"""EZMPI: A simple MPI processing pool for Python.

This package provides a simple MPI-based processing pool for parallel
task execution across multiple processes.
"""

from .parallel import MPIPool

__version__ = "0.1.0"
__all__ = ["MPIPool"]
