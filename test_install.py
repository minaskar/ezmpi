#!/usr/bin/env python3
"""Test script to verify ezmpi installation."""


def test_basic():
    """Test basic import and version."""
    from ezmpi import MPIPool, __version__

    print(f"EZMPI version: {__version__}")
    print(f"MPIPool class imported: {MPIPool.__name__}")
    print("Basic import test: PASSED")


def test_square(x):
    """Simple worker function for testing."""
    return x * x


if __name__ == "__main__":
    import sys

    try:
        test_basic()

        # Only run MPI test if we have multiple processes
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            if size > 1:
                print(f"\nRunning MPI test with {size} processes...")
                from ezmpi import MPIPool

                with MPIPool() as pool:
                    if pool.is_master():
                        tasks = [1, 2, 3, 4, 5]
                        results = pool.map(test_square, tasks)
                        print(f"Input: {tasks}")
                        print(f"Output: {results}")
                        print("MPI test: PASSED")
            else:
                print(
                    "\nMPI test skipped: Run with 'mpiexec -n 4 python test_install.py' to test MPI functionality"
                )

        except ImportError:
            print("\nMPI test skipped: mpi4py not available")

        print("\n✓ All tests completed successfully!")
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ Test failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
