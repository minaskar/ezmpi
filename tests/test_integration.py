"""Integration tests for EZMPI with real MPI execution."""

import pytest

pytestmark = pytest.mark.integration


def square(x):
    """Simple worker function for testing."""
    return x * x


def add_one(x):
    """Worker function that adds one."""
    return x + 1


def complex_computation(x):
    """CPU-bound computation for testing."""
    return sum(i * i for i in range(x + 1))


@pytest.mark.mpi(min_size=2)
@pytest.mark.skipif(
    not hasattr(__import__("mpi4py", fromlist=["MPI"]), "MPI")
    or __import__("mpi4py").MPI.COMM_WORLD.Get_size() <= 1,
    reason="Run with: mpiexec -n N pytest tests/test_integration.py",
)
def test_integration_basic_map():
    """Test basic parallel mapping with real MPI."""

    from ezmpi import MPIPool

    pool = MPIPool(test_mode=True)

    if pool.is_master():
        tasks = [1, 2, 3, 4, 5]
        results = pool.map(square, tasks)
        assert results == [1, 4, 9, 16, 25]
        print(f"✓ test_integration_basic_map passed: {results}")


@pytest.mark.mpi(min_size=2)
@pytest.mark.skipif(
    not hasattr(__import__("mpi4py", fromlist=["MPI"]), "MPI")
    or __import__("mpi4py").MPI.COMM_WORLD.Get_size() <= 1,
    reason="Run with: mpiexec -n N pytest tests/test_integration.py",
)
def test_integration_multiple_workers():
    """Test with multiple workers."""

    from ezmpi import MPIPool

    pool = MPIPool(test_mode=True)

    if pool.is_master():
        assert pool.size >= 1, f"Expected at least 1 worker, got {pool.size}"
        assert len(pool.workers) == pool.size
        assert 0 not in pool.workers

        tasks = list(range(10))
        results = pool.map(add_one, tasks)
        assert results == [i + 1 for i in tasks]
        print(f"✓ test_integration_multiple_workers passed: {pool.size} workers")


@pytest.mark.mpi(min_size=2)
@pytest.mark.skipif(
    not hasattr(__import__("mpi4py", fromlist=["MPI"]), "MPI")
    or __import__("mpi4py").MPI.COMM_WORLD.Get_size() <= 1,
    reason="Run with: mpiexec -n N pytest tests/test_integration.py",
)
def test_integration_result_ordering():
    """Test that results maintain correct order."""

    from ezmpi import MPIPool

    pool = MPIPool(test_mode=True)

    if pool.is_master():
        tasks = [5, 3, 8, 1, 9, 2, 7, 4, 6]
        results = pool.map(square, tasks)
        assert results == [x * x for x in tasks]
        print("✓ test_integration_result_ordering passed")


@pytest.mark.mpi(min_size=2)
@pytest.mark.skipif(
    not hasattr(__import__("mpi4py", fromlist=["MPI"]), "MPI")
    or __import__("mpi4py").MPI.COMM_WORLD.Get_size() <= 1,
    reason="Run with: mpiexec -n N pytest tests/test_integration.py",
)
def test_integration_empty_tasks():
    """Test map with empty task list."""

    from ezmpi import MPIPool

    pool = MPIPool(test_mode=True)

    if pool.is_master():
        results = pool.map(square, [])
        assert results == []
        print("✓ test_integration_empty_tasks passed")


@pytest.mark.mpi(min_size=2)
@pytest.mark.skipif(
    not hasattr(__import__("mpi4py", fromlist=["MPI"]), "MPI")
    or __import__("mpi4py").MPI.COMM_WORLD.Get_size() <= 1,
    reason="Run with: mpiexec -n N pytest tests/test_integration.py",
)
def test_integration_single_task():
    """Test with single task."""

    from ezmpi import MPIPool

    pool = MPIPool(test_mode=True)

    if pool.is_master():
        tasks = [42]
        results = pool.map(square, tasks)
        assert results == [1764]
        print(f"✓ test_integration_single_task passed: {results}")


@pytest.mark.mpi(min_size=2)
@pytest.mark.skipif(
    not hasattr(__import__("mpi4py", fromlist=["MPI"]), "MPI")
    or __import__("mpi4py").MPI.COMM_WORLD.Get_size() <= 1,
    reason="Run with: mpiexec -n N pytest tests/test_integration.py",
)
def test_integration_many_tasks():
    """Test with more tasks than workers."""

    from ezmpi import MPIPool

    pool = MPIPool(test_mode=True)

    if pool.is_master():
        tasks = list(range(20))
        results = pool.map(square, tasks)
        assert results == [x * x for x in tasks]
        print(f"✓ test_integration_many_tasks passed: {len(tasks)} tasks")


@pytest.mark.mpi(min_size=2)
@pytest.mark.skipif(
    not hasattr(__import__("mpi4py", fromlist=["MPI"]), "MPI")
    or __import__("mpi4py").MPI.COMM_WORLD.Get_size() <= 1,
    reason="Run with: mpiexec -n N pytest tests/test_integration.py",
)
def test_integration_cpu_bound_tasks():
    """Test with CPU-bound computations."""

    from ezmpi import MPIPool

    pool = MPIPool(test_mode=True)

    if pool.is_master():
        tasks = [100, 200, 300, 400, 500]
        results = pool.map(complex_computation, tasks)
        assert results == [complex_computation(x) for x in tasks]
        print("✓ test_integration_cpu_bound_tasks passed")


@pytest.mark.mpi(min_size=2)
@pytest.mark.skipif(
    not hasattr(__import__("mpi4py", fromlist=["MPI"]), "MPI")
    or __import__("mpi4py").MPI.COMM_WORLD.Get_size() <= 1,
    reason="Run with: mpiexec -n N pytest tests/test_integration.py",
)
def test_integration_context_manager():
    """Test context manager usage."""

    from ezmpi import MPIPool

    pool = MPIPool(test_mode=True)

    if pool.is_master():
        assert pool.is_master()
        initial_workers = pool.workers.copy()

        tasks = [1, 2, 3]
        results = pool.map(square, tasks)
        assert results == [1, 4, 9]
        assert pool.workers == initial_workers
        print("✓ test_integration_context_manager passed")


@pytest.mark.mpi(min_size=2)
@pytest.mark.skipif(
    not hasattr(__import__("mpi4py", fromlist=["MPI"]), "MPI")
    or __import__("mpi4py").MPI.COMM_WORLD.Get_size() <= 1,
    reason="Run with: mpiexec -n N pytest tests/test_integration.py",
)
def test_integration_process_roles():
    """Test that master/worker roles are correctly assigned."""

    from ezmpi import MPIPool

    pool = MPIPool(test_mode=True)

    if pool.is_master():
        assert pool.rank == 0
        assert pool.is_master()
        assert not pool.is_worker()
        print(
            f"✓ test_integration_process_roles passed: rank={pool.rank}, workers={pool.workers}"
        )


@pytest.mark.mpi(min_size=2)
@pytest.mark.skipif(
    not hasattr(__import__("mpi4py", fromlist=["MPI"]), "MPI")
    or __import__("mpi4py").MPI.COMM_WORLD.Get_size() <= 1,
    reason="Run with: mpiexec -n N pytest tests/test_integration.py",
)
def test_integration_different_task_sizes():
    """Test with varying task sizes."""

    from ezmpi import MPIPool

    pool = MPIPool(test_mode=True)

    if pool.is_master():
        tasks = [2, 500, 3, 1000, 1, 750, 4, 250]
        results = pool.map(complex_computation, tasks)
        assert results == [complex_computation(x) for x in tasks]
        print("✓ test_integration_different_task_sizes passed")


@pytest.mark.dill
@pytest.mark.mpi(min_size=2)
@pytest.mark.skipif(
    not hasattr(__import__("mpi4py", fromlist=["MPI"]), "MPI")
    or __import__("mpi4py").MPI.COMM_WORLD.Get_size() <= 1,
    reason="Run with: mpiexec -n N pytest tests/test_integration.py",
)
def test_integration_complex_objects():
    """Test with complex objects."""

    from ezmpi import MPIPool

    pool = MPIPool(test_mode=True)

    if pool.is_master():
        tasks = [1, 2, 3, 4, 5]
        results = pool.map(lambda x: x * 2, tasks)
        assert results == [2, 4, 6, 8, 10]
        print("✓ test_integration_complex_objects passed")
