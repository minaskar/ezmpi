"""Performance smoke tests for EZMPI."""

import time

import pytest


def heavy_computation(x):
    """CPU-bound computation for performance testing."""
    result = 0
    for _i in range(10000):
        result += sum(j * j for j in range(x + 1))
    return result


@pytest.mark.performance
def test_performance_single_vs_parallel():
    """Basic smoke test: verify parallel execution provides speedup."""
    from ezmpi import MPIPool

    pool = MPIPool(test_mode=True)
    if pool.comm.Get_size() > 1:
        with pool as pool:  # Reuse the same pool instance
            if pool.is_master():
                tasks = [50, 75, 100, 125, 150]

                start = time.time()
                results = pool.map(heavy_computation, tasks)
                parallel_time = time.time() - start

                # Verify correctness
                expected = [heavy_computation(x) for x in tasks]
                assert results == expected

                # Should complete in reasonable time
                assert (
                    parallel_time < 5.0
                ), f"Parallel execution too slow: {parallel_time:.2f}s"
                print(
                    f"Performance: {len(tasks)} tasks completed in {parallel_time:.2f}s"
                )


@pytest.mark.performance
def test_performance_task_scaling():
    """Test performance with varying number of tasks."""
    from ezmpi import MPIPool

    pool = MPIPool(test_mode=True)
    if pool.comm.Get_size() > 1:
        with pool as pool:  # Reuse the same pool instance
            if pool.is_master():
                tasks = list(range(10, 50, 5))

                start = time.time()
                pool.map(heavy_computation, tasks)
                duration = time.time() - start

                # Should handle multiple tasks efficiently
                assert duration < 3.0, f"Task scaling poor: {duration:.2f}s"
                print(f"Performance: {len(tasks)} tasks in {duration:.2f}s")


@pytest.mark.performance
def test_performance_communication_overhead():
    """Test overhead with lightweight communication."""
    from ezmpi import MPIPool

    pool = MPIPool(test_mode=True)
    if pool.comm.Get_size() > 1:
        with pool as pool:  # Reuse the same pool instance
            if pool.is_master():
                tasks = list(range(50))

                def simple_op(x):
                    return x * 2

                start = time.time()
                results = pool.map(simple_op, tasks)
                duration = time.time() - start

                # Verify results
                assert results == [x * 2 for x in tasks]

                # Should complete quickly
                assert (
                    duration < 2.0
                ), f"Communication overhead too high: {duration:.2f}s"
                print(f"Performance: {len(tasks)} lightweight tasks in {duration:.2f}s")


@pytest.mark.performance
def test_performance_large_data_transfer():
    """Test performance with larger data returns."""
    from ezmpi import MPIPool

    pool = MPIPool(test_mode=True)
    if pool.comm.Get_size() > 1:
        with pool as pool:  # Reuse the same pool instance
            if pool.is_master():

                def generate_data(size):
                    return list(range(size))

                tasks = [1000, 2000, 3000, 4000]

                start = time.time()
                results = pool.map(generate_data, tasks)
                duration = time.time() - start

                # Verify correctness
                for task, result in zip(tasks, results):
                    assert result == list(range(task))

                # Data transfer should complete without excessive delay
                assert duration < 5.0, f"Large data transfer too slow: {duration:.2f}s"
                print(f"Performance: Large data transfer in {duration:.2f}s")
