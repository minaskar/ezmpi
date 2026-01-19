#!/usr/bin/env python3
"""Test runner for EZMPI that works around pytest-mpi issues."""

import sys
from ezmpi import MPIPool


def square(x):
    return x * x


def add_one(x):
    return x + 1


def complex_computation(x):
    return sum(i * i for i in range(x + 1))


def test_basic_map():
    """Test basic parallel mapping."""
    tasks = [1, 2, 3, 4, 5]
    with MPIPool() as pool:
        if pool.is_master():
            results = pool.map(square, tasks)
            assert results == [1, 4, 9, 16, 25], (
                f"Expected [1, 4, 9, 16, 25], got {results}"
            )
    return True


def test_multiple_workers():
    """Test with multiple workers."""
    with MPIPool() as pool:
        if pool.is_master():
            assert pool.size >= 1, f"Expected at least 1 worker, got {pool.size}"
            assert len(pool.workers) == pool.size
    return True


def test_result_ordering():
    """Test that results maintain correct order."""
    tasks = [5, 3, 8, 1, 9, 2, 7, 4, 6]
    with MPIPool() as pool:
        if pool.is_master():
            results = pool.map(square, tasks)
            expected = [x * x for x in tasks]
            assert results == expected, f"Expected {expected}, got {results}"
    return True


def test_empty_tasks():
    """Test map with empty task list."""
    with MPIPool() as pool:
        if pool.is_master():
            results = pool.map(square, [])
            assert results == [], f"Expected [], got {results}"
    return True


def test_single_task():
    """Test with single task."""
    with MPIPool() as pool:
        if pool.is_master():
            tasks = [42]
            results = pool.map(square, tasks)
            assert results == [1764], f"Expected [1764], got {results}"
    return True


def test_many_tasks():
    """Test with more tasks than workers."""
    tasks = list(range(20))
    with MPIPool() as pool:
        if pool.is_master():
            results = pool.map(square, tasks)
            expected = [x * x for x in tasks]
            assert results == expected, f"Expected {expected}, got {results}"
    return True


def test_cpu_bound():
    """Test with CPU-bound computations."""
    tasks = [100, 200, 300]
    with MPIPool() as pool:
        if pool.is_master():
            results = pool.map(complex_computation, tasks)
            expected = [complex_computation(x) for x in tasks]
            assert results == expected, f"Expected {expected}, got {results}"
    return True


def test_context_manager():
    """Test context manager."""
    with MPIPool() as pool:
        if pool.is_master():
            assert pool.is_master()
            tasks = [1, 2, 3]
            results = pool.map(square, tasks)
            assert results == [1, 4, 9]
    return True


# List of all tests to run
ALL_TESTS = [
    ("basic_map", test_basic_map),
    ("multiple_workers", test_multiple_workers),
    ("result_ordering", test_result_ordering),
    ("empty_tasks", test_empty_tasks),
    ("single_task", test_single_task),
    ("many_tasks", test_many_tasks),
    ("cpu_bound", test_cpu_bound),
    ("context_manager", test_context_manager),
]


def run_all_tests():
    """Run all integration tests."""
    failed = []
    passed = []

    with MPIPool() as pool:
        if pool.is_master():
            print(
                f"Running integration tests with {pool.size + 1} processes ({pool.size} workers + 1 master)\n"
            )

            for test_name, test_func in ALL_TESTS:
                try:
                    print(f"Running {test_name}...", end=" ")
                    sys.stdout.flush()
                    test_func()
                    print("✓ PASSED")
                    passed.append(test_name)
                except Exception as e:
                    print(f"✗ FAILED: {e}")
                    failed.append((test_name, str(e)))

            print(f"\n{'=' * 60}")
            print(f"Tests run: {len(ALL_TESTS)}")
            print(f"Passed: {len(passed)}")

            if failed:
                print(f"Failed: {len(failed)}")
                print("\nFailed tests:")
                for name, error in failed:
                    print(f"  - {name}: {error}")
                sys.exit(1)
            else:
                print("\n✅ All integration tests passed!")
                sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
