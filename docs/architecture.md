# ezMPI Architecture

This document describes the technical architecture and design decisions behind ezMPI.

## Design Philosophy

ezMPI follows a simple master-worker pattern designed for ease of use and reliability. The primary goals are:

- **Simplicity**: Easy to understand and use API
- **Reliability**: Robust error handling and cleanup
- **Performance**: Efficient task distribution and communication
- **Compatibility**: Works with various MPI implementations and Python versions

## Core Components

### MPIPool Class

The `MPIPool` class is the main interface that coordinates parallel execution across MPI processes.

```python
class MPIPool:
    def __init__(self, comm=None, use_dill=True, test_mode=False):
        self.comm = comm or MPI.COMM_WORLD
        self.master = 0
        self.rank = self.comm.Get_rank()
        # ...
```

**Key Responsibilities:**
- Process role management (master vs worker)
- Task distribution via `map()`
- Result collection and ordering
- Cleanup and shutdown coordination

### Process Roles

**Master Process (rank 0):**
- Receives tasks from user
- Distributes tasks to workers
- Collects results
- Maintains result order
- Initiates shutdown

**Worker Processes (rank > 0):**
- Wait for tasks from master
- Execute tasks
- Return results to master
- Handle shutdown signals

### Communication Flow

```
┌─────────────┐
│  Master (0) │
└──────┬──────┘
       │ 1. Send task to worker
       ▼
┌─────────────┐
│ Worker (1+) │
└──────┬──────┘
       │ 2. Execute task
       │ 3. Return result
       ▼
┌─────────────┐
│  Master (0) │
└─────────────┘
       4. Collect and order results
```

### Task Distribution

The `map()` method implements a simple but effective task distribution:

```python
def map(self, worker, tasks):
    """Distribute tasks to workers and collect results."""
    # Master packages tasks
    # Workers execute when available
    # Results collected in order
```

**Key Features:**
- Synchronous task distribution
- Results maintained in original order
- Automatic worker load balancing via MPI
- Graceful shutdown handling

## Serialization

### Default: Standard Pickle

By default, ezMPI uses Python's standard `pickle` module for serialization, which works well for most basic Python objects.

### Optional: Dill Support

For complex objects (lambdas, closures, class instances), ezMPI supports `dill`:

```python
# Enable dill serialization
pool = MPIPool(use_dill=True)
```

Dill provides more comprehensive object serialization than standard pickle.

## Error Handling

### Initialization Errors

- **Single Process**: Raises `ValueError` if only one MPI process is available
- **Import Errors**: Clear messages for missing dependencies (mpi4py, dill)

### Runtime Errors

- **Worker Crashes**: Master detects worker failures via MPI communication
- **Task Failures**: Exceptions in worker tasks are propagated to master
- **Cleanup**: `atexit` handlers ensure proper MPI cleanup

## Testing Strategy

ezMPI uses a comprehensive testing approach:

### Unit Tests (`tests/test_unit.py`)
- Mock MPI environment using `unittest.mock`
- Test individual components without real MPI
- Fast execution, comprehensive coverage

### Integration Tests (`tests/test_integration.py`)
- Real MPI with multiple processes (2, 4, 8 nodes)
- Test end-to-end functionality
- Validate parallel execution and communication

### Performance Tests (`tests/test_performance.py`)
- Benchmark parallel speedup
- Test scalability with varying task counts
- Measure communication overhead

## Performance Considerations

### Communication Overhead

Task size vs communication overhead trade-offs:

- **Large Tasks**: Reduce relative communication cost
- **Small Tasks**: May have higher overhead, use batching
- **Data Transfer**: Minimize data movement, compute where possible

### Load Balancing

ezMPI relies on MPI's built-in load balancing:
- Workers pull tasks as they become available
- No explicit load balancing code needed
- Works well for homogeneous tasks

### Memory Management

- Each worker process has independent memory space
- Results are collected back to master
- Consider memory limits when working with large datasets

## Future Improvements

Potential enhancements for future versions:

1. **Asynchronous Task Distribution**: Non-blocking task submission
2. **Dynamic Load Balancing**: More sophisticated for heterogeneous tasks
3. **Fault Tolerance**: Handle worker failures more gracefully
4. **Progress Tracking**: Real-time task progress monitoring
5. **Memory Optimization**: Stream large datasets

## Compatibility

ezMPI is compatible with:
- **Python**: 3.8, 3.10, 3.12+
- **MPI**: OpenMPI, MPICH, MS-MPI
- **OS**: Linux, macOS, Windows (with MPI)
- **Pickle**: Standard library + optional dill support

## Learn More

- [Getting Started](getting_started.md) - Quick start guide
- [API Reference](api.md) - Complete API documentation
- [Contributing](contributing.md) - How to contribute

## Questions?

For questions about the architecture, please open an issue on [GitHub](https://github.com/minaskar/ezmpi/issues).
