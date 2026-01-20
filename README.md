# ezMPI

A simple MPI-based processing pool for Python that distributes tasks across multiple processes using MPI (Message Passing Interface).

## Features

- Simple API similar to Python's built-in `map()` function
- Automatic worker process management
- Support for complex objects via optional `dill` integration
- Context manager support for automatic cleanup
- Lightweight and easy to integrate into existing projects

## Requirements

- Python 3.8+
- MPI implementation (OpenMPI, MPICH, or similar)
- `mpi4py` library

## Installation

### Install from source

```bash
git clone https://github.com/minask/ezmpi.git
cd ezmpi
pip install -e .
```

### Install with development dependencies

```bash
pip install -e ".[dev]"
```

### Install with dill support

```bash
pip install -e ".[dill]"
```

## Quick Start

### Basic Usage

```python
from ezmpi import MPIPool

def square(x):
    return x * x

# Create a pool (must be run with mpiexec)
with MPIPool() as pool:
    results = pool.map(square, [1, 2, 3, 4, 5])
    print(results)  # [1, 4, 9, 16, 25]
```

Run with multiple processes:
```bash
mpiexec -n 4 python your_script.py
```

### Using dill for Complex Objects

```python
from ezmpi import MPIPool

def process_data(data):
    # Complex processing logic
    return sum(data) / len(data)

# Enable dill for better pickling support
with MPIPool(use_dill=True) as pool:
    results = pool.map(process_data, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(results)  # [2.0, 5.0, 8.0]
```

## MPI Execution

ezMPI requires at least 2 MPI processes (1 master + 1+ workers):

```bash
# Minimum: 2 processes
mpiexec -n 2 python script.py

# Typical: 4-8 processes
mpiexec -n 4 python script.py

# On HPC systems
srun -n 32 python script.py
```

**Note**: The master process (rank 0) coordinates tasks, all other processes are workers.

## API Reference

### MPIPool

```python
class MPIPool(comm=None, use_dill=True)
```

**Parameters:**
- `comm`: MPI communicator (optional, defaults to MPI.COMM_WORLD)
- `use_dill`: Use dill for pickling instead of standard pickle (default: True)

**Methods:**
- `map(worker, tasks)`: Execute worker function on each task in parallel
- `close()`: Shutdown worker processes
- `is_master()`: Check if current process is the master
- `is_worker()`: Check if current process is a worker

## Development

### Setup Development Environment

```bash
git clone https://github.com/minask/ezmpi.git
cd ezmpi
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run basic tests pytest -v

# Run MPI tests
mpiexec -n 4 pytest -v

# Run with coverage
pytest --cov=ezmpi --cov-report=html
```

### Code Quality

```bash
# Lint code
ruff check .

# Format code
ruff format .

# Type checking
mypy src/ezmpi

# Import sorting
isort src/
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This implementation is inspired by and adapted from similar MPI pool implementations in the scientific Python community.
