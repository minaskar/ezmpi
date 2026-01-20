# ezMPI Documentation

**A simple MPI processing pool for Python**

[![Documentation Status](https://readthedocs.org/projects/ezmpi/badge/?version=latest)](https://ezmpi.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/ezmpi.svg)](https://pypi.org/project/ezmpi/)

## What is ezMPI?

ezMPI is a simple, easy-to-use MPI processing pool for Python that helps you parallelize computationally intensive tasks across multiple processes. It follows a master-worker pattern where the master process distributes tasks to workers, which execute them in parallel and return results.

## Quick Start

Install ezMPI:

```bash
pip install ezmpi
```

Basic usage:

```python
from ezmpi import MPIPool

def square(x):
    return x * x

# Run with: mpiexec -n 4 python script.py
with MPIPool() as pool:
    results = pool.map(square, [1, 2, 3, 4, 5])
    print(results)  # [1, 4, 9, 16, 25]
```

## Features

- 🚀 **Simple API**: Just like Python's built-in `map()` function
- 🔄 **Automatic worker management**: No manual process management needed
- 🧩 **Flexible pickling**: Optional dill support for complex objects
- 🧹 **Context manager support**: Automatic cleanup with `with` statement
- 📈 **Scalable**: Works from laptops to clusters
- 🧪 **Well tested**: Comprehensive test suite with 31 tests

## Documentation Structure

- **[Getting Started](getting-started.md)**: Installation and basic usage
- **[API Reference](api.md)**: Complete API documentation
- **[Architecture](architecture.md)**: Design decisions and implementation details
- **[Contributing](contributing.md)**: How to contribute to ezMPI
- **[Changelog](changelog.md)**: Version history and changes

## Installation

### From PyPI (recommended)

```bash
pip install ezmpi
```

### From Source

```bash
git clone https://github.com/minaskar/ezmpi.git
cd ezmpi
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/minaskar/ezmpi.git
cd ezmpi
pip install -e ".[dev,test,docs]"
```

## Running with MPI

ezMPI requires at least 2 MPI processes (1 master + 1 worker):

```bash
# Minimum: 2 processes
mpiexec -n 2 python script.py

# Typical: 4-8 processes
mpiexec -n 4 python script.py

# On HPC clusters
srun -n 32 python script.py
```

## Requirements

- Python 3.8+
- MPI implementation (OpenMPI, MPICH, or similar)
- `mpi4py>=3.0.0`

Optional dependencies:
- `dill>=0.3.0` (for enhanced pickling)

## Examples

### Basic Mapping

```python
from ezmpi import MPIPool

def cube(x):
    return x ** 3

with MPIPool() as pool:
    results = pool.map(cube, [1, 2, 3, 4, 5])
    print(results)  # [1, 8, 27, 64, 125]
```

### Using dill for Lambda Functions

```python
from ezmpi import MPIPool

with MPIPool(use_dill=True) as pool:
    results = pool.map(lambda x: x * 2, [1, 2, 3, 4])
    print(results)  # [2, 4, 6, 8]
```

### CPU-Bound Computations

```python
from ezmpi import MPIPool
import math

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

numbers = list(range(2, 1000))
with MPIPool() as pool:
    primes = [n for n, is_p in zip(numbers, pool.map(is_prime, numbers)) if is_p]
    print(f"Found {len(primes)} primes between 2 and 1000")
```

## API Overview

The main API is simple and consists of just a few methods:

- `MPIPool(comm=None, use_dill=True)` - Create a pool
- `pool.map(worker, tasks)` - Execute worker on each task in parallel
- `pool.close()` - Shutdown worker processes
- `pool.is_master()` - Check if this is the master process
- `pool.is_worker()` - Check if this is a worker process

For detailed API documentation, see the [API Reference](api.md).

## Contributing

We welcome contributions! Please see our [Contributing Guide](contributing.md) for details on:
- Setting up a development environment
- Running tests
- Submitting pull requests
- Code style guidelines

## License

ezMPI is licensed under the MIT License. See the [LICENSE](https://github.com/minaskar/ezmpi/blob/main/LICENSE) file for details.

## Support

- **Documentation**: https://ezmpi.readthedocs.io/
- **GitHub Issues**: https://github.com/minaskar/ezmpi/issues
- **GitHub Discussions**: https://github.com/minaskar/ezmpi/discussions

## Citation

If you use ezMPI in your research, please cite:

```bibtex
@software{ezmpi,
  title = {ezMPI: A simple MPI processing pool for Python},
  author = {Karamanis, Minas},
  url = {https://github.com/minaskar/ezmpi},
  year = {2026}
}
```

---

**Enjoy using ezMPI!** 🚀
