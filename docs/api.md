# API Reference

## mpi4py.MPI

EZMPI uses `mpi4py.MPI` for MPI communication. For details, see the [mpi4py documentation](https://mpi4py.readthedocs.io/).

## MPIPool

```{eval-rst}
.. autoclass:: ezmpi.MPIPool
   :members:
   :undoc-members:
   :show-inheritance:
```

## Examples

### Using MPIPool as a context manager

```python
from ezmpi import MPIPool

def worker(x):
    return x * 2

with MPIPool() as pool:
    results = pool.map(worker, [1, 2, 3, 4, 5])
    print(results)  # [2, 4, 6, 8, 10]
```

### Using a custom communicator

```python
from mpi4py import MPI
from ezmpi import MPIPool

comm = MPI.COMM_WORLD
with MPIPool(comm=comm) as pool:
    if pool.is_master():
        results = pool.map(worker, tasks)
```

### Using dill for complex functions

```python
from ezmpi import MPIPool

with MPIPool(use_dill=True) as pool:
    results = pool.map(lambda x: x * 2, [1, 2, 3])
```

### Checking process roles

```python
from ezmpi import MPIPool

with MPIPool() as pool:
    if pool.is_master():
        print("This is the master process")
        results = pool.map(worker, tasks)
    else:
        # This is a worker process - wait() is called automatically
        pass
```
