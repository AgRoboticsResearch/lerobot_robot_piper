"""Lock-free FIFO shared memory queue.

Port from UMI (umi/shared_memory/shared_memory_queue.py).
"""

import numbers
from queue import Empty, Full
from typing import Dict, List, Union

import numpy as np
from multiprocessing.managers import SharedMemoryManager

from .ndarray import SharedNDArray
from .util import ArraySpec, SharedAtomicCounter


class SharedMemoryQueue:
    """Lock-free FIFO shared memory queue for inter-process commands."""

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        array_specs: List[ArraySpec],
        buffer_size: int,
    ):
        write_counter = SharedAtomicCounter(shm_manager)
        read_counter = SharedAtomicCounter(shm_manager)

        shared_arrays = dict()
        for spec in array_specs:
            key = spec.name
            assert key not in shared_arrays
            array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(buffer_size,) + tuple(spec.shape),
                dtype=spec.dtype,
            )
            shared_arrays[key] = array

        self.buffer_size = buffer_size
        self.array_specs = array_specs
        self.write_counter = write_counter
        self.read_counter = read_counter
        self.shared_arrays = shared_arrays

    @classmethod
    def create_from_examples(
        cls,
        shm_manager: SharedMemoryManager,
        examples: Dict[str, Union[np.ndarray, numbers.Number]],
        buffer_size: int,
    ):
        specs = list()
        for key, value in examples.items():
            if isinstance(value, np.ndarray):
                shape = value.shape
                dtype = value.dtype
                assert dtype != np.dtype("O")
            elif isinstance(value, numbers.Number):
                shape = tuple()
                dtype = np.dtype(type(value))
            else:
                raise TypeError(f"Unsupported type {type(value)}")

            spec = ArraySpec(name=key, shape=shape, dtype=dtype)
            specs.append(spec)

        return cls(
            shm_manager=shm_manager,
            array_specs=specs,
            buffer_size=buffer_size,
        )

    def qsize(self):
        return self.write_counter.load() - self.read_counter.load()

    def empty(self):
        return self.qsize() <= 0

    def clear(self):
        self.read_counter.store(self.write_counter.load())

    def put(self, data: Dict[str, Union[np.ndarray, numbers.Number]]):
        read_count = self.read_counter.load()
        write_count = self.write_counter.load()
        n_data = write_count - read_count
        if n_data >= self.buffer_size:
            raise Full()

        next_idx = write_count % self.buffer_size

        for key, value in data.items():
            arr = self.shared_arrays[key].get()
            if isinstance(value, np.ndarray):
                arr[next_idx] = value
            else:
                arr[next_idx] = np.array(value, dtype=arr.dtype)

        self.write_counter.add(1)

    def get(self, out=None) -> Dict[str, np.ndarray]:
        write_count = self.write_counter.load()
        read_count = self.read_counter.load()
        n_data = write_count - read_count
        if n_data <= 0:
            raise Empty()

        if out is None:
            out = self._allocate_empty()

        next_idx = read_count % self.buffer_size
        for key, value in self.shared_arrays.items():
            arr = value.get()
            np.copyto(out[key], arr[next_idx])

        self.read_counter.add(1)
        return out

    def get_k(self, k, out=None) -> Dict[str, np.ndarray]:
        write_count = self.write_counter.load()
        read_count = self.read_counter.load()
        n_data = write_count - read_count
        if n_data <= 0:
            raise Empty()
        assert k <= n_data

        out = self._get_k_impl(k, read_count, out=out)
        self.read_counter.add(k)
        return out

    def get_all(self, out=None) -> Dict[str, np.ndarray]:
        write_count = self.write_counter.load()
        read_count = self.read_counter.load()
        n_data = write_count - read_count
        if n_data <= 0:
            raise Empty()

        out = self._get_k_impl(n_data, read_count, out=out)
        self.read_counter.add(n_data)
        return out

    def peek_all(self, out=None) -> Dict[str, np.ndarray]:
        """Read all items without advancing read counter (for visualization)."""
        write_count = self.write_counter.load()
        read_count = self.read_counter.load()
        n_data = write_count - read_count
        if n_data <= 0:
            raise Empty()

        out = self._get_k_impl(n_data, read_count, out=out)
        return out

    def _get_k_impl(self, k, read_count, out=None) -> Dict[str, np.ndarray]:
        if out is None:
            out = self._allocate_empty(k)

        curr_idx = read_count % self.buffer_size
        for key, value in self.shared_arrays.items():
            arr = value.get()
            target = out[key]

            start = curr_idx
            end = min(start + k, self.buffer_size)
            target_start = 0
            target_end = end - start
            target[target_start:target_end] = arr[start:end]

            remainder = k - (end - start)
            if remainder > 0:
                start = 0
                end = start + remainder
                target_start = target_end
                target_end = k
                target[target_start:target_end] = arr[start:end]

        return out

    def _allocate_empty(self, k=None):
        result = dict()
        for spec in self.array_specs:
            shape = spec.shape
            if k is not None:
                shape = (k,) + shape
            result[spec.name] = np.empty(shape=shape, dtype=spec.dtype)
        return result
