"""Lock-free FILO shared memory ring buffer for state feedback.

Port from UMI (umi/shared_memory/shared_memory_ring_buffer.py).
"""

import numbers
import time
from queue import Empty
from typing import Dict, List, Union

import numpy as np
from multiprocessing.managers import SharedMemoryManager

from .ndarray import SharedNDArray
from .util import ArraySpec, SharedAtomicCounter


class SharedMemoryRingBuffer:
    """Lock-free FILO shared memory ring buffer for robot state feedback."""

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        array_specs: List[ArraySpec],
        get_max_k: int,
        get_time_budget: float,
        put_desired_frequency: float,
        safety_margin: float = 1.5,
    ):
        counter = SharedAtomicCounter(shm_manager)

        buffer_size = (
            int(
                np.ceil(
                    put_desired_frequency * get_time_budget * safety_margin
                )
            )
            + get_max_k
        )

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

        timestamp_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager,
            shape=(buffer_size,),
            dtype=np.float64,
        )
        timestamp_array.get()[:] = -np.inf

        self.buffer_size = buffer_size
        self.array_specs = array_specs
        self.counter = counter
        self.shared_arrays = shared_arrays
        self.timestamp_array = timestamp_array
        self.get_time_budget = get_time_budget
        self.get_max_k = get_max_k
        self.put_desired_frequency = put_desired_frequency

    @property
    def count(self):
        return self.counter.load()

    @classmethod
    def create_from_examples(
        cls,
        shm_manager: SharedMemoryManager,
        examples: Dict[str, Union[np.ndarray, numbers.Number]],
        get_max_k: int = 32,
        get_time_budget: float = 0.01,
        put_desired_frequency: float = 60,
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
            get_max_k=get_max_k,
            get_time_budget=get_time_budget,
            put_desired_frequency=put_desired_frequency,
        )

    def clear(self):
        self.counter.store(0)

    def put(
        self,
        data: Dict[str, Union[np.ndarray, numbers.Number]],
        wait: bool = True,
    ):
        count = self.counter.load()
        next_idx = count % self.buffer_size

        timestamp_lookahead_idx = (next_idx + self.get_max_k - 1) % self.buffer_size
        old_timestamp = self.timestamp_array.get()[timestamp_lookahead_idx]
        t = time.monotonic()
        if (t - old_timestamp) < self.get_time_budget:
            deltat = t - old_timestamp
            if wait:
                time.sleep(self.get_time_budget - deltat)
            else:
                past_iters = self.buffer_size - self.get_max_k
                hz = past_iters / deltat
                raise TimeoutError(
                    f"Put executed too fast {past_iters}items/{deltat:.4f}s ~= {hz}Hz"
                )

        for key, value in data.items():
            arr = self.shared_arrays[key].get()
            if isinstance(value, np.ndarray):
                arr[next_idx] = value
            else:
                arr[next_idx] = np.array(value, dtype=arr.dtype)

        self.timestamp_array.get()[next_idx] = time.monotonic()
        self.counter.add(1)

    def _allocate_empty(self, k=None):
        result = dict()
        for spec in self.array_specs:
            shape = spec.shape
            if k is not None:
                shape = (k,) + shape
            result[spec.name] = np.empty(shape=shape, dtype=spec.dtype)
        return result

    def get(self, out=None) -> Dict[str, np.ndarray]:
        if out is None:
            out = self._allocate_empty()
        start_time = time.monotonic()
        count = self.counter.load()
        curr_idx = (count - 1) % self.buffer_size
        for key, value in self.shared_arrays.items():
            arr = value.get()
            np.copyto(out[key], arr[curr_idx])
        end_time = time.monotonic()
        dt = end_time - start_time
        if dt > self.get_time_budget:
            raise TimeoutError(f"Get time out {dt} vs {self.get_time_budget}")
        return out

    def get_last_k(self, k: int, out=None) -> Dict[str, np.ndarray]:
        assert k <= self.get_max_k
        if out is None:
            out = self._allocate_empty(k)
        start_time = time.monotonic()
        count = self.counter.load()
        assert k <= count
        curr_idx = (count - 1) % self.buffer_size
        for key, value in self.shared_arrays.items():
            arr = value.get()
            target = out[key]

            end = curr_idx + 1
            start = max(0, end - k)
            target_end = k
            target_start = target_end - (end - start)
            target[target_start:target_end] = arr[start:end]

            remainder = k - (end - start)
            if remainder > 0:
                end = self.buffer_size
                start = end - remainder
                target_start = 0
                target_end = end - start
                target[target_start:target_end] = arr[start:end]
        end_time = time.monotonic()
        dt = end_time - start_time
        if dt > self.get_time_budget:
            raise TimeoutError(f"Get time out {dt} vs {self.get_time_budget}")
        return out

    def get_all(self) -> Dict[str, np.ndarray]:
        k = min(self.count, self.get_max_k)
        return self.get_last_k(k=k)
