"""SharedNDArray: numpy array backed by shared memory.

Port from UMI (umi/shared_memory/shared_ndarray.py).
"""

from __future__ import annotations

from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt

SharedMemoryLike = Union[str, SharedMemory]
SharedT = TypeVar("SharedT", bound=np.generic)


class SharedNDArray:
    """Numpy array backed by shared memory for inter-process communication."""

    shm: SharedMemory
    dtype: np.dtype

    def __init__(
        self,
        shm: SharedMemoryLike,
        shape: Tuple[int, ...],
        dtype: npt.DTypeLike,
    ):
        if isinstance(shm, str):
            shm = SharedMemory(name=shm, create=False)
        dtype = np.dtype(dtype)
        assert shm.size >= (dtype.itemsize * np.prod(shape))
        self.shm = shm
        self.dtype = dtype
        self._shape: Tuple[int, ...] = shape

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @classmethod
    def create_from_shape(
        cls,
        mem_mgr: SharedMemoryManager,
        shape: Tuple,
        dtype: npt.DTypeLike,
    ) -> SharedNDArray:
        dtype = np.dtype(dtype)
        shm = mem_mgr.SharedMemory(int(np.prod(shape)) * dtype.itemsize)
        return cls(shm=shm, shape=shape, dtype=dtype)

    @classmethod
    def create_from_array(
        cls,
        mem_mgr: SharedMemoryManager,
        arr: npt.NDArray[SharedT],
    ) -> SharedNDArray:
        shared_arr = cls.create_from_shape(mem_mgr, arr.shape, arr.dtype)
        shared_arr.get()[:] = arr[:]
        return shared_arr

    def get(self) -> npt.NDArray[SharedT]:
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def __del__(self):
        self.shm.close()
