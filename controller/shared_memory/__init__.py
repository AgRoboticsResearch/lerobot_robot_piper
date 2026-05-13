"""Shared memory IPC utilities — port from UMI.

Provides lock-free inter-process communication:
- SharedMemoryQueue: FIFO command queue
- SharedMemoryRingBuffer: FILO state feedback buffer
- SharedNDArray: numpy array backed by shared memory
- SharedAtomicCounter: atomic integer for lock-free coordination
"""

from .queue import SharedMemoryQueue
from .ring_buffer import SharedMemoryRingBuffer
from .ndarray import SharedNDArray
from .util import ArraySpec, SharedAtomicCounter
