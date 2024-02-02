from .prototypes import PrototypeBuffer
from itertools import count


class BasePrototypes:
    """
    A basic data structure to hold prototype based learning
    """
    def __init__(self):
        #self.epoch = count(start=1)
        self.epoch = 1
        self.buffer: PrototypeBuffer = PrototypeBuffer()
