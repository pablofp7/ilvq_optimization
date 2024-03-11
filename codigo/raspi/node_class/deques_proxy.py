import logging
import traceback
import inspect
from collections import deque
import multiprocessing
from multiprocessing.managers import BaseManager
import sys

# Configura el logging para que escriba en un archivo
log_filename = 'test_log.txt'
# logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s')


class DequesProxy:
    def __init__(self, num_deques, maxlen=None):
        self.deques = [(deque(maxlen=maxlen), multiprocessing.Lock()) for _ in range(num_deques)]
        self.log("DequesProxy initialized with num_deques: {} and maxlen: {}".format(num_deques, maxlen))

    def append(self, deque_index, item):
        deque, lock = self.deques[deque_index]
        self.log(f"Attempting to acquire lock for append deque at index {deque_index}")
        with lock:
            self.log(f"Acquired lock for append deque at index {deque_index}")
            deque.append(item)
            # self.log(f"Appended item to deque at index {deque_index}")
        self.log(f"Released lock for append deque at index {deque_index}")
            
            
    def appendleft(self, deque_index, item):
        deque, lock = self.deques[deque_index]
        self.log(f"Attempting to acquire lock for appendleft at index {deque_index}")
        with lock:
            self.log(f"Acquired lock for appendleft at index {deque_index}")
            deque.appendleft(item)
            self.log(f"Appended item to left of deque at index {deque_index}")
        self.log(f"Released lock for appendleft at index {deque_index}")

    def pop(self, deque_index):
        deque, lock = self.deques[deque_index]
        self.log(f"Attempting to acquire lock for pop at index {deque_index}")
        item = None
        with lock:
            self.log(f"Acquired lock for pop at index {deque_index}")
            if deque:
                item = deque.pop()
                self.log(f"Popped item from deque at index {deque_index}")
        self.log(f"Released lock for pop at index {deque_index}")
        return item

    def popleft(self, deque_index):
        deque, lock = self.deques[deque_index]
        self.log(f"Attempting to acquire lock for popleft at index {deque_index}")
        item = None
        with lock:
            self.log(f"Acquired lock for popleft at index {deque_index}")
            if deque:
                item = deque.popleft()
                self.log(f"Popped item from left of deque at index {deque_index}")
        self.log(f"Released lock for popleft at index {deque_index}")
        return item

    def getleft(self, deque_index):
        deque, lock = self.deques[deque_index]
        self.log(f"Attempting to acquire lock for getleft at index {deque_index}")
        item = None
        with lock:
            self.log(f"Acquired lock for getleft at index {deque_index}")
            if deque:
                item = deque[0]
                self.log(f"Got leftmost item from deque at index {deque_index}")
        self.log(f"Released lock for getleft at index {deque_index}")
        return item

    def get_length(self, deque_index):
        deque, lock = self.deques[deque_index]
        self.log(f"Attempting to acquire lock for get_length at index {deque_index}")
        length = 0
        with lock:
            self.log(f"Acquired lock for get_length at index {deque_index}")
            length = len(deque)
            self.log(f"Got length of deque at index {deque_index}")
        self.log(f"Released lock for get_length at index {deque_index}")
        return length

    def extendleft(self, deque_index, item_list):
        deque, lock = self.deques[deque_index]
        self.log(f"Attempting to acquire lock for extendleft at index {deque_index}")
        with lock:
            self.log(f"Acquired lock for extendleft at index {deque_index}")
            deque.extendleft(item_list)
            self.log(f"Extended left of deque at index {deque_index} with items")
        self.log(f"Released lock for extendleft at index {deque_index}")


    def log(self, message):
        process_name = multiprocessing.current_process().name
        # Start with the current frame, then find the caller's frame and go back up to 3 levels.
        frame = inspect.currentframe()
        caller_frames = []

        # Collect up to 3 levels of caller information.
        for _ in range(4):  # Current frame + 3 callers
            if frame is None:
                break
            caller_frames.append(frame)
            frame = frame.f_back

        # Format the collected caller information.
        caller_info = ''
        for f in caller_frames[1:]:  # Skip the first frame which is this log function itself.
            filename = f.f_code.co_filename
            lineno = f.f_lineno
            func_name = f.f_code.co_name
            caller_info += f"File \"{filename}\", line {lineno}, in {func_name}\n"

        # Trim the last newline character for cleaner formatting.
        if caller_info.endswith('\n'):
            caller_info = caller_info[:-1]

        # Log the message with caller information.
        # logging.debug(f"{message}\nProcess: {process_name}\nStack trace:\n{caller_info}")

        # Cleanup to prevent reference cycles.
        del caller_frames

class ListsProxy:
    def __init__(self, num_lists):
        self.lists = [([], multiprocessing.Lock()) for _ in range(num_lists)]
        self.log("ListsProxy initialized with num_lists: {}".format(num_lists))

    def append(self, list_index, item):
        lst, lock = self.lists[list_index]
        self.log(f"Attempting to acquire lock for append at index {list_index}")
        with lock:
            self.log(f"Acquired lock for append at index {list_index}")
            lst.append(item)
            self.log(f"Appended item to list at index {list_index}")
        self.log(f"Released lock for append at index {list_index}")

    def remove(self, list_index, item):
        lst, lock = self.lists[list_index]
        self.log(f"Attempting to acquire lock for remove at index {list_index}")
        with lock:
            self.log(f"Acquired lock for remove at index {list_index}")
            lst.remove(item)
            self.log(f"Removed item from list at index {list_index}")
        self.log(f"Released lock for remove at index {list_index}")

    def pop(self, list_index, index=-1):
        lst, lock = self.lists[list_index]
        self.log(f"Attempting to acquire lock for pop at index {list_index}")
        item = None
        with lock:
            self.log(f"Acquired lock for pop at index {list_index}")
            item = lst.pop(index)
            self.log(f"Popped item from list at index {list_index}")
        self.log(f"Released lock for pop at index {list_index}")
        return item

    def get_length(self, list_index):
        lst, lock = self.lists[list_index]
        self.log(f"Attempting to acquire lock for get_length at index {list_index}")
        length = 0
        with lock:
            self.log(f"Acquired lock for get_length at index {list_index}")
            length = len(lst)
            self.log(f"Got length of list at index {list_index}")
        self.log(f"Released lock for get_length at index {list_index}")
        return length

    def get_item(self, list_index, index):
        lst, lock = self.lists[list_index]
        self.log(f"Attempting to acquire lock for get_item at index {list_index}")
        item = None
        with lock:
            self.log(f"Acquired lock for get_item at index {list_index}")
            item = lst[index]
            self.log(f"Got item from list at index {list_index} at position {index}")
        self.log(f"Released lock for get_item at index {list_index}")
        return item

    def set_item(self, list_index, index, item):
        lst, lock = self.lists[list_index]
        self.log(f"Attempting to acquire lock for set_item at index {list_index}")
        with lock:
            self.log(f"Acquired lock for set_item at index {list_index}")
            lst[index] = item
            self.log(f"Set item in list at index {list_index} at position {index}")
        self.log(f"Released lock for set_item at index {list_index}")

    def get_slice(self, list_index, start=None, stop=None, step=None):
        lst, lock = self.lists[list_index]
        self.log(f"Attempting to acquire lock for get_slice at index {list_index}")
        slice_ = None
        with lock:
            self.log(f"Acquired lock for get_slice at index {list_index}")
            slice_ = lst[start:stop:step]
            self.log(f"Got slice from list at index {list_index}")
        self.log(f"Released lock for get_slice at index {list_index}")
        return slice_

    def clear(self, list_index):
        lst, lock = self.lists[list_index]
        self.log(f"Attempting to acquire lock for clear at index {list_index}")
        with lock:
            self.log(f"Acquired lock for clear at index {list_index}")
            lst.clear()
            self.log(f"Cleared list at index {list_index}")
        self.log(f"Released lock for clear at index {list_index}")

    def get_list(self, list_index):
        lst, lock = self.lists[list_index]
        self.log(f"Attempting to acquire lock for get_list at index {list_index}")
        list_copy = None
        with lock:
            self.log(f"Acquired lock for get_list at index {list_index}")
            list_copy = list(lst)
            self.log(f"Got copy of list at index {list_index}")
        self.log(f"Released lock for get_list at index {list_index}")
        return list_copy


    def log(self, message):
        process_name = multiprocessing.current_process().name
        # Start with the current frame, then find the caller's frame and go back up to 3 levels.
        frame = inspect.currentframe()
        caller_frames = []

        # Collect up to 3 levels of caller information.
        for _ in range(4):  # Current frame + 3 callers
            if frame is None:
                break
            caller_frames.append(frame)
            frame = frame.f_back

        # Format the collected caller information.
        caller_info = ''
        for f in caller_frames[1:]:  # Skip the first frame which is this log function itself.
            filename = f.f_code.co_filename
            lineno = f.f_lineno
            func_name = f.f_code.co_name
            caller_info += f"File \"{filename}\", line {lineno}, in {func_name}\n"

        # Trim the last newline character for cleaner formatting.
        if caller_info.endswith('\n'):
            caller_info = caller_info[:-1]

        # Log the message with caller information.
        logging.debug(f"{message}\nProcess: {process_name}\nStack trace:\n{caller_info}")

        # Cleanup to prevent reference cycles.
        del caller_frames

class DequeManager(BaseManager):
    pass

DequeManager.register('DequesProxy', DequesProxy, exposed=['append', 'appendleft', 'pop', 'popleft', 'get_length', 'extendleft', 'getleft'])
DequeManager.register('ListsProxy', ListsProxy, exposed=['append', 'remove', 'pop', 'get_length', 'get_item', 'set_item', 'get_slice', 'clear', 'get_list'])

@classmethod
def start_manager(cls):
    m = cls()
    m.start()
    return m

DequeManager.start_manager = start_manager
