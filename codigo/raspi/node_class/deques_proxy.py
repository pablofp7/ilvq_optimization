from collections import deque
import multiprocessing
from multiprocessing.managers import BaseManager

LOGGING = False


class DequesProxy:
    def __init__(self, num_deques, maxlen=None, id=None):
        self.deques = [(deque(maxlen=maxlen), multiprocessing.Lock()) for _ in range(num_deques)]
        self.id = id
        self.log(f"[NODO {self.id}]DequesProxy initialized with num_deques: {num_deques} and maxlen: {maxlen}")

    def append(self, deque_index, item, call_method = ""):
        deque, lock = self.deques[deque_index]
        self.log(f"[NODO {self.id}] - {call_method} - Attempting to acquire lock for APPEND deque at index {deque_index}")
        with lock:
            self.log(f"[NODO {self.id}] - {call_method} - Acquired lock for APPEND deque at index {deque_index}")
            deque.append(item)
            # self.self.log(f"Appended item to deque at index {deque_index}")
        self.log(f"[NODO {self.id}] - {call_method} - Released lock for APPEND deque at index {deque_index}")            
            
    def appendleft(self, deque_index, item, call_method = ""):
        deque, lock = self.deques[deque_index]
        self.log(f"[NODO {self.id}] - {call_method} - Attempting to acquire lock for APPENDLEFT deque at index {deque_index}")
        with lock:
            self.log(f"[NODO {self.id}] - {call_method} - Acquired lock for APPENDLEFT deque at index {deque_index}")
            deque.appendleft(item)
            self.log(f"[NODO {self.id}] - {call_method} - Released lock for APPENDLEFT deque at index {deque_index}")

    def pop(self, deque_index, item_index, call_method = ""):
        deque, lock = self.deques[deque_index]
        self.log(f"[NODO {self.id}] - {call_method} - Attempting to acquire lock for POP deque at index {deque_index}")
        item = None
        with lock:
            self.log(f"[NODO {self.id}] - {call_method} - Acquired lock for POP deque at index {deque_index}")
            if deque:
                item = deque.pop()
                self.log(f"[NODO {self.id}] - {call_method} - Released lock for POP deque at index {deque_index}")
        return item

    def popleft(self, deque_index, call_method = ""):
        deque, lock = self.deques[deque_index]
        self.log(f"[NODO {self.id}] - {call_method} - Attempting to acquire lock for POPLEFT deque at index {deque_index}")
        item = None
        with lock:
            self.log(f"[NODO {self.id}] - {call_method} - Acquired lock for POPLEFT deque at index {deque_index}")
            if deque:
                item = deque.popleft()
                self.log(f"[NODO {self.id}] - {call_method} - Released lock for POPLEFT deque at index {deque_index}")
        return item

    def getleft(self, deque_index, call_method = ""):
        deque, lock = self.deques[deque_index]
        self.log(f"[NODO {self.id}] - {call_method} - Attempting to acquire lock for GETLEFT deque at index {deque_index}")
        item = None
        with lock:
            self.log(f"[NODO {self.id}] - {call_method} - Acquired lock for GETLEFT deque at index {deque_index}")
            if deque:
                item = deque[0]
                self.log(f"[NODO {self.id}] - {call_method} - Released lock for GETLEFT deque at index {deque_index}")
        return item

    def get_length(self, deque_index, call_method = ""):
        deque, lock = self.deques[deque_index]
        self.log(f"[NODO {self.id}] - {call_method} - Attempting to acquire lock for GET_LENGTH deque at index {deque_index}")
        length = 0
        with lock:
            self.log(f"[NODO {self.id}] - {call_method} - Acquired lock for GET_LENGTH deque at index {deque_index}")
            length = len(deque)
            self.log(f"[NODO {self.id}] - {call_method} - Released lock for GET_LENGTH deque at index {deque_index}")
        return length

    def extendleft(self, deque_index, item_list, call_method = ""):
        deque, lock = self.deques[deque_index]
        self.log(f"[NODO {self.id}] - {call_method} - Attempting to acquire lock for EXTENDLEFT deque at index {deque_index}")
        with lock:
            self.log(f"[NODO {self.id}] - {call_method} - Acquired lock for EXTENDLEFT deque at index {deque_index}")
            deque.extendleft(item_list)
            self.log(f"[NODO {self.id}] - {call_method} - Released lock for EXTENDLEFT deque at index {deque_index}")
            
    def clear(self, deque_index, call_method = ""):
        deque, lock = self.deques[deque_index]
        self.log(f"[NODO {self.id}] - {call_method} - Attempting to acquire lock for CLEAR deque at index {deque_index}")
        with lock:
            self.log(f"[NODO {self.id}] - {call_method} - Acquired lock for CLEAR deque at index {deque_index}")
            deque.clear()
            self.log(f"[NODO {self.id}] - {call_method} - Released lock for CLEAR deque at index {deque_index}")

    def log(self, message):
        if not LOGGING:
            return
        
        if self.id == 0:        
            print(message)
        else:
            print(message)



class ListsProxy:
    def __init__(self, num_lists, id):
        self.lists = [([], multiprocessing.Lock()) for _ in range(num_lists)]
        self.id = id
        self.log(f"[NODO {self.id}]: ListsProxy initialized with num_lists: {num_lists}")

    def append(self, list_index, item, call_method = ""):
        lst, lock = self.lists[list_index]
        self.log(f"[NODO {self.id}] - {call_method} - Attempting to acquire lock for append at index {list_index}")
        with lock:
            self.log(f"[NODO {self.id}] - {call_method} - Acquired lock for append at index {list_index}")
            lst.append(item)
            self.log(f"[NODO {self.id}] - {call_method} - Appended item to list at index {list_index}")
        self.log(f"[NODO {self.id}] - {call_method} - Released lock for append at index {list_index}")

    def remove(self, list_index, item, call_method = ""):
        lst, lock = self.lists[list_index]
        self.log(f"[NODO {self.id}] - {call_method} - Attempting to acquire lock for remove at index {list_index}")
        with lock:
            self.log(f"[NODO {self.id}] - {call_method} - Acquired lock for remove at index {list_index}")
            lst.remove(item)
            self.log(f"[NODO {self.id}] - {call_method} - Removed item from list at index {list_index}")
        self.log(f"[NODO {self.id}] - {call_method} - Released lock for remove at index {list_index}")

    def pop(self, list_index, index=-1, call_method = ""):
        lst, lock = self.lists[list_index]
        self.log(f"[NODO {self.id}] - {call_method} - Attempting to acquire lock for pop at index {list_index}")
        item = None
        with lock:
            self.log(f"[NODO {self.id}] - {call_method} - Acquired lock for pop at index {list_index}")
            item = lst.pop(index)
            self.log(f"[NODO {self.id}] - {call_method} - Popped item from list at index {list_index}")
        self.log(f"[NODO {self.id}] - {call_method} - Released lock for pop at index {list_index}")
        return item

    def get_length(self, list_index, call_method = ""):
        lst, lock = self.lists[list_index]
        self.log(f"[NODO {self.id}] - {call_method} - Attempting to acquire lock for get_length at index {list_index}")
        length = 0
        with lock:
            self.log(f"[NODO {self.id}] - {call_method} - Acquired lock for get_length at index {list_index}")
            length = len(lst)
            self.log(f"[NODO {self.id}] - {call_method} - Got length of list at index {list_index}")
        self.log(f"[NODO {self.id}] - {call_method} - Released lock for get_length at index {list_index}")
        return length

    def get_item(self, list_index, index, call_method = ""):
        lst, lock = self.lists[list_index]
        self.log(f"[NODO {self.id}] - {call_method} - Attempting to acquire lock for get_item at index {list_index}")
        item = None
        with lock:
            self.log(f"[NODO {self.id}] - {call_method} - Acquired lock for get_item at index {list_index}")
            item = lst[index]
            self.log(f"[NODO {self.id}] - {call_method} - Got item from list at index {list_index} at position {index}")
        self.log(f"[NODO {self.id}] - {call_method} - Released lock for get_item at index {list_index}")
        return item

    def set_item(self, list_index, index, item, call_method = ""):
        lst, lock = self.lists[list_index]
        self.log(f"[NODO {self.id}] - {call_method} - Attempting to acquire lock for set_item at index {list_index}")
        with lock:
            self.log(f"[NODO {self.id}] - {call_method} - Acquired lock for set_item at index {list_index}")
            lst[index] = item
            self.log(f"[NODO {self.id}] - {call_method} - Set item in list at index {list_index} at position {index}")
        self.log(f"[NODO {self.id}] - {call_method} - Released lock for set_item at index {list_index}")

    def get_slice(self, list_index, call_method = "", start=None, stop=None, step=None):
        lst, lock = self.lists[list_index]
        self.log(f"[NODO {self.id}] - {call_method} - Attempting to acquire lock for get_slice at index {list_index}")
        slice_ = None
        with lock:
            self.log(f"[NODO {self.id}] - {call_method} - Acquired lock for get_slice at index {list_index}")
            slice_ = lst[start:stop:step]
            self.log(f"[NODO {self.id}] - {call_method} - Got slice from list at index {list_index}")
        self.log(f"[NODO {self.id}] - {call_method} - Released lock for get_slice at index {list_index}")
        return slice_

    def clear(self, list_index, call_method = ""):
        lst, lock = self.lists[list_index]
        self.log(f"[NODO {self.id}] - {call_method} - Attempting to acquire lock for clear at index {list_index}")
        with lock:
            self.log(f"[NODO {self.id}] - {call_method} - Acquired lock for clear at index {list_index}")
            lst.clear()
            self.log(f"[NODO {self.id}] - {call_method} - Cleared list at index {list_index}")
        self.log(f"[NODO {self.id}] - {call_method} - Released lock for clear at index {list_index}")

    def get_list(self, list_index, call_method = ""):
        lst, lock = self.lists[list_index]
        self.log(f"[NODO {self.id}] - {call_method} - Attempting to acquire lock for get_list at index {list_index}")
        list_copy = None
        with lock:
            self.log(f"[NODO {self.id}] - {call_method} - Acquired lock for get_list at index {list_index}")
            list_copy = list(lst)
            self.log(f"[NODO {self.id}] - {call_method} - Got copy of list at index {list_index}")
        self.log(f"[NODO {self.id}] - {call_method} - Released lock for get_list at index {list_index}")
        return list_copy

    def log(self, message):
        if not LOGGING:
            return
        if self.id == 0:   
            print(message)
        else:
            print(message)
        return

class DequeManager(BaseManager):
    pass

DequeManager.register('DequesProxy', DequesProxy, exposed=['append', 'appendleft', 'pop', 'popleft', 'get_length', 'extendleft', 'getleft', 'clear'])
DequeManager.register('ListsProxy', ListsProxy, exposed=['append', 'remove', 'pop', 'get_length', 'get_item', 'set_item', 'get_slice', 'clear', 'get_list'])

@classmethod
def start_manager(cls):
    m = cls()
    m.start()
    return m

DequeManager.start_manager = start_manager
