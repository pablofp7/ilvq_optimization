from collections import deque
import multiprocessing
from multiprocessing.managers import BaseManager
from multiprocessing import Manager

class DequesProxy:
    def __init__(self, num_deques, maxlen=None):
        self.deques = [(deque(maxlen=maxlen), multiprocessing.Lock()) for _ in range(num_deques)]

    def append(self, deque_index, item):
        deque, lock = self.deques[deque_index]
        with lock:
            deque.append(item)

    def appendleft(self, deque_index, item):
        deque, lock = self.deques[deque_index]
        with lock:
            deque.appendleft(item)

    def pop(self, deque_index):
        deque, lock = self.deques[deque_index]
        with lock:
            if deque:
                return deque.pop()
            else:
                pass
                # raise IndexError("pop from an empty deque")

    def popleft(self, deque_index):
        deque, lock = self.deques[deque_index]
        with lock:
            if deque:
                return deque.popleft()
            else:
                pass
                # raise IndexError("popleft from an empty deque")
    
    def getleft(self, deque_index):
        deque, lock = self.deques[deque_index]
        with lock:
            if deque:
                return deque[0]
            else:
                pass
                # raise IndexError("popleft from an empty deque")

    def get_length(self, deque_index):
        deque, lock = self.deques[deque_index]
        with lock:
            return len(deque)


    def extendleft(self, deque_index, item_list):
        deque, lock = self.deques[deque_index]
        for item in item_list:
            with lock:
                deque.appendleft(item)
                
class ListsProxy:
    def __init__(self, num_lists):
        self.lists = [([], multiprocessing.Lock()) for _ in range(num_lists)]

    def append(self, list_index, item):
        lst, lock = self.lists[list_index]
        lock.acquire()
        try:
            lst.append(item)
        finally:
            lock.release()

    def remove(self, list_index, item):
        lst, lock = self.lists[list_index]
        lock.acquire()
        try:
            lst.remove(item)
        finally:
            lock.release()

    def pop(self, list_index, index=-1):
        lst, lock = self.lists[list_index]
        lock.acquire()
        try:
            return lst.pop(index)
        finally:
            lock.release()

    def get_length(self, list_index):
        lst, lock = self.lists[list_index]
        lock.acquire()
        try:
            return len(lst)
        finally:
            lock.release()

    def get_item(self, list_index, index):
        lst, lock = self.lists[list_index]
        lock.acquire()
        try:
            return lst[index]
        finally:
            lock.release()

    def set_item(self, list_index, index, item):
        lst, lock = self.lists[list_index]
        lock.acquire()
        try:
            lst[index] = item
        finally:
            lock.release()

    def get_slice(self, list_index, start=None, stop=None, step=None):
        lst, lock = self.lists[list_index]
        lock.acquire()
        try:
            return lst[start:stop:step]
        finally:
            lock.release()

    def clear(self, list_index):
        lst, lock = self.lists[list_index]
        lock.acquire()
        try:
            lst.clear()
        finally:
            lock.release()

    def get_list(self, list_index):
        lst, lock = self.lists[list_index]
        lock.acquire()
        try:
            return list(lst)
        finally:
            lock.release()

        
# class ListsProxy:
#     def __init__(self, num_lists):
#         self.lists = [[] for _ in range(num_lists)]

#     def append(self, list_index, item):
#         self.lists[list_index].append(item)

#     def remove(self, list_index, item):
#         self.lists[list_index].remove(item)

#     def pop(self, list_index, index=-1):
#         return self.lists[list_index].pop(index)

#     def get_length(self, list_index):
#         return len(self.lists[list_index])

#     def get_item(self, list_index, index):
#         return self.lists[list_index][index]

#     def set_item(self, list_index, index, item):
#         self.lists[list_index][index] = item

#     def get_slice(self, list_index, start=None, stop=None, step=None):
#         return self.lists[list_index][start:stop:step]

#     def clear(self, list_index):
#         self.lists[list_index].clear()

#     def get_list(self, list_index):
#         # Retorna una copia de la lista
#         return list(self.lists[list_index])

        
        
        
class DequeManager(BaseManager):
    pass   

DequeManager.register('DequesProxy', DequesProxy, exposed=['append', 'appendleft', 'pop', 'popleft', 'get_length', 'extendleft', 'getleft'])
DequeManager.register('ListsProxy', ListsProxy, exposed=['append', 'remove', 'pop', 'get_length', 'get_item', 'set_item', 'get_slice', 'clear', 'get_list'])

# Añadir el método start_manager como método de clase
@classmethod
def start_manager(cls):
    m = cls()
    m.start()
    return m

# Asegúrate de añadir el método al DequeManager después de su definición
DequeManager.start_manager = start_manager