import sys
import os
ruta_directorio_main = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_directorio_main not in sys.path:
    sys.path.append(ruta_directorio_main)
    
from old_node_class.deques_proxy import DequeManager
import multiprocessing
import time
import random

def worker(shared_deques, deque_index, worker_id):
    # Añadir elementos al deque especificado usando appendleft
    for i in range(5):
        item = f'worker{worker_id}_item{i}'
        print(f"{multiprocessing.current_process().name} adding '{item}' to deque {deque_index}")
        shared_deques.appendleft(deque_index, item)
        time.sleep(random.uniform(0.1, 0.5))  # Espera un tiempo aleatorio para simular trabajo

    # Retirar elementos del deque usando popleft
    time.sleep(1)  # Espera para asegurar que todos los trabajadores hayan añadido elementos
    while True:
        try:
            item = shared_deques.popleft(deque_index)
            print(f"{multiprocessing.current_process().name} removed '{item}' from deque {deque_index}")
            time.sleep(random.uniform(0.1, 0.5))  # Espera un tiempo aleatorio para simular trabajo
        except IndexError:
            print(f"{multiprocessing.current_process().name} found deque {deque_index} empty.")
            break

if __name__ == '__main__':
    manager = DequeManager()
    manager.start()

    num_deques = 2  # Ejemplo: 2 deques
    shared_deques = manager.DequesProxy(num_deques)

    # Crear y ejecutar procesos trabajadores
    workers = [multiprocessing.Process(target=worker, args=(shared_deques, i % num_deques, i)) for i in range(4)]

    for w in workers:
        w.start()

    for w in workers:
        w.join()
