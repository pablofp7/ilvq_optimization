import zmq
import threading
import time
import sys
import random
import pickle
import queue
import socket as socket_lib



def get_node_id():
    hostname = socket_lib.gethostname()
    # Extraer el ID del nodo a partir del nombre de la máquina (nodoX)
    id_str = ''.join(filter(str.isdigit, hostname))
    return int(id_str)


def start_server():
    global to_write
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt_string(zmq.IDENTITY, f"{id}")
    socket.bind(f"tcp://*:{mi_puerto}")

    print(f"Nodo {id} escuchando en el mi_puerto {mi_puerto}...")

    while True:
        identidad, destinos, mensaje = socket.recv_multipart()  # Bloqueante
        # destinos = pickle.loads(destinos)
        string_recep = f"[RECIBIDO] {id} recibió: {mensaje.decode()}, de {identidad.decode()}. Con destino/s: {destinos}"
        print(string_recep) 
        to_write.put(string_recep)

def start_sender():
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt_string(zmq.IDENTITY, f"{id}")

    # Conectar con los demás nodos
    for dir, puerto in zip(dir_nodos_vecinos, puertos_vecinos):
        socket.connect(f"tcp://{dir}:{puerto}")

    time.sleep(3) # Esperar a que los nodos se conecten
    return socket

def send_messages(socket: zmq.Socket):
    global to_write
    for i in range(10):
        destinatarios = random.sample(vecinos, k=random.randint(1, len(vecinos)))
        destinatarios_encoded = pickle.dumps(destinatarios)
        mensaje = f"Mensaje de {id}, NUM:{random.randrange(0,100)} a nodo/s: {destinatarios}"
        to_write.put(f"[ENVIADO]: {mensaje}\n")
        
        # input("Enter para enviar mensaje")
        for dest in destinatarios:
            socket.send_multipart([bytes(f"{dest}", encoding="utf-8"), bytes(), mensaje.encode()])
            print(f"[ENVIADO]: {mensaje}\n")
            time.sleep(0.1)
        
    #Para esperar a que se reciban todos los mensajes
    time.sleep(3)

if __name__ == "__main__":

    id = get_node_id()
    mi_hostname = socket_lib.gethostname()
    N_NODOS = int(sys.argv[1])

    mi_puerto = 10000 + id
    vecinos = [i for i in range(N_NODOS) if i != id]
    puertos_vecinos = [10000 + i for i in vecinos]

    # Lista de nombres de host de los nodos
    dir_nodos = [f"nodo{i}.local" for i in range(N_NODOS)]
    dir_nodos_vecinos = [dir_nodos[i] for i in range(N_NODOS) if i != id]
    
    

    to_write = queue.Queue()

    # Iniciar hilo receptor
    hilo_receptor = threading.Thread(target=start_server, args=(), daemon=True)
    hilo_receptor.start()
    socket_enviar = start_sender()

    send_messages(socket_enviar)
    with open(f"nodo{id}.txt", "w") as f:
        while not to_write.empty():
            line = to_write.get()
            f.write(line + "\n")
    