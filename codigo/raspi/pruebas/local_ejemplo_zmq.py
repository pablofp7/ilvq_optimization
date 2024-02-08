import zmq
import threading
import time
import sys
import random
import pickle
import queue

# Número total de nodos, recibido como argumento
N_NODOS = int(sys.argv[1])

def get_node_id():
    # Simulación de obtención de ID de nodo basado en argumento de línea de comando, para testing en localhost
    # En un escenario real, usarías socket_lib.gethostname() y extraerías el ID como en tu código original
    return int(sys.argv[2])

def start_server(id, puerto, to_write):
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.IDENTITY, f"{id}".encode())
    socket.bind(f"tcp://*:{puerto}")

    print(f"Nodo {id} escuchando en el puerto {puerto}...")

    while True:
        identidad, destinos, mensaje = socket.recv_multipart()  # Bloqueante
        destinos = pickle.loads(destinos)
        string_recep = f"[RECIBIDO] {id} recibió: {mensaje.decode()}, de {identidad.decode()}. Con destino/s: {destinos}"
        print(string_recep)
        to_write.put(string_recep)

def start_sender(id, vecinos, puertos_vecinos):
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.IDENTITY, f"{id}".encode())

    # Conectar con los demás nodos
    for puerto in puertos_vecinos:
        socket.connect(f"tcp://localhost:{puerto}")

    time.sleep(3)  # Esperar a que los nodos se conecten
    return socket

def send_messages(id, socket, vecinos, to_write):
    for i in range(20):
        destinatarios = random.sample(vecinos, k=random.randint(1, len(vecinos)))
        destinatarios_encoded = pickle.dumps(destinatarios)
        mensaje = f"Mensaje de {id}, NUM:{random.randrange(0,100)} a nodo/s: {destinatarios}"
        to_write.put(f"[ENVIADO]: {mensaje}\n")

        for dest in destinatarios:
            socket.send_multipart([f"{dest}".encode(), destinatarios_encoded, mensaje.encode()])
            print(f"[ENVIADO]: {mensaje}\n")
            time.sleep(0.25)

        # Para esperar a que se reciban todos los mensajes
    time.sleep(3)

if __name__ == "__main__":
    id = get_node_id()
    puerto = 10000 + id
    vecinos = [i for i in range(N_NODOS) if i != id]
    puertos_vecinos = [10000 + i for i in vecinos]

    to_write = queue.Queue()

    # Iniciar hilo receptor
    hilo_receptor = threading.Thread(target=start_server, args=(id, puerto, to_write), daemon=True)
    hilo_receptor.start()

    socket_enviar = start_sender(id, vecinos, puertos_vecinos)

    send_messages(id, socket_enviar, vecinos, to_write)

    with open(f"nodo{id}.txt", "w") as f:
        while not to_write.empty():
            line = to_write.get()
            f.write(line + "\n")
