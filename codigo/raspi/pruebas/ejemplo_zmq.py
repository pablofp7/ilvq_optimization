import zmq
import threading
import time
import sys
import random
import pickle
import queue

# Número total de nodos conocido a priori
N_NODOS = 3  # Ajusta este valor según el número real de nodos en tu red

def start_server():
    global to_write
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

def start_sender():
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.IDENTITY, f"{id}".encode())

    # Conectar con los demás nodos
    for puerto in puertos_vecinos:
        socket.connect(f"tcp://localhost:{puerto}")

    time.sleep(3) # Esperar a que los nodos se conecten
    return socket

def send_messages(socket: zmq.Socket):
    global to_write
    for i in range(20):
        destinatarios = random.sample(vecinos, k=random.randint(1, len(vecinos)))
        destinatarios_encoded = pickle.dumps(destinatarios)
        mensaje = f"Mensaje de {id}, NUM:{random.randrange(0,100)} a nodo/s: {destinatarios}"
        to_write.put(f"[ENVIADO]: {mensaje}\n")
        
        for dest in destinatarios:
            socket.send_multipart([f"{dest}".encode(), destinatarios_encoded, mensaje.encode()])
            print(f"[ENVIADO]: {mensaje}\n")
            time.sleep(0.1)
    #Para esperar a que se reciban todos los mensajes
    time.sleep(3)

if __name__ == "__main__":

    id = int(sys.argv[1])
    puerto = 10000 + id
    vecinos = [i for i in range(N_NODOS) if i != id]
    puertos_vecinos = [10000 + i for i in vecinos]

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
    