import zmq
import threading
import time
import sys

# Número total de nodos conocido a priori
N_NODOS = 3  # Ajusta este valor según el número real de nodos en tu red

def receptor(id_nodo, puerto):
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.IDENTITY, f"{id_nodo}".encode())
    socket.bind(f"tcp://*:{puerto}")

    print(f"Nodo {id_nodo} escuchando en el puerto {puerto}...")

    while True:
        identidad, _, mensaje = socket.recv_multipart()  # Bloqueante
        print(f"Nodo {id_nodo} recibió: {mensaje.decode()}")

def emisor(id_nodo):
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.IDENTITY, f"{id_nodo}".encode())

    # Conectar con los demás nodos
    for i in range(N_NODOS):
        if i != id_nodo:  # No conectarse consigo mismo
            puerto = 10000 + i
            socket.connect(f"tcp://localhost:{puerto}")

    
    while True:
        for i in range(N_NODOS):
            if i != id_nodo:  # No enviarse mensajes a sí mismo
                mensaje = f"Mensaje de nodo{id_nodo} a nodo{i}"
                socket.send_multipart([f"{i}".encode(), b"", mensaje.encode()])
                print(f"Enviado: {mensaje}")
            time.sleep(3)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: nodo.py [ID del nodo]")
        sys.exit(1)

    id_nodo = int(sys.argv[1])
    puerto = 10000 + id_nodo

    # Iniciar hilo receptor
    hilo_receptor = threading.Thread(target=receptor, args=(id_nodo, puerto))
    hilo_receptor.start()

    # Ejecutar función de emisor en el hilo principal
    emisor(id_nodo)
