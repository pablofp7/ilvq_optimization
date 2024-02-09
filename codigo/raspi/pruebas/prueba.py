import zmq
import threading
import pickle
import socket as socket_lib

# Definir la cantidad de nodos
N_NODOS = 3

# Obtener el ID del nodo basado en el hostname (asumiendo que el hostname es 'nodoX' donde X es el ID)
hostname = socket_lib.gethostname()
mi_id = int(hostname[-1])  # Extraer el último carácter del hostname y convertirlo a int

# Función para actuar como servidor
def servidor(context, mi_id):
    socket = context.socket(zmq.ROUTER)
    puerto = 10000 + mi_id
    socket.setsockopt_string(zmq.IDENTITY, hostname)
    socket.bind(f"tcp://*:{puerto}")
    print(f"Servidor {hostname} escuchando en el puerto {puerto}")

    while True:
        try:
            req = socket.recv_multipart()
            print(f"[RECV] From: {req[0].decode()}. Message: {pickle.loads(req[2])}")
        except KeyboardInterrupt:
            break

# Función para actuar como cliente
def cliente(context, mi_id):
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt_string(zmq.IDENTITY, hostname)

    # Conectar a los vecinos
    for i in range(N_NODOS):
        if i != mi_id:  # No conectarse a sí mismo
            vecino_puerto = 10000 + i
            socket.connect(f"tcp://nodo{i}.local:{vecino_puerto}")
            print(f"Conectado a nodo{i} en el puerto {vecino_puerto}")

    while True:
        input("Enter para enviar mensaje")
        for i in range(N_NODOS):
            if i != mi_id:
                msg_serialized = pickle.dumps(f"hello nodo{i} desde {hostname}")
                socket.send_multipart([bytes(f"nodo{i}", encoding="utf-8"), bytes(), msg_serialized])

context = zmq.Context()

# Crear hilos para servidor y cliente
thread_servidor = threading.Thread(target=servidor, args=(context, mi_id))
thread_cliente = threading.Thread(target=cliente, args=(context, mi_id))

# Iniciar hilos
thread_servidor.start()
thread_cliente.start()

thread_servidor.join()
thread_cliente.join()
