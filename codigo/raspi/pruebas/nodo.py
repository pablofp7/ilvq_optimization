import zmq
import threading
import sys
import time
import random

def start_server(context, id):
    print("Iniciando el servidor...")
    socket = context.socket(zmq.ROUTER)
    socket.bind(f"tcp://*:{10000 + id}")
    
    receive_message(socket)
    
    return socket
    
def start_client(context, puertos_vecinos):
    print("Conectando al servidor...")
    socket = context.socket(zmq.DEALER)
    socket.identity = b'nodo0'
    for puerto in puertos_vecinos:
        socket.connect(f"tcp://localhost:{puerto}")
    
    send_message(socket)
    
    return socket


def receive_message(socket):
    while True:
        message = socket.recv_multipart()
        print(f"Mensaje recibido: {message}")

def send_message(socket):
    while True:
        numero = random.randint(0, 99)
        print(f"Numero generado: {numero}")
        socket.send_multipart([f"SRC:{id}".encode(), f"Numero generado: {numero}".encode('utf-8')])
        time.sleep(2)
        


if __name__ == "__main__":
    
    context = zmq.Context()
    id = int(sys.argv[1])
    N_NODOS = 2
    vecinos = [i for i in range(N_NODOS) if i != id]
    puertos_vecinos = [10000 + i for i in vecinos]

    threading.Thread(target=start_server, args=(context, id)).start()
    threading.Thread(target=start_client, args=(context, puertos_vecinos)).start()
    

    # socket_enviar = start_client(context, puertos_vecinos)
    # socket_recibir = start_server(context, id)
    
    # threading.Thread(target=receive_message, args=(socket_recibir,)).start()
    # threading.Thread(target=send_message, args=(socket_enviar,)).start()