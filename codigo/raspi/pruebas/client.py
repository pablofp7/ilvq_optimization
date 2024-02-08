    # client.py (Nodo0)

import zmq
import time

def main():
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    id_cliente = "nodo0".encode()  # Identificador del cliente
    socket.setsockopt(zmq.IDENTITY, id_cliente)
    nodos = ["nodo1"]
    for nodo in nodos:
        socket.connect(f"tcp://{nodo}.local:5555")

    while True:
        mensaje = "Hola desde nodo0"
        print(f"Enviando '{mensaje}' al servidor...")
        for nodo in nodos:
            socket.send_multipart([nodo.encode(), b"",mensaje.encode()])  # nodo1 es el identificador esperado del servidor

        # Espera por una respuesta
        identidad, respuesta = socket.recv_multipart()
        print(f"Respuesta recibida: '{respuesta.decode()}'")

        time.sleep(3)  # Espera 3 segundos antes de enviar el próximo mensaje

if __name__ == "__main__":
    main()
