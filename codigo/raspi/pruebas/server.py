# server.py (Nodo1)

import zmq
import time
import sys

def main():
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.IDENTITY, f"nodo{int(sys.argv[1])}".encode())  # Identificador del servidor
    socket.bind("tcp://*:5555")

    print("Servidor iniciado en el puerto 5555 esperando mensajes...")

    while True:
        # Espera por un mensaje
        identidad, mensaje = socket.recv_multipart()
        print(f"Recibido '{mensaje.decode()}' de {identidad.decode()}")

        # Envía una respuesta
        respuesta = f"Respuesta a {mensaje.decode()}"
        socket.send_multipart([identidad, respuesta.encode()])

if __name__ == "__main__":
    main()
