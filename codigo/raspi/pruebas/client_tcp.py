import zmq

def iniciar_receptor():
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.IDENTITY, b"dispositivo1")
    socket.bind("tcp://*:5555")

    print("Dispositivo 1 esperando mensajes...")

    while True:
        # Bloquea hasta que se recibe un mensaje
        identidad, _, mensaje = socket.recv_multipart()
        print(f"Mensaje de {identidad.decode()}: {mensaje.decode()}")

if __name__ == "__main__":
    iniciar_receptor()
