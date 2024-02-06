import zmq
import time

def iniciar_emisor():
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.IDENTITY, b"dispositivo2")
    # Asegúrate de que la dirección y puerto coincidan con el Dispositivo 1
    socket.connect("tcp://localhost:5555")

    print("Dispositivo 2 enviando mensajes...")

    while True:
        mensaje = "Hola desde el Dispositivo 2"
        # El primer frame es la identidad del receptor, el segundo es un delimitador vacío, y el tercero es el mensaje
        socket.send_multipart([b"dispositivo1", b"", mensaje.encode()])
        print(f"Enviado: {mensaje}")
        time.sleep(3)  # Esperar 3 segundos antes de enviar el siguiente mensaje

if __name__ == "__main__":
    iniciar_emisor()
