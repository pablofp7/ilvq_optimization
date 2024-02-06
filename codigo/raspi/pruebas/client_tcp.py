import socket
import time
import sys

HOST = 'localhost'  # Dirección IP del servidor
PORT = 12345       # Puerto del servidor

# Crea un socket TCP
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    client_socket.connect((HOST, PORT))  # Conecta al servidor

    while True:
        message = f"\nEste es un mensaje de prueba: {sys.argv[1]}"
        client_socket.sendall(message.encode())  # Envía datos al servidor
        client_socket.sendall(message.encode())  # Envía datos al servidor
        client_socket.sendall(message.encode())  # Envía datos al servidor
        time.sleep(3)  # Espera 3 segundos antes de enviar el siguiente mensaje
