import socket
import time

HOST = 'localhost'  # Dirección IP del servidor
PORT = 12345       # Puerto de escucha
MAX_BUFFER_SIZE = 5 * 1024 * 1024  # Tamaño máximo del buffer

# Crea un socket TCP
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))  # Enlaza el socket al host y puerto
    server_socket.listen(2)  # Espera por conexiones entrantes
    print(f"Servidor escuchando en {HOST}:{PORT}")

    conn, addr = server_socket.accept()  # Acepta la conexión entrante
    with conn:
        print(f"Conexión establecida desde {addr}")

        while True:
            time.sleep(2)
            data = conn.recv(MAX_BUFFER_SIZE)  # Recibe datos del cliente
            if not data:
                break
            print(f"Mensaje recibido del cliente: {data.decode()}")
