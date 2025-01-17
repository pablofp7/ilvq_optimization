import socket
import subprocess
import os
import time

# Configuración del socket UDP
HOST = '0.0.0.0'  # Escucha en todas las interfaces
PORT = 15000  # Puerto arbitrario para escuchar comandos UDP

# Crear el socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))


proceso = None  # Variable para almacenar el proceso lanzado

while True:
    print("Launcher esperando comandos...")
    data, addr = sock.recvfrom(1024)  # Buffer size de 1024 bytes
    comando = data.decode('utf-8').strip()
    print(f"Comando recibido: {comando}")

    if comando.startswith("start"):
        try:
            _, n_nodos = comando.split()
            if proceso:
                proceso.kill()  # Asegurarse de que no hay otro proceso corriendo
            # Ignorar la salida estándar y el error estándar

            archivo_salida = f"salida_nodo{n_nodos}.log"

            with open(archivo_salida, 'wb') as f:
                proceso = subprocess.Popen(['python3', 'ejemplo_zmq.py', n_nodos],
                                           stdout=f,
                                           stderr=subprocess.STDOUT)
            print(f"Proceso lanzado con N_NODOS={n_nodos}")
        except ValueError:
            print("Error: Formato de comando 'start' incorrecto. Uso esperado: 'start N_NODOS'")
    
    elif comando == "stop":
        if proceso:
            proceso.kill()
        print("Saliendo del launcher.")
        break

sock.close()
