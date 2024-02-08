import socket
import subprocess

# Configuración del socket UDP
HOST = '0.0.0.0'  # Escucha en todas las interfaces
PORT = 15000  # Puerto arbitrario para escuchar comandos UDP

# Crear el socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))

print("Launcher esperando comandos...")

proceso = None  # Variable para almacenar el proceso lanzado

while True:
    data, addr = sock.recvfrom(1024)  # Buffer size de 1024 bytes
    comando = data.decode('utf-8').strip()

    if comando.startswith("start"):
        # Extrae el número de nodos del mensaje, asumiendo formato "start N_NODOS"
        try:
            _, n_nodos = comando.split()
            if proceso:
                proceso.kill()  # Asegurarse de que no hay otro proceso corriendo
            # Lanza el script con el número de nodos como argumento
            proceso = subprocess.Popen(['python3', 'ejemplo_zmq.py', n_nodos])
            print(f"Proceso lanzado con N_NODOS={n_nodos}")
        except ValueError:
            print("Error: Formato de comando 'start' incorrecto. Uso esperado: 'start N_NODOS'")
    
    elif comando == "fin":
        if proceso:
            proceso.kill()
            print("Proceso terminado")
            proceso = None  # Reiniciar la variable del proceso
        else:
            print("No hay un proceso en ejecución para terminar.")
    
    elif comando == "exit":
        if proceso:
            proceso.kill()
        print("Saliendo del launcher.")
        break

sock.close()
