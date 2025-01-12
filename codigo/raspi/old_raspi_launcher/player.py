import argparse
import socket
import subprocess
import sys
import os

# Define un parser de argumentos
parser = argparse.ArgumentParser(description='Controlador de versiones de programa.')
# Agrega un argumento para la versión del programa, con opciones predefinidas
parser.add_argument('-v', '--version', choices=['', '_mp', '4_1', '4_1_mp', '4_2', '4_2_mp', '5'], default='', help='Especifica la versión del programa a ejecutar')

# Parsea los argumentos de la línea de comandos
args = parser.parse_args()

host = ''  # Escucha en todas las interfaces disponibles
port = 12345
program_process = None

def handle_command(command):
    global program_process
    if command == 'start':
        if program_process is None:
            # Lanza program.py como un proceso independiente
            program = f'programv{args.version}.py' if args.version else 'program.py'
            program_process = subprocess.Popen(['python3', program])
            print(f"{program} started.")
    elif command == 'stop':
        if program_process:
            # Construye el patrón de búsqueda basado en el nombre del archivo del programa
            program_pattern = f'programv{args.version}.py' if args.version else 'program.py'
            program_process.terminate()            
            # Usar comillas simples directamente. Si se necesita escape: os.system(f"pkill -f \\"{program_pattern}\\"")
            os.system(f"pkill -f '{program_pattern}'")
            
            program_process = None
            print(f"{program_pattern} stopped.")
            sys.exit()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((host, port))
    s.listen()
    print(f"Listening on port {port}...")
    while True:
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                handle_command(data.decode())
