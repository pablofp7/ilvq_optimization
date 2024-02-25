import socket
import subprocess

host = ''  # Escucha en todas las interfaces disponibles
port = 12345
program_process = None

def handle_command(command):
    global program_process
    if command == 'start':
        if program_process is None:
            # Lanza program.py como un proceso independiente
            program_process = subprocess.Popen(['python3', 'programv4.py'])
            print("program.py started.")
    elif command == 'stop':
        if program_process:
            program_process.terminate()  # Finaliza el proceso
            program_process = None
            print("program.py stopped.")
            exit()

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
