import socket

# Lista de direcciones IP de las Raspberry Pi (o hostnames si están en el mismo DNS)
raspberry_pis = ['nodo0.local', 'nodo1.local', 'nodo2.local', 'nodo3.local', 'nodo4.local']
port = 12345  # El puerto que estás usando para tus sockets

def send_command(command):
    for pi in raspberry_pis:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((pi, port))
                s.sendall(command.encode())
                print(f"Command {command} sent to {pi}")
        except ConnectionRefusedError:
            print(f"Failed to connect to {pi}")

if __name__ == "__main__":
    import sys
    command = sys.argv[1]  # start o stop
    send_command(command)
