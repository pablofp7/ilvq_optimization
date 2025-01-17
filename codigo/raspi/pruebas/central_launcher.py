import socket
import sys

# Puerto UDP en el que tus nodos están escuchando
PUERTO = 15000

def enviar_comando(comando, n_nodos):
    """Envía un comando a todos los nodos Raspberry Pi."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        if comando == "start":
            # Construir y enviar comando start a cada nodo
            for i in range(n_nodos):
                nodo_ip = f"nodo{i}.local"
                mensaje = f"{comando} {n_nodos}".encode('utf-8')
                sock.sendto(mensaje, (nodo_ip, PUERTO))
                print(f"Comando '{comando} {n_nodos}' enviado a {nodo_ip}:{PUERTO}")
                
        elif comando == "stop":
            print(f"Enviando comando '{comando}' a todos los nodos...")
            # Enviar comando fin a todos los nodos conocidos
            for i in range(n_nodos):
                nodo_ip = f"nodo{i}.local"
                mensaje = comando.encode('utf-8')
                sock.sendto(mensaje, (nodo_ip, PUERTO))
                print(f"Comando '{comando}' enviado a {nodo_ip}:{PUERTO}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python central_launcher.py <comando> [N_NODOS]")
        sys.exit(1)

    comando = sys.argv[1]
    n_nodos = int(sys.argv[2])

    enviar_comando(comando, n_nodos)
