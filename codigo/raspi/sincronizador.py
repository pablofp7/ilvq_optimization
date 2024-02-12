import socket
import time
import threading

def proceso_de_sincronizacion():
    N_NODOS = 5
    dir_nodos = [f"nodo{i}.local" for i in range(N_NODOS)]  # Direcciones de los nodos no centrales
    puerto = 11111
    buffer_size = 1024

    while True:
        print("Comienza la sincronización...")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(("0.0.0.0", puerto))
            lista_confirmaciones = [False] * N_NODOS
            
            while not all(lista_confirmaciones):
                data, addr = s.recvfrom(buffer_size)
                msg = data.decode()
                print(f"Nodo Sincro. Recibido: {msg}")
                if msg.startswith("LISTO"):
                    nodo_id = int(msg.split()[1])
                    lista_confirmaciones[nodo_id] = True
            
            time.sleep(5)
            # Enviar "COMENZAR" a todos los nodos excepto al nodo central
            for dir in dir_nodos:
                s.sendto("COMENZAR".encode(), (dir, puerto)) 
            print("Se le ha enviado COMENZAR a todos los slaves.")
            time.sleep(0.75)
            
            print("Nodo SINCRO: todos listos.")

        # Agrega un mecanismo de espera o condición para determinar si se debe continuar o no con la sincronización
        # Por ejemplo, esto podría ser una variable compartida o un evento que se chequea aquí.
        # Se omite en este ejemplo por simplicidad.

def escuchar_manager():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s_manager:
        s_manager.bind(("0.0.0.0", 12345))  # Escuchar en todas las interfaces en el puerto 12345
        
        while True:
            data, addr = s_manager.recvfrom(1024)
            mensaje = data.decode()
            print(f"Mensaje del manager recibido: {mensaje}")
            
            if mensaje == "PARAR":
                print("Recibido mensaje para PARAR. Terminando programa.")
                break  # Salir del bucle para terminar el programa

if __name__ == "__main__":
    # Iniciar el hilo secundario para la sincronización
    hilo_sincronizacion = threading.Thread(target=proceso_de_sincronizacion, daemon=True)
    hilo_sincronizacion.start()

    # El hilo principal se dedica a escuchar al manager
    escuchar_manager()
    
    # Nota: Como el hilo de sincronización es un daemon, terminará automáticamente cuando el programa principal termine.
