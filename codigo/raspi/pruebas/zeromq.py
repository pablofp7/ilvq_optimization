import zmq
import threading
import sys
import time
import random
import pickle

N_NODOS = 5

def setup_dealer(context, dealer_ports):
    """Configura el DEALER para enviar mensajes a los ROUTERs de otros nodos."""
    dealer = context.socket(zmq.DEALER)
    for port in dealer_ports:
        dealer.connect(f"tcp://localhost:{port}")
    return dealer

def dealer_thread(dealer):
    """Envía mensajes a través del DEALER."""
    while True:
        # Aquí implementarías la lógica de envío de mensajes.
        # Elegir dos vecinos al azar
        if id == 0:
            selected_neighbors = random.sample(vecinos, 2)
            message = f"EL NAAANO EL MEJOR DE TODOS LOS TIEMPOS"
            packet = pickle.dumps([id, selected_neighbors, message])
            print(f"Enviando {pickle.loads(packet)}....")
            dealer.send(packet)
            time.sleep(3)
        else:
            break

def setup_router(context, router_port):
    """Configura el ROUTER para recibir mensajes de los DEALERs de otros nodos."""
    router = context.socket(zmq.ROUTER)
    router.bind(f"tcp://*:{router_port}")
    return router

def router_thread(router):
    """Recibe mensajes a través del ROUTER."""
    while True:
        message = pickle.loads(router.recv_multipart()[1])
        print(f"MENSAAJE: {message}")
        orig = message[0]
        dest = message[1]
        content = message[2]
        # print(f"Quiero comparar id y ver si esta en des: {id}, {dest}")
        if id in dest:
            print(f"El nodo {id}, ha recibido un mensaje con origen {orig}, destino: {dest}. Contenido: {content}")

if __name__ == "__main__":

    id = int(sys.argv[1])
    print(f"Inicializando nodo {id}...")


    # Ejemplo de puertos para DEALER conectarse a otros nodos' ROUTERs
    vecinos = [i for i in range(N_NODOS)]
    vecinos.remove(id)
    
    context = zmq.Context()
    dealer_ports = [f'1000{i}' for i in vecinos]  # Asumiendo que este es el nodo 0
    router_port = f'1000{id}'  # Puerto en el que este nodo ROUTER escucha

    dealer = setup_dealer(context, dealer_ports)
    router = setup_router(context, router_port)

    # Iniciar hilos para DEALER y ROUTER
    threading.Thread(target=dealer_thread, args=(dealer,)).start()
    threading.Thread(target=router_thread, args=(router,)).start()
    
    
    
