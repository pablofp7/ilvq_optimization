import zmq
import threading
import sys
import time
import random
import pickle


N_NODOS = 3

def setup_router(port):
    """Configura el socket ROUTER para recibir mensajes."""
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.bind(f"tcp://*:{port}")
    print(f"Router escuchando en el puerto {port}")
    return router

def setup_dealer(ports, node_id):
    """Configura el socket DEALER para enviar mensajes."""
    context = zmq.Context()
    dealer = context.socket(zmq.DEALER)
    dealer.setsockopt(zmq.IDENTITY, f"nodo{node_id}".encode())

    # Conectar a los ROUTERs de los nodos vecinos
    for port in ports:
        dealer.connect(f"tcp://localhost:{port}")
    return dealer


def send_messages(dealer, node_id, ports):
    """Envía mensajes desde el socket DEALER."""
    while True:
        if node_id == 0:
            # n_vecinos = random.randrange(N_NODOS)
            n_vecinos = 2
            selec_vecinos = random.sample(vecinos, n_vecinos)
            packet = [f"nodo{node_id}".encode(),f"Destino hipotético: {selec_vecinos}".encode(), f"Hola desde nodo {node_id}".encode()]
            # packet_bytes = pickle.dumps(packet)
            # dealer.send(packet_bytes)
            dealer.send((f"nodo{node_id}." + f"Destino hipotético: {selec_vecinos}." + f"Hola desde nodo {node_id}").encode())
            print(f"Se ha enviado: {packet}")
            time.sleep(2)  # Espera un segundo antes de enviar el siguiente lote de mensajes
        else:
            time.sleep(10)


def receive_messages(router):
    """Recibe mensajes en el socket ROUTER."""
    while True:
        message = router.recv()
        print(f"Se ha recibido: {message}")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: script.py <node_id>")
        sys.exit(1)

    node_id = int(sys.argv[1])
    vecinos = [i for i in range(N_NODOS)]
    vecinos.remove(node_id)
    
    router_port = 10000 + node_id
    dealer_ports = [10000 + i for i in range(5) if i != node_id]

    # Configura y ejecuta el router y dealer en hilos separados
    router = setup_router(router_port)
    dealer = setup_dealer(dealer_ports, node_id)

    threading.Thread(target=receive_messages, args=(router,)).start()
    threading.Thread(target=send_messages, args=(dealer, node_id, dealer_ports)).start()
