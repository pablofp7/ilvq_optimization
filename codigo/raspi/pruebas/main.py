import asyncio
from quic_server import run_quic_server, message_queue  # Asume que message_queue es tu cola conjunta
from quic_client import QuicClient  # Tu clase de cliente modificada para soportar .send()
import threading
import ssl

# Supongamos que tienes la configuración de QUIC ya definida
from aioquic.quic.configuration import QuicConfiguration

# Función para iniciar el servidor en un hilo separado
def start_server_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_quic_server())

# Crear y gestionar clientes QUIC para cada vecino
async def manage_clients():
    configuration = QuicConfiguration(is_client=True)
    configuration.load_cert_chain("cert.pem", "key.pem")
    configuration.verify_mode = ssl.CERT_NONE  # Recuerda, solo para desarrollo

    # Diccionario para almacenar clientes QUIC, asumiendo que sabes los hosts y puertos de los vecinos
    quic_clients = {
        'cliente1': QuicClient('localhost', 4433, configuration),
        'cliente2': QuicClient('localhost', 4434, configuration),
    }

    # Ejemplo de cómo enviar datos a cada vecino
    await quic_clients['cliente1'].send(b'datos1')
    await quic_clients['cliente2'].send(b'datos1')

    # Suponiendo que tu método .send() es asíncrono, si no, ajusta la llamada correspondientemente

def main():
    # Iniciar el servidor en su propio hilo
    server_thread = threading.Thread(target=start_server_thread, daemon=True)
    server_thread.start()

    # Crear y gestionar clientes en el loop de eventos principal
    asyncio.run(manage_clients())

if __name__ == "__main__":
    main()
