import zmq

context = zmq.Context()
socket = context.socket(zmq.ROUTER)

# Configurar el tamaño del buffer de envío a 5MB
socket.setsockopt(zmq.SNDBUF, 5 * 1024 * 1024)  # 5 MB

# Configurar el tamaño del buffer de recepción a 2 GB (o 2000 MB)
socket.setsockopt(zmq.RCVBUF, 2000 * 1024 * 1024)  # 2000 MB

# Verificar los valores establecidos
sndbuf_size = socket.getsockopt(zmq.SNDBUF)
rcvbuf_size = socket.getsockopt(zmq.RCVBUF)
print(f"Buffer de envío configurado: {sndbuf_size / (1024 * 1024)} MB")
print(f"Buffer de recepción configurado: {rcvbuf_size / (1024 * 1024)} MB")
