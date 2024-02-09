import zmq
import sys
import pickle  # Importar el módulo pickle
import socket as socket_lib

context = zmq.Context()

socket = context.socket(zmq.ROUTER)

if sys.argv[1] == "s":
    hostname = socket_lib.gethostname()
    print(f"Hostname: {hostname}")
    socket.setsockopt_string(zmq.IDENTITY, hostname)
    socket.bind("tcp://*:10000")

    while True:
        print("waiting for msg")
        req = socket.recv_multipart()
        print(f"[RECV] From: {req[0].decode()}. Message: {pickle.loads(req[2])}")
        # Serializar el mensaje antes de enviarlo
        # msg_serialized = pickle.dumps("whatup")  # Serializar con pickle
        # #                      identity of receptionist   empty frame     serialized message content
        # socket.send_multipart([req[0],                    bytes(),      msg_serialized])

elif sys.argv[1] == "c":
    hostname = socket_lib.gethostname()
    socket.setsockopt_string(zmq.IDENTITY, hostname)
    socket.connect("tcp://nodo1.local:10000")
    socket.connect("tcp://nodo2.local:10000")

    while True:
        input("enter to send msg")
        # Serializar el mensaje antes de enviarlo
        msg_serialized1 = pickle.dumps("hello nodo1")  # Serializar con pickle
        msg_serialized2 = pickle.dumps("hello nodo2")  # Serializar con pickle
        #                       identity of receptionist          empty frame     serialized message content
        socket.send_multipart([bytes("nodo1", encoding="utf-8"), bytes(),     msg_serialized1])
        socket.send_multipart([bytes("nodo2", encoding="utf-8"), bytes(),     msg_serialized2])

        # rep = socket.recv_multipart()
        # # Deserializar el mensaje después de recibirlo
        # msg_deserialized = pickle.loads(rep[2])  # Deserializar con pickle
        # print(msg_deserialized)

# In real applications do not forget to disconnect when the connection is not needed anymore
