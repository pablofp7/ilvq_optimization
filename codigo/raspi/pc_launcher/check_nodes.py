#!/usr/bin/env python3

import subprocess
import threading

# Definir la función para enviar pings y verificar la respuesta
def check_host_online(hostname, n_pings, timeout, online_hosts):
    command = ['ping', '-c', str(n_pings), '-W', str(timeout), hostname]
    try:
        output = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if output.returncode == 0:
            online_hosts.append(hostname)
            print(f"{hostname} is online")
        else:
            print(f"{hostname} is offline")
    except Exception as e:
        print(f"Error pinging {hostname}: {e}")

# Define the number of nodes and other parameters
N = 5
n_pings = 3
timeout = 2  # segundos

online_hosts = []

# Crear y empezar los threads
threads = []
for i in range(N):
    hostname = f"nodo{i}.local"
    thread = threading.Thread(target=check_host_online, args=(hostname, n_pings, timeout, online_hosts))
    threads.append(thread)
    thread.start()

# Esperar a que todos los threads terminen
for thread in threads:
    thread.join()

# Imprimir los resultados
print(f"Hosts en línea: {online_hosts}")
