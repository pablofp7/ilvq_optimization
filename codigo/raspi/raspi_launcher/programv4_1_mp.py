import sys
import os
ruta_directorio_main = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_directorio_main not in sys.path:
    sys.path.append(ruta_directorio_main)

from prototypes import XuILVQ
import pandas as pd
from node_class.raspi_nodev4_1 import RaspiNodev4_1
import time
import threading
import numpy as np
import socket

# Cada raspi ejecuta su propio "Nodo"
##IMPLEMENTAR MECANISMOS DE SINCRONIZACION A LA HORA DE INICAR CADA ITERACION DEL SCRIPT(por ejemplo request al central y que este responda con un ok)


def read_dataset(name: str):
    filename = data_name[name]
    dataset = pd.read_csv(f"../dataset/{filename}")
    # Se cambia el 'UP' por 1 y el 'DOWN' por 0
    dataset.replace('UP', 1, inplace=True)
    dataset.replace('DOWN', 0, inplace=True)

    dataset.replace('True', 1, inplace=True)
    dataset.replace('False', 0, inplace=True)


    return dataset


def main(df: pd.DataFrame):

    #Se adapta para darle a cada nodo su parte
    init_index = 0
    tam_muestra = n_muestras * n_nodos
    if "phis" in dataset or "elec2" in dataset:
        tam_muestra = min(tam_muestra, 1250)
        max_init_index = len(df) - tam_muestra
        print(f"max_init_index: {max_init_index}, tam_muestra: {tam_muestra}, len(df): {len(df)}")
        init_index = np.random.randint(0, max_init_index) if max_init_index > 0 else 0

    df_short = df[init_index : init_index + tam_muestra]
    df_nodos = [df_short.iloc[i::n_nodos, :].reset_index(drop=True) for i in range(n_nodos)]


    nodo = RaspiNodev4_1(id, dataset=df_nodos[id], modelo_proto=XuILVQ(), nodos=n_nodos, s=s, T=t, media_llegadas=media_llegadas)

    hilo = threading.Thread(target=nodo.run)
    hilo.start()
    hilo.join()

    to_write = []
    # to_write.append(f" - TIEMPO EJECUCION: {(time.time() - tiempo_inicio) / 60} minutos.\n\n")
    #Vamos a guardar en una string lo que se va a escribir en el archivo

    tp = nodo.matriz_conf["TP"]
    fp = nodo.matriz_conf["FP"]
    fn = nodo.matriz_conf["FN"]
    precision = round(tp / (tp + fp), 3) if tp + fp != 0 else 0
    recall = round(tp / (tp + fn), 3) if tp + fn != 0 else 0
    f1 = round(2 * (precision * recall) / (precision + recall), 3) if precision + recall != 0 else 0


    if nodo.tiempo_learn_queue == 0:
        cap_ejec = 0
    else:
        cap_ejec = round(nodo.protos_train / nodo.tiempo_learn_queue, 3)

    to_write.append(f" - NODO {nodo.id}.\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}\n"
                    f"Se ha entrenado con {nodo.muestras_train} muestras.\nSe ha entrenado con {nodo.protos_train} prototipos.\n"
                    f"Ha compartido {nodo.shared_times} veces.\n"
                    f"Ha compartido {nodo.compartidos} prototipos a cada uno de los {s} vecino/s.\n"
                    f"Tiempo de aprendizaje (muestras): {nodo.tiempo_learn_data}\n"
                    f"Tiempo de aprendizaje (prototipos): {nodo.tiempo_learn_queue}\n"
                    f"Tiempo compartiendo prototipos: {nodo.tiempo_share}\n"
                    f"Tiempo total: {nodo.tiempo_final_total}\n"
                    f"Capacidad de ejecución: {cap_ejec}\n"
                    f"ID, Tamaño de lotes recibidos: {nodo.tam_lotes_recibidos}\n"
                    f"Tamaño conjunto de prototipos: {nodo.tam_conj_prot}\n"
                    f"\n")


    print("Se ha terminado de ejecutar todo.")

    with open(nombre_archivo, "w") as f:
        f.writelines(to_write)

def vaciar_buffer(socket):
    print("Vaciando buffer...")
    ahora = time.time()
    while time.time() - ahora < 1:
        try:
            socket.recv(1024)
        except:
            print("Buffer vaciado.")
            break

def sincronizar():

    print("Comienza la sincronización...")
    puerto = 11111  # Puerto común para la sincronización
    buffer_size = 1024  # Tamaño del buffer para recibir mensajes
    dir_server = "nodo0.local"  # Dirección del nodo central
    dir_nodos = [f"nodo{i}.local" for i in range(1, 5)]  # Direcciones de los nodos no centrales

    if id == 0:
        # Nodo central
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(("0.0.0.0", puerto))
            # vaciar_buffer(s)

            lista_confirmaciones = [False] * n_nodos
            lista_confirmaciones[id] = True

            while not all(lista_confirmaciones):
                data, addr = s.recvfrom(buffer_size)
                msg = data.decode()
                print(f"Nodo 0. Recibido: {msg}")
                option, nodo_id = check_mensaje(msg)
                if option:
                    lista_confirmaciones[nodo_id] = True
                # if msg.startswith("LISTO") and parametros in msg:
                #     nodo_id = int(msg.split()[1])
                #     lista_confirmaciones[nodo_id] = True

            # Enviar "COMENZAR" a todos los nodos excepto al nodo central
            for i, dir in enumerate(dir_nodos):
                # print(f"Se va e enviar COMENZAR: dir:{dir}, puerto: {puerto}")
                s.sendto("COMENZAR".encode(), (dir, puerto))
            print(f"Se le ha enviado COMENZAR a todos los slaves.")
            time.sleep(0.25)


            print("Nodo 0: todos listos.")
    else:
        # Nodo no central
        ready = False
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s_recepcion:
                s_recepcion.bind(("0.0.0.0", puerto))
                s_recepcion.settimeout(0.5)  # Establecer timeout
                while not ready:
                    try:
                        # Enviar "LISTO" al nodo central
                        s.sendto(f"LISTO {parametros}".encode(), (dir_server, puerto))
                        print(f"LISTO enviado desde {id}")

                        try:
                            data, _ = s_recepcion.recvfrom(buffer_size)
                            msg = data.decode()
                            if msg == "COMENZAR":
                                print(f"{id}: Recibido COMENZAR desde nodo0.")
                                ready = True
                                break
                        except socket.timeout:
                            print(f"{id}: Esperando a recibir COMENZAR del nodo 0.")
                            # No es necesario romper el bucle; sigue esperando
                    except Exception as e:
                        print(f"Error en nodo {id}: {e}")
                        time.sleep(0.5)  # Esperar un poco antes de reintentar

                print(f"Nodo {id} contestando a nodo0.")


def check_mensaje(mensaje):
    """
    Evalúa si el mensaje comienza con 'LISTO', contiene los parámetros esperados
    (sin considerar el sufijo '_nodoX'), y extrae el id del nodo. Devuelve True si
    cumple las condiciones, o False en caso contrario, junto con el id del nodo.
    """
    # Verificar si el mensaje comienza con "LISTO"
    if mensaje.startswith("LISTO"):
        try:
            # Extraer la parte después de "LISTO"
            _, resto_mensaje = mensaje.split(' ', 1)

            partes = resto_mensaje.split('_')
            nodo_id_str = partes[-1].replace("nodo", "")  # Extraer el ID del nodo, que está al final
            parametros_recibidos = "_".join(partes[:-1])  # Unir todas las partes excepto la última

            nodo_id = int(nodo_id_str)
            parametros_esperados_sin_nodo = "_".join(parametros.split('_')[:-1])

            # Verificar si los parámetros recibidos coinciden con los esperados (sin '_nodoX')
            if parametros_recibidos == parametros_esperados_sin_nodo:
                # print(f"Mensaje recibido: {mensaje}")
                # print(f"Parametros esperados vs recibidos: {parametros_esperados_sin_nodo} vs {parametros_recibidos}")
                return True, nodo_id
        except ValueError as e:
            # Manejar errores de conversión o de formato incorrecto
            print(f"Error al procesar el mensaje: {e}")

    # Si el mensaje no cumple las condiciones, devolver False y None
    return False, None

if __name__ == "__main__":

    try:

        hostname = socket.gethostname()
        id = int(''.join(filter(str.isdigit, hostname)))
        n_nodos = 5
        n_muestras = 1000

        S = [i for i in range(1, 5)]
        T = np.array([i for i in range(0, 1001, 50)])
        T = T / 1000
        tasa_llegadas = 5
        media_llegadas = 1 / tasa_llegadas

        iteraciones = 50
        datasets = ["elec", "phis", "elec2"]

        data_name = {"elec": "electricity.csv", "phis": "phishing.csv", "elec2": "electricity.csv"}

        directorio_resultados = "../resultados_raspi_indiv"

        if not os.path.exists(directorio_resultados):
            os.makedirs(directorio_resultados)

        for i in range(iteraciones):
            for dataset in datasets:
                data_frame = read_dataset(dataset)
                for s in S:
                    tiempo_s = time.time()
                    for t in T:
                        tiempo_inicio = time.time()
                        print(f"ITERACIÓN {i}, dataset: {dataset}, S: {s}, T:{t}")

                        parametros = f"{dataset}_s{s}_T{t}_it{i}_nodo{id}"
                        nombre_archivo = f"{directorio_resultados}/result_{parametros}.txt"
                        if os.path.isfile(nombre_archivo):
                            print(f"El archivo '{nombre_archivo}' ya existe. No es necesario generarlos de nuevo.")
                            continue  # Salta a la siguiente iteración si el archivo ya existe

                        sincronizar()
                        main(data_frame)
                        print(f"- Tiempo de ejecución: {(time.time() - tiempo_inicio) / 60} minutos.\n")

    except KeyboardInterrupt as e:
        os.system("ipcrm --all=msg")


