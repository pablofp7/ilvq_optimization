import sys
import os
ruta_directorio_main = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_directorio_main not in sys.path:
    sys.path.append(ruta_directorio_main)

from prototypes import XuILVQ
import pandas as pd
from node_class.raspi_nodev2 import RaspiNodev2
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


    nodo = RaspiNodev2(id, dataset=df_nodos[id], modelo_proto=XuILVQ(), nodos=n_nodos, s=s, T=t, media_llegadas=media_llegadas)

    hilo = threading.Thread(target=nodo.run)
    hilo.start()
    hilo.join(T_MAX_IT)
    
    if hilo.is_alive():
        print(f"EL HILO ESTÁ BLOQUEADO. CERRANDO PROGRAMA")
        exit()
        
    to_write = []
    # to_write.append(f" - TIEMPO EJECUCION: {(time.perf_counter() - tiempo_inicio) / 60} minutos.\n\n")
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
    ahora = time.perf_counter()
    while time.perf_counter() - ahora < 1:
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
        min_prov = None
        # Nodo central
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(("0.0.0.0", puerto))
            
            # Inicializar la lista de confirmaciones y esperar los mensajes "LISTO"
            lista_confirmaciones = [False] * n_nodos
            lista_confirmaciones[id] = True
            
            while not all(lista_confirmaciones):
                data, addr = s.recvfrom(buffer_size)
                mensaje = data.decode()
                check_mensaje(mensaje, lista_confirmaciones, contador_prints=0, min_prov=min_prov)  # Asumiendo contador_prints gestionado adecuadamente
            
            # Una vez todos los nodos están listos, enviar la combinación mínima
            combinacion_minima = min_prov  # Obtenida de los mensajes "LISTO"
            for _ in range(5):
                for i, dir in enumerate(dir_nodos):
                    # Convierte índices a parámetros si es necesario antes de enviar
                    mensaje_minimo = indices_a_parametros(combinacion_minima)
                    s.sendto(f"COMENZAR {mensaje_minimo}".encode(), (dir, puerto))
                print("Nodo 0: Se la ha enviado COMENZAR a todos los slaves.")
            time.sleep(0.05)
            print("Nodo 0: todos listos con min_prov.")
    else:
        ready = False
        # Nodo no central
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s_recepcion:
                s_recepcion.bind(("0.0.0.0", puerto))
                s_recepcion.settimeout(3)  # Establecer timeout
                
                while not ready:
                    # Enviar mensaje "LISTO" al nodo central
                    mensaje_listo = f"LISTO {parametros}"  # Asegúrate de que 'parametros' está definido correctamente
                    s.sendto(mensaje_listo.encode(), (dir_server, puerto))
                    
                    # Esperar el mensaje "COMENZAR" del nodo central
                    try:
                        data, _ = s_recepcion.recvfrom(buffer_size)
                        mensaje = data.decode()
                        if mensaje.startswith("COMENZAR"):
                            _, combinacion_minima = mensaje.split(' ', 1)
                            print(f"Recibido: {mensaje}")
                            ready = True
                            combinacion_minima = parsear_parametros(combinacion_minima)
                            break
                        
                    except socket.timeout:
                        time.sleep(1)
                        
    return combinacion_minima


def check_mensaje(mensaje, lista_confirmaciones, contador_prints, min_prov):

    if mensaje.startswith("LISTO"):
        try:
            _, parametros_mensaje = mensaje.split(' ', 1)
            indices_mensaje = parsear_parametros(parametros_mensaje)
            
            # Comprueba si el mensaje recibido tiene índices menores y actualiza min_prov
            if min_prov is None or indices_mensaje < min_prov:
                min_prov = indices_mensaje
            
            # Lógica para actualizar lista_confirmaciones (no se muestra por brevedad)
            nodo_id = int(parametros_mensaje.split('_')[-1].replace("nodo", ""))
            if not lista_confirmaciones[nodo_id]:
                lista_confirmaciones[nodo_id] = True
                if contador_prints % 50 == 0:
                    print(f"Nodo 0. Recibido: {mensaje}")
                contador_prints += 1

        except ValueError as e:
            print(f"Error al procesar el mensaje: {e}")


# def check_mensaje(mensaje, lista_confirmaciones, contador_prints):
#     """
#     Evalúa si el mensaje comienza con 'LISTO', contiene los parámetros esperados
#     (sin considerar el sufijo '_nodoX'), y extrae el id del nodo. Devuelve True si
#     cumple las condiciones, o False en caso contrario, junto con el id del nodo.
#     """
#     # Verificar si elmensaje comienza con "LISTO"
#     if mensaje.startswith("LISTO"):
#         try:
#             # Extraer la parte después de "LISTO"
#             _, resto_mensaje = mensaje.split(' ', 1)

#             partes = resto_mensaje.split('_')
#             nodo_id_str = partes[-1].replace("nodo", "")  # Extraer el ID del nodo, que está al final
#             parametros_recibidos = "_".join(partes[:-1])  # Unir todas las partes excepto la última

#             nodo_id = int(nodo_id_str)
#             parametros_esperados_sin_nodo = "_".join(parametros.split('_')[:-1])
            

#             if lista_confirmaciones[nodo_id] is True:
#                 return

#             if contador_prints % 50:
#                 print(f"Nodo 0. Recibido: {mensaje}")
#                 print(f"Parámetros recibidos: {parametros_recibidos}")
#                 print(f"Mis parametros sin nodo {parametros_esperados_sin_nodo}")
#             contador_prints += 1

#             # Verificar si los parámetros recibidos coinciden con los esperados (sin '_nodoX')
#             if parametros_recibidos == parametros_esperados_sin_nodo:
#                 lista_confirmaciones[nodo_id] = True
#                 # Aquí se obtiene 

#         except ValueError as e:
#             # Manejar errores de conversión o de formato incorrecto
#             print(f"Error al procesar el mensaje: {e}")

#     return 

def parsear_parametros(mensaje):
    partes = mensaje.split('_')
    dataset = partes[1]
    s_value = int(partes[2][1:])  # Extrae el número después de 's'
    t_value = float(partes[3][1:])  # Extrae el valor después de 'T'
    iteration = int(partes[4][2:])  # Extrae el número después de 'it'
    
    # Encuentra índices de los valores en sus respectivas listas
    dataset_index = datasets.index(dataset)
    s_index = S.index(s_value)
    t_index = np.where(T == t_value)[0][0]
    
    return (dataset_index, s_index, t_index, iteration)


def indices_a_parametros(indices):
    t_index, s_index, dataset_index, iteration = indices
    dataset = datasets[dataset_index]
    s = S[s_index]
    T_value = T[t_index]
    return f"{dataset}_s{s}_T{T_value}_it{iteration}"

if __name__ == "__main__":

    try:

        hostname = socket.gethostname()
        id = int(''.join(filter(str.isdigit, hostname)))
        n_nodos = 5
        n_muestras = 1000

        T_MAX_IT = 300  # Tiempo máximo de ejecución del hilo
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

        i_iter = 31  # Comienza en la iteración 31
        while i_iter < iteraciones:
            dataset_idx = 0
            while dataset_idx < len(datasets):
                dataset = datasets[dataset_idx]
                data_frame = read_dataset(dataset)
                s_idx = 0
                while s_idx < len(S):
                    s = S[s_idx]
                    t_idx = 0
                    while t_idx < len(T):
                        t = T[t_idx]
                        tiempo_inicio = time.perf_counter()
                        print(f"ITERACIÓN {i_iter}, dataset: {dataset}, S: {s}, T:{t}")

                        parametros = f"{dataset}_s{s}_T{t}_it{i_iter}_nodo{id}"
                        nombre_archivo = f"{directorio_resultados}/result_{parametros}.txt"
                        if os.path.isfile(nombre_archivo):
                            print(f"El archivo '{nombre_archivo}' ya existe. No es necesario generarlos de nuevo.")
                            t_idx += 1
                            continue  # Salta a la siguiente iteración si el archivo ya existe

                        dataset_idx, s_idx, t_idx, i_iter = sincronizar()
                        dataset = datasets[dataset_idx]
                        s = S[s_idx]
                        t = T[t_idx]
                        i_iter = i_iter
                        
                        main(data_frame)
                        print(f"- Tiempo de ejecución: {(time.perf_counter() - tiempo_inicio) / 60} minutos.\n")
                        t_idx += 1  # Avanza manualmente a la siguiente iteración de T
                    s_idx += 1  # Avanza manualmente a la siguiente iteración de S
                dataset_idx += 1  # Avanza manualmente a la siguiente iteración del dataset
            i_iter += 1  # Avanza manualmente a la siguiente iteración principal

    except KeyboardInterrupt as e:
        print(f"Se ha interrumpido la ejecución del programa: {e}")

