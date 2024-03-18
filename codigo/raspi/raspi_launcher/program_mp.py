import sys
import os
ruta_directorio_main = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_directorio_main not in sys.path:
    sys.path.append(ruta_directorio_main)

from prototypes import XuILVQ
import pandas as pd
from node_class.raspi_nodev2_mp import RaspiNodev2_mp
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


    nodo = RaspiNodev2_mp(id, dataset=df_nodos[id], modelo_proto=XuILVQ(), nodos=n_nodos, s=s, T=t, media_llegadas=media_llegadas)

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
                min_prov = check_mensaje(mensaje, lista_confirmaciones, contador_prints=0, min_prov=min_prov)  # Asumiendo contador_prints gestionado adecuadamente
            
            # Una vez todos los nodos están listos, enviar la combinación mínima
            combinacion_minima = min_prov  # Obtenida de los mensajes "LISTO"
            mensaje_minimo = indices_a_parametros(combinacion_minima)
            print(f"Se va enviar COMENZAR + parametros minimos: {mensaje_minimo}")
            for _ in range(5):
                for i, dir in enumerate(dir_nodos):
                    # Convierte índices a parámetros si es necesario antes de enviar
                    s.sendto(f"COMENZAR {mensaje_minimo}".encode(), (dir, puerto))
            time.sleep(0.05)
            print("Nodo 0: todos listos.")
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
             
    print(f"[SINCRONIZACIÓN] Combinación mínima: {combinacion_minima}")           
    return combinacion_minima


def check_mensaje(mensaje, lista_confirmaciones, contador_prints, min_prov):

    if mensaje.startswith("LISTO"):
        _, parametros_mensaje = mensaje.split(' ', 1)
        
        # Comprueba si el mensaje recibido tiene índices menores y actualiza min_prov
        
        nodo_id = int(parametros_mensaje.split('_')[-1].replace("nodo", ""))
        if not lista_confirmaciones[nodo_id]:
            lista_confirmaciones[nodo_id] = True

            indices_mensaje = parsear_parametros(parametros_mensaje)
            print(f"Min prov: {min_prov}, indices mensaje: {indices_mensaje}")
            if min_prov is None: 
                min_prov = indices_mensaje
                print(f"Nuevo min prov por None: {min_prov}")            
            elif indices_mensaje < min_prov:
                min_prov = indices_mensaje
                print(f"Nuevo min prov: {min_prov}")


            if contador_prints % 50 == 0:
                print(f"Nodo 0. Recibido: {mensaje}")
            contador_prints += 1
            
            return min_prov
            
        return min_prov


def parsear_parametros(mensaje):
    partes = mensaje.split('_')
    
    dataset = partes[0]  # 'elec'
    s_value = int(partes[1][1:])  # Extrae '1' de 's1' y convierte a entero
    t_value = float(partes[2][1:])  # Extrae '0.0' de 'T0.0' y mantiene como flotante
    iteration = int(partes[3][2:])  # Extrae '31' de 'it31' y convierte a entero

    # Encuentra índices de los valores en sus respectivas listas
    dataset_index = datasets.index(dataset)
    s_index = S.index(s_value)
    
    # Para encontrar el índice de t_value en T, usamos np.isclose para manejar precisión de punto flotante
    t_index = np.where(np.isclose(T, t_value))[0][0]

    return (iteration, dataset_index, s_index, t_index)


def indices_a_parametros(indices):
    iteration, dataset_index, s_index, t_index = indices
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
                        print(f"[ITERATION] Pre-SINCRO:  {i_iter}, dataset: {dataset}, S: {s}, T:{t}")

                        parametros = f"{dataset}_s{s}_T{t}_it{i_iter}_nodo{id}"
                        nombre_archivo = f"{directorio_resultados}/result_{parametros}.txt"
                        if os.path.isfile(nombre_archivo):
                            print(f"El archivo '{nombre_archivo}' ya existe. No es necesario generarlos de nuevo.")
                            t_idx += 1
                            continue  # Salta a la siguiente iteración si el archivo ya existe

                        i_iter, dataset_idx, s_idx, t_idx = sincronizar()
                        dataset = datasets[dataset_idx]
                        s = S[s_idx]
                        t = T[t_idx]
                        i_iter = i_iter
                        
                        print(f"[ITERATION] Post-SINCRO:  {i_iter}, dataset: {dataset}, S: {s}, T:{t}")
                        main(data_frame)
                        print(f"- Tiempo de ejecución: {(time.perf_counter() - tiempo_inicio) / 60} minutos.\n")
                        t_idx += 1  # Avanza manualmente a la siguiente iteración de T
                    s_idx += 1  # Avanza manualmente a la siguiente iteración de S
                dataset_idx += 1  # Avanza manualmente a la siguiente iteración del dataset
            i_iter += 1  # Avanza manualmente a la siguiente iteración principal

    except KeyboardInterrupt as e:
        print(f"Se ha interrumpido la ejecución del programa: {e}")

