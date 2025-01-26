import csv
import sys
import os
import time
import threading
import numpy as np
import pandas as pd
import socket
ruta_directorio_main = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_directorio_main not in sys.path:
    sys.path.append(ruta_directorio_main)
from node_class.vfdt_node import VFDTreev1

# Function to read and preprocess the dataset
def read_dataset(name: str):
    filename = data_name[name]
    dataset = pd.read_csv(f"../dataset/{filename}")
    dataset.replace(['UP', 'True'], 1, inplace=True)
    dataset.replace(['DOWN', 'False'], 0, inplace=True)
    dataset.infer_objects(copy=False)
    return dataset

# Main function to run the node
def main(df: pd.DataFrame, id: int, n_nodos: int, n_muestras: int, dataset: str, s: int, t: float, i_iter: int):
    # Prepare the dataset for the node
    init_index = 0
    tam_muestra = n_muestras * n_nodos
    if "phis" in dataset or "elec2" in dataset:
        tam_muestra = min(tam_muestra, 1250)
        max_init_index = len(df) - tam_muestra
        init_index = np.random.randint(0, max_init_index) if max_init_index > 0 else 0

    df_short = df[init_index: init_index + tam_muestra]
    df_nodos = [df_short.iloc[i::n_nodos, :].reset_index(drop=True) for i in range(n_nodos)]

    # Initialize the node
    nodo = VFDTreev1(
        id=id,
        dataset=df_nodos[id],
        nodos=n_nodos,
        s=s,
        T=t,
        media_llegadas=media_llegadas
    )

    # Start the node in a separate thread
    hilo = threading.Thread(target=nodo.run)
    hilo.start()
    hilo.join(T_MAX_IT)

    if hilo.is_alive():
        print(f"EL HILO ESTÁ BLOQUEADO. CERRANDO PROGRAMA")
        exit()

    # Collect statistics from the node
    tp = nodo.matriz_conf["TP"]
    fp = nodo.matriz_conf["FP"]
    fn = nodo.matriz_conf["FN"]
    precision = round(tp / (tp + fp), 3) if tp + fp != 0 else 0
    recall = round(tp / (tp + fn), 3) if tp + fn != 0 else 0
    f1 = round(2 * (precision * recall) / (precision + recall), 3) if precision + recall != 0 else 0

    # Calculate execution capacity
    if nodo.tiempo_aggregation == 0:
        cap_ejec = 0
    else:
        cap_ejec = round((nodo.muestras_train + nodo.params_aggregated) / (nodo.tiempo_learn_data + nodo.tiempo_aggregation), 3)


    # Prepare the row for the CSV file
    row = {
        "NODO": nodo.id,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Muestras entrenadas": nodo.muestras_train,
        "Parámetros agregados": nodo.params_aggregated,
        "Veces compartido": nodo.shared_times_final,
        "Tiempo aprendizaje (muestras)": nodo.tiempo_learn_data,
        "Tiempo agregación": nodo.tiempo_aggregation,
        "Tiempo compartiendo": nodo.tiempo_share_final,
        "Tiempo no compartiendo": nodo.tiempo_no_share_final,
        "Tiempo total espera activa": nodo.tiempo_espera_total,
        "Tiempo total": nodo.tiempo_final_total,
        "Capacidad de ejecución": cap_ejec
    }

    # Write the results to the CSV file
    parametros = f"{dataset}_s{s}_T{t}_it{i_iter}_nodo{id}"
    nombre_archivo = f"{directorio_resultados}/result_{parametros}.csv"
    with open(nombre_archivo, 'w', newline='') as csvfile:
        fieldnames = row.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    print("Se ha terminado de ejecutar todo y se ha generado el archivo CSV.")

def sincronizar():

    print("Comienza la sincronización...")
    puerto = 11111  # Puerto común para la sincronización
    buffer_size = 1024  # Tamaño del buffer para recibir mensajes
    dir_server = "nodo0.local"  # Dirección del nodo central
    dir_nodos = [f"nodo{i}.local" for i in range(1, 5)]  # Direcciones de los nodos no centrales

    check_availability(id, dir_nodos, puerto)
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
            for _ in range(5):  # Intenta enviar el mensaje un máximo de 5 veces para cada nodo
                for i, dir in enumerate(dir_nodos):
                    enviado = False
                    while not enviado:
                        try:
                            s.sendto(f"COMENZAR {mensaje_minimo}".encode(), (dir, puerto))
                            enviado = True  # Si se envía correctamente, establece enviado a True para salir del bucle
                        except socket.gaierror:
                            print(f"Error al enviar a {dir}:{puerto}. Reintentando...")
                            
            time.sleep(2)
            print("Nodo 0: todos listos.")
    else:
        ready = False
        # Nodo no central
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s_recepcion:
                s_recepcion.bind(("0.0.0.0", puerto))
                s_recepcion.settimeout(3)  # Establecer timeout
                
                while True:
                    try:
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
                                time.sleep(2)
                                break
                            
                        except socket.timeout:
                            time.sleep(1)

                    except socket.gaierror:
                        print(f"Error al enviar a {dir_server}:{puerto}. Reintentando...")
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

def check_availability(nodo_id, nodos, puerto):
    """
    Comprueba la disponibilidad de todos los nodos intentando establecer
    una conexión UDP. Reintenta hasta un máximo de veces especificado.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:    
        
        if nodo_id == 0:
            s.settimeout(10)  # Configura un timeout para las respuestas
            todos_disponibles = False
            while not todos_disponibles:
                respuestas_exitosas = 0
                
                for nodo in nodos:
                    try:
                        print(f"Enviando 'ping' a {nodo}")
                        s.sendto(b"ping", (nodo, puerto))
                        data, _ = s.recvfrom(1024)
                        if data.decode() == "pong":
                            print(f"{nodo} respondió 'pong'")
                            respuestas_exitosas += 1
                    except (socket.gaierror, socket.timeout):
                        print(f"{nodo} no respondió o timeout alcanzado.")
                
                # Verifica si todos los nodos respondieron en esta iteración
                if respuestas_exitosas == len(nodos):
                    todos_disponibles = True
                    print("Todos los nodos están disponibles.")
                    for nodo in nodos:
                        s.sendto(b"stop", (nodo, puerto))
                else:
                    print("No todos los nodos están disponibles. Reintentando...")
                    time.sleep(1)  # Espera un momento antes de intentar nuevamente

        
        else: 
            # Configurar el nodo no central para responder a pings
            s.bind(("0.0.0.0", puerto))
            while True:
                try:
                    data, addr = s.recvfrom(1024)
                    if data.decode() == "ping":
                        s.sendto(b"pong", addr)
                        print("Respondido 'pong'")
                    elif data.decode() == "stop":
                        print(f"Recibido 'stop'. Cerrando socket...")
                        break
                except socket.timeout:
                    print("No se recibieron pings en el tiempo esperado.")


# Main execution block
if __name__ == "__main__":
    try
        hostname = socket.gethostname()
        id = int(''.join(filter(str.isdigit, hostname)))
        n_nodos = 5
        n_muestras = 1000
        
        T_MAX_IT = 300  # Tiempo máximo de ejecución del hilo por iteración
        S = [i for i in range(1, 5)]
        T = np.array([i for i in range(0, 1001, 100)])
        T = T / 1000
        tasa_llegadas = 10
        media_llegadas = 1 / tasa_llegadas
        
        
        iteraciones = 50
        datasets = ["elec", "phis", "elec2"]

        data_name = {"elec": "electricity.csv", 
                    "phis": "phishing.csv",
                    "elec2": "electricity.csv",
                    }
        
        # Parámetros temporales para hacer pruebas no simulaciones
        S = [1, 2, 4] 
        iteraciones = 20


        directorio_resultados = "../resultados_raspi_indiv_tree"

        if not os.path.exists(directorio_resultados):
            os.makedirs(directorio_resultados)



        i_iter = 0
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
                        print(f"[ITERATION] Pre-SINCRO: {i_iter}, dataset: {dataset}, S: {s}, T: {t}")

                        parametros = f"{dataset}_s{s}_T{t}_it{i_iter}_nodo{id}"
                        nombre_archivo = f"{directorio_resultados}/result_{parametros}.csv"

                        if os.path.isfile(nombre_archivo):
                            print(f"El archivo '{nombre_archivo}' ya existe. No es necesario generarlos de nuevo.")
                            t_idx += 1
                            continue

                        # Synchronize here
                        i_iter, dataset_idx, s_idx, t_idx = sincronizar()
                        dataset = datasets[dataset_idx]
                        s = S[s_idx]
                        t = T[t_idx]

                        new_parametros = f"{dataset}_s{s}_T{t}_it{i_iter}_nodo{id}"
                        nombre_archivo = f"{directorio_resultados}/result_{new_parametros}.csv"

                        print(f"[ITERATION] Post-SINCRO: {i_iter}, dataset: {dataset}, S: {s}, T: {t}")
                        main(data_frame, id, n_nodos, n_muestras, dataset, s, t, i_iter)
                        print(f"- Tiempo de ejecución: {(time.perf_counter() - tiempo_inicio) / 60} minutos.\n")

                        t_idx += 1
                    s_idx += 1
                dataset_idx += 1
            i_iter += 1

    except KeyboardInterrupt as e:
        print(f"Se ha interrumpido la ejecución del programa: {e}")