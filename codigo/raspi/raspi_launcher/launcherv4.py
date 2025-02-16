import sys
import os
ruta_directorio_main = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_directorio_main not in sys.path:
    sys.path.append(ruta_directorio_main)

from prototypes_mod import XuILVQ
import pandas as pd
from node_class.nodev4 import Nodev4
import time
import threading
import numpy as np
import socket
import csv


def cyc_sampling_for_node(df: pd.DataFrame, node_id: int, step: int, cycle_shift: int, num_samples: int) -> pd.DataFrame:
    """
    Sample num_samples indices from df using a cyclical pattern based on node_id.
    """
    total_samples = len(df)
    indices = []
    offset = node_id  # Start offset depends on node
    # Continue until we have the desired number of samples
    while len(indices) < num_samples:
        # Determine how many samples per cycle (avoid division by zero)
        max_per_cycle = total_samples // step if step != 0 else total_samples
        for i in range(max_per_cycle):
            idx = (offset + i * step) % total_samples
            indices.append(idx)
            if len(indices) == num_samples:
                break
        offset += cycle_shift  # Shift for the next cycle
    sampled_df = df.iloc[indices].reset_index(drop=True)
    return sampled_df



def read_dataset(name: str):
    filename = data_name[name]
    dataset = pd.read_csv(f"../dataset/{filename}")
    # Se cambia el 'UP' por 1 y el 'DOWN' por 0
    dataset.replace('UP', 1, inplace=True)
    dataset.infer_objects(copy=False)
    dataset.replace('DOWN', 0, inplace=True)
    dataset.infer_objects(copy=False)

    dataset.replace('True', 1, inplace=True)
    dataset.infer_objects(copy=False)
    dataset.replace('False', 0, inplace=True)
    dataset.infer_objects(copy=False)


      
    dataset.infer_objects(copy=False)
    return dataset


def main(df: pd.DataFrame):
    if dataset == "lgr":
        step = 1000        
        cycle_shift = 5    
        num_samples = 500  
        df_nodos = cyc_sampling_for_node(df, id, step, cycle_shift, num_samples)
        print(f"[Node {id}] Using downsampled data: {df_nodos.shape[0]} samples for dataset '{dataset}'")
    
    elif "phis" in dataset or "elec2" in dataset:
        # For the 'phis' or 'elec2' datasets, adjust the total sample size:
        init_index = 0
        tam_muestra = n_muestras * n_nodos
        # Limit the sample size to 1250 if needed
        tam_muestra = min(tam_muestra, 1250)
        max_init_index = len(df) - tam_muestra
        print(f"[Node {id}] max_init_index: {max_init_index}, tam_muestra: {tam_muestra}, len(df): {len(df)}")
        init_index = np.random.randint(0, max_init_index) if max_init_index > 0 else 0
        df_short = df[init_index : init_index + tam_muestra]
        df_nodos = df_short.iloc[id::n_nodos, :].reset_index(drop=True)
        print(f"[Node {id}] Using 'phis'/'elec2' slicing: {df_nodos.shape[0]} samples")
    
    else:
        # For other datasets, use standard contiguous slicing:
        init_index = 0
        tam_muestra = n_muestras * n_nodos
        df_short = df[init_index : init_index + tam_muestra]
        df_nodos = df_short.iloc[id::n_nodos, :].reset_index(drop=True)
        print(f"[Node {id}] Using default slicing: {df_nodos.shape[0]} samples")


    nodo = Nodev4(id, dataset=df_nodos, modelo_proto=XuILVQ(max_pset_size=limit, target_size=target_range), nodos=n_nodos, s=s, T=t, media_llegadas=media_llegadas)

    hilo = threading.Thread(target=nodo.run)
    hilo.start()
    hilo.join(T_MAX_IT)
    
    if hilo.is_alive():
        print(f"EL HILO ESTÁ BLOQUEADO. CERRANDO PROGRAMA")
        exit()
        

    tp = nodo.matriz_conf["TP"]
    fp = nodo.matriz_conf["FP"]
    fn = nodo.matriz_conf["FN"]
    precision = round(tp / (tp + fp), 3) if tp + fp != 0 else 0
    recall = round(tp / (tp + fn), 3) if tp + fn != 0 else 0
    f1 = round(2 * (precision * recall) / (precision + recall), 3) if precision + recall != 0 else 0
    clust_time = round(nodo.clust_time, 1)


    if nodo.tiempo_learn_queue == 0:
        cap_ejec = 0
    else:
        cap_ejec = round((nodo.protos_train + nodo.muestras_train) / (nodo.tiempo_learn_queue + nodo.tiempo_learn_data), 3)
        

    # Crear una fila con los datos del nodo
    row = {
        "NODO": nodo.id,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Muestras entrenadas": nodo.muestras_train,
        "Prototipos entrenados": nodo.protos_train,
        "Veces compartido": nodo.shared_times_final,
        "Prototipos compartidos": nodo.compartidos_final,
        "Prototipos ahorrados": nodo.no_comp_jsd_final,
        "Prototipos descartados": nodo.protos_descartados_final,
        "Ejecuciones de clustering": nodo.clust_runs,
        "Tiempo clustering": clust_time,
        "Tiempo aprendizaje (muestras)": nodo.tiempo_learn_data,
        "Tiempo aprendizaje (prototipos)": nodo.tiempo_learn_queue,
        "Tiempo compartiendo prototipos": nodo.tiempo_share_final,
        "Tiempo no compartiendo prototipos": nodo.tiempo_no_share_final,
        "Tiempo total espera activa": nodo.tiempo_espera_total,
        "Tiempo total": nodo.tiempo_final_total,
        "Capacidad de ejecución": cap_ejec,
        "Tamaño de lotes recibidos": nodo.tam_lotes_recibidos,
        "Tamaño conjunto de prototipos": nodo.tam_conj_prot
    }


    # Escribir en el archivo CSV
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
        # Como base se ponen los parametros del nodo central
        min_prov = parsear_parametros(parametros)
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
    print(f"Se van a parsear los parametros: {mensaje}")
    partes = mensaje.split('_')
    
    dataset = partes[0]
    s_value = int(partes[1][1:])
    t_value = float(partes[2][1:])
    limite = int(partes[3][5:])
    ranges = partes[4].split('-')
    inf_range = float(ranges[0][5:])
    sup_range = float(ranges[1][:])
    target_range = (inf_range, sup_range)
    lim_range_searched = (limite, target_range)
    iteracion = int(partes[5][2:])   
    
    dataset_index = datasets.index(dataset)
    s_index = S.index(s_value)
    t_index = np.where(np.isclose(T, t_value))[0][0]
    target_index = lim_range.index(lim_range_searched)
    
    return (iteracion, dataset_index, s_index, t_index, target_index)



def indices_a_parametros(indices):
    iteration, dataset_index, s_index, t_index, limit_target_index = indices
    dataset = datasets[dataset_index]
    s = S[s_index]
    T_value = T[t_index]
    limit, target_range = lim_range[limit_target_index]
    target_range_str = "-".join(map(str, target_range))
    return f"{dataset}_s{s}_T{T_value}_limit{limit}_range{target_range_str}_it{iteration}"



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

        


if __name__ == "__main__":

    try:

        hostname = socket.gethostname()
        id = int(''.join(filter(str.isdigit, hostname)))
        n_nodos = 5
        n_muestras = 1000

        T_MAX_IT = 300  # Tiempo máximo de ejecución del hilo
        S = [i for i in range(1, 5)]
        T = np.array([i for i in range(0, 1001, 50)])
        T = np.array([i for i in range(0, 1001, 100)])
        T = T / 1000
        tasa_llegadas = 10
        media_llegadas = 1 / tasa_llegadas

        iteraciones = 50
        datasets = ["elec", "phis", "elec2", "lgr"] #, "nrr", "lar", "lrr", "ngcr", "nsch" ]
	datasets = ["lgr"]

        data_name = {"elec": "electricity.csv", 
                    "phis": "phishing.csv",
                    "elec2": "electricity.csv",
                    "lgr": "linear_gradual_rotation_noise_and_redunce.csv" , 
                    # "nrr": "nonlinear_recurrent_rollingtorus_noise_and_redunce.csv",
                    # "lar": "linear_abrupt_noise_and_redunce.csv",                          
                    # "lrr": "linear_recurrent_rotation_noise_and_redunce.csv",              
                    # "ngcr": "nonlinear_gradual_cakerotation_noise_and_redunce.csv",        
                    # "nsch": "nonlinear_sudden_chocolaterotation_noise_and_redunce.csv"     
                    }
        
        # Parámetros temporales para hacer pruebas no simulaciones
        # datasets = [ "lgr", "nrr", "lar", "lrr", "ngcr", "nsch" ]
        # S = [1, 4]
        # T = np.array([0.0, 0.1, 0.5, 1.0])
        # iteraciones = 10

        lim_range = [
            # (50, (50, 60)),
            # (150, (50, 60)),
            # (250, (50, 60)),
            (500, (72.5, 77.5))
        ] 
        
        directorio_resultados = "../resultados_raspi_indiv"

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
                        lim_range_idx = 0
                        while lim_range_idx < len(lim_range):
                            limit, target_range = lim_range[lim_range_idx]
                            tiempo_inicio = time.perf_counter()
                            print(f"[ITERATION] Pre-SINCRO: {i_iter}, dataset: {dataset}, S: {s}, T: {t}, Limit: {limit}, Target: {target_range}")

                            parametros = f"{dataset}_s{s}_T{t}_limit{limit}_range{'-'.join(map(str, target_range))}_it{i_iter}_nodo{id}"
                            nombre_archivo = f"{directorio_resultados}/result_{parametros}.csv"
                            
                            if os.path.isfile(nombre_archivo):
                                print(f"El archivo '{nombre_archivo}' ya existe. No es necesario generarlos de nuevo.")
                                lim_range_idx += 1
                                continue

                            # Synchronize here
                            i_iter, dataset_idx, s_idx, t_idx, lim_range_idx = sincronizar()
                            dataset = datasets[dataset_idx]
                            s = S[s_idx]
                            t = T[t_idx]
                            limit, target_range = lim_range[lim_range_idx]
                                    
                            new_parametros = f"{dataset}_s{s}_T{t}_limit{limit}_range{'-'.join(map(str, target_range))}_it{i_iter}_nodo{id}"
                            nombre_archivo = f"{directorio_resultados}/result_{new_parametros}.csv"

                            print(f"[ITERATION] Post-SINCRO: {i_iter}, dataset: {dataset}, S: {s}, T: {t}, Limit: {limit}, Target: {target_range}")
                            main(data_frame)
                            print(f"- Tiempo de ejecución: {(time.perf_counter() - tiempo_inicio) / 60} minutos.\n")

                            lim_range_idx += 1  # Move to the next combination of limit and target_range
                        t_idx += 1  # Move to the next T value
                    s_idx += 1  # Move to the next S value
                dataset_idx += 1  # Move to the next dataset
            i_iter += 1  # Move to the next overall iteration

    except KeyboardInterrupt as e:
        print(f"Se ha interrumpido la ejecución del programa: {e}")
