import sys
import os
ruta_directorio_main = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_directorio_main not in sys.path:
    sys.path.append(ruta_directorio_main)

import pandas as pd
from old_node_class.raspi_nodev4_2_local_mp import RaspiNodev4_2local_mp
import time
import numpy as np
import multiprocessing



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

def nodo_run_wrapper(args: list, cola_resultados: multiprocessing.Queue):

    id, df, n_nodos, s, t, media_llegadas, tam_colas = args
    nodo = RaspiNodev4_2local_mp(id=id, dataset=df, nodos=n_nodos, s=s, T=t, media_llegadas=media_llegadas, tam_colas = tam_colas)

    nodo.run()
    
    try:
        precision = nodo.matriz_conf["TP"] / (nodo.matriz_conf["TP"] + nodo.matriz_conf["FP"])
    except:
        precision = 0
    try:
        recall = nodo.matriz_conf["TP"] / (nodo.matriz_conf["TP"] + nodo.matriz_conf["FN"])
    except:
        recall = 0
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except:
        f1 = 0
        
    try:
        cap_ejec = (nodo.muestras_train + nodo.protos_train )/ nodo.tiempo_final_total
    except:
        cap_ejec = 0
        
    precision = round(precision, 3)
    recall = round(recall, 3)
    f1 = round(f1, 3)
    estadisticas = {
        'id': nodo.id,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'muestras_train': nodo.muestras_train,
        'protos_train': nodo.protos_train,
        'tiempo_learn_data': nodo.tiempo_learn_data,
        'tiempo_learn_queue': nodo.tiempo_learn_queue,
        'tiempo_final_total': nodo.tiempo_final_total,
        'cap_ejec': cap_ejec,
        'tam_conj_prot': nodo.tam_conj_prot,
        'tiempo_share': nodo.tiempo_share_final,
        'tam_lotes_recibidos': nodo.tam_lotes_recibidos,
        'shared_times': nodo.shared_times_final,
        'compartidos': nodo.compartidos_final,
    }
    
    cola_resultados.put(estadisticas)   
    
    print(f"[NODO {nodo.id}] - Ha vuelto del nodo.run() y se han obtenido las estadísticas, volviendo al join")
    
    return 
    
def main(df: pd.DataFrame): 

    max_retries = 3
    retries = 0
    
    while retries < max_retries:
        
        #Se adapta para darle a cada nodo su parte
        init_index = 0
        tam_muestra = n_muestras * n_nodos
        if "phis" in dataset or "elec2" in dataset:
            tam_muestra = min(tam_muestra, 1250)
            max_init_index = len(df) - tam_muestra
            print(f"max_init_index: {max_init_index}, tam_muestra: {tam_muestra}, len(df): {len(df)}")
            init_index = np.random.randint(0, max_init_index) if max_init_index > 0 else 0
        elif "http" in dataset:
            tam_muestra = len(df)
        
        df_short = df[init_index : init_index + tam_muestra]
        df_nodos = [df_short.iloc[i::n_nodos, :].reset_index(drop=True) for i in range(n_nodos)]

    
        nodos_args = []
        for id in range(n_nodos):
            args = [id, df_nodos[id], n_nodos, s, t, media_llegadas, TAM_COLAS]
            nodos_args.append(args)

        
        cola_resultados = multiprocessing.Queue()
        # Crear una lista para mantener un seguimiento de los procesos hijos
        procesos = []
        # Crear y empezar un nuevo proceso para cada conjunto de argumentos
        for args in nodos_args:
            p = multiprocessing.Process(target=nodo_run_wrapper, args=(args, cola_resultados), name=f"Proceso_Nodo_{args[0]}")
            # p = multiprocessing.Process(target=nodo_run_wrapper, args=(args, cola_resultados))
            p.start()
            procesos.append(p)
        
        
        tiempo_inicio = time.time()
        while time.time() - tiempo_inicio < T_MAX_IT:
            if all(not p.is_alive() for p in procesos):
                print(f"Todos los procesos han terminado EXITOSAMENTE.")
                break
            time.sleep(2)
        
        else:
            print(f"Se ha alcanzado el tiempo máximo de ejecución.")
            for p in procesos:
                if p.is_alive():
                    p.terminate()
                    print(f"Se ha terminado el proceso {p.name} por exceder el tiempo máximo de ejecución.")
                            # Esperar a que cada proceso hijo termine
            for p in procesos:
                p.join()
            
            retries += 1
            print(f"Se ha alcanzado el tiempo máximo de ejecución. Se va a intentar de nuevo. Intento {retries} de {max_retries}")
            continue
        
        
        # Esperar a que cada proceso hijo termine
        for p in procesos:
            p.join()
        
        resultados = []
        while not cola_resultados.empty():
            resultados.append(cola_resultados.get())
        
        resultados = sorted(resultados, key=lambda x: x['id'])
        
        for resultado in resultados:
            print(f"ID: {resultado['id']}, Precision: {resultado['precision']}, Recall: {resultado['recall']},"
                    f"F1: {resultado['f1']}, Muestras entrenadas: {resultado['muestras_train']}, Prototipos entrenados: {resultado['protos_train']},"
                    f"Capacidad de ejecución: {resultado['cap_ejec']}, Tiempo total: {resultado['tiempo_final_total']},"
                    f"Tiempo de aprendizaje (prototipos): {resultado['tiempo_learn_queue']}, Tiempo compartiendo prototipos: {resultado['tiempo_share']}, Tam. conjunto prototipos: {resultado['tam_conj_prot']},")
        
        to_write = []
        for stats in resultados:
                
            to_write.append(f" - NODO {stats['id']}.\nPrecision: {stats['precision']}\nRecall: {stats['recall']}\nF1: {stats['f1']}\n"
                                f"Se ha entrenado con {stats['muestras_train']} muestras.\nSe ha entrenado con {stats['protos_train']} prototipos.\n"
                                f"Ha compartido {stats['shared_times']} veces.\n"
                                f"Ha compartido {stats['compartidos']} prototipos a cada uno de los {s} vecino/s.\n"
                                f"Tiempo de aprendizaje (muestras): {stats['tiempo_learn_data']}\n"
                                f"Tiempo de aprendizaje (prototipos): {stats['tiempo_learn_queue']}\n"
                                f"Tiempo compartiendo prototipos: {stats['tiempo_share']}\n"
                                f"Tiempo total: {stats['tiempo_final_total']}\n"
                                f"Capacidad de ejecución: {stats['cap_ejec']}\n"
                                f"ID, Tamaño de lotes recibidos: {stats['tam_lotes_recibidos']}\n"
                                f"Tamaño conjunto de prototipos: {stats['tam_conj_prot']}\n"
                                f"\n")

        print("Se ha terminado de ejecutar todo.")    

        with open(nombre_archivo, "w") as f:
            f.writelines(to_write)
        
        break
    
    else: 
        print(f"BLOQUEO")
        raise KeyboardInterrupt
            

if __name__ == "__main__":
    nombre_programa = sys.argv[0]
    
    try:
        n_nodos = 5
        n_muestras = 1000
        t_max_minutes = 250000
        T_MAX_IT = t_max_minutes * 60
        TAM_COLAS = 500
                
        S = [i for i in range(1, 5)]
        T = np.array([i for i in range(0, 1001, 50)])
        T = T / 1000
        tasa_llegadas = 10
        media_llegadas = 1 / tasa_llegadas
        
        iteraciones = 50
        datasets = ["elec", "phis", "elec2"]
        
        datasets = ["http"]
        S = [4]
        T = [0.0, 0.2, 1.0] 

        data_name = {"elec": "electricity.csv", "phis": "phishing.csv", "elec2": "electricity.csv", "http": "http_proc.csv", "movie": "movie_proc.csv"}
        
        directorio_resultados = "../resultados_local"
        
        if not os.path.exists(directorio_resultados):
            os.makedirs(directorio_resultados)

        for i in range(iteraciones):
            for dataset in datasets:
                data_frame = read_dataset(dataset)
                for s in S:
                    tiempo_s = time.perf_counter()
                    for t in T:
                        # if t == 0 and i > 0 and s > 1:
                        #     continue
                        tiempo_inicio = time.perf_counter()
                        print(f"ITERACIÓN {i}, dataset: {dataset}, S: {s}, T:{t}")
                        
                        parametros = f"{dataset}_s{s}_T{t}_it{i}"
                        nombre_archivo = f"{directorio_resultados}/result_{parametros}.txt"
                        if os.path.isfile(nombre_archivo):
                            print(f"El archivo '{nombre_archivo}' ya existe. No es necesario generarlos de nuevo.")
                            continue  # Salta a la siguiente iteración si el archivo ya existe
                        
                        main(data_frame)
                        print(f"- Tiempo de ejecución: {(time.perf_counter() - tiempo_inicio) / 60} minutos.\n")
        
        
        # comando = f"pkill -f \"python3 {nombre_programa}\""
        # print(f"Se va a ejecutar el comando: {comando}")
        # os.system(comando)
        
    except KeyboardInterrupt as e:
        # comando = f"pkill -f \"python3 {nombre_programa}\""
        # print(f"Se va a ejecutar el comando: {comando}")
        # os.system(comando)
        exit()
        
        
