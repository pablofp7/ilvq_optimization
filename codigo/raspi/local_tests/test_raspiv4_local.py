import sys
import os
ruta_directorio_main = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_directorio_main not in sys.path:
    sys.path.append(ruta_directorio_main)

from prototypes import XuILVQ
import pandas as pd
from old_node_class.raspi_nodev4_local import RaspiNodev4local
import time
import threading
import numpy as np


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
    
    
    nodos = []
    for id in range(n_nodos):
        nodo = RaspiNodev4local(id, dataset=df_nodos[id], modelo_proto=XuILVQ(), nodos=n_nodos, s=s, T=t, media_llegadas=media_llegadas)
        nodos.append(nodo)    
    
    hilos = []
    for nodo in nodos:
        hilo = threading.Thread(target=nodo.run)
        hilos.append(hilo)
        hilo.start()

    for hilo in hilos:
        hilo.join()
        
        

        
    to_write = []
    # to_write.append(f" - TIEMPO EJECUCION: {(time.perf_counter() - tiempo_inicio) / 60} minutos.\n\n")
    #Vamos a guardar en una string lo que se va a escribir en el archivo
    for nodo in nodos:
        
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


if __name__ == "__main__":
    
    try:
        n_nodos = 5
        n_muestras = 1000
        
        S = [i for i in range(1, 5)]
        T = np.array([i for i in range(0, 1001, 50)])
        T = T / 1000
        tasa_llegadas = 20
        media_llegadas = 1 / tasa_llegadas
        
        iteraciones = 50
        datasets = ["elec", "phis", "elec2"]
        
        iteraciones = 20
        datasets = ["elec", "phis", "elec2"]

        data_name = {"elec": "electricity.csv", "phis": "phishing.csv", "elec2": "electricity.csv"}
        
        directorio_resultados = "../resultados_raspiv4"
        
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
        
    except KeyboardInterrupt as e:
        os.system("ipcrm --all=msg")
        
        
