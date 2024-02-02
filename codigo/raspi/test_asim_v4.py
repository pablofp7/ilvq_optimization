from prototypes import XuILVQ
import pandas as pd
from nodev4 import Nodev4
import os
import time
import threading
import numpy as np


def read_dataset(name: str):
    filename = data_name[name]
    dataset = pd.read_csv(f"dataset/{filename}")
    dataset.replace('UP', 1, inplace=True)
    dataset.replace('DOWN', 0, inplace=True) 
    return dataset


def main(df: pd.DataFrame): 
    init_index = 0
    tam_muestra = n_muestras * n_nodos
    if "phis" in dataset or "elec2" in dataset:
        tam_muestra = min(tam_muestra, 1250)
        max_init_index = len(df) - tam_muestra
        print(f"max_init_index: {max_init_index}, tam_muestra: {tam_muestra}, len(df): {len(df)}")
        init_index = np.random.randint(0, max_init_index) if max_init_index > 0 else 0

    df_short = df[init_index : init_index + tam_muestra]
        

    tam_parte = tam_muestra // 5
    df_nodos = [df_short.iloc[i * tam_parte : (i + 1) * tam_parte, :].reset_index(drop=True) for i in range(4)]
    tam_quinto_nodo = tam_parte // 5
    inicio_quinto_nodo = 4 * tam_parte
    df_nodos.append(df_short.iloc[inicio_quinto_nodo : inicio_quinto_nodo + tam_quinto_nodo, :].reset_index(drop=True))
    
    nodos = []
    for id in range(n_nodos):
        nodo = Nodev4(id, dataset=df_nodos[id], modelo_proto=XuILVQ(), nodos=n_nodos, s=s, T=t, media_llegadas=media_llegadas)
        nodos.append(nodo)    
    
    hilos = []
    for nodo in nodos:
        hilo = threading.Thread(target=nodo.run)
        hilos.append(hilo)
        hilo.start()

    for hilo in hilos:
        hilo.join()

    for cola in nodos[0].colas:
        cola.remove()

    to_write = []
    # to_write.append(f" - TIEMPO EJECUCION: {(time.time() - tiempo_inicio) / 60} minutos.\n\n")
    #Vamos a guardar en una string lo que se va a escribir en el archivo
    for nodo in nodos:
        try:       
            precision = round(nodo.matriz_conf["TP"] / (nodo.matriz_conf["TP"] + nodo.matriz_conf["FP"]), 3)
            recall = round(nodo.matriz_conf["TP"] / (nodo.matriz_conf["TP"] + nodo.matriz_conf["FN"]), 3)
            f1 = round(2 * (precision * recall) / (precision + recall), 3)     
        except ZeroDivisionError:
            precision = 0
            recall = 0
            f1 = 0
            
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
                        f"ID, Tamaño de lotes recibidos: {nodo.tam_lotes_recibidos}\n\n")

    print("Se ha terminado de ejecutar todo.")    
    with open(nombre_archivo, "w") as f:
        f.writelines(to_write)


if __name__ == "__main__":
    try:
        n_nodos = 5
        n_muestras = 1000

        S = [i for i in range(1, 5)]
        T = np.array([i for i in range(0, 1001, 25)])
        T = T / 1000
        tasa_llegadas = 4
        media_llegadas = 1/ tasa_llegadas

        iteraciones = 20
        datasets = ["elec", "phis", "elec2"]

        data_name = {"elec": "electricity.csv", "phis": "phishing.csv", "elec2": "electricity.csv"}

        directorio_resultados = "resultados_asim_v4"
        
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

                        parametros = f"{dataset}_s{s}_T{t}_it{i}"
                        nombre_archivo = f"{directorio_resultados}/result_{parametros}.txt"
                        if os.path.isfile(nombre_archivo):
                            print(f"El archivo '{nombre_archivo}' ya existe. No es necesario generarlos de nuevo.")
                            continue
                        
                        main(data_frame)
                        print(f"- Tiempo de ejecución: {(time.time() - tiempo_inicio) / 60} minutos.\n")
                        with open("tiempos.txt", "a") as f:
                            f.write(f"\nITERACIONES:\nS: {s}, T:{t} - Tiempo de ejecución: {(time.time() - tiempo_inicio) / 60}\n") if i == 0 else None

    except KeyboardInterrupt as e:
        os.system("ipcrm --all=msg")
