import sys
import os
import pandas as pd
import time
from tqdm import tqdm
import socket
ruta_directorio_main = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_directorio_main not in sys.path:
    sys.path.append(ruta_directorio_main)
from prototypes import XuILVQ


hostname = socket.gethostname()


def read_dataset(name: str):
    dataset_path = f"../dataset/{name.strip('2')}"
    dataset = pd.read_csv(dataset_path)
    dataset.replace({'UP': 1, 'DOWN': 0, 'True': 1, 'False': 0}, inplace=True)
    
    if "2" in name:
        return dataset.iloc[::5].iloc[:1000]
    return dataset

def main():
    num_runs = 3  # Ejecutar el experimento 30 veces
    datasets = ["electricity.csv", "http_proc.csv", "2electricity.csv"]
    resultados = {key: {'cap_ejec': [], 'tams': []} for key in datasets}

    for _ in range(num_runs):
        for dataset in datasets:
            modelo = XuILVQ()
            df = read_dataset(dataset)
            df_list = [(fila[:-1], fila[-1]) for fila in df.values]
            
            start = time.perf_counter()
            for x, y in tqdm(df_list, desc=f"Processing {dataset}"):
                x = dict(enumerate(x))
                modelo.learn_one(x, y)
            end = time.perf_counter()
            
            cap_ejec = len(df) / (end - start)
            tams = len(list(modelo.buffer.prototypes.values()))
            
            resultados[dataset]['cap_ejec'].append(cap_ejec)
            resultados[dataset]['tams'].append(tams)
    
    # Escribir los resultados promedio en un archivo
    with open(f"sbench_{hostname}.txt", "w") as f:
        f.write("Capacidad de ejecución promedio\n")
        for key in resultados:
            f.write(f"{key}: {sum(resultados[key]['cap_ejec']) / num_runs}\n")
        
        f.write("\nTamaños promedios de los conjuntos de prototipos generados\n")
        for key in resultados:
            f.write(f"{key}: {sum(resultados[key]['tams']) / num_runs}\n")
            
if __name__ == "__main__":
    main()
