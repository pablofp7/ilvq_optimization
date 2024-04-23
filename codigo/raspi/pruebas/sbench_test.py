import sys
import os
ruta_directorio_main = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_directorio_main not in sys.path:
    sys.path.append(ruta_directorio_main)

import pandas as pd
from prototypes import XuILVQ
import time
from tqdm import tqdm

def read_dataset(name: str):
    
    if "2" in name:
        name = name[1:]
        dataset = pd.read_csv(f"../dataset/{name}")
        # Se cambia el 'UP' por 1 y el 'DOWN' por 0
        dataset.replace('UP', 1, inplace=True)
        dataset.replace('DOWN', 0, inplace=True)
        dataset.replace('True', 1, inplace=True)
        dataset.replace('False', 0, inplace=True)
        #En el segundo caso de electricity se entrena "sólo" con mil muestras desde el principio pero intercaladas de 5 en 5
        dataset = dataset.iloc[::5].iloc[:1000]
        return dataset
        
    dataset = pd.read_csv(f"../dataset/{name}")
    # Se cambia el 'UP' por 1 y el 'DOWN' por 0
    dataset.replace('UP', 1, inplace=True)
    dataset.replace('DOWN', 0, inplace=True) 

    dataset.replace('True', 1, inplace=True)
    dataset.replace('False', 0, inplace=True) 


    return dataset


def main():
    
    datasets = ["electricity.csv", "http_proc.csv", "2electricity.csv"]
    tiempos = {key: None for key in datasets}
    tams = {key: None for key in datasets}
    
    modelo = XuILVQ()
    
    # Vamos a realizar train y contar cuantas muestras se han entrenado por segundo
    # Esto se hace entrenando con el dataset, diviendo número de muestras entrenadas entre los segundos que ha tardado
    
    for dataset in datasets:
        df = read_dataset(dataset)
        print(f"Dataset: {dataset}")
        df_list = [(fila[:-1], fila[-1]) for fila in df.values]

        start = time.perf_counter()
        for i, (x, y) in enumerate(tqdm(df_list, desc=f"Processing {dataset}")):
            x = {k: v for k, v in enumerate(x)}
            modelo.learn_one(x, y)
        end = time.perf_counter()
        tiempos[dataset] = len(df) / (end - start)
        tams[dataset] = len(list(modelo.buffer.prototypes.values()))
        
    
    with open("tiempos_sbench.txt", "w") as f:
        for key, value in tiempos.items():
            f.write(f"{key}: {value}\n")
            f.write("\n")
        
        for key, value in tams.items():
            f.write(f"{key}: {value}\n")
            f.write("\n")
            
            
if __name__ == "__main__":
    main()
                
