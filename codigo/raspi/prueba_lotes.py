# Este programa se va a entrenar el modelo XuILVQ como en los otros
# programas, guardar el tamaño del conjunto de prototipos en número de prototipos
# convertirlo a pickle como se hace en nodev3.py y luego se calcula el numero de bytes que es mensaje ocupa

import numpy as np
import pandas as pd     
from prototypes import XuILVQ
import pickle
import sys


def read_dataset():
    dataset = pd.read_csv(f"dataset/electricity.csv")
    # Se cambia el 'UP' por 1 y el 'DOWN' por 0
    dataset.replace('UP', 1, inplace=True)
    dataset.replace('DOWN', 0, inplace=True) 

    dataset.replace('True', 1, inplace=True)
    dataset.replace('False', 0, inplace=True) 


    return dataset




def main():
    
    df = read_dataset()
    #        datalist = [(fila[:-1], fila[-1]) for fila in dataset.values] 
    df_list = [(fila[:-1], fila[-1]) for fila in df.values]
    
    
    modelo = XuILVQ()
    lista_tam_bytes = []
    for i in range(len(df_list)):
        x, y = df_list.pop(0)            
        x = {k: v for k, v in enumerate(x)}    
        modelo.learn_one(x, y)
        if i % 100 == 0:
            print(f"Iteracion {i}")
            protos = list(modelo.buffer.prototypes.values())
            tam_num = len(protos)
            proto_to_share = pickle.dumps({"id": id, "protos": [{'x': proto['x'], 'y': proto['y']} for proto in protos]})
            tam_bytes = sys.getsizeof(proto_to_share)
            
            lista_tam_bytes.append((i, tam_num, tam_bytes))

    print(f"Lista de tamaños de prototipos y bytes: {lista_tam_bytes}")
    
    
main()