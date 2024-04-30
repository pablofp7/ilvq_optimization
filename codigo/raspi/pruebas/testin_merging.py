import sys
import os
ruta_directorio_main = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_directorio_main not in sys.path:
    sys.path.append(ruta_directorio_main)

import numpy as np
import pandas as pd     
from prototypes_mod import XuILVQ
from sklearn.cluster import DBSCAN




def read_dataset():
    dataset = pd.read_csv(f"../dataset/electricity.csv")
    # Se cambia el 'UP' por 1 y el 'DOWN' por 0
    dataset.replace('UP', 1, inplace=True)
    dataset.replace('DOWN', 0, inplace=True) 

    dataset.replace('True', 1, inplace=True)
    dataset.replace('False', 0, inplace=True) 


    return dataset


def dbscan_prototypes(prototypes, max_prototypes=100, target_range=(80, 90), eps_initial=0.3):
    current_count = prototypes.shape[0]
    eps = eps_initial

    # Realizar ajustes hasta que el número de prototipos esté dentro del rango deseado
    while not (target_range[0] <= current_count <= target_range[1]):
        # Aplicar DBSCAN con los parámetros actuales
        dbscan = DBSCAN(eps=eps, min_samples=1)
        labels = dbscan.fit_predict(prototypes)

        # Nuevos prototipos como centroides de los clusters formados
        unique_labels = np.unique(labels)
        new_prototypes = np.array([prototypes[labels == label].mean(axis=0) for label in unique_labels if label != -1])

        current_count = new_prototypes.shape[0]
        
        # Ajustar eps según si hay más prototipos de lo deseado o menos
        if current_count > target_range[1]:
            eps *= 1.1  # Hacer los clusters más grandes (menos clusters)
        elif current_count < target_range[0]:
            eps /= 1.1  # Hacer los clusters más pequeños (más clusters)

    return new_prototypes

    




df = read_dataset()
df_list = [(fila[:-1], fila[-1]) for fila in df.values]
df_list = df_list[:50]

LIMIT = 100
modelo = XuILVQ()
matriz_conf = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
predictions = []
true_labels = []

for i in range(len(df_list)):
    
    x, y = df_list.pop(0)            
    x = {k: v for k, v in enumerate(x)}    

    protos = list(modelo.buffer.prototypes.values())
    tam = len(protos)
    
    if tam > LIMIT:
        fusion(modelo)

    # Measure prediction time
    prediction = modelo.predict_one(x)

    if isinstance(prediction, dict):
        if 1.0 in prediction:
            prediction = prediction[1.0]
        else:
            prediction = 0.0

    if prediction is None:
        prediction = 0.0    
            
    predictions.append(prediction)
    true_labels.append(y)

    modelo.learn_one(x, y)
    
prototipos_orig = modelo.buffer.prototypes
print(f"Tipo de prototipos: {type(prototipos_orig)}")
print(f"Numero de prototipos: {len(prototipos_orig)}")
print(f"Formato de los prototipos: {prototipos_orig}")



