import sys
import os
import pandas as pd

# Configurando la ruta del directorio principal para asegurar el acceso a los módulos necesarios
ruta_directorio_main = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_directorio_main not in sys.path:
    sys.path.append(ruta_directorio_main)

# Importando módulos adicionales
import numpy as np
from prototypes import XuILVQ  # Asumo que este módulo es parte de tus archivos personalizados
import pickle
from matplotlib import pyplot as plt

def read_dataset():
    try:
        # Cargando el dataset HTTP
        http = pd.read_csv("../dataset/kdd99_http.csv", sep=",")
    except Exception as e:
        print(f"Error loading HTTP dataset: {e}")
        http = None

    try:
        # Cargando el dataset Movie Lens
        movie = pd.read_csv("../dataset/ml_100k.csv", sep='\t')
    except Exception as e:
        print(f"Error loading Movie dataset: {e}")
        movie = None

    return http, movie

# Cargando los datasets
http, movie = read_dataset()

# Verificando y mostrando información si la carga fue exitosa
if http is not None and movie is not None:
    print("Datasets loaded successfully.")
    print("HTTP Dataset Info:")
    print(http.head())  # Correcto uso de head()
    print("Movie Dataset Info:")
    print(movie.head())  # Correcto uso de head()
else:
    print("Failed to load one or both datasets.")
