# Este programa se va a entrenar el modelo XuILVQ como en los otros
# programas, guardar el tamaño del conjunto de prototipos en número de prototipos
# convertirlo a pickle como se hace en el otro archivo y ver cuantos bytes ocupa tambien

import numpy as np
import pandas as pd
import pickle
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from prototypes import XuILVQ




# Cargar los datos
def read_dataset():
    