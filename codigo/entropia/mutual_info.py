import numpy as np
import pandas as pd
from tqdm import tqdm


def read_dataset(n_muestras: int, data_file: str):
    
    dataset = pd.read_csv(data_file, nrows=n_muestras)
    # Se cambia el 'UP' por 1 y el 'DOWN' por 0
    dataset.replace('UP', 1, inplace=True)
    dataset.replace('DOWN', 0, inplace=True)
    return dataset


def entropia(conj_muestras: list):
    
    #Convertir la lista de listas a arrays de numpy
    array_proto = np.array(conj_muestras)
    
    #Transformar el conjunto de arrays de numpy en un array de numpy
    flat_array = array_proto.flatten()
    
    # print(f"flat_array: {flat_array}")
    
    _, value_counts = np.unique(flat_array, return_counts=True)
    prob_values = value_counts / len(flat_array)
    entropy = -np.sum(prob_values * np.log2(prob_values))
    
    return entropy

def info_mutua(conj_muestras: list, n_muestra: list):
    
    # X nueva muestra, Y conjunto de muestras
    conj_sin = np.delete(conj_muestras, n_muestra, axis=0)
    ent_total = entropia(conj_muestras)
    ent_Y = entropia(conj_sin)
    ent_X = entropia(conj_muestras[n_muestra])
    informacion_mutua = ent_X + ent_Y - ent_total
    return informacion_mutua

def reduce_dataset(dataset, conj_muestras: list, conj_inf: list, th: float, th_mayor: bool = True):

    conj_inf_np = np.array(conj_inf)
    posiciones_umbral = np.where(conj_inf_np > th)[0] if th_mayor else np.where(conj_inf_np <= th)[0]
    conjunto_muestras_filtrado = np.delete(conj_muestras, posiciones_umbral, axis=0)
    # print(f"Posiciones que tienen que tener un 1: {posiciones_umbral}")
    dataset['relevancia'] = 1
    dataset.loc[posiciones_umbral, 'relevancia'] = 0
    
    return dataset, conjunto_muestras_filtrado




if __name__ == '__main__':

    try:
        N_MUESTRAS = int(input("Introduce el número de muestras del dataset:\n"))
    except ValueError:
        print("El valor introducido no es un número entero.")
        quit()
    
    N_MUESTRAS = min(N_MUESTRAS, 45312)
    
    #Elegir el tipo de umbral
    option = input("\nQue umbral quieres usar para filtrar el dataset?\n1. Media\n2. Mediana\n")
    
    DATA_FILE = '../basic_transformer/IA/electricity.csv'
    dataset = read_dataset(N_MUESTRAS, DATA_FILE)
    # print(f"DATASET: {dataset}")
    lista_data = dataset.values.tolist()

    
    # N_MUESTRAS = len(lista_data) if N_MUESTRAS > len(lista_data) else N_MUESTRAS
    
    lista_data_np = np.array(lista_data)
    
    lista_inf_mutua = []
    for i in tqdm(range(N_MUESTRAS), desc="Progreso", unit="%"):
        lista_inf_mutua.append(info_mutua(lista_data_np, i))        
        
    
    max_inf = np.max(lista_inf_mutua)
    min_inf = np.min(lista_inf_mutua)
    media_inf = np.mean(lista_inf_mutua)
    mediana_inf = np.median(lista_inf_mutua)
    
    
    umbral = media_inf if option == '1' else mediana_inf 
    
    df_mod, lista_data_filtrada_mayor = reduce_dataset(dataset, lista_data, lista_inf_mutua, umbral, True)
    df_mod_menor, lista_data_filtrada_menor = reduce_dataset(dataset, lista_data, lista_inf_mutua, umbral, False)
    
    #Se guarda el dataset modificado para el transformer con etiquetas con las muestras más relevantes
    data_mod_file = "../basic_transformer/IA/electricity_modificado.csv"
    df_mod.to_csv(data_mod_file, index=False)
    
    
    # print(f"Posiciones umbral (no relevantes): {posiciones_mayor}")
    
    print(f"\nMÁXIMA información mutua: {round(max_inf, 7)}, número de muestra: {lista_inf_mutua.index(max_inf)}")
    print(f"MÍNIMA información mutua: {round(min_inf, 7)}, número de muestra: {lista_inf_mutua.index(min_inf)}")
    print(f"MEDIA información mutua: {round(media_inf, 7)}")
    print(f"MEDIANA información mutua: {round(mediana_inf, 7)}")

    print(f"\nTamaño dataset antes de filtrar: {len(lista_data)},"
        f" tamaño después de filtrar valores MAYORES a la métrica escogida: {len(lista_data_filtrada_mayor)},"
        f" tamaño después de filtrar valores MENORES a la métrica escogida: {len(lista_data_filtrada_menor)}")
    
    print(f"Entropía del dataset SIN FILTRAR: {entropia(lista_data)}")
    print(f"Entropía del dataset FILTRADO MAYOR: {entropia(lista_data_filtrada_mayor)}")
    print(f"Entropía del dataset FILTRADO MENOR: {entropia(lista_data_filtrada_menor)}\n")
    