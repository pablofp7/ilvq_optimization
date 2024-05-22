import sys
import os
ruta_directorio_main = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_directorio_main not in sys.path:
    sys.path.append(ruta_directorio_main)

import numpy as np
import pandas as pd     
from prototypes import XuILVQ
import pickle
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline



def read_dataset():
    pd.set_option('future.no_silent_downcasting', True)
    dataset = pd.read_csv(f"../dataset/electricity.csv")
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





df = read_dataset()
#        datalist = [(fila[:-1], fila[-1]) for fila in dataset.values] 
df_list = [(fila[:-1], fila[-1]) for fila in df.values]
df_list = df_list[:50000]


modelo = XuILVQ()
lista_tam_bytes = []
lista_tam_conj = []
lista_df_nbors = []

for i in range(len(df_list)):
    x, y = df_list.pop(0)            
    x = {k: v for k, v in enumerate(x)}    
    modelo.learn_one(x, y)
    protos = list(modelo.buffer.prototypes.values())
    tam_num = len(protos)
    lista_tam_conj.append((i, tam_num))
    if i % 100 == 0:
        print(f"Iteracion {i}")
        proto_to_share = pickle.dumps({"id": id, "protos": [{'x': proto['x'], 'y': proto['y']} for proto in protos]})
        tam_bytes = sys.getsizeof(proto_to_share)
        
        lista_tam_bytes.append((i, tam_num, tam_bytes))
        
        # Creamos un dataframe con la probability mass function del número de vecinos de cada uno de los prototipos y lo guardamos en la lista de dataframes
        
        # Obtener el número de vecinos para cada prototipo
        num_vecinos = [len(proto["neighbors"]) for proto in protos]
        conteo_vecinos = pd.value_counts(num_vecinos, normalize=True)
        # Luego, convertimos el conteo a un DataFrame
        df_pmf_nbors = pd.DataFrame({
            'Numero de Vecinos': conteo_vecinos.index,
            'Probabilidad': conteo_vecinos.values
        })
        
        # Agregar el DataFrame a la lista de DataFrames
        lista_df_nbors.append(df_pmf_nbors)


# print(f"Lista de tamaños de prototipos y bytes: {lista_tam_bytes}")

# val_x, val_y = zip(*lista_tam_conj)

# num_muestras = np.array(val_x)
# tam_protos = np.array(val_y)

# # Crear el modelo de regresión por splines
# spline = UnivariateSpline(num_muestras, tam_protos, s=1)  # Ajusta el parámetro de suavizado `s` según sea necesario

# # Generar puntos para la evaluación del modelo
# muestras_eval = np.linspace(min(num_muestras), max(num_muestras), 200)
# prototipos_pred = spline(muestras_eval)

# plt.figure(figsize=(10, 6))
# plt.plot(num_muestras, tam_protos, 'o', label='Datos reales')
# plt.plot(muestras_eval, prototipos_pred, '-', label='Ajuste por splines')
# plt.title("Regresión por Splines: Tamaño del Conjunto de Prototipos vs. Número de Muestras")
# plt.xlabel("Número de Muestras")
# plt.ylabel("Tamaño del Conjunto de Prototipos")
# plt.legend()
# plt.show()

# Asegúrate de que la lista tenga al menos 9 elementos, de lo contrario ajusta el rango
if len(lista_df_nbors) >= 9:
    indices = np.linspace(0, len(lista_df_nbors) - 1, 9, dtype=int)  # Índices equiespaciados
    selected_dfs = [lista_df_nbors[i] for i in indices]
else:
    selected_dfs = lista_df_nbors  # Si hay menos de 9, toma todos los que hay

# Configurar la figura y los ejes para una malla de subplots de 3x3
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
fig.subplots_adjust(hspace=0.5, wspace=0.5)


# Iterar sobre los DataFrames seleccionados para graficar PMF y CDF
for i, ax in enumerate(axes.flatten()):
    if i < len(selected_dfs):
        df = selected_dfs[i]
        # Asegurarse de que los datos estén ordenados por 'Numero de Vecinos'
        df = df.sort_values(by='Numero de Vecinos').reset_index(drop=True)
        # Calcular la CDF
        df['CDF'] = df['Probabilidad'].cumsum()
        # Corregir cualquier punto donde la CDF supere el valor de 1
        df['CDF'] = df['CDF'].clip(upper=1)
                
        # Graficar la PMF
        ax.bar(df['Numero de Vecinos'], df['Probabilidad'], color='skyblue', label='PMF')
        # Graficar la CDF
        ax.plot(df['Numero de Vecinos'], df['CDF'], color='red', marker='o', linestyle='-', 
                linewidth=2, markersize=5, label='CDF')
        
        # Configuración del subplot
        ax.set_title(f'PMF y CDF Index {indices[i]}')
        ax.set_xlabel('Número de Vecinos/Prototipo')
        ax.set_ylabel('Probabilidad')
        ax.set_xticks(df['Numero de Vecinos'].unique())  # Ajustar los ticks del eje X
        ax.legend()  # Mostrar la leyenda

    else:
        ax.set_visible(False)  # Ocultar ejes si no hay datos para graficar

# Mostrar la figura con los subplots

metricas = []
for i in range(len(lista_df_nbors)):
    df = lista_df_nbors[i]
    df = df.sort_values(by='Numero de Vecinos').reset_index(drop=True)
    # Calcular la CDF
    df['CDF'] = df['Probabilidad'].cumsum()
    # Corregir cualquier punto donde la CDF supere el valor de 1
    df['CDF'] = df['CDF'].clip(upper=1)
    resultado_filtrado = df[df['Numero de Vecinos'] == 1]

    # Verificar si hay resultados antes de intentar acceder a ellos
    if not resultado_filtrado.empty:
        valor_cdf = resultado_filtrado['CDF'].iloc[0]  # Obtiene el primer valor de CDF
        print(f"Resultado: {valor_cdf}")
        metricas.append(valor_cdf)  # Añade el valor numérico a la lista metricas


# Filtrar None y convertir a numpy array para facilitar los cálculos
metricas_filtradas = np.array([m for m in metricas])

# Calcular la media
media = round(np.mean(metricas_filtradas), 4)

# Calcular la desviación estándar
desviacion_tipica = round(np.std(metricas_filtradas), 4)

# Calcular el coeficiente de variación (CV) como la desviación estándar dividida por la media y multiplicado por 100 para obtenerlo en porcentaje
coeficiente_variacion = round((desviacion_tipica / media), 4)

print(f"METRICAS: \n")
print(f"Media : {media}")
print(f"Desviación típica: {desviacion_tipica}")
print(f"Coeficiente de variación: {coeficiente_variacion}")



plt.show()