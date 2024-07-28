import sys
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

# Directorio principal
ruta_directorio_main = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_directorio_main not in sys.path:
    sys.path.append(ruta_directorio_main)
from prototypes_mod import XuILVQ

# Leer y preparar el dataset
def read_dataset():
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
df_list_original = [(fila[:-1], fila[-1]) for fila in df.values][:50000]

# Configuración de las pruebas
lista_pset_size = [i * 25 for i in range(1, 41)]
target_size = (75, 85)
# target_percentage = 80

# Estructuras para guardar resultados
tiempos_entrenamiento = {size: [] for size in lista_pset_size}
tiempos_prediccion = {size:  [] for size in lista_pset_size}
tamaños_prototipos = {size: [] for size in lista_pset_size}
f1_scores = {size: [] for size in lista_pset_size}

def media_movil_adaptativa(datos, max_ventana=1000):
    n = len(datos)
    resultado = np.zeros(n)  # Array para guardar el resultado
    for i in range(n):
        ventana = min(i+1, max_ventana)
        resultado[i] = np.mean(datos[max(0, i-ventana+1):i+1])
    return resultado

# Pruebas para cada pset_size
for pset_size in lista_pset_size:
    # modelo = XuILVQ(target_percentage=target_percentage, max_pset_size=pset_size, merge_mode="kmeans")
    modelo = XuILVQ(target_size=target_size, max_pset_size=pset_size)
    print(f"INICIO: pset_size: {pset_size}, objetivo: [{target_size[0] * pset_size / 100}, {target_size[1] * pset_size / 100}")
    df_list = df_list_original.copy()
    matriz_conf = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    for i, (x, y) in enumerate(df_list):
        #Hacer print cada 500 iteraciones
        if i % 500 == 0:
            print(f"Iteración {i} - PSet Size {pset_size}")
            print(f"Matriz de confusión: {matriz_conf}")
            TP, TN, FP, FN = matriz_conf.values()
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = round(2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0, 5)
            print(f"F1 score: {f1_score}")

        # Preparación de datos
        x = {k: v for k, v in enumerate(x)}

        # Predicción
        start_pred = time.perf_counter_ns()
        prediccion = modelo.predict_one(x)
        end_pred = time.perf_counter_ns()
        tiempos_prediccion[pset_size].append((end_pred - start_pred) / 1e9)

        # Clasificar la predicción
        if isinstance(prediccion, dict):
            if 1.0 in prediccion:
                prediccion = prediccion[1.0]
            else:
                prediccion = 0.0      

        # Actualizar matriz de confusión
        if prediccion == 0 and y == 0:
            matriz_conf["TN"] += 1
        elif prediccion == 1 and y == 1:
            matriz_conf["TP"] += 1
        elif prediccion == 1 and y == 0:
            matriz_conf["FP"] += 1
        elif prediccion == 0 and y == 1:
            matriz_conf["FN"] += 1

        # Entrenamiento
        start_train = time.perf_counter_ns()
        modelo.learn_one(x, y)
        end_train = time.perf_counter_ns()
        tiempos_entrenamiento[pset_size].append((end_train - start_train) / 1e9)
        
        tamaños_prototipos[pset_size].append(len(list(modelo.buffer.prototypes.values())))

    # Calcular F1 score
    TP, TN, FP, FN = matriz_conf.values()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f1_scores[pset_size] = f1_score  # Guarda el F1 score final para cada pset_size



# Plots de tiempos medios y coeficientes de variación
metricas = [tiempos_entrenamiento, tiempos_prediccion]
nombres_metricas = ['Tiempo de entrenamiento', 'Tiempo de predicción']
for metric, name in zip(metricas, nombres_metricas):
    medias = [np.mean(metric[size]) for size in lista_pset_size]
    cvs = [np.std(metric[size]) / np.mean(metric[size]) if np.mean(metric[size]) > 0 else 0 for size in lista_pset_size]
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(lista_pset_size, medias, '-o', label='Media')
    plt.title(f'Media de {name}')
    plt.ylabel('Tiempo medio (s)')
    plt.subplot(2, 1, 2)
    plt.plot(lista_pset_size, cvs, '-o', label='Coef. de Variación')
    plt.title(f'Coeficiente de Variación para {name}')
    plt.xlabel('pset size')
    plt.ylabel('Coeficiente de Variación')
    plt.tight_layout()
    plt.savefig(f'{name.replace(" ", "_").lower()}_metrics.png')
    plt.close()

# Suma de tiempos
tiempos_sumados = [np.mean(np.array(tiempos_entrenamiento[size]) + np.array(tiempos_prediccion[size])) for size in lista_pset_size]
cv_sumados = [np.std(np.array(tiempos_entrenamiento[size]) + np.array(tiempos_prediccion[size])) / np.mean(np.array(tiempos_entrenamiento[size]) + np.array(tiempos_prediccion[size])) for size in lista_pset_size]
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.plot(lista_pset_size, tiempos_sumados, '-o', label='Tiempo Total Medio')
plt.title('Tiempo Total Medio')
plt.ylabel('Tiempo (s)')
plt.subplot(2, 1, 2)
plt.plot(lista_pset_size, cv_sumados, '-o', label='Coef. de Variación Total')
plt.title('Coeficiente de Variación Total')
plt.xlabel('pset size')
plt.ylabel('Coeficiente de Variación')
plt.tight_layout()
plt.savefig('tiempos_sumados_metrics.png')
plt.close()

# Evolución del tamaño de los prototipos
num_plots = len(lista_pset_size) // 2 + len(lista_pset_size) % 2
for i in range(num_plots):
    plt.figure(figsize=(10, 10))
    for j in range(2):
        pset_index = i * 2 + j
        if pset_index < len(lista_pset_size):
            pset_size = lista_pset_size[pset_index]
            plt.subplot(2, 1, j + 1)
            plt.plot(tamaños_prototipos[pset_size], '-o')
            media = np.mean(tamaños_prototipos[pset_size])
            cv = np.std(tamaños_prototipos[pset_size]) / media if media > 0 else 0
            plt.title(f'pset size {pset_size} - Media: {media:.2f}, CV: {cv:.2f}')
            plt.xlabel('Iteración')
            plt.ylabel('Tamaño de Prototipos')
    plt.tight_layout()
    plt.savefig(f'prototipos_evolucion_plot_{i + 1}.png')
    plt.close()
    
    
    
    
# # Generar gráficos
# for pset_size in lista_pset_size:
#     plt.figure(figsize=(10, 15))

#     for index, (title, data) in enumerate([
#         ('Tiempo de entrenamiento', tiempos_entrenamiento[pset_size]),
#         ('Tiempo de predicción', tiempos_prediccion[pset_size]),
#         ('Suma de tiempos', np.array(tiempos_entrenamiento[pset_size]) + np.array(tiempos_prediccion[pset_size]))
#     ]):
#         plt.subplot(3, 1, index+1)
#         media = np.mean(data)
#         desviacion = np.std(data)
#         coef_variacion = desviacion / media if media else 0

#         plt.plot(media_movil_adaptativa(data), label='Media Móvil Adaptativa')
#         plt.title(f"{title} (Media móvil adaptativa) - PSet Size {pset_size}\n"
#                   f"Media: {media:.4f}, Desv. Típica: {desviacion:.4f}, Coef. de Variación: {coef_variacion:.4f}")
#         plt.xlabel('Iteraciones')
#         plt.ylabel('Tiempo (s)')

#     plt.tight_layout()
#     plt.savefig(f'graficas_pset_{pset_size}.png')
#     plt.close()
    plt.close()