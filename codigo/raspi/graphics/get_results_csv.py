import os
import re
import pandas as pd
import ast

# Definimos los conjuntos de datos válidos
VALID_DATASETS = ["elec", "phis", "elec2", "lgr", "nrr", "lar", "lrr", "ngcr", "nsch"]

def get_results(test: str = "test1"):
    # Directorio donde se encuentran los archivos CSV
    directorio_resultados = os.path.expanduser(f'~/ilvq_optimization/codigo/raspi/{test}_resultados/')

    # Diccionario para almacenar los datos de cada conjunto
    datasets = {dataset: [] for dataset in VALID_DATASETS}

    # Recorrer todos los archivos en el directorio
    for nombre_archivo in os.listdir(directorio_resultados):
        if nombre_archivo.endswith('.csv'):
            # Extraer el nombre del conjunto de datos y los parámetros
            match = re.match(r'result_(' + '|'.join(VALID_DATASETS) + r')_s(\d+)_T([\d.]+)_it(\d+).csv', nombre_archivo)
            if match:
                dataset_name, s, T, it = match.groups()

                # Verificar si el conjunto de datos es válido
                if dataset_name not in datasets:
                    continue  # Si no es válido, saltar este archivo

                # Ruta completa del archivo
                ruta_archivo = os.path.join(directorio_resultados, nombre_archivo)

                # Leer el archivo CSV
                df = pd.read_csv(ruta_archivo)

                # Procesar campos especiales
                if 'Tamaño de lotes recibidos' in df.columns:
                    df['Tamaño de lotes recibidos'] = df['Tamaño de lotes recibidos'].apply(
                        lambda x: sum(tupla[1] for tupla in ast.literal_eval(x))
                    )
                # Calcular promedios y totales
                precision_prom = df['Precision'].mean()
                recall_prom = df['Recall'].mean()
                f1_prom = df['F1'].mean()
                capacidad_promedio = df['Capacidad de ejecución'].mean()
                tamaño_promedio_lotes = df['Tamaño de lotes recibidos'].mean() if 'Tamaño de lotes recibidos' in df.columns else 0
                promedio_prototipos_entrenados = df['Prototipos entrenados'].mean()
                promedio_prototipos_compartidos = df['Prototipos compartidos'].mean()
                promedio_mensajes_enviados = df['Veces compartido'].mean() * int(s)
                tiempo_total_prom = df['Tiempo total'].mean()
                ancho_banda_prom = (105 * promedio_prototipos_compartidos) / tiempo_total_prom

                # Crear el diccionario de resultados
                resultado = {
                    's': int(s),
                    'T': float(T),
                    'it': int(it),
                    'Precisión': precision_prom,
                    'Recall': recall_prom,
                    'F1': f1_prom,
                    'Mensajes_Enviados': promedio_mensajes_enviados,
                    'Prototipos_Entrenados': promedio_prototipos_entrenados,
                    'Prototipos_Compartidos': promedio_prototipos_compartidos,
                    'Capacidad_Ejecucion': capacidad_promedio,
                    'Tamaño_Promedio_Lotes': tamaño_promedio_lotes,
                    'Ancho_Banda': ancho_banda_prom
                }

                # Añadir el resultado a la lista correspondiente
                datasets[dataset_name].append(resultado)

    # Convertir las listas a DataFrames
    return {k: pd.DataFrame(v) for k, v in datasets.items()}

def get_results_4(test, filters, metric):
    # Directorio donde se encuentran los archivos CSV
    directorio_resultados = os.path.expanduser(f'~/ilvq_optimization/codigo/raspi/{test}_resultados/')

    # Diccionario para almacenar los datos de cada conjunto
    datasets = {dataset: [] for dataset in VALID_DATASETS}

    # Recorrer todos los archivos en el directorio
    for nombre_archivo in os.listdir(directorio_resultados):
        if nombre_archivo.endswith('.csv'):
            # Extraer el nombre del conjunto de datos y los parámetros
            match = re.match(r'result_(' + '|'.join(VALID_DATASETS) + r')_s(\d+)_T([\d.]+)_limit(\d+)_range(\d+(\.\d+)?)-(\d+(\.\d+)?)_it(\d+).csv', nombre_archivo)
            if match:
                dataset_name, s, T, limit, range_start, _, range_end, _, it = match.groups()

                # Verificar si el conjunto de datos es válido
                if dataset_name not in datasets:
                    continue  # Si no es válido, saltar este archivo

                # Ruta completa del archivo
                ruta_archivo = os.path.join(directorio_resultados, nombre_archivo)

                # Leer el archivo CSV
                df = pd.read_csv(ruta_archivo)

                # Procesar campos especiales
                if 'Tamaño de lotes recibidos' in df.columns:
                    df['Tamaño de lotes recibidos'] = df['Tamaño de lotes recibidos'].apply(
                        lambda x: sum(int(tupla.split(', ')[1].strip(')')) for tupla in ast.literal_eval(x))
                    )

                # Calcular promedios y totales
                precision_prom = df['Precision'].mean()
                recall_prom = df['Recall'].mean()
                f1_prom = df['F1'].mean()
                capacidad_promedio = df['Capacidad de ejecución'].mean()
                tamaño_promedio_lotes = df['Tamaño de lotes recibidos'].mean() if 'Tamaño de lotes recibidos' in df.columns else 0
                promedio_prototipos_entrenados = df['Prototipos entrenados'].mean()
                promedio_prototipos_compartidos = df['Prototipos compartidos'].mean()
                promedio_mensajes_enviados = df['Veces compartido'].mean() * int(s)
                promedio_clust_time = df['Tiempo clustering'].mean() if 'Tiempo clustering' in df.columns else 0
                promedio_clust_runs = df['Ejecuciones de clustering'].mean() if 'Ejecuciones de clustering' in df.columns else 0
                tiempo_total_prom = df['Tiempo total'].mean()
                ancho_banda_prom = (105 * promedio_prototipos_compartidos) / tiempo_total_prom

                # Crear el diccionario de resultados
                resultado = {
                    's': int(s),
                    'T': float(T),
                    'limit': int(limit),
                    'range_start': float(range_start),
                    'range_end': float(range_end),
                    'it': int(it),
                    'Precision': precision_prom,
                    'Recall': recall_prom,
                    'F1': f1_prom,
                    'Mensajes_Enviados': promedio_mensajes_enviados,
                    'Prototipos_Entrenados': promedio_prototipos_entrenados,
                    'Prototipos_Compartidos': promedio_prototipos_compartidos,
                    'Capacidad_Ejecucion': capacidad_promedio,
                    'Tamaño_Promedio_Lotes': tamaño_promedio_lotes,
                    'Clust_Time': promedio_clust_time,
                    'Clust_Runs': promedio_clust_runs,
                    'Ancho_Banda': ancho_banda_prom
                }

                # Añadir el resultado a la lista correspondiente
                datasets[dataset_name].append(resultado)

    # Convertir las listas a DataFrames
    datasets = {k: pd.DataFrame(v) for k, v in datasets.items()}

    # Aplicar filtros si se proporcionan
    for param, value in filters.items():
        for dataset_name, df in datasets.items():
            if param in df.columns:
                datasets[dataset_name] = df[df[param] == value]

    # Ordenar por la métrica especificada
    for dataset_name, df in datasets.items():
        if not df.empty:
            if metric == 'Clust':
                df = df.sort_values(by=['Clust_Time', 'Clust_Runs'], ascending=[False, False])
            else:
                df = df.sort_values(by=[metric], ascending=False)
            datasets[dataset_name] = df

    return datasets