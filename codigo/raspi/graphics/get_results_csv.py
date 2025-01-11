import os
import re
import pandas as pd
import ast  # Para convertir strings literales a objetos Python


def get_results(test: str = "test1"):
    # Directorio donde se encuentran los archivos CSV
    directorio_resultados = os.path.expanduser(f'~/ilvq_optimization/codigo/raspi/{test}_resultados/')

    # Listas para almacenar los datos de cada conjunto
    datos_elec = []
    datos_elec2 = []
    datos_phis = []

    # Recorrer todos los archivos en el directorio
    for nombre_archivo in os.listdir(directorio_resultados):
        if nombre_archivo.endswith('.csv'):
            # Determinar el tipo de conjunto de datos
            es_elec2 = "elec2" in nombre_archivo
            es_elec = "elec" in nombre_archivo if not es_elec2 else False
            es_phis = "phis" in nombre_archivo

            # Ruta completa del archivo
            ruta_archivo = os.path.join(directorio_resultados, nombre_archivo)

            # Extraer parámetros del nombre del archivo
            match = re.match(r'result_(elec|phis|elec2)_s(\d+)_T([\d.]+)_it(\d+).csv', nombre_archivo)
            if match:
                _, s, T, it = match.groups()

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
                if es_elec:
                    datos_elec.append(resultado)
                elif es_phis:
                    datos_phis.append(resultado)
                elif es_elec2:
                    datos_elec2.append(resultado)

    # Convertir las listas a DataFrames
    return pd.DataFrame(datos_elec), pd.DataFrame(datos_phis), pd.DataFrame(datos_elec2)


def get_results_4(test, filters, metric):
    # Directorio donde se encuentran los archivos CSV
    directorio_resultados = os.path.expanduser(f'~/ilvq_optimization/codigo/raspi/{test}_resultados/')

    # Listas para almacenar los datos de cada conjunto
    datos_elec = []
    datos_elec2 = []
    datos_phis = []

    # Recorrer todos los archivos en el directorio
    for nombre_archivo in os.listdir(directorio_resultados):
        if nombre_archivo.endswith('.csv'):
            # Determinar el tipo de conjunto de datos
            es_elec2 = "elec2" in nombre_archivo
            es_elec = "elec" in nombre_archivo if not es_elec2 else False
            es_phis = "phis" in nombre_archivo

            # Ruta completa del archivo
            ruta_archivo = os.path.join(directorio_resultados, nombre_archivo)

            # Extraer parámetros del nombre del archivo
            match = re.match(r'result_(elec|phis|elec2)_s(\d+)_T([\d.]+)_limit(\d+)_range(\d+(\.\d+)?)-(\d+(\.\d+)?)_it(\d+).csv', nombre_archivo)
            if match:
                _, s, T, limit, range_start, _, range_end, _, it = match.groups()

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
                if es_elec:
                    datos_elec.append(resultado)
                elif es_phis:
                    datos_phis.append(resultado)
                elif es_elec2:
                    datos_elec2.append(resultado)

    # Convertir las listas a DataFrames
    datos_elec_df = pd.DataFrame(datos_elec)
    datos_phis_df = pd.DataFrame(datos_phis)
    datos_elec2_df = pd.DataFrame(datos_elec2)

    # Aplicar filtros si se proporcionan
    for param, value in filters.items():
        if param in datos_elec_df.columns:
            datos_elec_df = datos_elec_df[datos_elec_df[param] == value]
            datos_phis_df = datos_phis_df[datos_phis_df[param] == value]
            datos_elec2_df = datos_elec2_df[datos_elec2_df[param] == value]

    # Ordenar por la métrica especificada
    if metric == 'Clust':
        # Ordenar por Clust_Time y Clust_Runs
        if not datos_elec_df.empty:
            datos_elec_df = datos_elec_df.sort_values(by=['Clust_Time', 'Clust_Runs'], ascending=[False, False])
        if not datos_phis_df.empty:
            datos_phis_df = datos_phis_df.sort_values(by=['Clust_Time', 'Clust_Runs'], ascending=[False, False])
        if not datos_elec2_df.empty:
            datos_elec2_df = datos_elec2_df.sort_values(by=['Clust_Time', 'Clust_Runs'], ascending=[False, False])
    else:
        # Ordenar por la métrica especificada
        if not datos_elec_df.empty:
            datos_elec_df = datos_elec_df.sort_values(by=[metric], ascending=False)
        if not datos_phis_df.empty:
            datos_phis_df = datos_phis_df.sort_values(by=[metric], ascending=False)
        if not datos_elec2_df.empty:
            datos_elec2_df = datos_elec2_df.sort_values(by=[metric], ascending=False)

    return datos_elec_df, datos_phis_df, datos_elec2_df