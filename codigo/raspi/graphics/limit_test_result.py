import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import re
import os
import argparse

def get_results(test, filters, metric):
    directorio_resultados = os.path.expanduser(f'~/ilvq_optimization/codigo/raspi/{test}_resultados/')
    datos_elec = []
    datos_elec2 = []
    datos_phis = []

    for nombre_archivo in os.listdir(directorio_resultados):
        if nombre_archivo.endswith('.txt'):
            es_elec2 = "elec2" in nombre_archivo
            es_elec = "elec" in nombre_archivo if not es_elec2 else False
            es_phis = "phis" in nombre_archivo

            ruta_archivo = os.path.join(directorio_resultados, nombre_archivo)

            match = re.match(r'result_(elec|phis|elec2)_s(\d+)_T([\d.]+)_limit(\d+)_range(\d+(\.\d+)?)-(\d+(\.\d+)?)_it(\d+).txt', nombre_archivo)
            
            if match:
                _, s, T, limit, range_start, _, range_end, _, it = match.groups()
                
                with open(ruta_archivo, 'r') as file:
                    contenido = file.readlines()
                    
                    precision_total, recall_total, f1_total = 0, 0, 0
                    prototipos_entrenados, prototipos_compartidos = 0, 0
                    mensajes_enviados = 0
                    cuenta_nodos = 0
                    capacidad_ejecucion_total, tamaño_lotes_total, cuenta_lotes = 0, 0, 0
                    clust_time_total, clust_runs_total = 0, 0
                    tiempo_total = 0

                    for linea in contenido:
                        precision_match = re.search(r'Precision: (\d.\d+)', linea)
                        if precision_match:
                            precision_total += float(precision_match.group(1))
                            cuenta_nodos += 1
                        recall_match = re.search(r'Recall: (\d.\d+)', linea)
                        if recall_match:
                            recall_total += float(recall_match.group(1))
                        f1_match = re.search(r'F1: (\d.\d+)', linea)
                        if f1_match:
                            f1_total += float(f1_match.group(1))
                        match_entrenados = re.search(r'Se ha entrenado con (\d+) prototipos.', linea)
                        if match_entrenados:
                            prototipos_entrenados += int(match_entrenados.group(1))
                        
                        match_compartidos = re.search(r'Ha compartido (\d+) prototipos', linea)
                        if match_compartidos:
                            prototipos_compartidos += int(match_compartidos.group(1))
                        
                        veces_compartido_match = re.search(r'Ha compartido (\d+) veces.', linea)
                        if veces_compartido_match:
                            veces_compartido = int(veces_compartido_match.group(1))
                            mensajes_nodo = veces_compartido * int(s)
                            mensajes_enviados += mensajes_nodo
                        
                        capacidad_match = re.search(r'Capacidad de ejecución: (\d+.\d+)', linea)
                        if capacidad_match:
                            capacidad_ejecucion_total += float(capacidad_match.group(1))
                                    
                        lotes_match = re.search(r'ID, Tamaño de lotes recibidos: \[(.*?)\]', linea)
                        if lotes_match:
                            lotes = lotes_match.group(1).split('), (')
                            for lote in lotes:
                                partes = lote.strip('()').split(', ')
                                if len(partes) == 2:
                                    _, tamaño = partes
                                    tamaño_lotes_total += int(tamaño)
                                    cuenta_lotes += 1
                                    
                        clust_time_match = re.search(r'Tiempo invertido en clustering: (\d+.\d+)', linea)
                        if clust_time_match:
                            clust_time_total += float(clust_time_match.group(1))

                        clust_runs_match = re.search(r'Número de ejecuciones de clustering: (\d+)', linea)
                        if clust_runs_match:
                            clust_runs_total += int(clust_runs_match.group(1))
                            
                        match_tiempo = re.search(r"Tiempo total: ([\d\.]+)", linea)
                        if match_tiempo:
                            tiempo_total = float(match_tiempo.group(1))

                                    
                    # Calculamos los promedios después de leer todo el archivo
                    precision_prom = precision_total / cuenta_nodos if cuenta_nodos else 0
                    recall_prom = recall_total / cuenta_nodos if cuenta_nodos else 0
                    f1_prom = f1_total / cuenta_nodos if cuenta_nodos else 0
                    capacidad_promedio = capacidad_ejecucion_total / cuenta_nodos if cuenta_nodos else 0
                    tamaño_promedio_lotes = tamaño_lotes_total / cuenta_lotes if cuenta_lotes else 0

                    promedio_prototipos_entrenados = prototipos_entrenados / cuenta_nodos if cuenta_nodos else 0
                    promedio_prototipos_compartidos = prototipos_compartidos / cuenta_nodos if cuenta_nodos else 0
                    promedio_mensajes_enviados = mensajes_enviados / cuenta_nodos if cuenta_nodos else 0
                    promedio_clust_time = clust_time_total / cuenta_nodos if cuenta_nodos else 0
                    promedio_clust_runs = clust_runs_total / cuenta_nodos if cuenta_nodos else 0
                    
                    tiempo_total_prom = tiempo_total / cuenta_nodos
                    ancho_banda_prom = (105 * promedio_prototipos_compartidos) / tiempo_total_prom



                    resultado = {
                        's': int(s), 'T': float(T), 'limit': int(limit), 
                        'range_start': float(range_start), 'range_end': float(range_end), 
                        'it': int(it), 'Precision': precision_prom, 'Recall': recall_prom, 
                        'F1': f1_prom, 'Mensajes_Enviados': promedio_mensajes_enviados, 
                        'Prototipos_Entrenados': promedio_prototipos_entrenados, 
                        'Prototipos_Compartidos': promedio_prototipos_compartidos, 
                        'Capacidad_Ejecucion': capacidad_promedio, 
                        'Tamaño_Promedio_Lotes': tamaño_promedio_lotes,
                        'Clust_Time': promedio_clust_time,
                        'Clust_Runs': promedio_clust_runs,
                        'Ancho_Banda': ancho_banda_prom
                    }
                    
                    if es_elec:
                        datos_elec.append(resultado)
                    if es_phis:
                        datos_phis.append(resultado)
                    if es_elec2:
                        datos_elec2.append(resultado)
                        
    datos_elec_df = pd.DataFrame(datos_elec)
    datos_phis_df = pd.DataFrame(datos_phis)
    datos_elec2_df = pd.DataFrame(datos_elec2)

    # Verificar si los DataFrames no están vacíos antes de intentar agruparlos
    if not datos_elec_df.empty:
        datos_elec_df = datos_elec_df.groupby(['s', 'T', 'limit', 'range_start', 'range_end']).mean().reset_index()
        datos_elec_df = datos_elec_df.drop('it', axis=1)

    # if not datos_phis_df.empty:
    #     datos_phis_df = datos_phis_df.groupby(['s', 'T', 'limit', 'range_start', 'range_end']).mean().reset_index()
    #     datos_phis_df = datos_phis_df.drop('it', axis=1)

    # if not datos_elec2_df.empty:
    #     datos_elec2_df = datos_elec2_df.groupby(['s', 'T', 'limit', 'range_start', 'range_end']).mean().reset_index()
    #     datos_elec2_df = datos_elec2_df.drop('it', axis=1)

    # Apply filters if provided
    for param, value in filters.items():
        if param in datos_elec_df.columns:
            datos_elec_df = datos_elec_df[datos_elec_df[param] == value]
            # datos_phis_df = datos_phis_df[datos_phis_df[param] == value]
            # datos_elec2_df = datos_elec2_df[datos_elec2_df[param] == value]
    
    # Sorting by specified metric in descending order after filtering
    if metric == 'Clust':
        # For the special case where the metric is 'Clust', sort by both Clust_Time and Clust_Runs
        if not datos_elec_df.empty:
            datos_elec_df = datos_elec_df.sort_values(by=['Clust_Time', 'Clust_Runs'], ascending=[False, False])
        # if not datos_phis_df.empty:
        #     datos_phis_df = datos_phis_df.sort_values(by=['Clust_Time', 'Clust_Runs'], ascending=[False, False])
        # if not datos_elec2_df.empty:
        #     datos_elec2_df = datos_elec2_df.sort_values(by=['Clust_Time', 'Clust_Runs'], ascending=[False, False])
    else:
        if not datos_elec_df.empty:
            datos_elec_df = datos_elec_df.sort_values(by=[metric], ascending=False)
        # if not datos_phis_df.empty:
        #     datos_phis_df = datos_phis_df.sort_values(by=[metric], ascending=False)
        # if not datos_elec2_df.empty:
        #     datos_elec2_df = datos_elec2_df.sort_values(by=[metric], ascending=False)

    return datos_elec_df #, datos_phis_df, datos_elec2_df
                
                
def plot_results_all(df, titulo_dataset, ax, metrica_seleccionada, y_min, y_max):
    
    not_gui = True

    if df.empty:
        ax.set_xlabel('T')
        ax.set_ylabel(metrica_seleccionada)
        ax.set_title(f'{titulo_dataset} - {metrica_seleccionada}=f(T,s). Sin datos disponibles.')
        return
    promedios = df.groupby(['s', 'T']).mean().reset_index()
    

    for key, grp in promedios.groupby('s'):
        etiqueta = f's = {key}'
        grp.plot(ax=ax, kind='line', x='T', y=metrica_seleccionada, label=etiqueta, marker='o')

    if metrica_seleccionada not in df.columns:
        print(f"Error: La métrica {metrica_seleccionada} no se encuentra en los datos proporcionados.")
        return

    if not_gui:
        resultados_imprimir = pd.DataFrame()

        for key, grp in promedios.groupby('s'):
            indice_s = key[0] if isinstance(key, tuple) else key

            serie_actual = grp.set_index('T')[metrica_seleccionada].rename(indice_s)

            if resultados_imprimir.empty:
                resultados_imprimir = serie_actual.to_frame()
            else:
                resultados_imprimir = resultados_imprimir.join(serie_actual, how='outer')

        # Aseguramos que las columnas sean enteros (esto es necesario si 'key' ya era un entero y no una tupla).
        resultados_imprimir.columns = [int(col) for col in resultados_imprimir.columns]

        # Agregar 'T/s' en la esquina superior izquierda
        resultados_imprimir.index.name = 'T/s'

        # Al imprimir, vamos a suprimir el índice para limpiar la salida.
        print(f"\nResultados para {titulo_dataset} - Métrica: {metrica_seleccionada}:\n")
        print(resultados_imprimir.to_string())
        print("\n")

        # return

    text_metrica_seleccionada = ""
    if "totipo" in metrica_seleccionada:
        text_metrica_seleccionada = "Number of Trained Prototypes"
    elif "ncho" in metrica_seleccionada:
        text_metrica_seleccionada = "Bandwidth"
    else:
        text_metrica_seleccionada = metrica_seleccionada

    ax.legend(loc='best')
    ax.set_xlabel('T')
    ax.set_ylabel(text_metrica_seleccionada)
    tam_fuente = 11
    ax.set_title(f"{titulo_dataset} - {text_metrica_seleccionada}=f(T,s). As a function of T, for different values of s.", fontsize=tam_fuente)            
                   
def main():
    # List of full metric names
    metrics = [
        'Prototipos_Entrenados',
        'Prototipos_Compartidos',
        'Mensajes_Enviados',
        'Capacidad_Ejecucion',
        'Tamaño_Promedio_Lotes',
        'Precision',
        'Recall',
        'F1',
        'Clust_Time',
        'Clust_Runs',
        'Ancho_Banda'
    ]

    parser = argparse.ArgumentParser(description='Filter results based on parameters and output specific metrics along with parameter data.')
    parser.add_argument('-s', '--s', type=int, help='Filter by s value')
    parser.add_argument('-T', '--T', type=float, help='Filter by T value')
    parser.add_argument('-l', '--limit', type=int, help='Filter by limit value')
    parser.add_argument('-rs', '--range_start', type=float, help='Filter by range start value')
    parser.add_argument('-re', '--range_end', type=float, help='Filter by range end value')
    parser.add_argument('-m', '--metric', required=True, help='Specify a metric to output (e.g., f1, prec, rec). This is mandatory.')

    args = parser.parse_args()

    # Function to find the full metric name from the inserted abbreviation
    def find_metric_name(abbreviation, metric_list):
        if abbreviation.lower() == 'clust':
            return 'Clust'
        matches = [metric for metric in metric_list if abbreviation.lower() in metric.lower()]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            print(f"Error: The abbreviation '{abbreviation}' is ambiguous. It matches multiple metrics: {matches}")
            return None
        else:
            print(f"Error: No matching metric found for abbreviation '{abbreviation}'.")
            return None

    full_metric_name = find_metric_name(args.metric, metrics)
    if full_metric_name is None:
        return  # Exit if no valid metric or too many matches are found

    filters = {k: v for k, v in vars(args).items() if v is not None and k != 'metric'}
    print(f"Filters : {filters}")

    test = "test" + input("Test? (4 o 5):\n")
    if full_metric_name == 'Clust':
        if "4" in test:
            print(f"Clustering metrics only available on Test 5.\n")
            return
        columns_to_display = ['s', 'T', 'limit', 'range_start', 'range_end', 'Clust_Time', 'Clust_Runs']

    else:
        columns_to_display = ['s', 'T', 'limit', 'range_start', 'range_end', full_metric_name]
    
    
    # elec_res, phis_res, elec2_res = get_results(test, filters, full_metric_name)  # Include metric for sorting
    elec_res = get_results(test, filters, full_metric_name)  # Include metric for sorting


    # Printing without the DataFrame index
    print(f"Filtered Results for the specified metric '{full_metric_name}':")
    print(f"\nElec Results:")
    print(elec_res[columns_to_display].to_string(index=False) if not elec_res.empty else "No data.")
    # print(f"\nPhis Results:")
    # print(phis_res[columns_to_display].to_string(index=False) if not phis_res.empty else "No data.")
    # print(f"\nElec2 Results:")
    # print(elec2_res[columns_to_display].to_string(index=False) if not elec2_res.empty else "No data.")
    
    exit() if len(filters) != 0 else None    
    
    combinaciones = [{'limit': 50, 'range_start': 50, 'range_end': 60}, 
                    {'limit': 150, 'range_start': 50, 'range_end': 60}, 
                    {'limit': 250, 'range_start': 50, 'range_end': 60}, 
                    {'limit': 500, 'range_start': 72.5, 'range_end': 77.5}
    ]

    y_min = 0
    y_max = float('-inf')

    # Loop through each combination to find the global y_max
    for combinacion in combinaciones:
        filtered_df = elec_res[
            (elec_res['limit'] == combinacion['limit']) &
            (elec_res['range_start'] == combinacion['range_start']) &
            (elec_res['range_end'] == combinacion['range_end'])
        ]
        
        if not filtered_df.empty:
            current_max = filtered_df[full_metric_name].max()
            y_max = max(y_max, current_max)

    # Increase y_max by a margin (e.g., 10%) for better visualization
    y_max += y_max * 0.1

    # Plot setup
    ancho = 20
    alto = 10
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(ancho, alto))
    axes = [ax1, ax2, ax3, ax4]

    # Loop through each combination and corresponding axis
    for combinacion, ax in zip(combinaciones, axes):
        filtered_df = elec_res[
            (elec_res['limit'] == combinacion['limit']) &
            (elec_res['range_start'] == combinacion['range_start']) &
            (elec_res['range_end'] == combinacion['range_end'])
        ]

        # Title for each subplot based on the filter conditions
        title = f"Limit: {combinacion['limit']}, Range: {combinacion['range_start']} to {combinacion['range_end']}"
        
        # Plot results for each filtered DataFrame
        plot_results_all(filtered_df, title, ax, metrica_seleccionada=full_metric_name, y_min=y_min, y_max=y_max)

    plt.subplots_adjust(top=0.85, hspace=0.4, wspace=0.3)
    plt.tight_layout()
    plt.show()

    
    

if __name__ == '__main__':
    main()

