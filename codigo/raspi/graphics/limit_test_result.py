import pandas as pd
import numpy as np
import re
import os
import argparse

def get_results(test, filters, metric):
    directorio_resultados = f'/home/pablo/trabajo/codigo/raspi/{test}_resultados/'
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
                        'Clust_Runs': promedio_clust_runs
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

    if not datos_phis_df.empty:
        datos_phis_df = datos_phis_df.groupby(['s', 'T', 'limit', 'range_start', 'range_end']).mean().reset_index()
        datos_phis_df = datos_phis_df.drop('it', axis=1)

    if not datos_elec2_df.empty:
        datos_elec2_df = datos_elec2_df.groupby(['s', 'T', 'limit', 'range_start', 'range_end']).mean().reset_index()
        datos_elec2_df = datos_elec2_df.drop('it', axis=1)

    # Apply filters if provided
    for param, value in filters.items():
        if param in datos_elec_df.columns:
            datos_elec_df = datos_elec_df[datos_elec_df[param] == value]
            datos_phis_df = datos_phis_df[datos_phis_df[param] == value]
            datos_elec2_df = datos_elec2_df[datos_elec2_df[param] == value]
    
    # Sorting by specified metric in descending order after filtering
    if metric == 'Clust':
        # For the special case where the metric is 'Clust', sort by both Clust_Time and Clust_Runs
        if not datos_elec_df.empty:
            datos_elec_df = datos_elec_df.sort_values(by=['Clust_Runs', 'Clust_Time'], ascending=[False, False])
        if not datos_phis_df.empty:
            datos_phis_df = datos_phis_df.sort_values(by=['Clust_Runs', 'Clust_Time'], ascending=[False, False])
        if not datos_elec2_df.empty:
            datos_elec2_df = datos_elec2_df.sort_values(by=['Clust_Runs', 'Clust_Time'], ascending=[False, False])
    else:
        if not datos_elec_df.empty:
            datos_elec_df = datos_elec_df.sort_values(by=[metric], ascending=False)
        if not datos_phis_df.empty:
            datos_phis_df = datos_phis_df.sort_values(by=[metric], ascending=False)
        if not datos_elec2_df.empty:
            datos_elec2_df = datos_elec2_df.sort_values(by=[metric], ascending=False)

    return datos_elec_df, datos_phis_df, datos_elec2_df
                            
                   
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
        'Clust_Runs'
    ]

    parser = argparse.ArgumentParser(description='Filter results based on parameters and output specific metrics along with parameter data.')
    parser.add_argument('-s', '--s', type=int, help='Filter by s value')
    parser.add_argument('-T', '--T', type=float, help='Filter by T value')
    parser.add_argument('-l', '--limit', type=int, help='Filter by limit value')
    parser.add_argument('-rs', '--range_start', type=int, help='Filter by range start value')
    parser.add_argument('-re', '--range_end', type=int, help='Filter by range end value')
    parser.add_argument('-m', '--metric', required=True, help='Specify a metric to output (e.g., f1, prec, rec). This is mandatory.')

    args = parser.parse_args()

    # Function to find the full metric name from the inserted abbreviation
    def find_metric_name(abbreviation, metric_list):
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

    test = "test4"
    elec_res, phis_res, elec2_res = get_results(test, filters, full_metric_name)  # Include metric for sorting

    if full_metric_name not in elec_res.columns:
        print(f"Error: The specified metric '{full_metric_name}' is not valid.")
        return

    # Create a list of columns to display: parameter columns and the selected metric
    if full_metric_name == 'Clust':
        columns_to_display = ['s', 'T', 'limit', 'range_start', 'range_end', 'Clust_Time', 'Clust_Runs']
    else:
        columns_to_display = ['s', 'T', 'limit', 'range_start', 'range_end', full_metric_name]

    # Printing without the DataFrame index
    print(f"Filtered Results for the specified metric '{full_metric_name}':")
    print(f"\nElec Results:")
    print(elec_res[columns_to_display].to_string(index=False) if not elec_res.empty else "No data.")
    print(f"\nPhis Results:")
    print(phis_res[columns_to_display].to_string(index=False) if not phis_res.empty else "No data.")
    print(f"\nElec2 Results:")
    print(elec2_res[columns_to_display].to_string(index=False) if not elec2_res.empty else "No data.")

if __name__ == '__main__':
    main()

