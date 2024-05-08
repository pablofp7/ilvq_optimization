import pandas as pd
import numpy as np
import re
import os
import argparse

def get_results(test):
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

            match = re.match(r'result_(elec|phis|elec2)_s(\d+)_T([\d.]+)_limit(\d+)_range(\d+)-(\d+)_it(\d+).txt', nombre_archivo)
            if match:
                _, s, T, limit, range_start, range_end, it = match.groups()
                
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
                                    
                                            # Calculamos los promedios después de leer todo el archivo
                    precision_prom = precision_total / cuenta_nodos if cuenta_nodos else 0
                    recall_prom = recall_total / cuenta_nodos if cuenta_nodos else 0
                    f1_prom = f1_total / cuenta_nodos if cuenta_nodos else 0
                    capacidad_promedio = capacidad_ejecucion_total / cuenta_nodos if cuenta_nodos else 0
                    tamaño_promedio_lotes = tamaño_lotes_total / cuenta_lotes if cuenta_lotes else 0

                    promedio_prototipos_entrenados = prototipos_entrenados / cuenta_nodos if cuenta_nodos else 0
                    promedio_prototipos_compartidos = prototipos_compartidos / cuenta_nodos if cuenta_nodos else 0
                    promedio_mensajes_enviados = mensajes_enviados / cuenta_nodos if cuenta_nodos else 0

                    resultado = {
                        's': int(s), 'T': float(T), 'limit': int(limit), 
                        'range_start': int(range_start), 'range_end': int(range_end), 
                        'it': int(it), 'Precision': precision_prom, 'Recall': recall_prom, 
                        'F1': f1_prom, 'Mensajes_Enviados': promedio_mensajes_enviados, 
                        'Prototipos_Entrenados': promedio_prototipos_entrenados, 
                        'Prototipos_Compartidos': promedio_prototipos_compartidos, 
                        'Capacidad_Ejecucion': capacidad_promedio, 
                        'Tamaño_Promedio_Lotes': tamaño_promedio_lotes
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

    if not datos_phis_df.empty:
        datos_phis_df = datos_phis_df.groupby(['s', 'T', 'limit', 'range_start', 'range_end']).mean().reset_index()

    if not datos_elec2_df.empty:
        datos_elec2_df = datos_elec2_df.groupby(['s', 'T', 'limit', 'range_start', 'range_end']).mean().reset_index()


    return datos_elec_df, datos_phis_df, datos_elec2_df
                            
                        
                        
                        
                        
                        
def main():
    datasets = ["elec"]
    S = [1, 4]
    T = np.array([0.0, 0.1, 1.0])
    lim_range = [
        (50, (50, 60)),
        (150, (50, 60)),
        (250, (50, 60)),
        (500, (72.5, 77.5))
    ] 
    
    test = "test4"
    elec_res, phis_res, elec2_res = get_results(test)
    
    
    
    print(f"")
    print(f"Resultados de elec")
    print(elec_res)
    print(f"")
    print(f"Resultados de phis")
    print(phis_res)
    print(f"")
    print(f"Resultados de elec2")
    print(elec2_res)
    print(f"")
            
    
if __name__ == '__main__':
    main()