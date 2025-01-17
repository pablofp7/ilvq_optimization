#!/usr/bin/env python3
import os
import re
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import tkinter as tk
from tkinter import ttk
from screeninfo import get_monitors
import argparse


def get_current_screen():
    cursor_x, cursor_y = ventana_principal.winfo_pointerxy()
    for monitor in get_monitors():
        if monitor.x <= cursor_x <= monitor.x + monitor.width and monitor.y <= cursor_y <= monitor.y + monitor.height:
            return monitor
    return None

def ajustar_tamaño_ventana():

    n_col, n_row = ventana_principal.grid_size()

    ancho = n_col * 250
    alto = n_row * 50

    ventana_principal.geometry(f"{ancho}x{alto}")
    
    return ancho, alto

def get_results():
    
    directorio_resultados = os.path.expanduser(f'~/ilvq_optimization/codigo/raspi/{test}_resultados/')

    datos_elec = []
    datos_elec2 = []
    datos_phis= []

    for nombre_archivo in os.listdir(directorio_resultados):
        if nombre_archivo.endswith('.txt'):
            es_elec2 = "elec2" in nombre_archivo
            es_elec = "elec" in nombre_archivo if not es_elec2 else False 
            es_phis = "phis" in nombre_archivo

            ruta_archivo = os.path.join(directorio_resultados, nombre_archivo)
            
            match = re.match(r'result_(elec|phis|elec2)_s(\d+)_T([\d.]+)_it(\d+).txt', nombre_archivo)
            if match:
                _, s, T, it = match.groups()
                
                with open(ruta_archivo, 'r') as file:
                    contenido = file.readlines()
                    
                    precision_total, recall_total, f1_total = 0, 0, 0
                    prototipos_entrenados, prototipos_compartidos = 0, 0
                    mensajes_enviados = 0
                    cuenta_nodos = 0
                    capacidad_ejecucion_total, tamaño_lotes_total, cuenta_lotes = 0, 0, 0
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
                            
                        match_compartidos = re.search(r'Ha compartido (\d+) (prototipos\.|en total\.)', linea)
                        if match_compartidos:
                            prototipos_compartidos += int(match_compartidos.group(1))
                            
                        match_compartidos = re.search(r'Ha compartido (\d+) veces.', linea)
                        if match_compartidos:
                            veces_compartido = int(match_compartidos.group(1))
                            
                            mensajes_nodo = veces_compartido * int(s)
                            mensajes_enviados += mensajes_nodo
                            
                        capacidad_match = re.search(r'Capacidad de ejecución: (\d+)', linea)
                        if capacidad_match:
                            capacidad_ejecucion_total += float(capacidad_match.group(1))
                            
                        match_tiempo = re.search(r"Tiempo total: ([\d\.]+)", linea)
                        if match_tiempo:
                            tiempo_total = float(match_tiempo.group(1))

                        lotes_match = re.search(r'ID, Tamaño de lotes recibidos: \[(.*?)\]', linea)
                        if lotes_match:
                            lotes = lotes_match.group(1).split('), (')
                            for lote in lotes:
                                partes = lote.strip('()').split(', ')
                                if len(partes) == 2:
                                    _, tamaño = partes
                                    tamaño_lotes_total += int(tamaño)
                                    cuenta_lotes += 1  
                   

                    precision_prom = precision_total / cuenta_nodos
                    recall_prom = recall_total / cuenta_nodos
                    f1_prom = f1_total / cuenta_nodos
                    capacidad_promedio = capacidad_ejecucion_total / cuenta_nodos
                    if cuenta_lotes == 0:
                        tamaño_promedio_lotes = 0
                    else:
                        tamaño_promedio_lotes = tamaño_lotes_total / cuenta_lotes if cuenta_lotes else 0
                        

                    promedio_prototipos_entrenados = prototipos_entrenados / cuenta_nodos
                    promedio_prototipos_compartidos = prototipos_compartidos / cuenta_nodos
                    promedio_mensajes_enviados = mensajes_enviados / cuenta_nodos
                    
                    tiempo_total_prom = tiempo_total / cuenta_nodos
                    ancho_banda_prom = (105 * promedio_prototipos_compartidos) / tiempo_total_prom
                    
                    if es_elec:
                        datos_elec.append({'s': int(s), 'T': float(T), 'it': int(it), 
                                    'Precisión': precision_prom, 
                                    'Recall': recall_prom, 
                                    'F1': f1_prom,
                                    'Mensajes_Enviados' : promedio_mensajes_enviados,
                                    'Prototipos_Entrenados': promedio_prototipos_entrenados, 
                                    'Prototipos_Compartidos': promedio_prototipos_compartidos,
                                    'Capacidad_Ejecucion': capacidad_promedio, 
                                    'Tamaño_Promedio_Lotes': tamaño_promedio_lotes,
                                    'Ancho_Banda': ancho_banda_prom})

                    if es_phis:
                        datos_phis.append({'s': int(s), 'T': float(T), 'it': int(it), 
                                    'Precisión': precision_prom, 
                                    'Recall': recall_prom, 
                                    'F1': f1_prom,
                                    'Mensajes_Enviados' : promedio_mensajes_enviados,
                                    'Prototipos_Entrenados': promedio_prototipos_entrenados, 
                                    'Prototipos_Compartidos': promedio_prototipos_compartidos,
                                    'Capacidad_Ejecucion': capacidad_promedio, 
                                    'Tamaño_Promedio_Lotes': tamaño_promedio_lotes,
                                    'Ancho_Banda': ancho_banda_prom})
                        
                    if es_elec2:
                        datos_elec2.append({'s': int(s), 'T': float(T), 'it': int(it), 
                                    'Precisión': precision_prom, 
                                    'Recall': recall_prom, 
                                    'F1': f1_prom,
                                    'Mensajes_Enviados' : promedio_mensajes_enviados,
                                    'Prototipos_Entrenados': promedio_prototipos_entrenados, 
                                    'Prototipos_Compartidos': promedio_prototipos_compartidos,
                                    'Capacidad_Ejecucion': capacidad_promedio, 
                                    'Tamaño_Promedio_Lotes': tamaño_promedio_lotes,
                                    'Ancho_Banda': ancho_banda_prom})
    
    return pd.DataFrame(datos_elec), pd.DataFrame(datos_phis), pd.DataFrame(datos_elec2)


def plot_results(df, titulo_dataset, ax, metrica_seleccionada, not_gui: bool):

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
        
        return
        
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
    ax.set_title(f"{titulo_dataset} - {text_metrica_seleccionada}=f(T,s). As a function of T, for different values of s.")

def get_metric_and_plot(metrica: str = None):

    global datos_elec, datos_phis, datos_elec2

    not_gui=False
    if not metrica:
        metrica_seleccionada = combo_metrics.get()
    else: 
        metrica_seleccionada = metrica
        not_gui=True
        
    
    try:
        ancho = 20
        alto = 10
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(ancho, alto))

        plot_results(datos_elec, "Electricity", ax1, metrica_seleccionada, not_gui)
        plot_results(datos_phis, "Phishing", ax2, metrica_seleccionada, not_gui)
        plot_results(datos_elec2, "Electricity2", ax3, metrica_seleccionada, not_gui)
	    
        plt.tight_layout()
        plt.show()
	
    except Exception as e:
        print(e)

def close_window():
    ventana_principal.destroy()
    

def parse_args():
    parser = argparse.ArgumentParser(description="Analizador de métricas")
    parser.add_argument(
        '--metrica',
        '-m',
        type=str,
        default=None,  # Esto asegura que si no hay una métrica, se inicie la GUI.
        help="Especifica la métrica para trazar. Si no se especifica, se iniciará la interfaz gráfica. \
              Opciones válidas son: precision, recall, f1, mensajes, entrenados, compartidos, banda."
    )

    # Define a custom type for checking the format "testX" or "testX.Y" where X is 1 to 10 and Y is 0 to 9
    def test_type(value):
        if not value.startswith("test"):
            raise argparse.ArgumentTypeError(f"Value must start with 'test' or 'all'. You entered: {value}")
        
        number_part = value[4:]  # Extract the number part after "test"

        # Check for both integer and decimal formats
        try:
            num = float(number_part)  # Attempt to convert to float to allow decimal formats

            # Validate the range
            if not 1 <= num <= 10:
                raise argparse.ArgumentTypeError("The number must be between 1 and 10, inclusive.")

            # Additional check for the decimal part, if exists, must be between .0 to .9
            if '.' in number_part:
                decimal_part = number_part.split('.')[1]
                if len(decimal_part) > 1 or not decimal_part.isdigit():
                    raise argparse.ArgumentTypeError("Decimal part must be a single digit between 0 and 9.")

        except ValueError:
            raise argparse.ArgumentTypeError("The format must be 'testX' or 'testX.Y' where X is 1 to 10 and Y is 0 to 9.")
        
        return value

    parser.add_argument(
        '-t',
        type=test_type,
        default='test1',  # Set 'test1' as the default value
        help="Specify the test case to run. Acceptable values are from 'test1' to 'test10', including 'testX.Y' where X is 1 to 10 and Y is 0 to 9. If not specified, 'test1' is used as the default."
    )

    return parser.parse_args()


def find_matching_metric(input_metric):
    metricas = {
        'precision': 'Precisión',
        'recall': 'Recall',
        'f1': 'F1',
        'mensajes': 'Mensajes_Enviados',
        'entrenados': 'Prototipos_Entrenados',
        'compartidos': 'Prototipos_Compartidos',
        'capacidad': 'Capacidad_Ejecucion',
        'lotes': 'Tamaño_Promedio_Lotes',
        'banda': 'Ancho_Banda'
    }

    matching_metrics = [metric for metric in metricas if metric.startswith(input_metric)]

    if matching_metrics:
        return metricas[matching_metrics[0]]
    else:
        print(f"No se encontró una métrica que coincida con '{input_metric}'. Se utilizará 'F1' por defecto.")
        return 'F1'


if __name__ == '__main__':
    
    args = parse_args()
    test = args.t
    datos_elec, datos_phis, datos_elec2 = get_results()
    
    
    if args.metrica:
        metricas = {
            'precision': 'Precisión',
            'recall': 'Recall',
            'f1': 'F1',
            'mensajes': 'Mensajes_Enviados',
            'entrenados': 'Prototipos_Entrenados',
            'compartidos': 'Prototipos_Compartidos',
            'capacidad': 'Capacidad_Ejecucion',
            'lotes': 'Tamaño_Promedio_Lotes',
            'banda': 'Ancho_Banda'
        }
        # Si la métrica está en el diccionario, la seleccionamos. Si no, por defecto es 'F1'.
        metrica_seleccionada = find_matching_metric(args.metrica)
        get_metric_and_plot(metrica_seleccionada)
        sys.exit()
    
    ventana_principal = tk.Tk()
    ventana_principal.title("VISUALIZADOR DE MÉTRICAS")
    
    ancho = 400
    alto = 100
    
    monitor_actual = get_current_screen()
    if monitor_actual:
        x = monitor_actual.x + (monitor_actual.width - ancho) // 2
        y = monitor_actual.y + (monitor_actual.height - alto) // 2
        ventana_principal.geometry(f"{ancho}x{alto}+{x}+{y}")
    else:
        ventana_principal.geometry(f"{ancho}x{alto}+0+0")
        

    frame = ttk.Frame(ventana_principal, padding="3 3 12 12")
    frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)

    ttk.Label(frame, text="Selecciona una métrica:").grid(column=0, row=1, sticky=tk.W)
    metric_var = tk.StringVar()
    combo_metrics = ttk.Combobox(frame, textvariable=metric_var)
    combo_metrics['values'] = ('F1', 'Prototipos_Entrenados', 'Precisión', 'Recall', 'Mensajes_Enviados', 'Prototipos_Compartidos', 
                               'Capacidad_Ejecucion', 'Tamaño_Promedio_Lotes', 'Ancho_Banda')
    combo_metrics.grid(column=1, row=1, sticky=(tk.W, tk.E), pady=20)

    frame_botones = ttk.Frame(frame)
    frame_botones.grid(column=1, row=2, sticky=(tk.E, tk.S))

    boton_trazar = tk.Button(frame_botones, text="Confirmar", command=get_metric_and_plot, bg='green', fg='white')
    boton_trazar.grid(column=0, row=0, sticky=tk.E, padx=5)

    boton_salir = tk.Button(frame_botones, text="Salir", command=close_window, bg='red', fg='white')
    boton_salir.grid(column=1, row=0, sticky=tk.W, padx=5)

    for child in frame.winfo_children():
        child.grid_configure(padx=5, pady=5)

    combo_metrics.focus()
    combo_metrics.current(0)
    
    ventana_principal.mainloop()
