import os
import re
import pandas as pd
import ast
import numpy as np

# Definir los datasets válidos
VALID_DATASETS = ["elec", "phis", "elec2", "lgr"]  # Se pueden agregar más si es necesario

# Directorio base donde están los resultados
RESULTS_DIR = os.path.expanduser('~/ilvq_optimization/codigo/raspi/')


def extract_results_from_txt(file_path, s, T):
    print(f"Processing TXT file: {file_path}")
    with open(file_path, 'r') as file:
        contenido = file.readlines()

    precision_total, recall_total, f1_total = 0, 0, 0
    protos_entrenados, mensajes_enviados, capacidad_ejecucion_total = 0, 0, 0
    cuenta_nodos, tamaño_lotes_total, cuenta_lotes = 0, 0, 0
    tiempo_total = 0
    prototipos_compartidos = 0

    for linea in contenido:
        if match := re.search(r'Precision: ([\d.]+)', linea):
            precision_total += float(match.group(1))
            cuenta_nodos += 1

        if match := re.search(r'Recall: ([\d.]+)', linea):
            recall_total += float(match.group(1))

        if match := re.search(r'F1: ([\d.]+)', linea):
            f1_total += float(match.group(1))

        if match := re.search(r'Se ha entrenado con (\d+) prototipos.', linea):
            protos_entrenados += int(match.group(1))

        if match := re.search(r'Ha compartido (\d+) veces.', linea):
            mensajes_enviados += int(match.group(1)) * int(s)

        if match := re.search(r'Capacidad de ejecución: (\d+)', linea):
            capacidad_ejecucion_total += float(match.group(1))

        if match := re.search(r"Tiempo total: ([\d.]+)", linea):
            tiempo_total = float(match.group(1))

        if match := re.search(r'Ha compartido (\d+) (prototipos\.|en total\.)', linea):
            prototipos_compartidos += int(match.group(1))

    if cuenta_nodos == 0:
        print("No valid nodes found in TXT file.")
        return None

    ancho_banda = (105 * prototipos_compartidos) / tiempo_total if tiempo_total else 0

    return {
        's': int(s),
        'T': float(T),
        'F1': round(f1_total / cuenta_nodos, 4),
        'Protos_Entrenados': round(protos_entrenados / cuenta_nodos, 4),
        'Ancho_Banda': round(ancho_banda, 4)
    }


def extract_results_from_csv(file_path, s, T):
    print(f"Processing CSV file: {file_path}")
    df = pd.read_csv(file_path)
    if df.empty:
        print("CSV file is empty.")
        return None

    # Ajustar valores según estructura CSV
    df['Protos_Entrenados'] = df.get('Prototipos entrenados', df.get('Parámetros agregados', 0))

    # Asegurar que 'Tiempo total' no sea cero para evitar división por cero
    df['Tiempo total'] = df['Tiempo total'].replace(0, np.nan)  # Sustituir ceros por NaN para evitar errores

    # Determinar cómo calcular el ancho de banda
    if 'Bytes enviados' in df.columns:
        df['Ancho_Banda'] = df['Bytes enviados'] / df['Tiempo total']
        print("Using 'Bytes enviados' for bandwidth calculation.")
    elif 'Prototipos compartidos' in df.columns:
        df['Ancho_Banda'] = (105 * df['Prototipos compartidos']) / df['Tiempo total']
        print("Using 'Prototipos compartidos' with formula for bandwidth calculation.")
    else:
        df['Ancho_Banda'] = 0  # Si no hay ninguna de las dos columnas, poner 0
        print("No valid bandwidth columns found, setting Ancho_Banda to 0.")

    # Reemplazar posibles NaN en 'Ancho_Banda' por 0
    df['Ancho_Banda'] = df['Ancho_Banda'].fillna(0)

    return {
        's': int(s),
        'T': float(T),
        'F1': round(df['F1'].mean(), 4),
        'Protos_Entrenados': round(df['Protos_Entrenados'].mean(), 4),
        'Ancho_Banda': round(df['Ancho_Banda'].mean(), 4)
    }


def process_results():
    for test_num in range(1, 7):
        test_folder = os.path.join(RESULTS_DIR, f'test{test_num}_resultados')
        if not os.path.exists(test_folder):
            print(f"Skipping test{test_num}: Folder does not exist.")
            continue

        print(f"\nProcessing test{test_num}...")
        results = {dataset: [] for dataset in VALID_DATASETS}

        for dataset in VALID_DATASETS:
            output_path = os.path.join(RESULTS_DIR, f'final_results/test{test_num}_{dataset}.csv')

            if os.path.exists(output_path):
                print(f"Skipping dataset {dataset} for test{test_num}, final file already exists: {output_path}")
                continue

            for filename in os.listdir(test_folder):
                print(f"Checking file: {filename}")
                match = re.match(r'result_(' + '|'.join(VALID_DATASETS) + r')_s(\d+)_T([\d.]+)_it(\d+).(txt|csv)', filename)
                if not match:
                    print(f"Skipping file: {filename} (does not match pattern)")
                    continue

                dataset_name, s, T, it, file_ext = match.groups()
                file_path = os.path.join(test_folder, filename)

                if dataset_name != dataset:
                    continue  

                if file_ext == 'txt':
                    result = extract_results_from_txt(file_path, s, T)
                else:
                    result = extract_results_from_csv(file_path, s, T)

                if result:
                    results[dataset_name].append(result)
                else:
                    print(f"Warning: No data extracted from {filename}")

            if results[dataset]:  
                print(f"Aggregating results for dataset: {dataset}")
                df = pd.DataFrame(results[dataset])
                df = df.groupby(['s', 'T']).mean().reset_index()
                df.to_csv(output_path, index=False)
                print(f"Saved aggregated results to {output_path}")

        print(f"Finished processing test{test_num}.\n")

if __name__ == "__main__":
    process_results()
    print("\nProcessing completed. CSV files saved.")
