import os
import re
import sys

# Directorios de origen y destino
results_dir = 'resultados_raspi_indiv'
try:
    test = sys.argv[1]     
except:
    test = "test1"

target_dir = f'{test}_resultados'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)


N_NODOS = 5
datasets = ['elec', 'phis', 'elec2']
s_values = [1, 2, 3, 4]
T_values = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0]
T_values = T_values[::-1]
nodos = [i for i in range(N_NODOS)]
it_range = range(50)  # Desde 0 hasta 49
all_files = os.listdir(results_dir)
all_combined = os.listdir(target_dir)



    
def process_combinations():
    for it in it_range:
        for dataset in datasets:
            for s in s_values:
                for T in T_values:
                    pattern = f"result_{dataset}_s{s}_T{T}_it{it}_nodo(\\d+)\\.txt"
                    compiled_pattern = re.compile(pattern)
                    files = [f for f in all_files if compiled_pattern.match(f)]

                    if not files:
                        if it > 19:  # Si `it` es mayor a 19 y no se encuentran archivos, se detiene la búsqueda
                            break
                        continue  # Si aún no se encuentran archivos, continúa buscando
                    
                    # Identificar nodos faltantes
                    nodes_found = [int(compiled_pattern.search(f).group(1)) for f in files]
                    missing_nodes = [n for n in range(max(nodes_found)+1) if n not in nodes_found]
                    
                    if missing_nodes:
                        print(f"Faltan archivos para dataset={dataset}, s={s}, T={T}, it={it} en los nodos: {missing_nodes}")
                        continue  # Pasa al siguiente conjunto de parámetros

                    # Ordenar archivos para mantener el orden de los nodos
                    files.sort(key=lambda x: int(re.search(r'nodo(\d+)', x).group(1)))

                    contents = []
                    for file_name in files:
                        file_path = os.path.join(results_dir, file_name)
                        with open(file_path, 'r') as file:
                            contents.append(file.read().strip())

                    final_content = "\n\n".join(contents) + "\n"

                    new_file_name = f"result_{dataset}_s{s}_T{T}_it{it}.txt"
                    new_file_path = os.path.join(target_dir, new_file_name)

                    with open(new_file_path, 'w') as new_file:
                        new_file.write(final_content)
                    print(f"Archivo combinado creado: {new_file_path}")


def check_latest():
    last_it = -1
    last_data = ""
    last_s = -1
    last_T_index = -1  # Usaremos el índice de T_values

    # Buscar la última iteración con archivos
    for it in it_range:
        pattern = f"_it{it}_"
        compiled_pattern = re.compile(pattern)
        files = [f for f in all_combined if compiled_pattern.search(f)]
        if files:
            last_it = it
        else:
            break  # Sale del bucle si no encuentra archivos para una iteración

    if last_it == -1:
        return None  # No se encontraron archivos

    # Basado en la última iteración encontrada, buscar el último dataset
    for data in datasets:
        pattern = f"_{data}_s"
        compiled_pattern = re.compile(pattern)
        files = [f for f in all_combined if compiled_pattern.search(f) and f"it{last_it}" in f]
        if files:
            last_data = data
        else:
            break  # Sale del bucle si no encuentra archivos para un dataset

    # Basado en el último dataset encontrado, buscar el último valor de s
    for s in s_values:
        pattern = f"_s{s}_T"
        compiled_pattern = re.compile(pattern)
        files = [f for f in all_combined if compiled_pattern.search(f) and f"it{last_it}" in f and last_data in f]
        if files:
            last_s = s
        else:
            break  # Sale del bucle si no encuentra archivos para un valor de s

    # Basado en el último valor de s encontrado, buscar el último valor de T
    for i, T in enumerate(T_values):
        pattern = f"_T{T}_"
        compiled_pattern = re.compile(pattern)
        files = [f for f in all_combined if compiled_pattern.search(f) and f"it{last_it}" in f and last_data in f and f"_s{last_s}_" in f]
        if files:
            last_T_index = i  # Actualizamos el índice de T_values al último T válido encontrado

    if last_T_index == -1:
        return None  # No se encontró un valor de T válido con archivos

    # Comprobar si están todos los nodos para el último T encontrado
    all_nodes_present = True
    for nodo in range(5):  # Asumiendo que los nodos van de 0 a 4
        pattern = f"result_{last_data}_s{last_s}_T{T_values[last_T_index]}_it{last_it}_nodo{nodo}.txt"
        if not os.path.isfile(os.path.join(results_dir, pattern)):
            all_nodes_present = False
            break

    # Si no están todos los nodos, se retrocede al valor de T anterior automáticamente
    if not all_nodes_present and last_T_index > 0:
        last_T_index -= 1

    return last_it, last_data, last_s, T_values[last_T_index] if last_T_index >= 0 else None

def process_combinations_from_latest():
    # Obtener el último punto de procesamiento
    result = check_latest()
    if result:
        last_it, last_data, last_s, last_T = result
        print(f"Continuando desde la última combinación encontrada: it={last_it}, dataset={last_data}, s={last_s}, T={last_T}")
        start_it = last_it
        start_dataset_index = datasets.index(last_data)
        start_s_index = s_values.index(last_s)
        start_T_index = T_values.index(last_T)
    else:
        print("No se encontraron archivos previos, comenzando desde el inicio.")
        start_it = 0
        start_dataset_index = 0
        start_s_index = 0
        start_T_index = 0

    for it in it_range[start_it:]:
        for dataset_index, dataset in enumerate(datasets[start_dataset_index:], start=start_dataset_index):
            for s_index, s in enumerate(s_values[start_s_index:], start=start_s_index):
                for T_index, T in enumerate(T_values[start_T_index:], start=start_T_index):
                    # Restablecer los índices de inicio para los siguientes bucles
                    if it > start_it or dataset_index > start_dataset_index:
                        start_s_index = 0
                    if it > start_it or dataset_index > start_dataset_index or s_index > start_s_index:
                        start_T_index = 0
                    
                    pattern = f"result_{dataset}_s{s}_T{T}_it{it}_nodo(\\d+)\\.txt"
                    compiled_pattern = re.compile(pattern)
                    files = [f for f in all_files if compiled_pattern.match(f)]
                    
                    if not files:
                        if it > 19:  # Si `it` es mayor a 19 y no se encuentran archivos, se detiene la búsqueda
                            break
                        continue  # Si aún no se encuentran archivos, continúa buscando
                    
                    # Identificar nodos faltantes
                    nodes_found = [int(compiled_pattern.search(f).group(1)) for f in files]
                    missing_nodes = [n for n in range(max(nodes_found)+1) if n not in nodes_found]
                    
                    if missing_nodes:
                        print(f"Faltan archivos para dataset={dataset}, s={s}, T={T}, it={it} en los nodos: {missing_nodes}")
                        exit()  # Pasa al siguiente conjunto de parámetros

                    # Ordenar archivos para mantener el orden de los nodos
                    files.sort(key=lambda x: int(re.search(r'nodo(\d+)', x).group(1)))

                    contents = []
                    for file_name in files:
                        file_path = os.path.join(results_dir, file_name)
                        with open(file_path, 'r') as file:
                            contents.append(file.read().strip())

                    final_content = "\n\n".join(contents) + "\n"

                    new_file_name = f"result_{dataset}_s{s}_T{T}_it{it}.txt"
                    new_file_path = os.path.join(target_dir, new_file_name)

                    with open(new_file_path, 'w') as new_file:
                        new_file.write(final_content)
                    print(f"Archivo combinado creado: {new_file_path}")


# process_combinations()
process_combinations_from_latest()

