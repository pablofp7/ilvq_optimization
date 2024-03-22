import os
import re
import sys


class ParameterCombinations:
    def __init__(self, it_range, datasets, s_values, T_values, start_it=0, start_dataset=0, start_s=0, start_T=0):
        self.it_range = it_range
        self.datasets = datasets
        self.s_values = s_values
        self.T_values = T_values
        self.current_it_index = start_it
        self.current_dataset_index = start_dataset
        self.current_s_index = start_s
        self.current_T_index = start_T

    def next(self):
        # Incrementa el parámetro más interno y maneja el acarreo usando el operador %
        self.current_T_index = (self.current_T_index + 1) % len(self.T_values)
        if self.current_T_index == 0:  # Acarreo para s
            self.current_s_index = (self.current_s_index + 1) % len(self.s_values)
            if self.current_s_index == 0:  # Acarreo para datasets
                self.current_dataset_index = (self.current_dataset_index + 1) % len(self.datasets)
                if self.current_dataset_index == 0:  # Acarreo para iteraciones
                    self.current_it_index = (self.current_it_index + 1) % len(self.it_range)
                    if self.current_it_index == 0:  # Ha completado un ciclo completo
                        return False
        return True

    def get_current_parameters(self):
        it = self.it_range[self.current_it_index]
        dataset = self.datasets[self.current_dataset_index]
        s = self.s_values[self.current_s_index]
        T = self.T_values[self.current_T_index]
        return it, dataset, s, T

    def reset(self, start_it=0, start_dataset=0, start_s=0, start_T=0):
        self.current_it_index = start_it
        self.current_dataset_index = start_dataset
        self.current_s_index = start_s
        self.current_T_index = start_T




# Directorios de origen y destino
results_dir = 'resultados_raspi_indiv'
try:
    test = sys.argv[1]     
except:
    test = "test2"

target_dir = f'{test}_resultados'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)


N_NODOS = 5
it_range = range(50)  # Desde 0 hasta 49
datasets = ['elec', 'phis', 'elec2']
s_values = [1, 2, 3, 4]
T_values = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0]
T_values = T_values[::-1]
nodos = [i for i in range(N_NODOS)]
all_files = os.listdir(results_dir)
all_combined = os.listdir(target_dir)



def check_latest():
    last_it = -1
    last_data = ""
    last_s = -1
    last_T_index = -1  # Usaremos el índice de T_values

    # Buscar la última iteración con archivos
    for it in it_range:
        pattern = f"_it{it}"
        compiled_pattern = re.compile(pattern)
        files = [f for f in all_combined if compiled_pattern.search(f)]
        if files:
            last_it = it
        else:
            break  # Sale del bucle si no encuentra archivos para una iteración

    if last_it == -1:
        print(f"ITERACION. No se encontraron archivos combinados en {target_dir}...")
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
        print(f"ÑAST  T INDEX. No se encontraron archivos combinados en {target_dir}...")
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

    print(f"Últimos valores encontrados: it={last_it}, dataset={last_data}, s={last_s}, T={T_values[last_T_index]}")
    return last_it, last_data, last_s, T_values[last_T_index] if last_T_index >= 0 else None



def process_combinations_from_latest():

    combinacion = check_latest()
    try:
        last_it, last_data, last_s, last_T = combinacion
    except:
        print("No hay archivos previos")
        last_it = None
        last_data = None
        last_s = None
        last_T = None

    if last_it is not None:
        # Convertir los valores a índices
        start_it_index = it_range.index(last_it)
        start_dataset_index = datasets.index(last_data)
        start_s_index = s_values.index(last_s)
        start_T_index = T_values.index(last_T)
    else:
        # Si no hay archivos previos, empezar desde el principio
        start_it_index = 0
        start_dataset_index = 0
        start_s_index = 0
        start_T_index = 0

    print(f"Indices: it={start_it_index}, dataset={start_dataset_index}, s={start_s_index}, T={start_T_index}")
    print(f"Valores: it={it_range[start_it_index]}, dataset={datasets[start_dataset_index]}, s={s_values[start_s_index]}, T={T_values[start_T_index]}")

    param_combinations = ParameterCombinations(it_range, datasets, s_values, T_values, start_it_index, start_dataset_index, start_s_index, start_T_index)
    param_combinations.next()

    more_combinations = True
    while more_combinations:
        it, dataset, s, T = param_combinations.get_current_parameters()

        pattern = f"result_{dataset}_s{s}_T{T}_it{it}_nodo(\\d+)\\.txt"
        compiled_pattern = re.compile(pattern)
        files = [f for f in all_files if compiled_pattern.match(f)]

        # Si no hay archivos, saltar a la siguiente combinación
        if not files and it > 19:
            more_combinations = param_combinations.next()
            continue

        # Identificar nodos faltantes
        nodes_found = [int(compiled_pattern.search(f).group(1)) for f in files]
        missing_nodes = [n for n in range(N_NODOS) if n not in nodes_found]

        if missing_nodes:
            print(f"Faltan archivos para it={it}, dataset={dataset}, s={s}, T={T} en los nodos: {missing_nodes}")
            more_combinations = param_combinations.next()
            break

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

        more_combinations = param_combinations.next()




process_combinations_from_latest()