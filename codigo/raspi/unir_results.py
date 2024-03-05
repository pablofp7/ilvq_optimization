import os
import re
import sys
from itertools import cycle, islice


class ParameterCombinations:
    def __init__(self, datasets, s_values, T_values, start_dataset, start_s, start_T):
        self.datasets = datasets
        self.s_values = s_values
        self.T_values = T_values
        self.current_dataset_index = datasets.index(start_dataset) if start_dataset in datasets else 0
        self.current_s_index = s_values.index(start_s) if start_s in s_values else 0
        self.current_T_index = T_values.index(start_T) if start_T in T_values else 0

    def reset(self, start_dataset=None, start_s=None, start_T=None):
        # Restablecer o inicializar los índices basados en los parámetros dados o al principio si no se especifican
        self.current_dataset_index = self.datasets.index(start_dataset) if start_dataset in self.datasets else 0
        self.current_s_index = self.s_values.index(start_s) if start_s in self.s_values else 0
        self.current_T_index = self.T_values.index(start_T) if start_T in self.T_values else 0


    def next(self):
        # Incrementa el parámetro más interno y maneja el acarreo
        self.current_T_index += 1
        if self.current_T_index >= len(self.T_values):
            self.current_T_index = 0
            self.current_s_index += 1
            if self.current_s_index >= len(self.s_values):
                self.current_s_index = 0
                self.current_dataset_index += 1
                if self.current_dataset_index >= len(self.datasets):
                    self.current_dataset_index = 0  # Reinicio para la siguiente iteración de 'it'
        
        return self.get_current_parameters()

    def get_current_parameters(self):
        # Obtiene los parámetros actuales
        dataset = self.datasets[self.current_dataset_index]
        s = self.s_values[self.current_s_index]
        T = self.T_values[self.current_T_index]
        return dataset, s, T



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
it_range = range(50)  # Desde 0 hasta 49
datasets = ['elec', 'phis', 'elec2']
s_values = [1, 2, 3, 4]
T_values = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0]
T_values = T_values[::-1]
nodos = [i for i in range(N_NODOS)]
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
        pattern = f"_it{it}"
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
    
    last_it, last_data, last_s, last_T = check_latest()  # Obtiene los últimos valores procesados
    param_combinations = ParameterCombinations(datasets, s_values, T_values, last_data, last_s, last_T)
    param_combinations.next()

    dataset, s, T = param_combinations.get_current_parameters()

    if dataset == datasets[0] and s == s_values[0] and T == T_values[0]:
        start_it = last_it + 1
    else:
        start_it = last_it

    for it in range(start_it, 50):  # Asumiendo que it_range va de 0 a 49
        while True:
            
            dataset, s, T = param_combinations.get_current_parameters()

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

            if dataset == datasets[0] and s == s_values[0] and T == T_values[0]:
                # Si después de un 'next()' volvemos al inicio, significa que hemos procesado todas las combinaciones para este 'it'
                break
            
            param_combinations.next()  # Avanza a la siguiente combinación de parámetros
            
        param_combinations.reset()  # Restablece los parámetros para la siguiente iteración de 'it' 
    

# process_combinations()
process_combinations_from_latest()

