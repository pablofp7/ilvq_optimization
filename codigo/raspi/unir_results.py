import os
import re
import sys


class ParameterCombinations:
    def __init__(self, it_range, datasets, s_values, T_values, lim_range, start_it=0, start_dataset=0, start_s=0, start_T=0, start_lim_range=0, is_test4=False):
        self.it_range = it_range
        self.datasets = datasets
        self.s_values = s_values
        self.T_values = T_values
        self.lim_range = lim_range
        self.current_it_index = start_it
        self.current_dataset_index = start_dataset
        self.current_s_index = start_s
        self.current_T_index = start_T
        self.current_lim_range_index = start_lim_range
        self.is_test4 = is_test4

    def next(self):
        if self.is_test4:
            self.current_lim_range_index = (self.current_lim_range_index + 1) % len(self.lim_range)
            if self.current_lim_range_index == 0:
                return self._increment_indices()
        else:
            return self._increment_indices()
        return True

    def _increment_indices(self):
        self.current_T_index = (self.current_T_index + 1) % len(self.T_values)
        if self.current_T_index == 0:
            self.current_s_index = (self.current_s_index + 1) % len(self.s_values)
            if self.current_s_index == 0:
                self.current_dataset_index = (self.current_dataset_index + 1) % len(self.datasets)
                if self.current_dataset_index == 0:
                    self.current_it_index = (self.current_it_index + 1) % len(self.it_range)
                    if self.current_it_index == 0:
                        return False  # Completa todas las combinaciones
        return True

    def get_current_parameters(self):
        it = self.it_range[self.current_it_index]
        dataset = self.datasets[self.current_dataset_index]
        s = self.s_values[self.current_s_index]
        T = self.T_values[self.current_T_index]
        if self.is_test4:
            limit, range_ = self.lim_range[self.current_lim_range_index]
            return it, dataset, s, T, limit, range_
        return it, dataset, s, T

    def reset(self, start_it=0, start_dataset=0, start_s=0, start_T=0, start_lim_range=0):
        self.current_it_index = start_it
        self.current_dataset_index = start_dataset
        self.current_s_index = start_s
        self.current_T_index = start_T
        self.current_lim_range_index = start_lim_range
        
        
# Directorios de origen y destino
try:
    test = sys.argv[1]     
except:
    test = "test4"

target_dir = f'{test}_resultados'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

suffix = ""
if test == "test5":
    suffix = "_tree"
elif test == "test6":
    suffix = "_nb"
results_dir = f"resultados_raspi_indiv{suffix}"


N_NODOS = 5
it_range = range(50)  # Desde 0 hasta 49
datasets = ['elec', 'phis', 'elec2', 'lgr']
s_values = [1, 2, 3, 4]
T_values = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0]
T_values = T_values[::-1]
nodos = [i for i in range(N_NODOS)]
lim_range = [
    (50, (50, 60)),
    (150, (50, 60)),
    (250, (50, 60)),
    (500, (72.5, 77.5))
] 
all_files = os.listdir(results_dir)
all_combined = os.listdir(target_dir)



def check_latest(is_test4=False):
    last_it = -1
    last_data = ""
    last_s = -1
    last_T_index = -1  # Usaremos el índice de T_values
    last_limit = None
    last_range = None

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

    # Basado en el último valor de s encontrado, buscar el último valor de T y limit/range si es test4
    for i, T in enumerate(T_values):
        if is_test4:
            for limit, range in lim_range:
                pattern = f"result_{last_data}_s{last_s}_T{T}_limit{limit}_range{range[0]}-{range[1]}_it{last_it}"
                if any(pattern in f for f in all_combined):
                    last_T_index = i
                    last_limit = limit
                    last_range = range
                    break
            if last_T_index != -1:
                break  # Salimos cuando encontramos la primera coincidencia
        else:
            pattern = f"_T{T}_"
            compiled_pattern = re.compile(pattern)
            files = [f for f in all_combined if compiled_pattern.search(f) and f"it{last_it}" in f and last_data in f and f"_s{last_s}_" in f]
            if files:
                last_T_index = i
                break

    if last_T_index == -1:
        print(f"LAST T INDEX. No se encontraron archivos combinados en {target_dir}...")
        return None  # No se encontró un valor de T válido con archivos

    if is_test4 and (last_limit is None or last_range is None):
        print("No se encontraron valores de limit o range para los archivos más recientes.")
        return None

    print(f"Últimos valores encontrados: it={last_it}, dataset={last_data}, s={last_s}, T={T_values[last_T_index]}, limit={last_limit}, range={last_range}")
    if is_test4:
        return last_it, last_data, last_s, T_values[last_T_index], last_limit, last_range
    else:
        return last_it, last_data, last_s, T_values[last_T_index]



def process_combinations_from_latest(is_test4=False):

    combinacion = check_latest(is_test4)
    try:
        if is_test4:
            last_it, last_data, last_s, last_T, last_limit, last_range = combinacion
        else:
            last_it, last_data, last_s, last_T = combinacion
    except:
        print("No hay archivos previos")
        last_it = None
        last_data = None
        last_s = None
        last_T = None
        if is_test4:
            last_limit = None
            last_range = None

    if last_it is not None:
        start_it_index = it_range.index(last_it)
        start_dataset_index = datasets.index(last_data)
        start_s_index = s_values.index(last_s)
        start_T_index = T_values.index(last_T)
        if is_test4:
            start_lim_range_index = lim_range.index((last_limit, last_range))
        else:
            start_lim_range_index = 0
    else:
        start_it_index = 0
        start_dataset_index = 0
        start_s_index = 0
        start_T_index = 0
        start_lim_range_index = 0

    param_combinations = ParameterCombinations(
        it_range, datasets, s_values, T_values, lim_range,
        start_it=start_it_index, start_dataset=start_dataset_index, 
        start_s=start_s_index, start_T=start_T_index, 
        start_lim_range=start_lim_range_index, is_test4=is_test4
    )

    param_combinations.next()

    more_combinations = True
    while more_combinations:
        if is_test4:
            it, dataset, s, T, limit, range_ = param_combinations.get_current_parameters()
            pattern = f"result_{dataset}_s{s}_T{T}_limit{limit}_range{range_[0]}-{range_[1]}_it{it}_nodo(\\d+)\\.txt"
        else:
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
            # print(f"Faltan archivos para it={it}, dataset={dataset}, s={s}, T={T} en los nodos: {missing_nodes}")
            more_combinations = param_combinations.next()
            continue

        # Ordenar archivos para mantener el orden de los nodos
        files.sort(key=lambda x: int(re.search(r'nodo(\d+)', x).group(1)))

        contents = []
        for file_name in files:
            file_path = os.path.join(results_dir, file_name)
            with open(file_path, 'r') as file:
                contents.append(file.read().strip())

        final_content = "\n\n".join(contents) + "\n"

        new_file_name = f"result_{dataset}_s{s}_T{T}_it{it}.txt" if not is_test4 else f"result_{dataset}_s{s}_T{T}_limit{limit}_range{range_[0]}-{range_[1]}_it{it}.txt"
        new_file_path = os.path.join(target_dir, new_file_name)

        with open(new_file_path, 'w') as new_file:
            new_file.write(final_content)
        print(f"Archivo combinado creado: {new_file_path}")

        more_combinations = param_combinations.next()





process_combinations_from_latest(is_test4=True)