import os
import re
import sys
import pandas as pd

class ParameterCombinations:
    def __init__(self, it_range, datasets, s_values, T_values, lim_range,
                 start_it=0, start_dataset=0, start_s=0, start_T=0, start_lim_range=0, is_test4=False):
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
                        return False  # Se completaron todas las combinaciones
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
    if not test:
        raise Exception("Falta definir el test.")
except Exception as e:
    print(f"Debes indicar el test (ilvq, vfdt, nb, etc.). Error: {e}")
    exit(1)

if "test4" in test:
    is_test4 = True
else:
    is_test4 = False
    
results_dir = 'resultados_raspi_indiv'
if "vfdt" in test:
    results_dir += "_tree"
if "nb" in test:
    results_dir += "_nb"

target_dir = f'{test}_resultados'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

N_NODOS = 5
it_range = range(50)  # 0 a 49
datasets = ["elec", "phis", "elec2", "lgr"] #, "nrr", "lar", "lrr", "ngcr", "nsch"]
s_values = [1, 2, 3, 4]
T_values = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4,
            0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0]
T_values = T_values[::-1]
nodos = list(range(N_NODOS))
lim_range = [
    (50, (50, 60)),
    (150, (50, 60)),
    (250, (50, 60)),
    (500, (72.5, 77.5))
]
all_files = os.listdir(results_dir)
all_combined = os.listdir(target_dir)


def check_latest_for_dataset(dataset, is_test4=False):
    """
    Busca la última combinación ya combinada para el dataset dado (filtrando por su subcadena en el nombre).
    Si no se encuentra nada, se devuelve None, lo que indicará que es un dataset nuevo.
    """
    last_it = -1
    last_s = -1
    last_T_index = -1  # índice en T_values
    last_limit = None
    last_range = None

    # Buscar la última iteración con archivos COMBINADOS para ESTE dataset
    for it in it_range:
        pattern = f"_it{it}"
        cp = re.compile(pattern)
        files = [f for f in all_combined if cp.search(f) and f"_{dataset}_" in f]
        if files:
            last_it = it
        else:
            break
    if last_it == -1:
        print(f"ITERACION: No se encontraron archivos combinados para {dataset} en {target_dir}...")
        return None

    # Buscar el último valor de s para ESTE dataset en la iteración encontrada
    for s in s_values:
        pattern = f"_{dataset}_s{s}_T"
        cp = re.compile(pattern)
        files = [f for f in all_combined if cp.search(f) and f"it{last_it}" in f]
        if files:
            last_s = s
        else:
            break

    # Buscar el último valor de T (y si es test4, también limit y range)
    for i, T in enumerate(T_values):
        if is_test4:
            for limit, rng in lim_range:
                pattern = f"result_{dataset}_s{last_s}_T{T}_limit{limit}_range{rng[0]}-{rng[1]}_it{last_it}"
                if any(pattern in f for f in all_combined):
                    last_T_index = i
                    last_limit = limit
                    last_range = rng
                    break
            if last_T_index != -1:
                break
        else:
            pattern = f"_T{T}_"
            cp = re.compile(pattern)
            files = [f for f in all_combined if cp.search(f)
                     and f"it{last_it}" in f and f"_{dataset}_" in f and f"_s{last_s}_" in f]
            if files:
                last_T_index = i
                break

    if last_T_index == -1:
        print(f"LAST T INDEX: No se encontraron archivos combinados para {dataset} en {target_dir}...")
        return None

    print(f"Últimos valores para {dataset}: it={last_it}, s={last_s}, T={T_values[last_T_index]}, limit={last_limit}, range={last_range}")
    if is_test4:
        return last_it, dataset, last_s, T_values[last_T_index], last_limit, last_range
    else:
        return last_it, dataset, last_s, T_values[last_T_index]


def process_combinations_from_latest(is_test4=False):
    # Procesamos cada dataset de forma independiente
    for dataset in datasets:
        combinacion = check_latest_for_dataset(dataset, is_test4)
        if combinacion is not None:
            if is_test4:
                last_it, last_data, last_s, last_T, last_limit, last_range = combinacion
            else:
                last_it, last_data, last_s, last_T = combinacion
        else:
            print(f"No hay archivos previos para {dataset}. Se procesará desde cero.")
            last_it = None
            last_s = None
            last_T = None
            if is_test4:
                last_limit = None
                last_range = None

        # Si se encontró información previa, arrancamos desde esos índices; si no, desde cero.
        if last_it is not None:
            start_it_index = it_range.index(last_it)
            start_s_index = s_values.index(last_s)
            start_T_index = T_values.index(last_T)
            start_dataset_index = 0  # Solo se procesa un dataset en cada instancia
            if is_test4:
                start_lim_range_index = lim_range.index((last_limit, last_range))
            else:
                start_lim_range_index = 0
        else:
            start_it_index = 0
            start_s_index = 0
            start_T_index = 0
            start_dataset_index = 0
            start_lim_range_index = 0

        param_combinations = ParameterCombinations(
            it_range, [dataset], s_values, T_values, lim_range,
            start_it=start_it_index, start_dataset=start_dataset_index,
            start_s=start_s_index, start_T=start_T_index,
            start_lim_range=start_lim_range_index, is_test4=is_test4
        )
        # Si ya había archivos previos para este dataset, avanzamos para no repetir la última combinación.
        if last_it is not None:
            param_combinations.next()

        more_combinations = True
        while more_combinations:
            if is_test4:
                it, ds, s, T, limit, rng = param_combinations.get_current_parameters()
                pattern = f"result_{dataset}_s{s}_T{T}_limit{limit}_range{rng[0]}-{rng[1]}_it{it}_nodo(\\d+)\\.csv"
            else:
                it, ds, s, T = param_combinations.get_current_parameters()
                pattern = f"result_{dataset}_s{s}_T{T}_it{it}_nodo(\\d+)\\.csv"
            cp = re.compile(pattern)
            files = [f for f in all_files if cp.match(f)]

            # Si no se encuentran archivos y la iteración es mayor a 19, saltamos esta combinación.
            if not files and it > 19:
                more_combinations = param_combinations.next()
                continue

            # Verificar que se tengan archivos de todos los nodos
            nodes_found = [int(cp.search(f).group(1)) for f in files]
            missing_nodes = [n for n in range(N_NODOS) if n not in nodes_found]
            if missing_nodes:
                print(f"Faltan archivos para it={it}, dataset={dataset}, s={s}, T={T} en nodos: {missing_nodes}")
                more_combinations = param_combinations.next()
                continue

            # Ordenar archivos y combinarlos usando pandas
            files.sort(key=lambda x: int(re.search(r'nodo(\d+)', x).group(1)))
            dfs = [pd.read_csv(os.path.join(results_dir, f_name)) for f_name in files]
            combined_df = pd.concat(dfs, ignore_index=True)

            if is_test4:
                new_file_name = f"result_{dataset}_s{s}_T{T}_limit{limit}_range{rng[0]}-{rng[1]}_it{it}.csv"
            else:
                new_file_name = f"result_{dataset}_s{s}_T{T}_it{it}.csv"
            new_file_path = os.path.join(target_dir, new_file_name)
            combined_df.to_csv(new_file_path, index=False)
            print(f"Archivo combinado creado: {new_file_path}")

            more_combinations = param_combinations.next()


# Ejecutar el procesamiento
process_combinations_from_latest(is_test4)
