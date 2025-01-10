import os
import re
import sys
import csv

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
results_dir = 'resultados_raspi_indiv'
valid_tests = {'test1', 'test2', 'test3', 'test4'}
test = sys.argv[1] if len(sys.argv) > 1 else None
if test not in valid_tests:
    print(f"Test '{test}' no válido. Debe ser uno de: {', '.join(valid_tests)}.")
    sys.exit(1)
print(f"Test válido: {test}")

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
lim_range = [
    (50, (50, 60)),
    (150, (50, 60)),
    (250, (50, 60)),
    (500, (72.5, 77.5))
] 
all_files = os.listdir(results_dir)
all_combined = os.listdir(target_dir)