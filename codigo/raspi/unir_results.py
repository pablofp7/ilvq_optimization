import os
import re

# Directorios de origen y destino
results_dir = 'resultados_raspi_indiv'
target_dir = 'test1_resultados'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

datasets = ['elec', 'phis', 'elec2']
s_values = [1, 2, 3, 4]
T_values = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0]
it_range = range(50)  # Desde 0 hasta 49

def process_combinations():
    for dataset in datasets:
        for s in s_values:
            for T in T_values:
                for it in it_range:
                    pattern = f"result_{dataset}_s{s}_T{T}_it{it}_nodo(\\d+)\\.txt"
                    compiled_pattern = re.compile(pattern)
                    files = [f for f in os.listdir(results_dir) if compiled_pattern.match(f)]

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

# Ejecutar la función para procesar todas las combinaciones
process_combinations()
