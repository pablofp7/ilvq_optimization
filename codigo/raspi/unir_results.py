import os
import re

# Directory containing the result files and target directory for new files
results_dir = 'resultados_raspi_indiv'
target_dir = 'Test1/'

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

datasets = ['elec', 'phis', 'elec2']
s_values = [1, 2, 3, 4]
T_values = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0]
it_range = range(50)  # From 0 to 49

def process_combinations():
    for dataset in datasets:
        for s in s_values:
            for T in T_values:
                for it in it_range:
                    pattern = f"result_{dataset}_s{s}_T{T}_it{it}_nodo\\d+\\.txt"
                    compiled_pattern = re.compile(pattern)
                    files = [f for f in os.listdir(results_dir) if compiled_pattern.match(f)]

                    # If no files found for the current `it`, break the loop (assuming files are sequentially numbered)
                    if not files:
                        if it > 19:  # If `it` goes beyond 19 and no files are found, stop searching for higher `it` values
                            break
                        else:
                            continue  # If `it` is 19 or below, it's possible we're just missing this particular `it`, so keep going

                    # Sort files to ensure node order
                    files.sort()

                    contents = []
                    for file_name in files:
                        node_number = re.search(r'_nodo(\d+)', file_name).group(1)
                        file_path = os.path.join(results_dir, file_name)
                        with open(file_path, 'r') as file:
                            file_content = file.read().strip()
                            contents.append(f" - NODO {node_number}.\n{file_content}")

                    final_content = "\n\n".join(contents)

                    new_file_name = f"result_comb_{dataset}_s{s}_T{T}_it{it}.txt"
                    new_file_path = os.path.join(target_dir, new_file_name)

                    with open(new_file_path, 'w') as new_file:
                        new_file.write(final_content)
                    print(f"Combined file created: {new_file_path}")

# Call the function to process all combinations
process_combinations()
