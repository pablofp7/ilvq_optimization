import subprocess
import os

# Nombre del script principal (asumiendo que está en el mismo directorio)
script_name = "plot_final_results.py"

# Directorio donde se guardarán las imágenes
output_dir = os.path.join("..", "plottedResults")
os.makedirs(output_dir, exist_ok=True)

# Combinaciones solicitadas:
# - Test 1 y 4 con lgr.
# - Test 1 y 5 con elec y lgr.
# - Test 1 y 6 con elec y lgr.
# - Test 4 y 5 con elec y lgr.
# - Test 4 y 6 con elec y lgr.
combinations = [
    {"tests": [1, 4], "datasets": ["lgr"]},
    {"tests": [1, 5], "datasets": ["elec", "lgr"]},
    {"tests": [1, 6], "datasets": ["elec", "lgr"]},
    {"tests": [4, 5], "datasets": ["elec", "lgr"]},
    {"tests": [4, 6], "datasets": ["elec", "lgr"]},
]

# Métricas a utilizar
metrics = ["f1", "bandwidth", "protos"]

# Función para definir el nombre de la métrica en el fichero (para "protos" usamos "trained_protos")
def file_metric_name(metric):
    return "trained_protos" if metric == "protos" else metric

# Recorrer las combinaciones y llamar al script para cada una
for combo in combinations:
    tests = combo["tests"]
    tests_str = "-".join(str(t) for t in tests)
    for dataset in combo["datasets"]:
        for metric in metrics:
            # Nombre del fichero de salida: p.ej. f1_tests1-4_lgr.png
            image_file = f"{file_metric_name(metric)}_tests{tests_str}_{dataset}.png"
            image_path = os.path.join(output_dir, image_file)
            
            # Construir el comando:
            #   -t <tests>, -d <dataset>, -m <metric>,
            #   --plot_mode test_marker --markers --save_image <image_path>
            cmd = ["python3", script_name,
                   "-t"] + [str(t) for t in tests] + [
                   "-d", dataset,
                   "-m", metric,
                   "--plot_mode", "test_marker",
                   "--markers",
                   "--save_image", image_path
            ]
            
            print("Ejecutando comando:", " ".join(cmd))
            subprocess.run(cmd)
