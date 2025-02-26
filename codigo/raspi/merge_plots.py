import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Directorio donde están las imágenes
directorio_imagenes = "../plottedResults/"

# Combinaciones de tests y datasets
combinations = [
    {"tests": [1], "datasets": ["elec", "lgr"]},
    {"tests": [2], "datasets": ["elec", "lgr"]},
    {"tests": [3], "datasets": ["elec", "lgr"]},
    {"tests": [4], "datasets": ["elec", "lgr"]},
]

# Inferir datasets automáticamente de las combinaciones
datasets = sorted(set(dataset for combo in combinations for dataset in combo["datasets"]))

# Inferir número de tests automáticamente
nrows = len(combinations)  # Una fila por test
ncols = len(datasets)      # Una columna por dataset

# Títulos de las columnas (usando nombres formateados)
column_titles = ["Electricity Fixed" if ds == "elec" else "Linear Gradual Rotation" for ds in datasets]

# Métrica seleccionada
metric = "trained_protos"

# Generar el diccionario de asignación de imágenes
asignacion = {}
for i, combo in enumerate(combinations):
    tests = combo["tests"]
    for j, dataset in enumerate(datasets):  # Iterar sobre los datasets inferidos
        if dataset in combo["datasets"]:  # Solo asignar si el dataset está en la combinación
            tests_str = "-".join(str(t) for t in tests)
            filename = f"{metric}_tests{tests_str}_{dataset}.png"
            asignacion[(i + 1, j + 1)] = filename  # Índices 1-based

# Factores de escala para ajustar dinámicamente el tamaño de los subplots
base_width = 4   
base_height = 3  

# Calcular el tamaño de la figura dinámicamente
fig_width = max(ncols * base_width * 0.8, 6)
fig_height = max(nrows * base_height * 0.8, 4)

# Crear la figura y los subplots con `constrained_layout=True`
fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), constrained_layout=True)

# Asegurar que `axes` sea un array bidimensional
if nrows == 1 or ncols == 1:
    axes = axes.flatten()

# Añadir títulos a las columnas con más espacio para evitar que se corten
for col in range(ncols):
    axes[0, col].set_title(column_titles[col], fontsize=12, fontweight="bold", pad=30)  # Aumentamos `pad`

# Ajustar espacio en la parte superior para que los títulos no se corten
plt.subplots_adjust(top=0.88)  # Ajusta este valor si sigue cortándose

# Colocar cada imagen en la posición asignada
for (fila, col), archivo in asignacion.items():
    ax = axes[fila - 1, col - 1] if isinstance(axes, list) else axes[fila - 1, col - 1]
    
    # Verificar si el archivo existe antes de cargarlo
    image_path = os.path.join(directorio_imagenes, archivo)
    if os.path.exists(image_path):
        img = mpimg.imread(image_path)
        ax.imshow(img)
    
    # No quitar los títulos ni ejes para mantener la apariencia original
    ax.set_xticks([])
    ax.set_yticks([])

# Ocultar subplots vacíos
for i in range(nrows):
    for j in range(ncols):
        if (i + 1, j + 1) not in asignacion:
            axes[i, j].axis("off")

plt.show()
