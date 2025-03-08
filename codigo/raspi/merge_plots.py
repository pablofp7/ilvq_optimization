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
        if dataset in combo["datasets"]:
            tests_str = "-".join(str(t) for t in tests)
            filename = f"{metric}_tests{tests_str}_{dataset}.png"
            asignacion[(i + 1, j + 1)] = filename  # Índices 1-based

# Factores de escala para ajustar dinámicamente el tamaño de los subplots
base_width = 6  
base_height = 3  

# Calcular el tamaño de la figura
fig_width = max(ncols * base_width, 6)
fig_width *= 1.5
fig_height = max(nrows * base_height, 4)

# Crear la figura y los subplots
fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
if nrows == 1 or ncols == 1:
    axes = axes.reshape(nrows, ncols)

# Añadir títulos a las columnas
for col in range(ncols):
    axes[0, col].set_title(column_titles[col], fontsize=12, fontweight="bold", pad=20)  # Reducido el pad

# Ajustar espacios entre subplots: eliminar espacios entre ellos
plt.subplots_adjust(wspace=0, hspace=0, top=0.85)

# Colocar cada imagen en la posición asignada
for (fila, col), archivo in asignacion.items():
    ax = axes[fila - 1, col - 1]
    image_path = os.path.join(directorio_imagenes, archivo)
    if os.path.exists(image_path):
        img = mpimg.imread(image_path)
        ax.imshow(img, aspect="auto")  # Permitir ajuste automático de aspecto
    ax.set_xticks([])
    ax.set_yticks([])

    # Quitar el borde (spines) alrededor del subplot
    for spine in ax.spines.values():
        spine.set_visible(False)

# Ocultar subplots vacíos
for i in range(nrows):
    for j in range(ncols):
        if (i + 1, j + 1) not in asignacion:
            axes[i, j].axis("off")

plt.show()
