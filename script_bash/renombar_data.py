import os

# Define el directorio donde se encuentran los archivos
directory = '/home/pablo/trabajo/codigo/new_exp/resultados'

def rename_files():
    # Lista todos los archivos en el directorio
    for filename in os.listdir(directory):
        # Asegúrate de que el archivo tiene el formato correcto antes de intentar cambiarle el nombre
        if filename.startswith("result_s") and filename.endswith(".txt"):
            # Divide el nombre del archivo en las partes necesarias
            prefix, rest = filename.split("_s", 1)
            # Crea el nuevo nombre del archivo
            new_filename = f"{prefix}_elec_s{rest}"
            # Define las rutas completas para el antiguo y el nuevo nombre del archivo
            old_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)
            # Cambia el nombre del archivo
            os.rename(old_filepath, new_filepath)

# Llama a la función para cambiar el nombre de los archivos
rename_files()
