import subprocess
import os
import sys

# Check if the first argument is provided
if len(sys.argv) < 2:
    raise ValueError("The first argument is required. ilvq, tree (vfdt), nb ...")

arg = sys.argv[1]

# Configuración
raspis = {
    "nodo0": "nodo0.local",
    "nodo1": "nodo1.local",
    "nodo2": "nodo2.local",
    "nodo3": "nodo3.local",
    "nodo4": "nodo4.local",
} 

usuario = "pablo"  # Usuario en las Raspberry Pis

# Modify directories based on the argument
if arg != "ilvq":
    directorio_remoto = f"/home/pablo/ilvq_optimization/codigo/raspi/resultados_raspi_indiv_{arg}"
    directorio_local = f"/home/pablo/ilvq_optimization/codigo/raspi/resultados_raspi_indiv_{arg}"
else:
    directorio_remoto = "/home/pablo/ilvq_optimization/codigo/raspi/resultados_raspi_indiv"
    directorio_local = "/home/pablo/ilvq_optimization/codigo/raspi/resultados_raspi_indiv"

# Crear el directorio local si no existe
os.makedirs(directorio_local, exist_ok=True)

# Función para sincronizar archivos desde una Raspberry Pi
def sincronizar_nodo(nodo, ip):
    print(f"Sincronizando archivos desde {nodo} ({ip})...")

    # Comando rsync
    comando = [
        "rsync",
        "-avz",
        "--progress",
        f"{usuario}@{ip}:{directorio_remoto}/",
        f"{directorio_local}/",
    ]

    # Ejecutar el comando
    try:
        subprocess.run(comando, check=True)
        print(f"Sincronización completada para {nodo}.")
    except subprocess.CalledProcessError as e:
        print(f"Error al sincronizar archivos desde {nodo}: {e}")

# Sincronizar archivos desde cada Raspberry Pi
for nodo, ip in raspis.items():
    sincronizar_nodo(nodo, ip)

print("Proceso de sincronización terminado.")
