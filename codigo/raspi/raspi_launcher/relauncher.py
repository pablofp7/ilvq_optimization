#!/usr/bin/env python3
import os
import time
import subprocess
import sys
import fnmatch
from datetime import datetime

# Directorios y rutas de configuración
PROJECT_DIR   = "/home/pablo/ilvq_optimization/codigo/raspi"
LAUNCHER_DIR  = os.path.join(PROJECT_DIR, "raspi_launcher")
RESULTS_DIR   = os.path.join(PROJECT_DIR, "resultados_raspi_indiv")
VENV_PATH     = "/home/pablo/.pyenv/versions/3.10.12/envs/raspi_env/bin/activate"

# Patrón para detectar el archivo de iteración final (comodín)
# LAST_ITERATION_FILE = "result_elec2_s4_T1.0_it49_nodo*.csv"
LAST_ITERATION_FILE = "result_lgr_s4_T1.0_it49_nodo*.csv"


# Variable para llevar el seguimiento del último archivo procesado
LAST_FILE_CHECKED = None

def get_script_name():
    """Recupera el nombre del script a ejecutar a partir de los argumentos de línea de comandos."""
    if len(sys.argv) < 2:
        print("Error: No se proporcionó el nombre del script. Uso: python super_launcher.py <prefijo_script>")
        sys.exit(1)

    script_prefix = sys.argv[1]
    script_name = f"{script_prefix}.py"
    script_path = os.path.join(LAUNCHER_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"Error: Script '{script_name}' no se encontró en {LAUNCHER_DIR}")
        sys.exit(1)

    return script_name

def get_latest_file_info():
    """
    Devuelve el nombre y la marca de tiempo del último archivo modificado en el directorio de resultados.
    Se utiliza el comando 'ls -lt' para obtener el archivo más reciente.
    """
    try:
        result = subprocess.run(
            f'ls -lt "{RESULTS_DIR}" | head -n 2',
            shell=True, text=True, capture_output=True, check=True
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            return None, None

        parts = lines[1].split()
        filename = parts[-1]
        file_time_str = " ".join(parts[-4:-1])
        file_time = datetime.strptime(file_time_str, "%b %d %H:%M")
        file_time = file_time.replace(year=datetime.now().year)

        return filename, file_time

    except Exception as e:
        print(f"Error al obtener el último archivo: {e}")
        return None, None

def time_difference_in_minutes(file_time):
    """Calcula la diferencia en minutos entre el tiempo actual y la marca de tiempo del archivo."""
    if not file_time:
        return float('inf')
    now = datetime.now()
    return (now - file_time).total_seconds() / 60

def stop_simulation(script_name):
    """
    Si se detecta el archivo final, se mata el proceso de la simulación y se termina el script de supervisión.
    """
    print(f"Iteración final detectada ({LAST_ITERATION_FILE}). Deteniendo {script_name} permanentemente.")
    subprocess.run(f"pkill -f '{script_name}'", shell=True)
    sys.exit(0)

def restart_simulation(script_name):
    """
    Mata el proceso actual de la simulación y lo relanza en segundo plano usando el entorno virtual.
    """
    global LAST_FILE_CHECKED

    print(f"Reiniciando la simulación {script_name}...")
    # Matar el proceso actual del script de simulación
    subprocess.run(f"pkill -f '{script_name}'", shell=True)

    # Construir y ejecutar el comando para lanzar el script en segundo plano
    command = f"""
    cd '{LAUNCHER_DIR}' &&
    source '{VENV_PATH}' &&
    nohup python3 {script_name} > /dev/null 2>&1 &
    """
    subprocess.run(command, shell=True, executable="/bin/bash")
    print(f"Simulación {script_name} reiniciada.")

def main():
    global LAST_FILE_CHECKED

    # Obtener el nombre del script a ejecutar (por ejemplo, 'mi_script.py')
    script_name = get_script_name()

    # Lanzar la simulación inicialmente
    restart_simulation(script_name)

    while True:
        time.sleep(300)  # Espera 5 minutos

        latest_filename, latest_file_time = get_latest_file_info()
        if not latest_filename or not latest_file_time:
            print("No se encontraron resultados, reiniciando la simulación.")
            restart_simulation(script_name)
            continue

        # Si el último archivo coincide con el patrón de iteración final, detener la simulación
        if fnmatch.fnmatch(latest_filename, LAST_ITERATION_FILE):
            stop_simulation(script_name)

        # Evitar reiniciar si ya se procesó el último archivo
        if latest_filename == LAST_FILE_CHECKED:
            print(f"El último archivo {latest_filename} ya fue procesado. No se necesita reiniciar.")
            continue

        minutes_since_last_update = time_difference_in_minutes(latest_file_time)
        if minutes_since_last_update >= 5:
            print(f"No se han generado nuevos resultados por {minutes_since_last_update:.1f} minutos. Reiniciando la simulación.")
            restart_simulation(script_name)
            LAST_FILE_CHECKED = latest_filename
        else:
            print(f"Última actualización de resultados hace {minutes_since_last_update:.1f} minutos. No se reinicia.")

if __name__ == "__main__":
    main()
