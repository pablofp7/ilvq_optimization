import os
import time
import subprocess
import sys
import fnmatch
from datetime import datetime

PROJECT_DIR = "/home/pablo/ilvq_optimization/codigo/raspi"
LAUNCHER_DIR = os.path.join(PROJECT_DIR, "raspi_launcher")
RESULTS_DIR = os.path.join(PROJECT_DIR, "resultados_raspi_indiv")
VENV_PATH = "/home/pablo/.pyenv/versions/3.10.12/envs/raspi_env/bin/activate"

LAST_ITERATION_FILE = "result_elec2_s4_T1.0_it49_nodo*.csv"

LAST_FILE_CHECKED = None

def get_script_name():
    """Retrieve script name from command-line arguments."""
    if len(sys.argv) < 2:
        print("Error: No script name provided. Usage: python super_launcher.py <script_suffix>")
        sys.exit(1)

    script_suffix = sys.argv[1]
    script_name = f"{script_suffix}.py"
    script_path = os.path.join(LAUNCHER_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"Error: Script '{script_name}' not found in {LAUNCHER_DIR}")
        sys.exit(1)

    return script_name

def get_latest_file_info():
    """Returns the last modified file's name and timestamp from the results folder."""
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
        print(f"Error retrieving latest file: {e}")
        return None, None

def time_difference_in_minutes(file_time):
    """Calculate time difference between current time and file timestamp."""
    if not file_time:
        return float('inf') 
    now = datetime.now()
    return (now - file_time).total_seconds() / 60 

def stop_simulation(script_name):
    """Kills the current simulation and does NOT restart."""
    print(f"Final iteration detected ({LAST_ITERATION_FILE}). Stopping {script_name} permanently.")
    

    subprocess.run(f"pkill -f '{script_name}'", shell=True)


    sys.exit(0)

def restart_simulation(script_name):
    """Kills the current simulation and starts a new one."""
    global LAST_FILE_CHECKED

    print(f"Restarting simulation {script_name}...")
    

    subprocess.run(f"pkill -f '{script_name}'", shell=True)


    command = f"""
    cd '{LAUNCHER_DIR}' &&
    source '{VENV_PATH}' &&
    nohup python3 {script_name} > /dev/null 2>&1 &
    """
    subprocess.run(command, shell=True, executable="/bin/bash")

    print(f"Simulation {script_name} restarted.")

def main():
    global LAST_FILE_CHECKED


    script_name = get_script_name()


    restart_simulation(script_name)

    while True:
        time.sleep(300) 

        latest_filename, latest_file_time = get_latest_file_info()
        if not latest_filename or not latest_file_time:
            print("No results found, restarting simulation.")
            restart_simulation(script_name)
            continue


        if fnmatch.fnmatch(latest_filename, LAST_ITERATION_FILE):
            stop_simulation(script_name)

        if latest_filename == LAST_FILE_CHECKED:
            print(f"Latest file {latest_filename} already processed. No restart needed.")
            continue

        minutes_since_last_update = time_difference_in_minutes(latest_file_time)

        if minutes_since_last_update >= 5:
            print(f"No new results for {minutes_since_last_update:.1f} minutes. Restarting simulation.")
            restart_simulation(script_name)
            LAST_FILE_CHECKED = latest_filename 
        else:
            print(f"Last result updated {minutes_since_last_update:.1f} minutes ago. No restart needed.")

if __name__ == "__main__":
    main()
