import subprocess
import sys

N_NODOS = 5  # Número de procesos a lanzar

procesos = []  # Lista para almacenar los objetos de proceso

for i in range(N_NODOS):
    # Lanza un nuevo proceso sin bloquear
    # Redirige stdout y stderr a subprocess.PIPE para capturarlos si es necesario
    proc = subprocess.Popen(["python3", "nodo.py", str(i)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    procesos.append(proc)  # Almacena el objeto de proceso para su posterior gestión

# Opcional: Código para manejar la salida de los procesos después de lanzarlos todos
for i, proc in enumerate(procesos):
    stdout, stderr = proc.communicate()  # Esto esperará a que cada proceso termine
    print(f"Proceso {i} terminado. Salida:\n{stdout.decode()}")
    if stderr:
        print(f"Errores:\n{stderr.decode()}", file=sys.stderr)
