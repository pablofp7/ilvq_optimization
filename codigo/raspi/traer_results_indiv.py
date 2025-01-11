import subprocess
import os

# Configuración
raspis = {
    "nodo0": "192.168.1.145",  # Cambia por la IP del nodo 0
    # "nodo1": "192.168.1.102",  # Cambia por la IP del nodo 1
}

usuario = "pablo"  # Usuario en las Raspberry Pis
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


# ------------------------------------------------------------------------------------
# INSTRUCCIONES PARA CONFIGURAR AUTENTICACIÓN SIN CONTRASEÑA
# ------------------------------------------------------------------------------------
# Para evitar que rsync pida contraseña, configura la autenticación sin contraseña
# usando claves SSH. Sigue estos pasos:
#
# 1. Generar un par de claves SSH en tu PC central (si no lo has hecho ya):
#    Ejecuta en una terminal:
#      ssh-keygen -t rsa -b 4096
#    Presiona Enter para aceptar la ubicación predeterminada y no establezcas una frase
#    de contraseña.
#
# 2. Copiar la clave pública a cada Raspberry Pi:
#    Usa el comando ssh-copy-id para copiar la clave pública a cada Raspberry Pi:
#      ssh-copy-id pablo@192.168.1.101  # Para nodo0
#      ssh-copy-id pablo@192.168.1.102  # Para nodo1
#    Introduce la contraseña del usuario pablo en la Raspberry Pi cuando te lo pida.
#
# 3. Verificar la conexión sin contraseña:
#    Prueba la conexión SSH desde tu PC central a una Raspberry Pi:
#      ssh pablo@192.168.1.101
#    Si todo está configurado correctamente, no te pedirá la contraseña.
#
# ¡Listo! Ahora el script rsync funcionará sin pedir contraseña.
# ------------------------------------------------------------------------------------