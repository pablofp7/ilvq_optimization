#!/bin/bash

# Ruta del archivo de configuración de Netplan
CONFIG_FILE="/etc/netplan/60-wifi-config.yaml"

# Verificar si el archivo ya existe
if [ -f "$CONFIG_FILE" ]; then
    echo "El archivo $CONFIG_FILE ya existe. No se realizarán cambios."
    exit 1
fi

# Crear el archivo y escribir la configuración
echo "Creando el archivo $CONFIG_FILE y añadiendo la configuración..."

cat <<EOF | sudo tee "$CONFIG_FILE" > /dev/null
network:
  version: 2
  renderer: NetworkManager
  wifis:
    wlan0:
      access-points:
        TP-Link_6367:
          password: 89615256
      dhcp4: true
      optional: true
EOF

# Aplicar la configuración de Netplan
echo "Aplicando la configuración de Netplan..."
sudo netplan apply

echo "¡Configuración aplicada correctamente!"
