#!/bin/bash


NETWORK_ID="${ZT_NETWORK_ID}"  


function show_error() {
  echo -e "====================================================\n"
  echo -e "¡Error! $1 \n"
  echo -e "====================================================\n"
  echo "$2"
  echo ""
  echo "Ejemplo de uso:"
  echo "ZT_NETWORK_ID=abc123 sudo ./zerotier_installer.sh"
  echo ""
  echo "Reemplaza 'abc123' con tu Network ID de ZeroTier."
  exit 1
}


if [ "$EUID" -ne 0 ]; then
  show_error "Este script debe ejecutarse con permisos de superusuario (sudo)."
fi


if [ -z "$NETWORK_ID" ]; then
  show_error "La variable ZT_NETWORK_ID no está definida."
fi


echo "Actualizando el sistema..."
apt-get update && apt-get upgrade -y


echo "Instalando dependencias..."
apt-get install -y curl


echo "Instalando ZeroTier..."
curl -s https://install.zerotier.com | bash


echo "Iniciando el servicio de ZeroTier..."
systemctl start zerotier-one
systemctl enable zerotier-one


echo "Uniéndose a la red ZeroTier con Network ID: $NETWORK_ID..."
zerotier-cli join $NETWORK_ID


echo "Mostrando el estado de ZeroTier..."
zerotier-cli status


echo "Obteniendo la dirección IP asignada por ZeroTier..."
zerotier-cli listnetworks


echo "¡Instalación y configuración completadas!"
echo "Por favor, autoriza este dispositivo en ZeroTier Central: https://my.zerotier.com/"
echo "Network ID: $NETWORK_ID"