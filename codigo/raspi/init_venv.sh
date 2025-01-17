#!/bin/bash

# Ruta base del proyecto
PROJECT_DIR="$HOME/ilvq_optimization/codigo/raspi"

# Instalar dependencias necesarias para pyenv
echo "Instalando dependencias para pyenv..."
sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
liblzma-dev python3-openssl git

# Instalar pyenv (si no está instalado)
if ! command -v pyenv &> /dev/null
then
    echo "pyenv no está instalado. Instalando pyenv..."
    curl https://pyenv.run | bash

    # Configurar pyenv en el shell
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"

    # Añadir configuraciones a .bashrc
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
    source ~/.bashrc
fi

# Instalar Python 3.10.12
echo "Instalando Python 3.10.12..."
pyenv install -s 3.10.12  # -s evita reinstalar si ya está instalado

# Crear un entorno virtual con Python 3.10.12
echo "Creando entorno virtual en $PROJECT_DIR/venv..."
pyenv virtualenv 3.10.12 raspi_env || echo "El entorno virtual ya existe."

# Activar el entorno virtual
echo "Activando entorno virtual 'raspi_env'..."
eval "$(pyenv init -)"
pyenv activate raspi_env

# Instalar pip 22.0.2
echo "Instalando pip 22.0.2..."
pip install --upgrade pip==22.0.2

# Instalar dependencias desde requirements.txt (si existe)
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    echo "Instalando dependencias desde requirements.txt..."
    pip install -r "$PROJECT_DIR/requirements.txt"
else
    echo "No se encontró requirements.txt en $PROJECT_DIR. Omitiendo instalación de dependencias."
fi

# Desactivar el entorno virtual
pyenv deactivate

# Instalar direnv (si no está instalado)
if ! command -v direnv &> /dev/null
then
    echo "direnv no está instalado. Instalando direnv..."
    sudo apt-get install -y direnv
    echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
    source ~/.bashrc
fi

# Crear un archivo .envrc para activar automáticamente el entorno virtual
echo "Configurando activación automática del entorno virtual..."
echo -e 'export VIRTUAL_ENV="/home/pablo/.pyenv/versions/3.10.12/envs/raspi_env"\nexport PATH="$VIRTUAL_ENV/bin:$PATH"' > ~/ilvq_optimization/codigo/raspi/.envrc
direnv allow "$PROJECT_DIR"

echo "¡Configuración completada!"
echo "El entorno virtual 'raspi_env' con Python 3.10.12 y pip 22.0.2 está listo."
echo "Al entrar en el directorio $PROJECT_DIR, el entorno virtual se activará automáticamente."
