
# Proyecto: Configuración y Ejecución en Raspberry Pi

## Nota Importante

Todo el proyecto debe ejecutarse desde el directorio `codigo/raspi/`, que contiene el código fuente y los scripts necesarios.

## Inicialización

Una vez instalado el sistema operativo en las Raspberry Pi y configurada la conexión a la red con el servidor SSH activo, sigue los siguientes pasos para preparar el entorno:

### 1. Configurar autenticación sin contraseña usando claves SSH

1.1. **Generar un par de claves SSH en el PC central** (si no se ha hecho previamente):

- Ejecuta en una terminal:
  ```bash
  ssh-keygen -t rsa -b 4096
  ```
- Presiona Enter para aceptar la ubicación predeterminada y no establezcas una frase de contraseña.

1.2. **Copiar la clave pública a cada Raspberry Pi**:

Usa el comando `ssh-copy-id` para copiar la clave pública a cada Raspberry Pi:

```bash
ssh-copy-id raspi-user@raspiHostname
```

Introduce la contraseña del usuario cuando te lo solicite.

1.3. **Verificar la conexión sin contraseña**:

Prueba la conexión SSH desde tu PC central a una Raspberry Pi:

```bash
ssh raspi-user@raspiHostname
```

Si todo está configurado correctamente, no se solicitará la contraseña.

### 2. Clonar el repository

```bash
git clone https://github.com/pablofp7/ilvq_optimization
```

### 3. Crear el entorno virtual:

Ejecuta el script `init_venv.py` para crear un entorno virtual con las librerías y la versión de Python necesarias.

### 4. Conectar a la VPN:

Lanza el script `zerotier_installer.py` para conectar las Raspberry Pi a la VPN. El administrador debe aceptar la solicitud de unión para que las Raspberry Pi puedan comunicarse con el PC central a través de la red local.

### 5. Lanzar el Launcher en las Raspberry Pi:

Puedes ejecutar el launcher utilizando las siguientes opciones:

**Forma 1**: Para ejecutar el script en segundo plano y guardar la salida en un archivo de registro:

```bash
nohup python3 script.py > output.log 2>&1 &
```

- **nohup**: Evita que el proceso se detenga si cierras la sesión.
- **python3 script.py**: Ejecuta el script en Python 3. En este caso cualquiera de los launchers, p.ej. `python3 laucherv1.py`.
- **> output.log**: Redirige la salida estándar (stdout) a un archivo llamado `output.log`.
- **2>&1**: Redirige la salida de error (stderr) al mismo lugar que la salida estándar.
- **&**: Ejecuta el comando en segundo plano.

**Forma 2**: Si no necesitas guardar ninguna salida, descarta tanto la salida estándar como los errores:

```bash
nohup python3 script.py > /dev/null 2>&1 &
```

**Comprobación**: Para verificar que el script sigue ejecutándose, usa:

```bash
ps aux | grep script.py
```

---

## Clases de Modelos

En la carpeta `node_class` se encuentran las versiones de clases de nodo utilizadas en cada prueba:

1. `nodev1` -> **BaseTest**
2. `nodev2` -> **JSDTest**
3. `nodev3` -> **LimitQueueSizeTest**
4. `nodev4` -> **ClusteringTest**

En la carpeta `raspi_launcher` se encuentran los scripts lanzadores correspondientes a cada versión.

---

## Generación de Gráficos (Todos los Tests)

**Objetivo**: Crear gráficos para el dataset de electricidad y otros datasets.

**Ubicación**: Los gráficos generados se guardan en `graphics/new plot test results`.

### Modo de Uso:

- Para visualizar las gráficas de un test específico, ejecutar el script con el parámetro `-t testX`, donde `X` es el número del test.
- Para representar todas las gráficas agrupadas, usa el parámetro `-t all`:
  - **Dataset de Electricidad**: Los 4 gráficos se organizan en una cuadrícula de 2x2.
  - **Otros Datasets**: Generar gráficos para los otros dos dataset (un por columna), sin incluir la prueba de clustering (3 tests, una fila por cada test, resultando en un 3x2).

### Personalización:

- Modifica la variable `opcion` en la línea 500 del código para cambiar las configuraciones de visualización.
- Ajusta manualmente los rangos del eje `y` para una mejor visualización, pues para poder comparar visualmente entre tests deben de tener los mismos rangos de valores.

---

## Transferencia y Combinación de Resultados

### Traer Resultados desde las Raspberry Pi al PC

**Script**: `traer_resutls_indiv.py`

**Objetivo**: Copiar los resultados generados por cada Raspberry Pi al PC central.

**Detalles**:

- Los archivos generados por cada nodo se transfieren al PC para luego ser combinados usando el script `unir_csv.py`.

### Unificación de Resultados CSV

**Script**: `unir_csv.py`

**Objetivo**: Combinar los archivos CSV generados por cada nodo.

**Detalles**:

- Los archivos CSV se unen según parámetros comunes como `dataset`, `ejecucion`, `s`, `T`, `it`, etc.
- La unificación facilita el análisis de los datos.

---

## Nueva Clase de Nodo

**Carpeta**: `node_class`

**Objetivo**: Implementar versiones actualizadas de las clases de nodo.

**Detalles**:

- Cada versión corresponde a un número de prueba específico.
- Esta estructura mejora la organización y el mantenimiento en comparación con la implementación anterior.

---

## Nuevo Lanzador para Raspberry Pi

**Carpeta**: `raspi_launcher`

**Objetivo**: Facilitar la ejecución de pruebas y la recopilación de datos en formato CSV.

**Detalles**:

- Los lanzadores están diseñados para trabajar con las clases de nodo actualizadas.

---

## Visualización de Resultados CSV

**Script**: `plot_csv.py`

**Objetivo**: Generar gráficos a partir de los archivos CSV unificados.

**Detalles**:

- Este script adapta las funcionalidades de `plot_test_results` para trabajar con datos en formato CSV.
- Permite visualizar los resultados de las pruebas almacenadas en CSV.
- Para visualizar datos antiguos en formato TXT, usa `plot_test_results`.

---

# Resumen General

- **Inicialización**: Configura la autenticación SSH sin contraseña, crea un entorno virtual con `init_venv.py` y conecta las Raspberry Pi a la VPN usando `zerotier_installer.py`.
- **Clases de Modelos**: Las clases necesarias para las pruebas se encuentran en `node_class`.
- **Lanzadores**: Usa los scripts en `raspi_launcher` para ejecutar pruebas y generar resultados en CSV.
- **Generación de Gráficos**: Emplea `plot_test_results` para datos TXT y `plot_csv.py` para CSV.
- **Transferencia y Unificación de Resultados**: Copia resultados con `traer_resutls_indiv.py` y unifícalos con `unir_csv.py`.
