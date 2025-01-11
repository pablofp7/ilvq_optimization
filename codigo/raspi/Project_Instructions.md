
# Instrucciones del Proyecto

## Generación de Gráficos (Todos los Tests)
**Objetivo**: Crear gráficos para el dataset de electricidad y otros datasets.

**Ubicación**: Los gráficos generados se guardan en `graphics/new plot test results`.

### Modo de Uso:

- Para utilizar la interfaz gráfica tradicional y visualizar las gráficas de un test específico, ejecutar el script con el parámetro `-t testX`, donde `X` es el número del test que deseas visualizar.

- Para representar todas las gráficas agrupadas:

  - Usar el parámetro `-t all`.

  - **Dataset de Electricidad**: Los 4 gráficos de prueba se organizan en una cuadrícula de (2,2).

  - **Otros Datasets**: Generar gráficos para las dos columnas sin incluir la prueba de clustering.


### Personalización:
- Para cambiar las opciones de visualización, modificar el valor de `opcion` en la línea 500 del código. Esta variable permite seleccionar entre diferentes configuraciones de gráficos.
- Ajustar manualmente los rangos de los valores del eje `y` para que los gráficos se vean correctamente en cada configuración.

## Unificación de Resultados CSV
**Script**: `unir_csv.py`

**Objetivo**: Unificar los archivos CSV generados por cada nodo en la Raspberry Pi.

### Detalles:
- Cada nodo genera un archivo CSV con sus resultados. Estos archivos deben unirse en función de parámetros comunes como `dataset`, `ejecucion`, `s`, `T`, `it`, etc.
- El script agrupa los resultados para facilitar su análisis posterior.

## Nueva Clase de Nodo
**Carpeta**: `new_node_class`

**Objetivo**: Implementar versiones actualizadas de las clases de nodo.

### Detalles:
- Cada versión de la clase de nodo corresponde a un número de prueba específico.
- Esta nueva estructura reemplaza la implementación anterior, que era menos organizada y más difícil de mantener.

## Nuevo Lanzador para Raspberry Pi
**Carpeta**: `new_raspi_launcher`

**Objetivo**: Crear nuevos lanzadores para las clases de nodo actualizadas.

### Detalles:
- Los lanzadores utilizan archivos CSV para almacenar los resultados de cada prueba.
- Esta estructura facilita la ejecución de pruebas en la Raspberry Pi y la recopilación de datos.

## Visualización de Resultados CSV
**Script**: `plot_result_csv.py`

**Objetivo**: Generar gráficos a partir de los archivos CSV unificados.

### Detalles:
- Este script es una adaptación de `new_plot_results` para trabajar con archivos CSV.
- Permite visualizar los resultados de las pruebas almacenadas en formato CSV.
- Las opciones de visualización (como la selección de tests y la configuración de gráficos) siguen siendo las mismas que en `new_plot_results`.
- Para visualizar datos antiguos en formato TXT, seguir utilizando `new_plot_results` como se indicó anteriormente.

## Resumen
- **Gráficos**: Usar `new_plot_results` para generar gráficos a partir de datos en formato TXT o `plot_result_csv.py` para datos en formato CSV.

- **Unificación de CSV**: Usar `unir_csv.py` para combinar los resultados de los nodos.

- **Nuevas Clases de Nodo**: Implementar y utilizar las clases actualizadas en `new_node_class`.

- **Lanzadores en Raspberry Pi**: Usar los nuevos lanzadores en `new_raspi_launcher` para ejecutar pruebas y almacenar resultados en CSV.
