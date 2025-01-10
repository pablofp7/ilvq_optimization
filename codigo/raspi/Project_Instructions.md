
# Instrucciones del Proyecto

## Resultados de los Plots (Todos)
- **Objetivo**: Generar gráficos para el dataset de electricidad y otros datasets.
- **Ubicación**: `graphics/new plot test results`
- Para usar como antiguamente, con la pequeña interfaz gráfica y demás, llamar con parámetro -t testX, dónde X es el test del que quieres ver las gráficas.
- Para representar las gráficas agrupadas:
  - Usar el parámetro -t all
  - Para el **dataset de electricidad**, usar los 4 gráficos de prueba organizados en una cuadrícula de (2,2).
  - Para los otros dos datasets, generar gráficos para las dos columnas sin añadir la prueba de clustering.
  - **Cómo cambiar opciones**: Modificar el valor de `opcion` en la **línea 500 aproximadmente** del código para seleccionar entre estas configuraciones.
  - Para los rangos de los valores del eje y, ajustar manualmente para que se ajuste bien dentro de cada opción.


## Unir Resultados CSV
- unir_csv.py
- Cada nodo genera sus resultados en un csv estos hay que unirlos agrupados según parametros (dataset,ejecucion,s,T,it...).
- **Objetivo**: Unir los archivos CSV que contienen los resultados individuales generados por cada nodo en la Raspberry Pi.

## Nueva Clase de Nodo
- new_node_class
- **Objetivo**: Implementar versiones actualizadas de las clases de nodo.
- **Detalles**:
  - Cada versión corresponde a un número de prueba específico.
  - Esta estructura reemplaza la implementación previa que era caótica.

## Nuevo Lanzador para Raspberry Pi
- new_raspi_launcher
- **Objetivo**: Crear nuevos lanzadores para las clases de nodo actualizadas.
- **Detalles**:
  - Los lanzadores utilizan archivos CSV para almacenar los resultados de cada prueba.
