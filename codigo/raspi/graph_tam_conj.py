import matplotlib.pyplot as plt
import numpy as np
import re
import os

def leer_datos(directorio_resultados, dataset_especifico, s_especifico, T_especifico):
    datos_filtrados = []
    for nombre_archivo in os.listdir(directorio_resultados):
        if nombre_archivo.endswith('.txt') and dataset_especifico in nombre_archivo:
            match = re.match(rf'result_{dataset_especifico}_s(\d+)_T([\d.]+)_it(\d+).txt', nombre_archivo)
            if match:
                s, T, _ = match.groups()
                if s == s_especifico and T == T_especifico:
                    ruta_archivo = os.path.join(directorio_resultados, nombre_archivo)
                    with open(ruta_archivo, 'r') as file:
                        for linea in file:
                            match_tamaño_prototipos = re.search(r'Tamaño conjunto de prototipos: \[(.*?)\]', linea)
                            if match_tamaño_prototipos:
                                tamaño_prototipos_str = match_tamaño_prototipos.group(1).split('), (')
                                datos_prototipos = [(int(x.split(', ')[0].strip('()')), int(x.split(', ')[1].strip('()'))) for x in tamaño_prototipos_str]
                                datos_filtrados.extend(datos_prototipos)
    return datos_filtrados

def calcular_promedios(datos_filtrados):
    datos_por_x = {}
    for x, y in datos_filtrados:
        if x not in datos_por_x:
            datos_por_x[x] = []
        datos_por_x[x].append(y)

    x_vals = sorted(datos_por_x.keys())
    y_proms = [np.mean(datos_por_x[x]) for x in x_vals]
    return x_vals, y_proms

def plotear(x_vals, y_proms, dataset_especifico, s_especifico, T_especifico):
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_proms, linestyle='-', color='b')
    plt.title(f'Tamaño del conjunto de prototipos para {dataset_especifico}, s={s_especifico} y T={T_especifico}')
    plt.xlabel('X')
    plt.ylabel('Promedio de Y')
    plt.grid(True)
    plt.show()

def main():
    directorio_resultados = "resultados_raspi"
    print("Valores posibles para el conjunto de datos: elec, phis, elec2")
    dataset_especifico = input("Ingrese el conjunto de datos específico (por ejemplo, 'elec'): ")
    print("Valores posibles para s: 1, 2, 3, 4")
    s_especifico = input("Ingrese el valor específico de s (por ejemplo, '1'): ")
    print("Valores posibles para T: 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0")
    T_especifico = input("Ingrese el valor específico de T (por ejemplo, '0.05'): ")

    datos_filtrados = leer_datos(directorio_resultados, dataset_especifico, s_especifico, T_especifico)
    x_vals, y_proms = calcular_promedios(datos_filtrados)
    plotear(x_vals, y_proms, dataset_especifico, s_especifico, T_especifico)

if __name__ == "__main__":
    main()
