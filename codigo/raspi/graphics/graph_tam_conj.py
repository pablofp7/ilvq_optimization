import matplotlib.pyplot as plt
import numpy as np
import re
import os
import sys

def leer_datos(directorio_resultados, dataset_especifico, s_especifico, T_especifico, g_especifico=None):
    datos_filtrados = []
    for nombre_archivo in os.listdir(directorio_resultados):
        if nombre_archivo.endswith('.txt') and dataset_especifico in nombre_archivo:
            # Define la expresión regular base
            base_regex = rf'result_{dataset_especifico}_s{s_especifico}_T{T_especifico}'
            
            # Ajusta la expresión regular según si g_especifico es proporcionado o no
            if g_especifico:
                # Si g_especifico no es None, incluye en la expresión regular
                regex = base_regex + rf'_g{g_especifico}_it\d+.txt'
            else:
                # Si g_especifico es None, busca archivos sin el componente 'g'
                regex = base_regex + r'_it\d+.txt'
            
            match = re.match(regex, nombre_archivo)
            if match:
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

def plotear(x_vals, y_proms, dataset_especifico, s_especifico, T_especifico, g_especifico = None):
    
    plt.figure(figsize=(10, 6))
    
    # Plotear los datos originales
    plt.plot(x_vals, y_proms, 'o-', color='b', label='Original Data')  # 'o-' crea puntos con líneas
    
    # Ajuste polinomial de grado 3 (puedes cambiar el grado para experimentar)
    coeficientes = np.polyfit(x_vals, y_proms, 3)
    polinomio = np.poly1d(coeficientes)
    x_linea = np.linspace(min(x_vals), max(x_vals), 100)  # Generar puntos x para la línea ajustada
    y_linea = polinomio(x_linea)
    
    # Plotear la curva ajustada
    plt.plot(x_linea, y_linea, color='r', label='Polynomial Fitting')
    
    title = f'Size of the prototype set for '
    if g_especifico:
        title += f'\u03B3={g_especifico}'  # Usando el símbolo Unicode de gamma
    else:
        title += f'{dataset_especifico}, s={s_especifico} y T={T_especifico}'
        
    plt.title(title)
    plt.xlabel('Number of samples/prototypes trained')
    plt.ylabel('Size of the model\'s prototype set')
    plt.legend()
    plt.grid(True)
    if g_especifico:
        plt.savefig(f"../gamma_pruebas/tam_conj_gamma{g_especifico}.png")
    else:
        plt.show()
    

def main():
    try:
        if "replica"in sys.argv[1]:
            directorio_resultados = "../resultados_servidor/replica1_tam_test"
        else:
            directorio_resultados = "../resultados_servidor/old_resultados_raspi"
    except:
        directorio_resultados = "../resultados_servidor/old_resultados_raspi"
        
    directorio_resultados = "../resultados_local"
        
    g_especifico = None
    g_list = None
    if len(sys.argv) > 1:
        if "g" in sys.argv[1]:
            g_list = ["30", "50", "70", "90", "110", "130", "150"]
            dataset_especifico = "elec"
            s_especifico = "4"
            T_especifico = "1.0"
            directorio_resultados = "../gamma_pruebas"  

            if not g_list:
                print("Valores posibles para gamma: 30, 50, 70, 90, 110, 130, 150")
                g_especifico = input("Ingrese el valor específico de gamma (por ejemplo, '30'): ")
                # Si introduce un g_especifico no valido por defecto se establece a 30
                if g_especifico not in ["30", "50", "70", "90", "110", "130", "150"]:
                    g_especifico = "30"
        
        else:
            print("Valores posibles para el conjunto de datos: elec, phis, elec2")
            dataset_especifico = input("Ingrese el conjunto de datos específico (por ejemplo, 'elec'): ")
            print("Valores posibles para s: 1, 2, 3, 4")
            s_especifico = input("Ingrese el valor específico de s (por ejemplo, '1'): ")
            print("Valores posibles para T: 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0")
            T_especifico = input("Ingrese el valor específico de T (por ejemplo, '0.05'): ")

    else:
        print("Valores posibles para el conjunto de datos: elec, phis, elec2")
        dataset_especifico = input("Ingrese el conjunto de datos específico (por ejemplo, 'elec'): ")
        print("Valores posibles para s: 1, 2, 3, 4")
        s_especifico = input("Ingrese el valor específico de s (por ejemplo, '1'): ")
        print("Valores posibles para T: 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0")
        T_especifico = input("Ingrese el valor específico de T (por ejemplo, '0.05'): ")
    

    if g_list:
        for g in g_list:
            datos_filtrados = leer_datos(directorio_resultados, dataset_especifico, s_especifico, T_especifico, g)
            x_vals, y_proms = calcular_promedios(datos_filtrados)
            plotear(x_vals, y_proms, dataset_especifico, s_especifico, T_especifico, g)
        
    else:
        datos_filtrados = leer_datos(directorio_resultados, dataset_especifico, s_especifico, T_especifico, g_especifico)
        x_vals, y_proms = calcular_promedios(datos_filtrados)
        plotear(x_vals, y_proms, dataset_especifico, s_especifico, T_especifico, g_especifico)

if __name__ == "__main__":
    main()
