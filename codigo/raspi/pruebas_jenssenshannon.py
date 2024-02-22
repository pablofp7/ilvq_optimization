from entropia import jsd
import numpy as np
import time

def prueba1():

    # Ejemplo de uso
    size = 10000

    mean1 = 0
    std_dev1 = 1  
    dimension1 = np.random.normal(loc=mean1, scale=std_dev1, size=size)
    dimension2 = np.random.uniform(low=0, high=1, size=size)
    data1 = np.column_stack([dimension1, dimension2])
    
    mean2 = 0
    std_dev2 = 1  
    dimension1 = np.random.normal(loc=mean2, scale=std_dev2, size=size)
    dimension2 = np.random.uniform(low=0, high=1, size=size)
    data2 = np.column_stack([dimension1, dimension2])
    
    distancia = jsd.monte_carlo_jsd(data1, data2)
    print(f"Distancia de Jensen-Shannon: {distancia}")
    

def prueba2():

    n_samples = 10000
    data1 = np.random.uniform(low=0, high=0.25, size=n_samples).reshape(-1, 1)
    data2 = np.random.uniform(low=0.75, high=1, size=n_samples).reshape(-1, 1)
    distancia = jsd.monte_carlo_jsd(data1, data2)
    
    
def prueba3():
    # Configuración de los datos
    size = 1000  # Número de muestras para cada dimensión
    n_dim = 1 # Número de dimensiones
    n_iterations = 1  # Número de iteraciones para la comparación

    # Listas para almacenar los resultados de JSD y los tiempos de ejecución
    jsd_values_method1 = []
    jsd_values_method2 = []
    jsd_values_method3 = []

    time_method1 = 0
    time_method2 = 0
    time_method3 = 0

    low1 = 0.5
    high1 = 1
    low2 = 0
    high2 = 0.5

    # Generar datos para data1 y data2
    dim = np.random.uniform(low=low1, high=high1, size=size)
    data1 = np.column_stack([dim for _ in range(n_dim)])
    dim2 = np.random.uniform(low=low2, high=high2, size=size)
    data2 = np.column_stack([dim2 for _ in range(n_dim)])

    for _ in range(n_iterations):

        # Método 1: compute_js_distance_multidimensional (condicionalmente basado en n_dim)
        if n_dim < 6:
            inicio = time.time()
            js_distance1 = jsd.compute_js_distance_multidimensional(data1, data2)
            fin = time.time() - inicio
            jsd_values_method1.append(js_distance1)
            time_method1 += fin

        # Método 2: monte_carlo_jsd
        inicio = time.time()
        js_distance2 = jsd.monte_carlo_jsd(data1, data2)
        fin = time.time() - inicio
        jsd_values_method2.append(js_distance2)
        time_method2 += fin

        # Método 3: adaptive_sampling_jsd
        inicio = time.time()
        js_distance3 = jsd.adaptive_sampling_jsd(data1, data2)
        fin = time.time() - inicio
        jsd_values_method3.append(js_distance3)
        time_method3 += fin

    # Imprimir los resultados
    if n_dim < 6:
        print(f"1) Jensen-Shannon Distance Multidimensional - Tiempo Total: {time_method1} segundos, Valores de JSD: Mean={np.mean(jsd_values_method1)}, Typical Deviation={np.std(jsd_values_method1)}.")
    print(f"2) Monte Carlo Jensen-Shannon Distance - Tiempo Total: {time_method2} segundos, Valores de JSD: Mean={np.mean(jsd_values_method2)}, Typical Deviation={np.std(jsd_values_method2)}.")
    print(f"3) Adaptive Sampling Jensen-Shannon Distance - Tiempo Total: {time_method3} segundos, Valores de JSD: Mean={np.mean(jsd_values_method3)}, Typical Deviation={np.std(jsd_values_method3)}.")
    
    
def prueba4():
    
    
    
    return


if __name__ == "__main__":
    
    prueba4()
    