import sys
import os
ruta_directorio_main = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_directorio_main not in sys.path:
    sys.path.append(ruta_directorio_main)

from entropia import jsd
import numpy as np
import time
import pandas as pd
from prototypes import XuILVQ

def read_dataset():
    pd.set_option('future.no_silent_downcasting', True)
    dataset = pd.read_csv(f"../dataset/electricity.csv")
    # Se cambia el 'UP' por 1 y el 'DOWN' por 0
    dataset.replace('UP', 1, inplace=True)
    dataset.infer_objects(copy=False)
    dataset.replace('DOWN', 0, inplace=True) 
    dataset.infer_objects(copy=False)

    dataset.replace('True', 1, inplace=True)
    dataset.infer_objects(copy=False)
    dataset.replace('False', 0, inplace=True) 
    dataset.infer_objects(copy=False)


      
    dataset.infer_objects(copy=False)
    return dataset


def prueba1():
    """
    Esta función calcula la distancia de jenssen shannon entre dos conjuntos de datos bidimensionales.
    Su primera dimensión es una distribución normal y la segunda dimensión es una distribución uniforme.
    """
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
    """
    Esta función calcula la distancia de jenssen shannon entre dos conjuntos de datos unidimensionales.
    Ambas distribuciones son uniformes.
    """
    n_samples = 10000
    data1 = np.random.uniform(low=0, high=0.25, size=n_samples).reshape(-1, 1)
    data2 = np.random.uniform(low=0.75, high=1, size=n_samples).reshape(-1, 1)
    distancia = jsd.monte_carlo_jsd(data1, data2)
    print(f"Distancia de Jensen-Shannon: {distancia}")
    
    
def prueba3():
    """
    Esta función compara tres métodos para calcular la distancia de Jensen-Shannon entre dos conjuntos de datos.
    1) Jensen-Shannon Distance Multidimensional (condicionalmente basado en n_dim)
    2) Monte Carlo Jensen-Shannon Distance
    3) Adaptive Sampling Jensen-Shannon Distance
    """
    
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

    low1 = 0
    high1 = 1
    low2 = 0
    high2 = 1

    n_samples = 10000
    
    # Generar datos para data1 y data2
    dim = np.random.uniform(low=low1, high=high1, size=size)
    data1 = np.column_stack([dim for _ in range(n_dim)])
    dim2 = np.random.uniform(low=low2, high=high2, size=size)
    data2 = np.column_stack([dim2 for _ in range(n_dim)])

    for _ in range(n_iterations):

        # Método 1: compute_js_distance_multidimensional (condicionalmente basado en n_dim)
        if n_dim < 6:
            inicio = time.perf_counter()
            js_distance1 = jsd.compute_js_distance_multidimensional(data1, data2, n_samples)
            fin = time.perf_counter() - inicio
            jsd_values_method1.append(js_distance1)
            time_method1 += fin

        # Método 2: monte_carlo_jsd
        inicio = time.perf_counter()
        js_distance2 = jsd.monte_carlo_jsd(data1, data2, n_samples)
        fin = time.perf_counter() - inicio
        jsd_values_method2.append(js_distance2)
        time_method2 += fin

        # Método 3: adaptive_sampling_jsd
        inicio = time.perf_counter()
        js_distance3 = jsd.adaptive_sampling_jsd(data1, data2, n_samples)
        fin = time.perf_counter() - inicio
        jsd_values_method3.append(js_distance3)
        time_method3 += fin

    # Imprimir los resultados
    if n_dim < 6:
        print(f"1) Jensen-Shannon Distance Multidimensional - Tiempo Total: {time_method1} segundos, Valores de JSD: Mean={np.mean(jsd_values_method1)}, Typical Deviation={np.std(jsd_values_method1)}.")
    print(f"2) Monte Carlo Jensen-Shannon Distance - Tiempo Total: {time_method2} segundos, Valores de JSD: Mean={np.mean(jsd_values_method2)}, Typical Deviation={np.std(jsd_values_method2)}.")
    print(f"3) Adaptive Sampling Jensen-Shannon Distance - Tiempo Total: {time_method3} segundos, Valores de JSD: Mean={np.mean(jsd_values_method3)}, Typical Deviation={np.std(jsd_values_method3)}.")
    
    
def prueba4(mode: int = 0):
    """
    Esta función trata de comparar dos conjuntos de prototipos generados por modelos ILVQ
    Consta de varios modos determinados por el parámetro 1
    1) mode = 0: se entrenan dos modelos ILVQ con las 1000 primeras muestras del dataset y se comparan sus prototipos
    2) mode = 1: se entrenan los dos modelos ILVQ con 1000 muestras cada uno pero ahora intercalando muestras: muestras 0,2,4,6,8,... y muestras 1,3,5,7,9,...
    3) mode = 2: se entrena un modelo ILVQ con las primeras 1000 muestras y otro con las últimas 1000 muestras del dataset, para ver cómo afecta el concept drift
    4) mode = 3: como el modo 2 pero ahora con 3 modelos ILVQ, se cogen las primeras 1000 muestras, las 1000 muestras intermedias (inicia 1/3 total) y las últimas 1000 muestras (inicia 2/3 total) y se hace la distancia entre cada uno de estos
    :mode elige el modo de la prueba
    """
    df = read_dataset()
    df_list = [(fila[:-1], fila[-1]) for fila in df.values]
    modelo1 = XuILVQ()
    modelo2 = XuILVQ()
    
    if mode == 0:
        for i in range(1000):
            x, y = df_list[i]
            x = {k: v for k, v in enumerate(x)}
            modelo1.learn_one(x, y)
        
        for i in range(1000):
            x, y = df_list[i]
            x = {k: v for k, v in enumerate(x)}
            modelo2.learn_one(x, y)
        
        data1 = np.array([np.append(proto['x'], proto['y']) for proto in list(modelo1.buffer.prototypes.values())])
        data2 = np.array([np.append(proto['x'], proto['y']) for proto in list(modelo2.buffer.prototypes.values())])
        distancia = jsd.monte_carlo_jsd(data1, data2)
        print(f"Distancia de Jensen-Shannon: {distancia}. Modo: {mode}")


    elif mode == 1:
        for i in range(0, 2000, 2):
            x, y = df_list[i]
            x = {k: v for k, v in enumerate(x)}
            modelo1.learn_one(x, y)
            
            # Verifica si i+1 está dentro del rango de la lista para evitar errores
            if i+1 < len(df_list):
                x, y = df_list[i+1]
                x = {k: v for k, v in enumerate(x)}
                modelo2.learn_one(x, y)
        
        data1 = np.array([np.append(proto['x'], proto['y']) for proto in list(modelo1.buffer.prototypes.values())])
        data2 = np.array([np.append(proto['x'], proto['y']) for proto in list(modelo2.buffer.prototypes.values())])
        distancia = jsd.monte_carlo_jsd(data1, data2)
        print(f"Distancia de Jensen-Shannon: {distancia}. Modo: {mode}")
        

    
        
    elif mode == 2:
        for i in range(1000):
            x, y = df_list[i]
            x = {k: v for k, v in enumerate(x)}
            modelo1.learn_one(x, y)
        
        # No es necesario recrear df_list. Directamente accede a las últimas 1000 muestras
        for i in range(-1000, 0, 1):
            x, y = df_list[i]
            x = {k: v for k, v in enumerate(x)}
            modelo2.learn_one(x, y)
            
        data1 = np.array([np.append(proto['x'], proto['y']) for proto in list(modelo1.buffer.prototypes.values())])
        data2 = np.array([np.append(proto['x'], proto['y']) for proto in list(modelo2.buffer.prototypes.values())])
        distancia = jsd.monte_carlo_jsd(data1, data2)
        print(f"Distancia de Jensen-Shannon: {distancia}. Modo: {mode}")
        
    elif mode == 3:
        modelo3 = XuILVQ()  # Inicializa el tercer modelo
        
        num_muestras = len(df_list)
        inicio_segundo_tercio = num_muestras // 3
        inicio_tercer_tercio = 2 * num_muestras // 3
        
        # Entrena el primer modelo con las primeras 1000 muestras
        for i in range(1000):
            x, y = df_list[i]
            x = {k: v for k, v in enumerate(x)}
            modelo1.learn_one(x, y)
            
        # Entrena el segundo modelo con las 1000 muestras intermedias
        for i in range(inicio_segundo_tercio, inicio_segundo_tercio + 1000):
            x, y = df_list[i]
            x = {k: v for k, v in enumerate(x)}
            modelo2.learn_one(x, y)
            
        # Entrena el tercer modelo con las últimas 1000 muestras
        for i in range(inicio_tercer_tercio, inicio_tercer_tercio + 1000):
            x, y = df_list[i]
            x = {k: v for k, v in enumerate(x)}
            modelo3.learn_one(x, y)
            
        # Genera los arrays de numpy para cada modelo
        data1 = np.array([np.append(proto['x'], proto['y']) for proto in list(modelo1.buffer.prototypes.values())])
        data2 = np.array([np.append(proto['x'], proto['y']) for proto in list(modelo2.buffer.prototypes.values())])
        data3 = np.array([np.append(proto['x'], proto['y']) for proto in list(modelo3.buffer.prototypes.values())])
        
        # Calcula la distancia de Jensen-Shannon entre cada par de modelos
        # Calcular el tiempo que se tarda en calcular cada distancia
        tic = time.perf_counter()
        distancia12 = jsd.monte_carlo_jsd(data1, data2)
        toc = time.perf_counter()
        print(f"Tiempo para calcular la distancia 12: {toc-tic}. Modo: {mode}")
        tic = time.perf_counter()
        distancia13 = jsd.monte_carlo_jsd(data1, data3)
        toc = time.perf_counter()
        print(f"Tiempo para calcular la distancia 13: {toc-tic}. Modo: {mode}")
        tic = time.perf_counter()
        distancia23 = jsd.monte_carlo_jsd(data2, data3)
        toc = time.perf_counter()
        print(f"Tiempo para calcular la distancia 23: {toc-tic}. Modo: {mode}")
        
        print(f"Distancia de Jensen-Shannon entre modelo1 y modelo2: {distancia12}. Modo: {mode}")
        print(f"Distancia de Jensen-Shannon entre modelo1 y modelo3: {distancia13}. Modo: {mode}")
        print(f"Distancia de Jensen-Shannon entre modelo2 y modelo3: {distancia23}. Modo: {mode}")
        

    elif mode == 4:
        modelo3 = XuILVQ()  # Inicializa el tercer modelo
        modelo4 = XuILVQ()  # Inicializa el cuarto modelo
        
        num_muestras = len(df_list)
        medio_dataset = num_muestras // 2
        
        # Entrena el primer modelo con las primeras 1000 muestras
        for i in range(1000):
            x, y = df_list[i]
            x = {k: v for k, v in enumerate(x)}
            modelo1.learn_one(x, y)
            
        # Entrena el segundo modelo con las segundas 1000 muestras
        for i in range(1000, 2000):
            x, y = df_list[i]
            x = {k: v for k, v in enumerate(x)}
            modelo2.learn_one(x, y)
            
        # Entrena el tercer modelo con 1000 muestras a partir de la mitad del dataset
        for i in range(medio_dataset, medio_dataset + 1000):
            x, y = df_list[i]
            x = {k: v for k, v in enumerate(x)}
            modelo3.learn_one(x, y)
            
        # Entrena el cuarto modelo con las últimas 1000 muestras
        for i in range(num_muestras - 1000, num_muestras):
            x, y = df_list[i]
            x = {k: v for k, v in enumerate(x)}
            modelo4.learn_one(x, y)
            
        # Genera los arrays de numpy para cada modelo
        data1 = np.array([np.append(proto['x'], proto['y']) for proto in list(modelo1.buffer.prototypes.values())])
        data2 = np.array([np.append(proto['x'], proto['y']) for proto in list(modelo2.buffer.prototypes.values())])
        data3 = np.array([np.append(proto['x'], proto['y']) for proto in list(modelo3.buffer.prototypes.values())])
        data4 = np.array([np.append(proto['x'], proto['y']) for proto in list(modelo4.buffer.prototypes.values())])
        
        # Calcula y muestra la distancia de Jensen-Shannon entre cada par de modelos
        distancia12 = jsd.monte_carlo_jsd(data1, data2)
        print(f"Distancia de Jensen-Shannon entre modelo1 y modelo2: {distancia12}. Modo: {mode}")
        
        distancia13 = jsd.monte_carlo_jsd(data1, data3)
        print(f"Distancia de Jensen-Shannon entre modelo1 y modelo3: {distancia13}. Modo: {mode}")
        
        distancia14 = jsd.monte_carlo_jsd(data1, data4)
        print(f"Distancia de Jensen-Shannon entre modelo1 y modelo4: {distancia14}. Modo: {mode}")
        
        distancia23 = jsd.monte_carlo_jsd(data2, data3)
        print(f"Distancia de Jensen-Shannon entre modelo2 y modelo3: {distancia23}. Modo: {mode}")
        
        distancia24 = jsd.monte_carlo_jsd(data2, data4)
        print(f"Distancia de Jensen-Shannon entre modelo2 y modelo4: {distancia24}. Modo: {mode}")
        
        distancia34 = jsd.monte_carlo_jsd(data3, data4)
        print(f"Distancia de Jensen-Shannon entre modelo3 y modelo4: {distancia34}. Modo: {mode}")

                



if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        arg = 1
    else: 
        arg = int(sys.argv[1])
    
    if arg == 1:
        prueba1()
    elif arg == 2:
        prueba2()
    elif arg == 3:
        prueba3()
    elif arg == 4:
        if len(sys.argv) == 2:
            mode = 0
        else:
            mode = int(sys.argv[2])
        
        prueba4(mode)
    
    
    
    