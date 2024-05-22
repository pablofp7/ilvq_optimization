import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.neighbors import KernelDensity


def read_dataset():
    pd.set_option('future.no_silent_downcasting', True)
    filename = "electricity.csv"
    dataset = pd.read_csv(f"../dataset/{filename}")
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

def kde_est(columna, bandwidths=[0.01, 0.05, 0.1, 0.25, 0.5]):
    # Generar un rango de valores para el eje x
    x_values = np.linspace(columna.min(), columna.max(), 100)
    
    # Plotear el histograma
    plt.hist(columna, bins=30, density=True, alpha=0.5, label="Histograma")
    
    # Plotear las curvas KDE para diferentes anchos de banda preseleccionados
    for bandwidth in bandwidths:
        kde = stats.gaussian_kde(columna, bw_method=bandwidth)
        y_values = kde(x_values)
        plt.plot(x_values, y_values, label=f"KDE Ancho de banda: {bandwidth}")
    
    # Estimación de ancho de banda utilizando los métodos de Scott y Silverman
    bw_methods = ['scott', 'silverman']
    for bw_method in bw_methods:
        kde = stats.gaussian_kde(columna, bw_method=bw_method)
        y_values = kde(x_values)
        plt.plot(x_values, y_values, label=f"KDE {bw_method.capitalize()}", linestyle="--")
        print(f"Estimación de ancho de banda usando {bw_method}: {kde.factor * columna.std(ddof=1)}")
    
    # Calcular la media y la desviación estándar para la columna original
    mu, sigma = columna.mean(), columna.std()
    
    # Plotear una curva de distribución normal para comparar
    plt.plot(x_values, stats.norm.pdf(x_values, mu, sigma), label="Distribución Normal", linestyle="--")
    
    plt.title(f"Estimación de Densidad Kernel para la Columna '{columna.name}'")
    plt.xlabel('Valores')
    plt.ylabel('Densidad')
    plt.legend()
    plt.show()
    


def plot_kde_histogram_combined(columna):
    # Convertir la columna a numpy array si es necesario
    if not isinstance(columna, np.ndarray):
        columna_data = columna.to_numpy().reshape(-1, 1)
    else:
        columna_data = columna.reshape(-1, 1)
    
    # Calcular el ancho de banda de Scott como referencia
    n = len(columna_data)
    sigma = np.std(columna_data, ddof=1)
    bandwidth = np.power(n, -1./5) * sigma
    
    # Preparar el espacio de valores para las estimaciones de densidad
    x_d = np.linspace(np.min(columna_data), np.max(columna_data), 1000).reshape(-1, 1)

    # Plotear el histograma de los datos
    plt.hist(columna_data[:, 0], bins=30, density=True, alpha=0.5, color='gray', label='Histograma')

    # Kernels a comparar
    kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']

    # Realizar y plotear la estimación KDE para cada kernel
    for kernel in kernels:
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(columna_data)
        log_dens = kde.score_samples(x_d)
        plt.plot(x_d[:, 0], np.exp(log_dens), '-', label=f'KDE {kernel}')

    plt.title("Histograma y KDE con Diferentes Kernels")
    plt.xlabel("Valor")
    plt.ylabel("Densidad")
    plt.legend()
    plt.show()
        
if __name__ == "__main__":
    dataset = read_dataset()
    df = pd.DataFrame(dataset.values, columns=dataset.columns)
    for columna in df.columns:
        kde_est(df[columna])
        plot_kde_histogram_combined(df[columna])