import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad, IntegrationWarning
import warnings



def plot_kde_histogram_epanechnikov(columna):
    # Convertir la columna a numpy array si es necesario
    columna_data = columna.to_numpy().reshape(-1, 1) if hasattr(columna, 'to_numpy') else columna.reshape(-1, 1)
    
    # Para minimizar la discretización en el histograma, determina los bins basados en los valores únicos
    bins = 1000
    
    # Calcular el ancho de banda de Scott como referencia
    n = len(columna_data)
    sigma = np.std(columna_data, ddof=1)
    bandwidth = np.power(n, -1./5) * sigma
    
    # Preparar el espacio de valores para las estimaciones de densidad
    # x_d = np.linspace(columna_data.min(), columna_data.max(), 1000).reshape(-1, 1)
    x_d = np.linspace(0, 1, 1000).reshape(-1, 1)

    # Plotear el "histograma" que en este caso actúa más como una función de masa de probabilidad
    plt.hist(columna_data[:, 0], bins=bins, density=True, alpha=0.5, color='gray', label='FMP')

    # Estimación KDE utilizando el kernel Epanechnikov
    kde = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth).fit(columna_data)
    log_dens = kde.score_samples(x_d)
    plt.plot(x_d[:, 0], np.exp(log_dens), '-', label='KDE Epanechnikov')

    plt.title(f"FMP y KDE Epanechnikov para '{columna.name if hasattr(columna, 'name') else 'Columna'}'")
    plt.xlabel("Valor")
    plt.ylabel("Densidad")
    plt.legend()    
    plt.show()
    
def multivariate_kde_estimation(data, bandwidth='scott'):
    # Ajustar KDE multivariante
    # Nota: 'data' debe ser un DataFrame de pandas o un array de NumPy con múltiples columnas
    kde = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth).fit(data)
    return kde

# Función para evaluar la densidad estimada en un punto x dada una KDE
def eval_kde(kde, x):
    # Ensure x is an array
    x_array = np.atleast_2d(x)
    # Evaluate the density at x_array
    log_density = kde.score_samples(x_array)
    return np.exp(log_density)

# Calcular la distancia de Jensen-Shannon usando KDEs
def jensen_shannon_distance_kde(kde_p, kde_q, lower_bound=-np.inf, upper_bound=np.inf):
    warnings.simplefilter('ignore', IntegrationWarning)
    # Función para calcular la media de las densidades KDE evaluadas en x
    def m_kde(x):
        return 0.5 * (eval_kde(kde_p, x) + eval_kde(kde_q, x))

    # Modificar 'kl_divergence_kde' para aceptar directamente una función de densidad media
    def kl_divergence_direct(kde, m_func, lower_bound, upper_bound):
        def integrand(x):
            p_x = eval_kde(kde, x)
            m_x = m_func(x)
            return p_x * np.log(p_x / m_x) if p_x > 0 and m_x > 0 else 0
        
        # Suprimir IntegrationWarning dentro de este contexto
        
        result, _ = quad(integrand, lower_bound, upper_bound)        
        return result

    kl_pm = kl_divergence_direct(kde_p, m_kde, lower_bound, upper_bound)
    kl_qm = kl_divergence_direct(kde_q, m_kde, lower_bound, upper_bound)
    return np.sqrt(0.5 * (kl_pm + kl_qm))
