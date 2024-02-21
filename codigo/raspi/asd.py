from scipy.stats import entropy as scipy_entropy
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon


def kl_divergence(p, q):
    return scipy_entropy(p, qk=q)

def jensen_shannon_distance(p, q):
    m = 0.5 * (p + q)
    return np.sqrt(0.5 * (kl_divergence(p, m) + kl_divergence(q, m)))

def fit_kde(data, bandwidth='scott'):
    kde = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth).fit(data)
    return kde

def calculate_density(kde, points):
    return np.exp(kde.score_samples(points))

def compute_js_distance_multidimensional(data1, data2, num_points=10):  # Usar un num_points más pequeño para alta dimensión
    # Determinar los rangos comunes para cada dimensión
    min_ranges = np.min(np.vstack([data1, data2]), axis=0)
    max_ranges = np.max(np.vstack([data1, data2]), axis=0)
    
    # Crear una malla de puntos para evaluar las densidades
    axis_ranges = [np.linspace(min_ranges[dim], max_ranges[dim], num_points) for dim in range(data1.shape[1])]
    meshgrid = np.meshgrid(*axis_ranges)
    flat_grid = np.array([axis.ravel() for axis in meshgrid]).T
    
    # Ajustar KDEs y calcular las densidades en la malla de puntos
    kde1 = fit_kde(data1)
    kde2 = fit_kde(data2)
    prob_density_estimate1 = calculate_density(kde1, flat_grid)
    prob_density_estimate2 = calculate_density(kde2, flat_grid)
    
    # Normalizar las densidades
    prob_density_estimate1_normalized = prob_density_estimate1 / sum(prob_density_estimate1)
    prob_density_estimate2_normalized = prob_density_estimate2 / sum(prob_density_estimate2)
    
    # Calcular la distancia de Jensen-Shannon
    js_distance = jensenshannon(prob_density_estimate1_normalized, prob_density_estimate2_normalized, base=2)
    
    return js_distance



# Ejemplo de uso
size = 1000

mean1 = 0
std_dev1 = 1  
dimension1 = np.random.normal(loc=mean1, scale=std_dev1, size=size)
dimension1 = np.random.uniform(low=0, high=1, size=size)
dimension2 = np.random.uniform(low=0, high=0.25, size=size)
data1 = np.column_stack((dimension1, dimension2))

mean2 = 0
std_dev2 = 1  
dimension1 = np.random.normal(loc=mean2, scale=std_dev2, size=size)
dimension1 = np.random.uniform(low=0, high=1, size=size)
dimension2 = np.random.uniform(low=0.75, high=1, size=size)
data2 = np.column_stack((dimension1, dimension2))

# n_samples = 10000
# data1 = np.random.uniform(low=0, high=0.75, size=n_samples).reshape(-1, 1)
# data2 = np.random.uniform(low=0.25, high=1, size=n_samples).reshape(-1, 1)

js_distance = compute_js_distance_multidimensional(data1, data2)
print("Jensen-Shannon Distance:", js_distance)

# # Plot histograms
# plt.figure(figsize=(10, 6))
# plt.hist(data1, bins=50, alpha=0.5, label='Data1: U[0, 0.75]')
# plt.hist(data2, bins=50, alpha=0.5, label='Data2: U[0.25, 1]')
# plt.legend(loc='upper right')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Histograms of Two Uniform Distributions')
# plt.grid(True)
# plt.show()