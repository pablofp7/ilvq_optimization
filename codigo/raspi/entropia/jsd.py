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

def compute_js_distance_multidimensional(data1, data2, num_points=1000):  # Usar un num_points más pequeño para alta dimensión
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

def monte_carlo_jsd(data1, data2, num_samples=1000):
    # Combine data to find common range
    combined_data = np.vstack([data1, data2])
    dimensions = data1.shape[1]
    min_ranges = np.min(combined_data, axis=0)
    max_ranges = np.max(combined_data, axis=0)
    
    # Calculate total range across all dimensions
    total_range = np.sum(max_ranges - min_ranges)
    
    # Adjust the number of samples based on dimensions and total range
    # This formula is heuristic; feel free to adjust it based on your needs or empirical testing
    num_samples = int(num_samples * 2**(dimensions / 2) * (total_range / (2 * dimensions)))
    
    # print(f"Using {num_samples} samples for Monte Carlo estimation")

    
    # Sample points uniformly within the range
    samples = np.random.uniform(min_ranges, max_ranges, (num_samples, data1.shape[1]))
    
    # Fit KDEs to data
    kde1 = KernelDensity(kernel='epanechnikov', bandwidth='scott').fit(data1)
    kde2 = KernelDensity(kernel='epanechnikov', bandwidth='scott').fit(data2)
    
    # Evaluate densities at sampled points
    log_density1 = kde1.score_samples(samples)
    log_density2 = kde2.score_samples(samples)
    density1 = np.exp(log_density1)
    density2 = np.exp(log_density2)
    
    # Normalize densities
    if density1.sum() > 0 and density2.sum() > 0:

        density1 /= density1.sum()
        density2 /= density2.sum()
        
        # Compute Jensen-Shannon distance
        js_distance = jensenshannon(density1, density2, base = 2)
        
        return js_distance

    else:
        print(f"[WARNING] : One of the densities is zero. Densities: {density1.sum()}, {density2.sum()}")

        # Sample points uniformly within the range
        samples = np.random.uniform(min_ranges, max_ranges, (2* num_samples, data1.shape[1]))
        
        # Fit KDEs to data
        kde1 = KernelDensity(kernel='epanechnikov', bandwidth='scott').fit(data1)
        kde2 = KernelDensity(kernel='epanechnikov', bandwidth='scott').fit(data2)
        
        # Evaluate densities at sampled points
        log_density1 = kde1.score_samples(samples)
        log_density2 = kde2.score_samples(samples)
        density1 = np.exp(log_density1)
        density2 = np.exp(log_density2)
        
        if density1.sum() > 0 and density2.sum() > 0:
            density1 /= density1.sum()
            density2 /= density2.sum()
            js_distance = jensenshannon(density1, density2, base = 2)
            return js_distance
        
        else: 
            print(f"[WARNING] : Second TRY. One of the densities is zero. Densities: {density1.sum()}, {density2.sum()}")
            print(f"[WARNING] : Data1: {data1[:20]}, Data2: {data2[:20]}")
            #Calcular con adaptative sampling
            js_distance = adaptive_sampling_jsd(data1, data2, num_iterations=5)
            
            return js_distance


def adaptive_sampling_jsd(data1, data2, num_samples=5000, num_iterations=5):
    # Combine data for range
    combined_data = np.vstack([data1, data2])
    min_ranges = np.min(combined_data, axis=0)
    max_ranges = np.max(combined_data, axis=0)

    # Initialize samples
    samples = np.random.uniform(low=min_ranges, high=max_ranges, size=(num_samples, data1.shape[1]))

    for _ in range(num_iterations):
        # Fit KDEs to data
        kde1 = KernelDensity(kernel='epanechnikov', bandwidth='scott').fit(data1)
        kde2 = KernelDensity(kernel='epanechnikov', bandwidth='scott').fit(data2)

        # Evaluate log densities at samples for both distributions
        log_dens1 = kde1.score_samples(samples)
        log_dens2 = kde2.score_samples(samples)

        # Find points with highest variance in densities and sample more around them
        density_diff = np.abs(np.exp(log_dens1) - np.exp(log_dens2))
        high_variance_indices = density_diff.argsort()[-num_samples//10:]

        # Sample more points around high variance points
        new_samples = samples[high_variance_indices] + np.random.randn(num_samples//10, data1.shape[1]) * (max_ranges - min_ranges) / num_samples
        samples = np.vstack([samples, new_samples])

    # Final KDE fit and density calculation
    final_kde1 = KernelDensity(kernel='epanechnikov', bandwidth='scott').fit(data1)
    final_kde2 = KernelDensity(kernel='epanechnikov', bandwidth='scott').fit(data2)
    log_dens1_final = final_kde1.score_samples(samples)
    log_dens2_final = final_kde2.score_samples(samples)

    # Compute Jensen-Shannon distance
    dens1 = np.exp(log_dens1_final)
    dens2 = np.exp(log_dens2_final)
    if dens1.sum() == 0 or dens2.sum() == 0:
        print(f"[WARNING] : Adpative Sampling. One of the densities is zero. Densities: {dens1.sum()}, {dens2.sum()}")
        return 1
    
    dens1_norm = dens1 / dens1.sum()
    dens2_norm = dens2 / dens2.sum()
    js_distance = jensenshannon(dens1_norm, dens2_norm, base=2)

    return js_distance