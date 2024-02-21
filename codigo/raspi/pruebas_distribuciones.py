from prototypes import XuILVQ
from scipy.stats import entropy as scipy_entropy  # Renaming the scipy.stats entropy function
import pandas as pd
import numpy as np
import time
import scipy.spatial.distance as dist
import scipy.stats as stats
from sklearn.neighbors import KernelDensity 


def read_dataset():
    dataset = pd.read_csv(f"dataset/electricity.csv")
    dataset.replace('UP', 1, inplace=True)
    dataset.replace('DOWN', 0, inplace=True) 
    dataset.replace('True', 1, inplace=True)
    dataset.replace('False', 0, inplace=True) 


    return dataset


# # Generar datos de muestra para dos distribuciones diferentes
# np.random.seed(0)  # Para reproducibilidad
# # Generar distribución uniforme para P entre 0 y 0.5
# data_p = pd.Series(np.random.uniform(0, 0.5, size=1000))

# # Generar distribución uniforme para Q entre 0.5 y 1
# data_q = pd.Series(np.random.uniform(0.5, 1, size=1000))

# # entropy.plot_kde_histogram_epanechnikov(data_p)
# # entropy.plot_kde_histogram_epanechnikov(data_q)

# # Estimar KDE para cada conjunto de datos
# kde_p = entropy.multivariate_kde_estimation(data_p.to_numpy().reshape(-1, 1))
# kde_q = entropy.multivariate_kde_estimation(data_q.to_numpy().reshape(-1, 1))

# # Calcular la distancia de Jensen-Shannon entre las dos KDEs
# # Es importante definir límites razonables para la integración basados en los datos
# lower_bound = min(data_p.min(), data_q.min())
# upper_bound = max(data_p.max(), data_q.max())

# inicio = time.time()
# js_distance = entropy.jensen_shannon_distance_kde(kde_p, kde_q, lower_bound, upper_bound)
# fin = time.time() - inicio
# print("Tiempo de cálculo de JS: ", fin)
# print(f"Distancia de Jensen-Shannon: {js_distance}")


def kl_divergence(p, q):
    return scipy_entropy(p, qk=q)

def continuous_entropy(prob_density):
    return -np.sum(prob_density * np.log(prob_density))

def jensen_shannon_distance(p, q):
    m = 0.5 * (p + q)
    return np.sqrt(0.5 * (kl_divergence(p, m) + kl_divergence(q, m)))

def multivariate_kde_estimation(data, bandwidth='scott'):
    kde = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth).fit(data)
    
    def density(points):
        return np.exp(kde.score_samples(points))
    
    return kde, density


# Generate data1
mean1 = 0
std_dev1 = 1  
size1 = 1000  

dimension1 = np.random.normal(loc=mean1, scale=std_dev1, size=size1)
dimension2 = np.random.uniform(low=0, high=0.25, size=size1)
data1 = np.column_stack((dimension1, dimension2))

# Generate data2
mean2 = 0
std_dev2 = 1  
size2 = 1000  

dimension1 = np.random.normal(loc=mean2, scale=std_dev2, size=size2)
dimension2 = np.random.uniform(low=0.75, high=1, size=size2)
data2 = np.column_stack((dimension1, dimension2))

# Estimate KDEs for each dataset
kde1, density1 = multivariate_kde_estimation(data1)
kde2, density2 = multivariate_kde_estimation(data2)

# Generate a set of points for evaluation
# points_to_evaluate = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=100)
# Define the domain range for each dimension
x_min, x_max = -5, 10  # Assuming these cover the range of your data in the first dimension
y_min, y_max = 0, 1    # Assuming these cover the range of your data in the second dimension
num_points = 100  # Number of points in each dimension

# Generate equally spaced points in each dimension
x = np.linspace(x_min, x_max, num_points)
y = np.linspace(y_min, y_max, num_points)

# Create a meshgrid for the bivariate domain
X, Y = np.meshgrid(x, y)
points_to_evaluate = np.column_stack([X.ravel(), Y.ravel()])


# Compute the probability densities for the two datasets
prob_density_estimate1 = density1(points_to_evaluate)
prob_density_estimate2 = density2(points_to_evaluate)

sum_density1 = np.sum(prob_density_estimate1)
sum_density2 = np.sum(prob_density_estimate2)

prob_density_estimate1_normalized = prob_density_estimate1 / sum_density1
prob_density_estimate2_normalized = prob_density_estimate2 / sum_density2


# Compute the Jensen-Shannon distance
js_distance = jensen_shannon_distance(prob_density_estimate1_normalized, prob_density_estimate2_normalized)

print("Jensen-Shannon Distance:", js_distance)

# Sum of probability density estimates
sum_density1 = np.sum(prob_density_estimate1_normalized)
sum_density2 = np.sum(prob_density_estimate2_normalized)

print(f"Sum of densities for dataset 1: {sum_density1}")
print(f"Sum of densities for dataset 2: {sum_density2}")
