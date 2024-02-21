from scipy.stats import entropy as scipy_entropy  # Renaming the scipy.stats entropy function
import numpy as np
from sklearn.neighbors import KernelDensity 

def kl_divergence(p, q):
    return scipy_entropy(p, qk=q)

def jensen_shannon_distance(p, q):
    m = 0.5 * (p + q)
    return np.sqrt(0.5 * (kl_divergence(p, m) + kl_divergence(q, m)))

def multivariate_kde_estimation(data, bandwidth='scott'):
    kde = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth).fit(data)
    def density(points):
        return np.exp(kde.score_samples(points))
    return kde, density

def compute_js_distance_between_datasets(data1, data2):
    # Estimate KDEs for each dataset
    kde1, density1 = multivariate_kde_estimation(data1)
    kde2, density2 = multivariate_kde_estimation(data2)

    # Define the domain range for each dimension based on the data
    x_min, x_max = -5, 10  # Adjust these ranges as needed
    y_min, y_max = 0, 1    # Adjust these ranges as needed
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

    # Optional: Sum of probability density estimates, if needed for further analysis
    sum_density1 = np.sum(prob_density_estimate1_normalized)
    sum_density2 = np.sum(prob_density_estimate2_normalized)

    print(f"Sum of densities for dataset 1: {sum_density1}")
    print(f"Sum of densities for dataset 2: {sum_density2}")

# Example usage:
# mean1 = 0
# std_dev1 = 1  
# size1 = 1000  

# dimension1 = np.random.normal(loc=mean1, scale=std_dev1, size=size1)
# dimension2 = np.random.uniform(low=0, high=0.25, size=size1)
# data1 = np.column_stack((dimension1, dimension2))

# # Generate data2
# mean2 = 0
# std_dev2 = 1  
# size2 = 1000  

# dimension1 = np.random.normal(loc=mean2, scale=std_dev2, size=size2)
# dimension2 = np.random.uniform(low=0.75, high=1, size=size2)
# data2 = np.column_stack((dimension1, dimension2))

data1 = np.random.rand(0, 1, 1000)
data2 = np.random.rand(0, 1, 1000)  

compute_js_distance_between_datasets(data1, data2)
