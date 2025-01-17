import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
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
    return dataset

def kde_est(column, bandwidths=[0.01, 0.05, 0.1, 0.25, 0.5]):
    print(f"Processing column: {column.name}")
    # Generate a range of values for the x-axis
    x_values = np.linspace(column.min(), column.max(), 100)
    
    # Plot the histogram
    plt.hist(column, bins=30, density=True, alpha=0.5, label="Histogram")
    
    # Plot the KDE curves for different preselected bandwidths
    for bandwidth in bandwidths:
        kde = stats.gaussian_kde(column, bw_method=bandwidth)
        y_values = kde(x_values)
        plt.plot(x_values, y_values, label=f"KDE Bandwidth: {bandwidth}")
    
    # Bandwidth estimation using Scott and Silverman methods
    bw_methods = ['scott', 'silverman']
    for bw_method in bw_methods:
        kde = stats.gaussian_kde(column, bw_method=bw_method)
        y_values = kde(x_values)
        plt.plot(x_values, y_values, label=f"KDE {bw_method.capitalize()}", linestyle="--")
        print(f"Bandwidth estimation using {bw_method}: {kde.factor * column.std(ddof=1)}")
    
    # # Calculate the mean and standard deviation for the original column
    # mu, sigma = column.mean(), column.std()
    
    # # Plot a normal distribution curve for comparison
    # plt.plot(x_values, stats.norm.pdf(x_values, mu, sigma), label="Normal Distribution", linestyle="--")
    
    plt.title(f"Kernel Density Estimation for Column '{column.name}'")
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f"kde_est_{column.name}.png")
    plt.clf()
    # # plt.show()


def plot_kde_histogram_combined(column):
    print(f"Processing column: {column.name}")
    # Convert the column to numpy array if necessary
    if not isinstance(column, np.ndarray):
        column_data = column.to_numpy().reshape(-1, 1)
    else:
        column_data = column.reshape(-1, 1)
    
    # Calculate Scott's bandwidth as reference
    n = len(column_data)
    sigma = np.std(column_data, ddof=1)
    bandwidth = np.power(n, -1./5) * sigma
    
    # Prepare the space of values for density estimations
    x_d = np.linspace(np.min(column_data), np.max(column_data), 1000).reshape(-1, 1)

    # Plot the histogram of the data
    plt.hist(column_data[:, 0], bins=30, density=True, alpha=0.5, color='gray', label='Histogram')

    # Kernels to compare
    kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']

    # Perform and plot the KDE estimation for each kernel
    for kernel in kernels:
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(column_data)
        log_dens = kde.score_samples(x_d)
        plt.plot(x_d[:, 0], np.exp(log_dens), '-', label=f'KDE {kernel}')

    plt.title("Histogram and KDE with Different Kernels")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(f"kde_histogram_combined_{column.name}.png")
    plt.clf()
    # plt.show()
        
if __name__ == "__main__":
    dataset = read_dataset()
    for column in dataset.columns:
        kde_est(dataset[column])
        plot_kde_histogram_combined(dataset[column])
