import os
import sys
ruta_directorio_main = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_directorio_main not in sys.path:
    sys.path.append(ruta_directorio_main)
import entropia.jsd as jsd
import numpy as np
import time
import matplotlib.pyplot as plt

data1_list = [
    [ 1.71946384e+00,  5.86878618e-01,  1.61371310e+00,  3.60120534e-01, -7.30714393e-01,  3.17651643e-01,  2.15660326e+00,  1.02650298e+00, -9.13365955e-03,  0.00000000e+00],
    [ 9.84821577e-01, -9.14302284e-02,  4.10849377e-01, -3.10828296e-02,  1.01579127e+00,  1.08613007e+00,  1.62395570e+00,  1.00729817e+00, -1.05928868e-02,  1.00000000e+00],
    [ 2.71137304e+00,  7.89785407e-01,  2.11430894e+00, -1.46551975e+00,  1.29224284e+00,  7.40121982e-01, -7.76888680e-01,  1.33346988e+00,  8.46622922e-02,  0.00000000e+00],
    [ 9.62329718e-01,  4.57249877e-01,  9.20327678e-01,  5.35466819e-01,  1.11980268e+00,  1.73158974e+00, -7.07568227e-01, -1.15583787e-01, -7.74974697e-02,  1.00000000e+00],
    [ 9.28994837e-01, -6.92062100e-01, -1.06545604e-01,  1.56672004e+00,  1.10717491e+00,  4.76406152e-01, -6.66868469e-01,  1.10934246e+00,  3.92802349e-02,  1.00000000e+00],
    [ 1.02285195e+00,  8.78634867e-01,  9.77551974e-01,  5.09814602e-01,  2.73286656e-02,  1.35967985e-01,  9.09773437e-01,  8.31760397e-01,  4.34551871e-02,  0.00000000e+00],
    [ 9.66592804e-01,  1.51547355e-01,  4.56988592e-01, -4.19432233e-01, -7.19885993e-01,  4.33957200e-01,  8.86534114e-01, -1.72349761e-01, -1.77979227e-02,  1.00000000e+00],
    [ -1.44558439e-01, -3.06511060e-01,  7.65900870e-02, -1.71852157e-01, -3.80407471e-01,  6.90062221e-01, -3.87725387e-01,  1.01132624e+00,  1.56704531e-01,  1.00000000e+00],
    [ 1.05926390e+00,  8.69631109e-01,  8.55577676e-01,  1.00698192e+00,  9.84339588e-01,  9.40104302e-01,  6.51905239e-01,  3.16419447e-01,  9.76930513e-01,  0.00000000e+00],
    [ 9.92767027e-01, -4.46081613e-02,  4.63082509e-01, -5.20453212e-03,  1.00912283e+00,  9.96805342e-01,  1.53647205e+00,  1.03098444e+00, -5.03197866e-03,  1.00000000e+00],
    [ 4.53291913e-01, -1.60137050e+00,  4.14440152e-01,  1.54825381e+00,  1.09136054e+00,  4.83667220e-01, -1.09749889e+00,  2.06935158e+00,  1.02851954e+00,  1.00000000e+00],
    [ -1.04227025e+00, -9.24757747e-01,  1.05553424e+00, -1.24574658e-02,  -1.95208601e-02,  1.01385772e+00, -9.82741345e-01,  9.99993539e-01, -5.56811412e-04,  1.00000000e+00],
    [ -8.44611215e-01, -1.15256019e-02, -4.27479562e-01,  9.08493744e-01,  -2.21201009e-02,  5.06193771e-01, -5.29410842e-01,  9.99926497e-01, -8.49813052e-04,  1.00000000e+00],
    [ -8.45666261e-02, -1.91238951e-02,  1.12980412e+00,  1.15795351e+00,  -1.17632982e-01,  1.05192977e+00,  5.10408702e-02, -9.12464541e-02,   1.49727316e-02,  0.00000000e+00],
    [ 1.30564433e+00,  3.94581816e-01,  1.11569358e+00, -3.46122581e-03,  1.18013303e+00,  9.43274891e-01,  7.05329525e-01, -1.38686685e-01,  5.58985069e-03,  0.00000000e+00],
    [ 1.94676193e+00,  1.58279502e+00,  1.02099930e+00, -2.73714228e-01,  1.07962742e+00, -4.17920187e-01,  9.57074119e-01,  9.41346393e-01,  2.68638508e-01,  0.00000000e+00],
    [ 5.84361020e-01,  2.04227753e-01,  2.15770884e-01,  7.32088910e-02, -1.35029311e-01,  5.94302786e-01,  6.80618015e-01, -2.95561302e-02, -1.73789239e-02,  1.00000000e+00],
    [ -8.08891820e-01, -1.51854090e-03, -4.04724095e-01,  9.00648481e-01,  -4.51562456e-03,  5.00632307e-01, -5.05481794e-01,  9.99994511e-01, -1.57149832e-04,  1.00000000e+00],
    [ 1.05501434e+00,  5.94609074e-01,  1.01016837e+00,  1.87385319e-01,  6.34298967e-01,  8.84690771e-01,  3.08326570e-01,  2.73997081e-03,  8.38856120e-01,  0.00000000e+00],
    [ -4.07176346e-02,  4.40691428e-01, -2.27777744e-02,  4.39552401e-01,  -2.19080700e-02,  4.99183662e-01,  4.80377361e-01,  1.00563332e+00, -3.57605429e-03,  1.00000000e+00]
]

# Data2
data2_list = [
    [ 7.97707615e-01,  1.83570142e-01,  3.26674426e-01,  1.21183043e-01, -1.29261011e-01,  6.11888848e-01,  6.00391507e-01, -3.49730317e-02, -1.07220388e-02,  1.00000000e+00],
    [ 2.03352738e-02,  1.61716537e-01,  1.29271736e-01,  2.17813795e-01,  2.19629935e-02,  3.51549235e-01,  1.00188464e+00,  9.28761364e-01,  8.89446105e-01,  1.00000000e+00],
    [ 1.01413642e+00,  8.95747752e-01,  8.91130389e-01,  7.39395712e-01,  4.56234337e-01, -5.55200919e-03,  3.49392032e-01,  1.02229475e+00,  1.01348000e+00,  0.00000000e+00],
    [ -8.10681748e-02,  4.46587436e-01,  1.63936512e-01,  1.85540312e-01, -5.04616417e-02,  7.35503137e-01, -2.57428989e-02,  1.00711042e+00, -1.09928306e-02,  1.00000000e+00],
    [ 1.42388200e+00,  6.30277882e-02,  1.24366704e+00, -4.00330323e-01,  1.01494815e+00, -8.11410112e-03, -2.01128173e-02,  1.02010676e+00,  4.03641023e-02,  0.00000000e+00],
    [ -8.57925411e-02,  4.63459135e-01, -4.37981128e-02,  9.28510012e-01,  1.02196178e+00,  4.91242265e-01,  4.50942651e-01, -1.31900690e-03, -9.40629740e-03,  1.00000000e+00],
    [ 1.43738602e+00,  1.01732082e-01,  9.82670335e-01,  1.07197292e-01,  1.17870946e+00,  1.00566041e+00,  8.59665117e-01, -3.14112139e-02, -1.02234624e-02,  0.00000000e+00],
    [ 1.01429607e+00,  8.54612935e-01,  9.46179244e-01,  1.01064452e+00,  9.97688747e-01,  3.83962836e-01,  8.41721739e-01,  9.98740796e-01, -2.39418448e-03,  0.00000000e+00],
    [ -8.05897858e-01, -9.05042840e-04, -4.03068661e-01,  9.00301885e-01, -2.99868320e-03,  5.00352670e-01, -5.03622589e-01,  9.99996486e-01, -1.08965132e-04,  1.00000000e+00],
    [ 9.35216383e-01, -6.30187785e-01, -5.76725479e-02,  1.52836915e+00,  1.03670825e+00,  4.90874161e-01, -5.71538791e-01,  1.08778624e+00,  4.14602924e-02,  1.00000000e+00],
    [ 9.70971560e-01, -1.03342018e-01,  4.17010671e-01, -3.64980666e-02,  1.02120187e+00,  1.08228631e+00,  1.66725329e+00,  1.02616100e+00, -2.66267768e-02,  1.00000000e+00],
    [ 1.23868125e+00,  6.34169954e-01,  1.00663195e+00,  1.02981341e+00,  9.88623500e-01,  8.32640052e-01,  5.52450302e-01,  5.75816472e-02,  3.86580607e-02,  0.00000000e+00],
    [ 2.09902057e+00,  7.91666707e-01, -7.19036645e-01,  2.00063800e+00,  6.19325318e-01,  1.54434012e+00,  1.65018906e+00,  1.61593173e-02, -9.69847782e-03,  0.00000000e+00],
    [ 3.38754383e-01,  1.07037796e-01, -1.59570239e-02,  4.45991761e-01,  8.78885648e-01,  2.13479218e-01,  1.20298641e-01,  9.67643538e-01, -7.39665847e-03,  1.00000000e+00],
    [ -3.95685534e-02, -1.46834999e-02,  1.08042411e+00,  1.09439092e+00, -3.91615160e-02,  1.04625102e+00,  4.16390522e-02, -3.97115959e-02,  5.07705685e-04,  0.00000000e+00],
    [ 2.27549807e+00,  9.98909265e-01,  1.14067138e+00,  9.63637562e-01,  9.46969316e-01,  4.50032772e-01,  3.67470847e-01,  5.31312730e-01,  2.54700440e-02,  0.00000000e+00],
    [ 1.04686063e+00,  5.45037409e-01,  1.04337679e+00,  2.59768579e-01,  1.74247971e-02,  1.01950915e+00,  1.04820788e+00,  1.02224517e+00,  8.43201625e-02,  1.00000000e+00]
]

data1_array = np.array(data1_list)
data2_array = np.array(data2_list)

# print(len(data1_list))

# Methods to test
methods = {
    "multidimensional": jsd.compute_js_distance_multidimensional,
    "monte_carlo": jsd.monte_carlo_jsd,
    "adaptive_sampling": jsd.adaptive_sampling_jsd
}

num_samples_values = [500, 1000, 1500, 2000, 2500, 4000, 10000, 25000, 50000]
n_iterations = 250  # Number of iterations per num_samples value for averaging

# Loop over each method
for method_name, method in methods.items():
    # Initialize lists to store the metrics
    times = []
    means = []
    std_devs = []

    # Loop over each num_samples value
    for num_samples in num_samples_values:
        iteration_times = []
        distances = []
        print(f"Calculating for {method_name} with num_samples = {num_samples} ...")

        # Run the simulation n_iterations times for each num_samples
        for _ in range(n_iterations):
            start_time = time.perf_counter_ns()
            if method_name == "adaptive_sampling":
                distance = method(data1_array, data2_array, num_samples=num_samples)
            else:
                distance = method(data1_array, data2_array, num_points=num_samples)
            iteration_time = time.perf_counter_ns() - start_time

            iteration_times.append(iteration_time / 1e9)
            distances.append(distance)

        # Calculate the mean of the metrics for the current num_samples
        avg_time = np.mean(iteration_times)
        mean_distance = np.mean(distances)
        std_dev_distance = np.std(distances)

        # Append the calculated metrics to their respective lists
        times.append(avg_time)
        means.append(mean_distance)
        std_devs.append(std_dev_distance)

    # Plotting
    plt.figure(figsize=(10, 15))

    # Plot for average time per calculation
    plt.subplot(3, 1, 1)
    plt.plot(num_samples_values, times, marker='o')
    plt.xlabel('Number of Samples')
    plt.ylabel('Average Time per Calculation (s)')
    plt.xscale('log')
    plt.title(f'{method_name.capitalize()} - Time vs. Number of Samples')

    # Plot for mean distance
    plt.subplot(3, 1, 2)
    plt.plot(num_samples_values, means, marker='o')
    plt.xlabel('Number of Samples')
    plt.ylabel('Mean Distance')
    plt.xscale('log')
    plt.title(f'{method_name.capitalize()} - Mean Distance vs. Number of Samples')

    # Plot for standard deviation of distance
    plt.subplot(3, 1, 3)
    plt.plot(num_samples_values, std_devs, marker='o')
    plt.xlabel('Number of Samples')
    plt.ylabel('Standard Deviation of Distance')
    plt.xscale('log')
    plt.title(f'{method_name.capitalize()} - Std Dev vs. Number of Samples')

    plt.tight_layout()
    plt.savefig(f"./{method_name}_test.png")
    # plt.show()
