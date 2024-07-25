import math

import numpy as np
import matplotlib.pyplot as plt
from lu_decomposition import solve_linear_system_with_lu_decomposition

def newton_method(F, J, x0, x_true, tolerance=1e-6, max_iterations=1000):
    x = x0
    x_values = [x0]
    current_iteration = 0
    # Save error values for plotting
    errors = []

    for i in range(max_iterations):
        # Compute y for the current x
        Fx = F(x)
        # Compute the Jacobi Matrix J at the point x
        Jx = J(x)

        # print(Fx)
        # print(Jx)
        # Compute the x delta with our LU-Decomposition Algorithm
        # F(x_k+1) ≈ F(x_k) + J(x_k) * delta_x_k  (= 0) <=> -F(x_k) = J(x_k) * delta_x_k
        delta_x, message = solve_linear_system_with_lu_decomposition(Jx, -Fx)

        print(delta_x)
        x_new = x + delta_x

        # print(x_new)
        x_values.append(x_new)
        error = np.linalg.norm(delta_x)

        errors.append(error)
        current_iteration += 1

        if error < tolerance:
            # The error converged, zero point is found. Compute the eoc and plot all relevant values
            eoc_values = calculate_experimental_order_of_convergence(x_values, x_true)
            plot_eoc_values(eoc_values)
            return x, current_iteration, errors

        x = x_new

    eoc_values = calculate_experimental_order_of_convergence(x_values, x_true)
    plot_eoc_values(eoc_values)
    return x, current_iteration, errors

def calculate_experimental_order_of_convergence(x_values, x_true):
    eoc_values = []
    for i in range(len(x_values)-2):
        # Now calculate with x_values[i], x_values[i+1], x_values[i+2] and x_true a numerical estimation of the order of convergence
        eoc_k = np.log(np.linalg.norm(x_values[i+1] - x_true) / np.linalg.norm(x_values[i+2] - x_true)) / np.log(np.linalg.norm(x_values[i] - x_true) / np.linalg.norm(x_values[i+1] - x_true))
        eoc_values.append(eoc_k)
    return eoc_values

def plot_convergence(errors):
    plt.figure()
    plt.plot(range(len(errors)), errors, marker='o')
    # plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Convergence Plot')
    plt.show()

def plot_eoc_values(eoc_values):
    print("EOC-Values")
    print(eoc_values)
    # Plot for the EOC Values
    plt.figure()
    plt.plot(range(1, len(eoc_values) + 1), eoc_values, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('EOC')
    plt.title('Newton EOC Plot')
    plt.show()

