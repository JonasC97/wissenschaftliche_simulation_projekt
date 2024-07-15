import matplotlib
from matplotlib.ticker import MaxNLocator

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# Global variable to hold the animation object
ani = None

def get_zeroes_with_newton_iteration(f, f_derivative, x0, x_true=None, tolerance=1e-6, max_iterations=1000,
                                     show_animation=False):

    x = x0
    # Relevant for plotting
    x_values = [x]
    f_values = [f(x)]

    print(f"Current x: {x}")
    for i in range(max_iterations):
        # In this case the method cannot continue because there is no zero for a horizontal tangent
        if f_derivative(x) == 0:
            print(
                "The iteration cannot be continued because it went into a point with gradient 0. Try a different start value.")
            plot_standard_plots(x_values, f_values, f, f_derivative, show_animation)
            return False, i + 1, None, None

        # Remember: The new x is the zero of the tangent of f in (x|f(x))
        # y = f'(x) * (x - x_i) + f(x_i) = 0 <=> x - x_i = f(x_i) / f'(x_i) <=> x = x_i - f(x_i) / f'(x_i)
        x = x - f(x) / f_derivative(x)
        x_values.append(x)
        f_values.append(f(x))

        print(f"Current x: {x}")
        if x_true is not None:
            print(f"Current Difference to Groundtruth Value: {abs(x_true - x)}")

        if abs(f(x)) < tolerance:
            # The error converged, zero point is found. Compute the eoc and plot all relevant values
            eoc_values = calculate_experimental_order_of_convergence(x_values, x_true)
            plot_standard_plots(x_values, f_values, f, f_derivative, show_animation)
            plot_eoc_values(eoc_values)
            return True, i + 1, x, eoc_values

    # The error did not converge, zero point not found. Compute the eoc and plot all relevant values
    eoc_values = calculate_experimental_order_of_convergence(x_values, x_true)
    plot_standard_plots(x_values, f_values, f, f_derivative, show_animation)
    plot_eoc_values(eoc_values)
    return False, max_iterations, None, eoc_values


def plot_standard_plots(x_values, f_values, f, f_derivative, show_animation):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    print("Newton x-Values")
    print(range(len(f_values)))
    # Plot for f(x) over iterations
    ax1.plot(range(len(f_values)), f_values, 'bo-', label='y = f(x)')
    ax1.axhline(0, color='red', linestyle='--', linewidth=1)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('y')
    ax1.set_title('Convergence of Newton Method (y-values)')
    ax1.legend()
    ax1.grid(True)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))


    # Plot for x values over iterations
    ax2.plot(range(len(x_values)), x_values, 'go-', label='x')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('x')
    ax2.set_title('Convergence of Newton Method (x-values)')
    ax2.legend()
    ax2.grid(True)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show(block=False)

    if show_animation:
        plt.show(block=False)
        plot_animation(x_values, f_values, f, f_derivative)


def plot_animation(x_values, f_values, f, f_derivative):
    global ani  # Reference to the global variable

    fig, ax = plt.subplots(figsize=(10, 6))
    x_min, x_max = min(x_values) - 1, max(x_values) + 1
    x_range = np.linspace(x_min, x_max, 400)
    y_range = f(x_range)

    def update(num):
        ax.clear()
        ax.plot(x_range, y_range, label='f(x)')
        ax.axhline(0, color='black', linewidth=1.5)  # X-Achse markieren

        if num < len(x_values):
            x = x_values[num]
            # Plot the tangent
            tangent = f(x) + f_derivative(x) * (x_range - x)
            ax.plot(x_range, tangent, linestyle='--', label=f'Tangent at x={x:.2f}')

            # Mark point where the tangent touches f
            ax.plot(x, f(x), 'ro')
            y_shift = 0.05 * (max(y_range) - min(y_range))  # Relative Verschiebung basierend auf dem y-Bereich
            ax.text(x, f(x) + y_shift, f'({x:.2f}, {f(x):.2f})', color='red')

            # Mark the zero point of the tangent
            tangent_x_intercept = x - f(x) / f_derivative(x)
            ax.plot(tangent_x_intercept, 0, 'go')
            ax.text(tangent_x_intercept, -y_shift, f'({tangent_x_intercept:.2f}, 0)', color='green')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(min(y_range), max(y_range))
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Function and Tangents')
        ax.legend()
        ax.grid(True)

    # Animate the plot
    ani = animation.FuncAnimation(fig, update, frames=len(x_values) * 2, interval=2000, repeat=True)
    plt.tight_layout()
    plt.show(block=False)  # Blockierender Plot


def calculate_experimental_order_of_convergence(x_values, x_true):
    eoc_values = []
    for i in range(len(x_values)-2):
        # Now calculate with x_values[i], x_values[i+1], x_values[i+2] and x_true a numerical estimation of the order of convergence
        eoc_k = np.log(np.linalg.norm(x_values[i+1] - x_true) / np.linalg.norm(x_values[i+2] - x_true)) / np.log(np.linalg.norm(x_values[i] - x_true) / np.linalg.norm(x_values[i+1] - x_true))
        eoc_values.append(eoc_k)
    return eoc_values

def plot_eoc_values(eoc_values):
    # Plot for the EOC Values
    plt.figure()
    plt.plot(range(1, len(eoc_values) + 1), eoc_values, marker='o')
    # plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('EOC')
    plt.title('Newton EOC Plot')
    plt.show(block=False)