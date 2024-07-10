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
    x_values = []
    f_values = []

    print(f"Current x: {x}")
    for i in range(max_iterations):
        if f_derivative(x) == 0:
            print(
                "The iteration cannot be continued because it went into a point with gradient 0. Try a different start value.")
            plot_standard_plots(x_values, f_values, f, f_derivative, show_animation)
            return False, i + 1, None

        x = x - f(x) / f_derivative(x)
        x_values.append(x)
        f_values.append(f(x))

        print(f"Current x: {x}")
        if x_true is not None:
            print(f"Current Difference to Groundtruth Value: {abs(x_true - x)}")

        if abs(f(x)) < tolerance:
            plot_standard_plots(x_values, f_values, f, f_derivative, show_animation)
            return True, i + 1, x

    plot_standard_plots(x_values, f_values, f, f_derivative, show_animation)
    return False, max_iterations, None


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
    plt.show(block=False)  # Nicht-blockierender Plot

    if show_animation:
        plt.show(block=False)  # Blockierender Plot für Animation
        plot_animation(x_values, f_values, f, f_derivative)


def plot_animation(x_values, f_values, f, f_derivative):
    global ani  # Referenz auf die globale Variable

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
            tangent = f(x) + f_derivative(x) * (x_range - x)
            ax.plot(x_range, tangent, linestyle='--', label=f'Tangent at x={x:.2f}')

            # Punkt markieren, an dem die Tangente den Graphen berührt
            ax.plot(x, f(x), 'ro')
            y_shift = 0.05 * (max(y_range) - min(y_range))  # Relative Verschiebung basierend auf dem y-Bereich
            ax.text(x, f(x) + y_shift, f'({x:.2f}, {f(x):.2f})', color='red')

            # Punkt markieren, an dem die Tangente die x-Achse schneidet
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

    ani = animation.FuncAnimation(fig, update, frames=len(x_values) * 2, interval=2000, repeat=True)
    plt.tight_layout()
    plt.show(block=False)  # Blockierender Plot


