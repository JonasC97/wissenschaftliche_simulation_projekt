import matplotlib
from matplotlib.ticker import MaxNLocator

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# Global variable to hold the animation object
ani = None

class NewtonIteration:
    def __init__(self, f, f_derivative, x0, x_true=None, tolerance=1e-6, max_iterations=1000, show_animation=False):
        self.f = f
        self.f_derivative = f_derivative
        self.x0 = x0
        self.x_true = x_true
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.show_animation = show_animation
        self.x_values = [x0]
        self.f_values = [f(x0)]
        self.eoc_values = []
        self.iterations = 0
        self.zero_found = False


    def get_zeroes_with_newton_iteration(self):
        x = self.x0

        print(f"Current x: {x}")
        for i in range(self.max_iterations):
            # In this case the method cannot continue because there is no zero for a horizontal tangent
            if self.f_derivative(x) == 0:
                print("The iteration cannot be continued because it went into a point with gradient 0. Try a different start value.")
                self.iterations = i + 1
                return

            # Remember: The new x is the zero of the tangent of f in (x|f(x))
            # y = f'(x) * (x - x_i) + f(x_i) = 0 <=> x - x_i = f(x_i) / f'(x_i) <=> x = x_i - f(x_i) / f'(x_i)
            x = x - self.f(x) / self.f_derivative(x)
            self.x_values.append(x)
            self.f_values.append(self.f(x))

            print(f"Current x: {x}")
            if self.x_true is not None:
                print(f"Current Difference to Groundtruth Value: {abs(self.x_true - x)}")

            if abs(self.f(x)) < self.tolerance:
                # The error converged, zero point is found. Compute the eoc and return all relevant values
                self.eoc_values = self.calculate_experimental_order_of_convergence()
                self.iterations = i + 1
                self.zero_found = True
                return

        # The error did not converge, zero point not found. Compute the eoc and return all relevant values
        self.eoc_values = self.calculate_experimental_order_of_convergence()
        self.iterations = self.max_iterations
        self.zero_found = False

    def plot_standard_plots(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        print("Newton x-Values")
        print(range(len(self.f_values)))
        # Plot for f(x) over iterations
        ax1.plot(range(len(self.f_values)), self.f_values, 'bo-', label='y = f(x)')
        ax1.axhline(0, color='red', linestyle='--', linewidth=1)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('y')
        ax1.set_title('Convergence of Newton Method (y-values)')
        ax1.legend()
        ax1.grid(True)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Plot for x values over iterations
        ax2.plot(range(len(self.x_values)), self.x_values, 'go-', label='x')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('x')
        ax2.set_title('Convergence of Newton Method (x-values)')
        ax2.legend()
        ax2.grid(True)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        plt.show(block=False)

        if self.show_animation:
            self.plot_animation()

    def plot_animation(self):
        global ani  # Reference to the global variable

        fig, ax = plt.subplots(figsize=(10, 6))
        x_min, x_max = min(self.x_values) - 1, max(self.x_values) + 1
        x_range = np.linspace(x_min, x_max, 400)
        y_range = self.f(x_range)

        def update(num):
            ax.clear()
            ax.plot(x_range, y_range, label='f(x)')
            ax.axhline(0, color='black', linewidth=1.5)  # X-Achse markieren

            if num < len(self.x_values):
                x = self.x_values[num]
                # Plot the tangent
                tangent = self.f(x) + self.f_derivative(x) * (x_range - x)
                ax.plot(x_range, tangent, linestyle='--', label=f'Tangent at x={x:.2f}')

                # Mark point where the tangent touches f
                ax.plot(x, self.f(x), 'ro')
                y_shift = 0.05 * (max(y_range) - min(y_range))  # Relative Verschiebung basierend auf dem y-Bereich
                ax.text(x, self.f(x) + y_shift, f'({x:.2f}, {self.f(x):.2f})', color='red')

                # Mark the zero point of the tangent
                tangent_x_intercept = x - self.f(x) / self.f_derivative(x)
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
        ani = animation.FuncAnimation(fig, update, frames=len(self.x_values) * 2, interval=2000, repeat=True)
        plt.tight_layout()
        plt.show(block=False)  # Blockierender Plot

    def calculate_experimental_order_of_convergence(self):
        eoc_values = []
        if self.x_true is None:
            return eoc_values
        for i in range(len(self.x_values) - 2):
            # Now calculate with x_values[i], x_values[i+1], x_values[i+2] and x_true a numerical estimation of the order of convergence
            eoc_k = np.log(np.linalg.norm(self.x_values[i + 1] - self.x_true) / np.linalg.norm(self.x_values[i + 2] - self.x_true)) / np.log(np.linalg.norm(self.x_values[i] - self.x_true) / np.linalg.norm(self.x_values[i + 1] - self.x_true))
            eoc_values.append(eoc_k)
        return eoc_values

    def plot_eoc_values(self):
        # Plot for the EOC Values
        plt.figure()
        plt.plot(range(1, len(self.eoc_values) + 1), self.eoc_values, marker='o')
        # plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('EOC')
        plt.title('Newton EOC Plot')
        plt.show(block=False)

# Beispiel-Funktion und Ableitung
def f(x):
    return x**3 - 2*x + 2

def f_derivative(x):
    return 3*x**2 - 2


