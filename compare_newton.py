import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from newtonIterationMethod import NewtonIteration

def f(x):
    return x**2 - 2

def f_derivative(x):
    return 2*x

def test_newton_method(f, f_derivative, x0_values, x_true=None, tolerance=1e-6, max_iterations=1000):
    all_x_values = []
    all_f_values = []
    all_iterations = []

    for x0 in x0_values:
        newton_instance = NewtonIteration(f, f_derivative, x0, x_true, tolerance, max_iterations, show_animation=False)
        newton_instance.get_zeroes_with_newton_iteration()
        all_x_values.append(newton_instance.x_values)
        all_f_values.append(newton_instance.f_values)
        all_iterations.append(len(newton_instance.x_values))


    fig, axs = plt.subplots(2, 1, figsize=(12, 12))


    for i, x_values in enumerate(all_x_values):
        iterations = range(len(x_values))
        axs[0].plot(iterations, x_values, label=f'x0 = {x0_values[i]}')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('x')
    axs[0].set_title('Newton Method: x-values for different x0')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))


    for i, f_values in enumerate(all_f_values):
        iterations = range(len(f_values))
        axs[1].plot(iterations, f_values, label=f'x0 = {x0_values[i]}')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Error')
    axs[1].set_title('Newton Method: Errors for different x0')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()

# Beispiel für die Verwendung der Testfunktion
x0_values = [0.1, 0.5, 1.0, 1.5, 5.0]
test_newton_method(f, f_derivative, x0_values, x_true=None)
