import math
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.ticker import MaxNLocator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from newtonIterationMethod import NewtonIteration

def f(x):
    return (1/4)*x**3 - x + 1/5

def f_derivate(x):
    return (3/4)*x**2 - 1

def test_newton_method(f, f_derivative, x0_values, x_true=None, tolerance=1e-6, max_iterations=1000):
    for x0 in x0_values:
        newton_instance = NewtonIteration(f, f_derivative, x0, x_true, tolerance, max_iterations, show_animation=False)
        newton_instance.get_zeroes_with_newton_iteration()
        newton_instance.plot_standard_plots()
        newton_instance.plot_errors()
        if newton_instance.eoc_values:
            newton_instance.plot_eoc_values()
        plt.show()


x0_values = [0.1, 0.5, 0.9]
test_newton_method(f, f_derivate, x0_values, x_true=0.202062517)
