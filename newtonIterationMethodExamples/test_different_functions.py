import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from newtonIterationMethod import NewtonIteration

def f1(x):
    return x**2 - 4

def f1_derivative(x):
    return 2*x

    return 3*x**2 - 12*x + 11

def f7(x):
    return x**3

def f7_derivate(x):
    return 3*x**2

def f8(x):
    return x**(1/3)

def f8_derivate(x):
    return 1 / (3 * x**(2/3))

def f9(x):
    return 1 / (1 + x**2) - 1 / 2

def f9_derivate(x):
    return -2 * x / (1 + x**2)**2

def f10(x):
    return (x-2)**2-1

def f10_derivate(x):
    return 2*(x-2)


# Testfunktion für das Newton-Verfahren mit einer gegebenen Funktion
def test_newton_method(f, f_derivative, x0, show_animation, x_true=None, tolerance=1e-6, max_iterations=1000):
    newton_instance = NewtonIteration(f, f_derivative, x0, x_true, tolerance, max_iterations, show_animation)
    newton_instance.get_zeroes_with_newton_iteration()
    newton_instance.plot_standard_plots()
    newton_instance.plot_errors()
    if newton_instance.eoc_values:
      newton_instance.plot_eoc_values()
      plt.show()
    plt.show()

# Beispielhafte Tests mit verschiedenen Funktionen und Startwerten
functions = [
    (f1, f1_derivative, 1, 2, True),    # Function that works well 
    (f7, f7_derivate, 1, 0, True),      # Function that converges slower, due to low gradient at x=0          
    (f8, f8_derivate, 2, 0, False),     # Case where the function diverge for x!=0
    (f9, f9_derivate, 2, 1, True),      # Converges to x=-1 instead of x=1
    (f10, f10_derivate, 2, 3, True)     # Bad starting point results in horizontal tangent
]

for f, f_derivative, x0, x_true, show_animation in functions:
    try:
        test_newton_method(f, f_derivative, x0, show_animation, x_true)
    except Exception as e:
        print(f"Fehler bei der Funktion {f.__name__} mit Startwert {x0}: {e}")
