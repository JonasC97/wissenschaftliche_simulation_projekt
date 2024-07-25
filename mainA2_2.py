import math
import matplotlib.pyplot as plt


from bisectionMethod import BisectionMethod

def f1(x):
    return x**3 + 2*x - 5

def f1_derivative(x):
    return 3*x**2 + 2

def f2(x):
    return math.cos(x) - x

def f3(x):
    return 2 * x

def f4(x):
    return x**4 - 2


bisection_instance = BisectionMethod(f1, 0, 3)
bisection_instance.get_zeroes_with_bisection()
bisection_instance.plot_bisection()

# Blockierende Anzeige aller geöffneten Plots
plt.show()
