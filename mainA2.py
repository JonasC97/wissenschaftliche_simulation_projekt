import math
import matplotlib.pyplot as plt


import bisectionMethod
import fixpointIterationMethod
import newtonIterationMethod



def f1(x):
    return x**3 + 2*x - 5
def f1_derivative(x):
    return 3*x**2 + 2


def f2(x):
    return math.cos(x) - x

fixpointIterationMethod.get_zeroes_with_fixpoint_iteration(f2, 0)
bisectionMethod.get_zeroes_with_bisection(f1, 0,3)
newtonIterationMethod.get_zeroes_with_newton_iteration(f1, f1_derivative, 0.5, x_true=1.328268856, show_animation=True)


plt.show()  # Blockierende Anzeige aller geöffneten Plots
