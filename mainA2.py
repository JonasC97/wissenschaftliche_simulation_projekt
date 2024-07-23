import math
import matplotlib.pyplot as plt


from bisectionMethod import BisectionMethod
from fixpointIterationMethod import FixpointIteration
from newtonIterationMethod import NewtonIteration



def f1(x):
    return x**3 + 2*x - 5

def f1_derivative(x):
    return 3*x**2 + 2

def f2(x):
    return math.cos(x) - x

def f3(x):
    return 2 * x


fixpoint_instance = FixpointIteration(f2, 0)
fixpoint_instance.get_zeroes_with_fixpoint_iteration()
fixpoint_instance.plot_fixpoint_iteration()


bisection_instance = BisectionMethod(f1, 0, 3)
bisection_instance.get_zeroes_with_bisection()
bisection_instance.plot_bisection()


newton_instance = NewtonIteration(f1, f1_derivative, 0.5, x_true=1.328268856, show_animation=True)
newton_instance.get_zeroes_with_newton_iteration()
newton_instance.plot_standard_plots()
if newton_instance.eoc_values:
    newton_instance.plot_eoc_values()

# Blockierende Anzeige aller geöffneten Plots
plt.show()