import math
import matplotlib.pyplot as plt


from bisectionMethod import BisectionMethod
from fixpointIterationMethod import FixpointIteration
from newtonIterationMethod import NewtonIteration



def f1(x):
    return x**3 + 2*x - 5

def f1_derivative(x):
    return 3*x**2 + 2

# Will be transformed to g(x) = cos(x) with fix point at about (0.7391, 0.7391)
# This function is pretty good for fixpoint iteration because g is everywhere a contraction
# (g'(x) = -sin(x) has always absolute values of <= 1). The convergence is therefore guaranteed for every starting point x0
# It furthermore does not really make a difference for the number of iterations, if x0 is at 0 or at -100000000
# because of the pretty constrained image area of cos(x)
def f2(x):
    return math.cos(x) - x

def f3(x):
    return 2 * x

# Will be transformed to g(x) = x^3 with fix points (0,0), (1,1), (-1,-1)
# g'(x) = 3x^2 => g'(0) = 0, g'(-1)=g'(1) = 1 => Lokale Kontraktion um 0 => Konvergiert z.B. für x0 = -0.5 mit g'(-0.5) = -0.75 (betragsmäßig < 1)
def f4(x):
    return x**3 - x

# Will be transformed to g(x) = x^3 - 3x with fix points (0,0), (2,2), (-2,-2).
# BUT: g'(x)=3x^2 - 3 => g'(0) = -3, g'(2) = g'(-2) = 9 => alle betragsmäßig größer 1
# => Keine lokalen Kontraktionen um die Fixpunkte => Fixpunktiteration konvergiert nicht.
def f5(x):
    return x**3 - 4*x

# Special Function that starts in a non-contractive area for x0 = 0.1, but still manages to jump into a contractive area.
# It also has an attracting fix point, so the iteration still converges.
# Shows that contraction criterion is only a sufficient but not necessary condition.
# f has a zero at around x=0.2236
def f6(x):
    return -2*x**3 + 0.1*x


fixpoint_instance_cos = FixpointIteration(f2, 0, plot_title="cos(x) - x, x0 = 0")
fixpoint_instance_cos.get_zeroes_with_fixpoint_iteration()
fixpoint_instance_cos.plot_fixpoint_iteration()


fixpoint_instance_cos_irrelevant_start = FixpointIteration(f2, -10000000, plot_title="cos(x) - x, x0 = -10000000")
fixpoint_instance_cos_irrelevant_start.get_zeroes_with_fixpoint_iteration()
fixpoint_instance_cos_irrelevant_start.plot_fixpoint_iteration()


fixpoint_instance_bad_non_contractive_polynom = FixpointIteration(f5, 2.5,  plot_title="x^3 - 4x")
fixpoint_instance_bad_non_contractive_polynom.get_zeroes_with_fixpoint_iteration()
fixpoint_instance_bad_non_contractive_polynom.plot_fixpoint_iteration()


fixpoint_instance_good_non_contractive_polynom = FixpointIteration(f6, 0.1, plot_title="-2x^3 + 0.1x, x0=0.1")
fixpoint_instance_good_non_contractive_polynom.get_zeroes_with_fixpoint_iteration()
fixpoint_instance_good_non_contractive_polynom.plot_fixpoint_iteration()

# bisection_instance = BisectionMethod(f1, 0, 3)
# bisection_instance.get_zeroes_with_bisection()
# bisection_instance.plot_bisection()


# newton_instance = NewtonIteration(f1, f1_derivative, 0.5, x_true=1.328268856, show_animation=True)
# newton_instance.get_zeroes_with_newton_iteration()
# newton_instance.plot_standard_plots()
# if newton_instance.eoc_values:
#     newton_instance.plot_eoc_values()

# Blockierende Anzeige aller geöffneten Plots
plt.show()