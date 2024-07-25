import math
import matplotlib.pyplot as plt


from bisectionMethod import BisectionMethod
from fixpointIterationMethod import FixpointIteration
from newtonIterationMethod import NewtonIteration

# Will be transformed to g(x) = cos(x) with fix point at about (0.7391, 0.7391)
# This function is pretty good for fixpoint iteration because g is everywhere a contraction
# (g'(x) = -sin(x) has always absolute values of <= 1). The convergence is therefore guaranteed for every starting point x0
# It furthermore does not really make a difference for the number of iterations, if x0 is at 0 or at -100000000
# because of the pretty constrained image area of cos(x)
def f1(x):
    return math.cos(x) - x


# Will be transformed to g(x) = x^3 - 3x with fix points (0,0), (2,2), (-2,-2).
# BUT: g'(x)=3x^2 - 3 => g'(0) = -3, g'(2) = g'(-2) = 9 => alle betragsmäßig größer 1
# => Keine lokalen Kontraktionen um die Fixpunkte => Fixpunktiteration konvergiert nicht.
def f2(x):
    return x**3 - 4*x

# Special Function that starts in a non-contractive area for x0 = 0.1, but still manages to jump into a contractive area.
# It also has an attracting fix point, so the iteration still converges.
# Shows that contraction criterion is only a sufficient but not necessary condition.
# f has a zero at around x=0.2236
def f3(x):
    return -2*x**3 + 0.1*x

fixpoint_instance_cos = FixpointIteration(f1, 0, plot_title="cos(x) - x, x0 = 0")
fixpoint_instance_cos.get_zeroes_with_fixpoint_iteration()
fixpoint_instance_cos.plot_fixpoint_iteration()

fixpoint_instance_cos_irrelevant_start = FixpointIteration(f1, -10000000, plot_title="cos(x) - x, x0 = -10000000")
fixpoint_instance_cos_irrelevant_start.get_zeroes_with_fixpoint_iteration()
fixpoint_instance_cos_irrelevant_start.plot_fixpoint_iteration()

fixpoint_instance_bad_non_contractive_polynom = FixpointIteration(f2, 2.5,  plot_title="x^3 - 4x")
fixpoint_instance_bad_non_contractive_polynom.get_zeroes_with_fixpoint_iteration()
fixpoint_instance_bad_non_contractive_polynom.plot_fixpoint_iteration()

fixpoint_instance_good_non_contractive_polynom = FixpointIteration(f3, 0.1, plot_title="-2x^3 + 0.1x, x0=0.1")
fixpoint_instance_good_non_contractive_polynom.get_zeroes_with_fixpoint_iteration()
fixpoint_instance_good_non_contractive_polynom.plot_fixpoint_iteration()


# Blockierende Anzeige aller geöffneten Plots
plt.show()