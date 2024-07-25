import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from bisectionMethod import BisectionMethod
from fixpointIterationMethod import FixpointIteration
from newtonIterationMethod import NewtonIteration

def f(x):
    return math.cos(x) - x

def f_derivate(x):
    return -math.sin(x) - 1

def f1(x):
    return x**2 - 2

def f1_derivate(x):
    return 2*x

def f3(x):
    return (1/4)*x**3 - x + 1/5

def f3_derivate(x):
    return (3/4)*x**2 - 1

def f4(x):
    return x**(1/3)

def f4_derivate(x):
    return 1 / (3 * x**(2/3))

def compare_methods(f, f_derivative=None, a=0, b=3, x0=None, x_true=None, tolerance=1e-6, max_iterations=1000):
    results = []

    fixpoint_instance = FixpointIteration(f, x0, tolerance, max_iterations)
    fixpoint_instance.get_zeroes_with_fixpoint_iteration()
    fixpoint_instance.plot_fixpoint_iteration()
    fixpoint_result = {
        'Method': 'Fixpoint Iteration',
        'Iterations': fixpoint_instance.iterations,
        'x': fixpoint_instance.x_values[-1] if fixpoint_instance.x_values else None,
        'EOC': None,
        'Error': None
    }
    results.append(fixpoint_result)

    bisection_instance = BisectionMethod(f, a, b, tolerance, max_iterations)
    bisection_instance.get_zeroes_with_bisection()
    bisection_instance.plot_bisection()
    bisection_result = {
        'Method': 'Bisection',
        'Iterations': bisection_instance.iterations,
        'x': bisection_instance.mid_points[-1] if bisection_instance.mid_points else None,
        'EOC': None,
        'Error': None
    }
    results.append(bisection_result)

    newton_instance = NewtonIteration(f, f_derivative, x0, x_true, tolerance, max_iterations, show_animation=False)
    newton_instance.get_zeroes_with_newton_iteration()
    newton_instance.plot_standard_plots()
    newton_instance.plot_errors()
    newton_instance.plot_eoc_values()
    newton_result = {
        'Method': 'Newton',
        'Iterations': newton_instance.iterations,
        'x': newton_instance.x_values[-1] if newton_instance.x_values else None,
        'EOC': newton_instance.eoc_values[-1] if newton_instance.eoc_values else None,
        'Error': newton_instance.errors[-1] if newton_instance.errors else None
    }
    results.append(newton_result)

    plt.show()
    return results

functions = [
(f, f_derivate, 0, 3, 1, 0.7390851086421),
(f1, f1_derivate, 0, 3, 1, math.sqrt(2)), 
(f3, f3_derivate, 0, 1, 0.5, 0.20206236349401457),
 (f4, f4_derivate, 0, 3, 2, 0.0)   
]

results = []

for f, f_derivative, a, b, x0, x_true, in functions:
    try:
        result = compare_methods(f, f_derivative, a=a, b=b, x0=x0, x_true=x_true)
        results.extend(result)
    except Exception as e:
        print(f"Fehler bei der Funktion {f.__name__} mit Startwert {x0}: {e}")

df = pd.DataFrame(results)
df.to_csv('results.csv', index=False)

print(df)


