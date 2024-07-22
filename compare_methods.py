import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from bisectionMethod import BisectionMethod
from fixpointIterationMethod import FixpointIteration
from newtonIterationMethod import NewtonIteration

def f(x):
    return math.cos(x) - x

def f_derivative(x):
    return -math.sin(x) - 1

def f1(x):
    return x**2 - 2

def f1_derivative(x):
    return 2*x

def compare_methods(f, f_derivative=None, a=None, b=None, x0=None, x_true=None, tolerance=1e-6, max_iterations=1000):
    
    fixpoint_instance = FixpointIteration(f, x0, tolerance, max_iterations)
    fixpoint_instance.get_zeroes_with_fixpoint_iteration()

    
    bisection_instance = BisectionMethod(f, a, b, tolerance, max_iterations)
    bisection_instance.get_zeroes_with_bisection()

    
    newton_instance = NewtonIteration(f, f_derivative, x0, x_true, tolerance, max_iterations, show_animation=False)
    newton_instance.get_zeroes_with_newton_iteration()

    
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

   
    iterations_fixpoint = range(len(fixpoint_instance.x_values))
    iterations_bisection = range(len(bisection_instance.mid_points))
    iterations_newton = range(len(newton_instance.x_values))

    axs[0].plot(iterations_fixpoint, fixpoint_instance.x_values, 'bo-', label='Fixpoint Iteration')
    axs[0].plot(iterations_bisection, bisection_instance.mid_points, 'go-', label='Bisection Method')
    axs[0].plot(iterations_newton, newton_instance.x_values, 'ro-', label='Newton Method')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('x')
    axs[0].set_title('Comparison of x-values')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plotting errors
    axs[1].plot(iterations_fixpoint, fixpoint_instance.errors, 'bo-', label='Fixpoint Iteration')
    axs[1].plot(iterations_bisection, bisection_instance.mid_points_y_values, 'go-', label='Bisection Method')
    axs[1].plot(iterations_newton, newton_instance.f_values, 'ro-', label='Newton Method')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Error')
    axs[1].set_title('Comparison of Errors')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()


compare_methods(f, f_derivative, a=0, b=1, x0=1, x_true=1.328268856)

compare_methods(f1, f1_derivative, a=0, b=2, x0=2, x_true=1.328268856)