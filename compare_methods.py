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

def f3(x):
    return (1/4)*x**3 - x + 1/5

def f3_derivative(x):
    return (3/4)*x**2 - 1

def f4(x):
    return x**2

def f4_derivative(x):
    return 2*x

def f5(x):
    return x**2 - 2

def f5_derivative(x):
    return 2*x

def f6(x):
    return x**3 + 2*x

def f7(x):
    return x**(1/3)

def f7_derivative(x):
    return 1 / (3*x**(2/3))


def compare_methods(f, f_derivative=None, a=None, b=None, x0=None, x_true=None, tolerance=1e-6, max_iterations=1000, plot_title="f"):
    
    fixpoint_instance = FixpointIteration(f, x0, tolerance=tolerance, max_iterations=max_iterations, plot_title=plot_title)
    fixpoint_does_converge = fixpoint_instance.get_zeroes_with_fixpoint_iteration()

    
    bisection_instance = BisectionMethod(f, a, b, tolerance, max_iterations)
    bisection_instance.get_zeroes_with_bisection()

    
    newton_instance = NewtonIteration(f, f_derivative, x0, x_true, tolerance, max_iterations, show_animation=False)
    newton_instance.get_zeroes_with_newton_iteration()

    
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    if not fixpoint_does_converge:
        fixpoint_x_values = []
        fixpoint_errors = []
        iterations_fixpoint = []
    else:
        fixpoint_x_values = fixpoint_instance.x_values
        fixpoint_errors = fixpoint_instance.errors
        iterations_fixpoint = range(len(fixpoint_instance.x_values))

    iterations_bisection = range(len(bisection_instance.mid_points))
    iterations_newton = range(len(newton_instance.x_values))

    fixpoint_instance.x_values = []
    axs[0].plot(iterations_fixpoint, fixpoint_x_values, 'bo-', label='Fixpoint Iteration')
    axs[0].plot(iterations_bisection, bisection_instance.mid_points, 'go-', label='Bisection Method')
    axs[0].plot(iterations_newton, newton_instance.x_values, 'ro-', label='Newton Method')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('x')
    axs[0].set_title(f'Comparison of x-values for {plot_title}')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plotting errors
    axs[1].plot(iterations_fixpoint, fixpoint_errors, 'bo-', label='Fixpoint Iteration')
    axs[1].plot(iterations_bisection, bisection_instance.mid_points_y_values, 'go-', label='Bisection Method')
    axs[1].plot(iterations_newton, newton_instance.f_values, 'ro-', label='Newton Method')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Error')
    axs[1].set_title(f'Comparison of Errors for {plot_title}')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()


# compare_methods(f3, f3_derivative, a=0, b=1, x0=0.9, x_true=0.202062517, plot_title="f(x)=1/4x^2 - x + 1/5, x0=0.9")

compare_methods(f1, f1_derivative, a=0, b=2, x0=2, x_true=None, plot_title="f(x)=x^2 - 2, x0=2")