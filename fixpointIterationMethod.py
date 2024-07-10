import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


# f is a function whose zero we want to determine. Since this is potentially very complex or perhaps even impossible
# to determine directly, we convert this problem of calculating the zero into a fixed point calculation, which we can approach iteratively
# iteratively.

# So instead of f(x) = 0, we want to have a function of the form g(x) = x.
# We now define g(x) := f(x) + x <=> x = f(x) + x <=> f(x) = 0
# The fixed points of g(x) are now the zeros of f(x)
def g(f, x):
    return f(x) + x

# This is where the fixed point iteration takes place. Starting from an initial value x0, the next
# x-values are calculated successively. If the distance between two consecutive x-values is smaller than the passed
# tolerance value, the function evaluates this as convergence and returns the corresponding x value.
# A maximum of 1000 iteration steps are checked.
def get_zeroes_with_fixpoint_iteration(f, x0, tolerance=1e-6, max_iterations=1000):
    print("---- Computing Zero with fixpoint iteration method ----")
    x = x0

    # Keep track of values for the plotting
    x_values = [x0]
    errors = []

    for i in range(max_iterations):
        try:
            print(f"Step {i+1}. x = {x}")
            # Get the next x by computing the y for it with g
            x_next = g(f, x)
            # Error = How large is the deviation of the last x-value from the new x-value
            # If this deviation is smaller than the tolerance range, g has a fixpoint and therefore f has a zero point
            error = abs(x_next - x)
            errors.append(error)
            # print(f"Result: {x_next} (Error: {error})")
            # print("-------")
            if error < tolerance:
                print(f"Found zero at {x_next}")
                # Plot the results
                plot_fixpoint_iteration(x_values, errors)
                return True
            if np.isnan(x_next) or np.isinf(x_next):
                print("Numerical instability detected. Stopping iteration.")
                break
            x = x_next
            x_values.append(x)
        # The method is very susceptible to error explosions, especially if f, g and x0 are chosen inappropriately
        # With steeply increasing functions, this can very quickly cause the values to fly out of the validity ranges
        # This should be intercepted.
        except OverflowError as e:
            print(f"Overflow error at step {i+1}: {e}")
            break
        except ValueError as e:
            print(f"Value error at step {i+1}: {e}")
            break
        except Exception as e:
            print(f"Unexpected error at step {i+1}: {e}")
            break

    print(f"No zero found after {max_iterations} iterations")
    # Plot the results
    plot_fixpoint_iteration(x_values, errors)
    return False

def plot_fixpoint_iteration(x_values, errors):
    iterations = range(len(x_values))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for x values over iterations
    # Plot for mid_points to show to which x values the function is heading (-> Zero Point, if it converges)
    ax1.plot(iterations, x_values, 'bo-', label='x values')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('x')
    ax1.set_title('Convergence of Fixpoint Iteration (x-values)')
    ax1.legend()
    ax1.grid(True)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot for errors over iterations to show to which final error
    # the function is heading (-> 0, if it converges)
    ax2.plot(iterations, errors, 'go-', label='errors')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Error')
    ax2.set_title('Convergence of Fixpoint Iteration (errors)')
    ax2.legend()
    ax2.grid(True)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show(block=False)
