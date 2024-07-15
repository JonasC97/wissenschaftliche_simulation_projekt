import numpy as np
import math
from newtonMultidimensional import newton_method, plot_convergence



# Example function f(x,y) = (x^2 + y^2 - 1, x - y)
def F_example(x):
    return np.array([x[0]**2 + x[1]**2 - 1, x[0] - x[1]])

# and its Jacobi Matrix J(x,y) = ((2x, 2y), (1, -1))
def J_example(x):
    return np.array([[2*x[0], 2*x[1]], [1, -1]])

# Start value
x0 = np.array([0.5, 0.5])

# True value (for eoc-calculation)
x_true = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)])

# Execute Newton Method
solution, iterations, errors = newton_method(F_example, J_example, x0, x_true)

print(f"Solution: {solution}")
print(f"Number of Iterations: {iterations}")
print(f"Errors: {errors}")

# if len(errors) > 2:
#     conv_order = experimental_convergence_order(errors)
#     print(f"Experimentelle Konvergenzordnung: {conv_order}")
#
plot_convergence(errors)