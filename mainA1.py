import numpy as np
from lu_decomposition import lu_decomposition, solve_linear_system_with_lu_decomposition

# Gleichungs-System: 2x - y + 3z = 2, -6x - 3y - 7z = -2, 4x + 4y + 5z = -5, 8x + 2y + 12z = 2
A = np.array([
    [2, -1, 1],
    [-3, -1, 2],
    [-2, 1, 2]
], dtype=float)
b = np.array([8, -11, -3], dtype=float)
print(f"--- Solve Linear System ---")
print("with A = ")
print(A)
print(f"and b = {b}")

x, returnMessage = solve_linear_system_with_lu_decomposition(A,b)
print(returnMessage)
print(f"Solution: x = {x}")
print("\n")


print("--- Test irregular matrix (det = 0) ---")
# Irreguläre Matrix
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
x, returnMessage = solve_linear_system_with_lu_decomposition(A,b)
print(A)
print(returnMessage)
print("\n")


print("--- Test 4x3 matrix ---")
# Irreguläre Matrix
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [4, 3, 8]], dtype=float)
x, returnMessage = solve_linear_system_with_lu_decomposition(A, b)
print(A)
print(returnMessage)
