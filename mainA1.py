import numpy as np
from lu_decomposition import lu_decomposition, solve_linear_system_with_lu_decomposition

A = np.array([[1,6,1],[2,3,2],[4,2,1]], dtype=float)
print("-- Teste LR-Zerlegung für: --")
print(A)
L,U,P = lu_decomposition(A)

print("L:")
print(L)

print("U:")
print(U)

print("P:")
print(P)

print("Test: L*U = ")
print(np.matmul(L,U))

print("Cross-check: P*A = ")
print(np.matmul(P,A))
print("\n")


#######################################


A = np.array([[2, -2, 4], [1, 3, 6], [-1, 2, 1]], dtype=float)
b = np.array([10, 25, 6], dtype=float)
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
