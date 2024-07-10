import numpy as np

def swap_rows(A, row1, row2):
    A_copy = np.copy(A)
    for i in range(len(A)):
        A[row1, i] = A[row2, i]
        A[row2, i] = A_copy[row1, i]
    return A

def lu_decomposition(A):
    L = np.zeros_like(A)
    U = np.copy(A)
    P = np.identity(len(A))
    for i in range(len(A)):
        # print(f"Schritt {i}")

        # L[i,i] = 1
        pivot_index = i
        max = A[i, i]
        # print(max)
        for j in range(i + 1, len(A)):
            if (abs(U[j, i]) > abs(max)):
                max = U[j, i]
                pivot_index = j
        # print(f"Tausche {i} mit {pivot_index}")
        P_i = np.identity(len(A))
        swap_rows(U, i, pivot_index)
        swap_rows(L, i, pivot_index)
        swap_rows(P_i, i, pivot_index)
        P = np.matmul(P_i, P)
        # print("Nach Tausch")
        # print(R)
        # print(L)

        for j in range(i + 1, len(A)):
            L[j, i] = U[j, i] / U[i, i]
            for k in range(len(A)):
                # print("----")
                # print(U[j, k])
                # print(L[j, i])
                # print(U[i, k])
                U[j, k] = U[j, k] - L[j, i] * U[i, k]

        # print(L)
        # print(U)

        # aktuellerWert * lWert * pivotElementDarüber
    for i in range(len(A)):
        L[i, i] = 1
    return L,U,P



# A = np.array([[2,4,3,5],[-4,-7,-5,-8], [6,8,2,9],[4,9,-2,14]], dtype=float)
# A = np.array([[4,2,1],[1,4,2], [2,2,4]], dtype=float)

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

def is_square(mat):
    return mat.shape[0] == mat.shape[1]

def is_regular(mat):
    # Überprüft, ob die Determinante der Matrix ungleich null ist.
    return np.linalg.det(mat) != 0

def solve_linear_system_with_lu_decomposition(A, b):
    if not is_square(A):
        return -1, "Cannot be solved. Matrix must be square!"
    if not is_regular(A):
        return -1, "Cannot be solved. Matrix must be regular!"

    L,U,P = lu_decomposition(A)

    # Vorwärtseinsetzen für die Lösung von Ly = Pb
    Pb = np.dot(P,b)
    y = np.zeros_like(b)
    for i in range(len(b)):
        y[i] = Pb[i] - np.dot(L[i, :i], y[:i])

    # Rückwärtseinsetzen für die Lösung von Rx = y
    x = np.zeros_like(b)
    for i in range(len(b) - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x, "Solvable"


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
print(returnMessage)
print("\n")


print("--- Test 4x3 matrix ---")
# Irreguläre Matrix
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [4, 3, 8]], dtype=float)
x, returnMessage = solve_linear_system_with_lu_decomposition(A, b)
print(returnMessage)
