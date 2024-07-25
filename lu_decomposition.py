import numpy as np

def swap_rows(A, row1, row2):
    A_copy = np.copy(A)
    for i in range(len(A)):
        A[row1, i] = A[row2, i]
        A[row2, i] = A_copy[row1, i]
    return A


def lu_decomposition(A):
    # Initialisiere L als Nullmatrix mit derselben Form wie A
    L = np.zeros_like(A)

    # Kopiere A in U, da U modifiziert wird
    U = np.copy(A)

    # Initialisiere die Permutationsmatrix P als Identitätsmatrix
    P = np.identity(len(A))

    print("Initialer Zustand:")
    print("A:\n", A)
    print("L:\n", L)
    print("U:\n", U)
    print("P:\n", P)
    print("")

    # Iteriere über jede Spalte von A
    for i in range(len(A)):
        # Initialisiere den Pivot-Index auf die aktuelle Spalte
        pivot_index = i

        # Setze den initialen Maximalwert auf das Element der Diagonale in der aktuellen Zeile
        max = A[i, i]

        # Finde die Zeile mit dem größten absoluten Wert in der aktuellen Spalte unterhalb der Diagonale
        for j in range(i + 1, len(A)):
            if abs(U[j, i]) > abs(max):
                max = U[j, i]
                pivot_index = j

        # Erzeuge eine Identitätsmatrix, die für die Zeilenvertauschung genutzt wird
        P_i = np.identity(len(A))

        # Tausche die aktuelle Zeile mit der Pivot-Zeile in U
        swap_rows(U, i, pivot_index)

        # Tausche die gleiche Zeile in L, um Konsistenz zu gewährleisten
        swap_rows(L, i, pivot_index)

        # Tausche die Zeilen in der Identitätsmatrix P_i, um die Permutation zu repräsentieren
        swap_rows(P_i, i, pivot_index)

        # Aktualisiere die Permutationsmatrix P durch Multiplikation mit P_i
        P = np.matmul(P_i, P)

        print(f"Nach Zeilentausch bei Schritt {i}:")
        print("U:\n", U)
        print("L:\n", L)
        print("P:\n", P)
        print("")

        # Aktualisiere die Elemente unterhalb der Diagonale in U und fülle die entsprechenden Elemente in L
        for j in range(i + 1, len(A)):
            # Berechne den Multiplikationsfaktor für die aktuelle Zeile und speichere ihn in L
            L[j, i] = U[j, i] / U[i, i]

            # Aktualisiere die Zeile j in U
            for k in range(len(A)):
                U[j, k] = U[j, k] - L[j, i] * U[i, k]
            print(f"Nach Schritt {i} und Zeile {j}:")
            print("U:\n", U)
            print("L:\n", L)
            print("P:\n", P)

    # Setze die Diagonalelemente von L auf 1
    for i in range(len(A)):
        L[i, i] = 1

    print("Finaler Zustand:")
    print("L:\n", L)
    print("U:\n", U)
    print("P:\n", P)

    return L, U, P


def is_square(mat):
    # Überprüft, ob die Anzahl der Zeilen und Spalten der Matrix gleich sind
    return mat.shape[0] == mat.shape[1]


def is_regular(mat):
    # Überprüft, ob die Determinante der Matrix ungleich null ist.
    return np.linalg.det(mat) != 0


def solve_linear_system_with_lu_decomposition(A, b):
    # Überprüfe, ob die Matrix quadratisch ist
    if not is_square(A):
        return -1, "Cannot be solved. Matrix must be a square!"

    # Überprüfe, ob die Matrix regulär ist (Determinante ungleich null)
    if not is_regular(A):
        return -1, "Cannot be solved. Matrix must be regular!"

    # Führe die LU-Zerlegung durch
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
