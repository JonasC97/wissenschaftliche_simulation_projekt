import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class BisectionMethod:
    def __init__(self, f, a, b, tolerance=1e-6, max_iterations=1000):
        # Initialisiert die BisectionMethod-Klasse mit den übergebenen Parametern.
        # f ist die Funktion, für die die Nullstelle gesucht wird.
        # a und b sind die Intervalgrenzen.
        # tolerance gibt die Genauigkeit der Lösung an.
        # max_iterations ist die maximale Anzahl an Iterationen.
        self.f = f
        self.a = a
        self.b = b
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.mid_points = []  # Liste zur Speicherung der berechneten Mittelpunkte
        self.mid_points_y_values = []  # Liste zur Speicherung der Funktionswerte an den Mittelpunkten

    def get_zeroes_with_bisection(self):
        # Methode zur Berechnung der Nullstelle mittels der Bisektionsmethode.
        print("---- Computing Zero with bisection method ----")

        # Überprüft, ob in dem gegebenen Intervall eine Nullstelle liegt.
        if self.f(self.a) * self.f(self.b) > 0:
            print("No zero in the given interval")
            return False
        else:
            iteration = 0

            while iteration < self.max_iterations:
                # Berechnet den neuen Mittelpunkt des Intervalls.
                mid_point = (self.a + self.b) / 2

                # Berechnet die Funktionswerte an den Intervallgrenzen und am Mittelpunkt.
                fa, fb, fmid = self.f(self.a), self.f(self.b), self.f(mid_point)

                # Speichert die Mittelpunkte und deren Funktionswerte.
                self.mid_points.append(mid_point)
                self.mid_points_y_values.append(fmid)

                print(f"Iteration {iteration}:")
                print(f"a = {self.a}, f(a) = {fa}")
                print(f"b = {self.b}, f(b) = {fb}")
                print(f"mid_point = {mid_point}, f(mid_point) = {fmid}")

                # Überprüft, ob der Funktionswert am Mittelpunkt innerhalb der Toleranz liegt.
                if abs(fmid) < self.tolerance:
                    print(f"Found Zero at {mid_point} after {iteration + 1} iterations")
                    return True

                # Bestimmt, ob die Nullstelle im linken oder rechten Teilintervall liegt und passt das Intervall entsprechend an.
                if fa * fmid < 0:
                    self.b = mid_point
                else:
                    self.a = mid_point

                iteration += 1

            print(f"No zero found after {self.max_iterations} iterations")
            return False

    def plot_bisection(self):
        # Methode zur Visualisierung der Konvergenz der Bisektionsmethode.
        iterations = range(len(self.mid_points))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot der Funktionswerte der Mittelpunkte über die Iterationen.
        ax1.plot(iterations, self.mid_points_y_values, 'bo-', label='y = f(mid_point)')
        ax1.axhline(0, color='red', linestyle='--', linewidth=1)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('y')
        ax1.set_title('Convergence of Bisection Method (y-values)')
        ax1.legend()
        ax1.grid(True)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Plot der Mittelpunkte über die Iterationen.
        ax2.plot(iterations, self.mid_points, 'go-', label='x = mid_point')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('x')
        ax2.set_title('Convergence of Bisection Method (x-values)')
        ax2.legend()
        ax2.grid(True)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        plt.show(block=False)
