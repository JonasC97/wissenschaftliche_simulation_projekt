import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class BisectionMethod:
    def __init__(self, f, a, b, tolerance=1e-6, max_iterations=1000):
        self.f = f
        self.a = a
        self.b = b
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.mid_points = []
        self.mid_points_y_values = []

    def get_zeroes_with_bisection(self):
        print("---- Computing Zero with bisection method ----")

        # Check, if there is even a zero in the given interval
        if self.f(self.a) * self.f(self.b) > 0:
            print("No zero in the given interval")
            return False
        else:
            iteration = 0
            # Relevant for plotting the results

            while iteration < self.max_iterations:
                # Compute the new mid point
                mid_point = (self.a + self.b) / 2

                # Save values, so they do not have to be calculated over and over again
                fa, fb, fmid = self.f(self.a), self.f(self.b), self.f(mid_point)

                self.mid_points.append(mid_point)
                self.mid_points_y_values.append(fmid)

                print(f"Iteration {iteration}:")
                print(f"a = {self.a}, f(a) = {fa}")
                print(f"b = {self.b}, f(b) = {fb}")
                print(f"mid_point = {mid_point}, f(mid_point) = {fmid}")

                # We found the Zero, if fmid is closer to zero than our tolerance
                if abs(fmid) < self.tolerance:
                    print(f"Found Zero at {mid_point} after {iteration + 1} iterations")
                    return True

                # fmid is not within the tolerance area. Check, if the zero is within the left or within the right interval
                if fa * fmid < 0:
                    # Zero in left interval => Keep a, Reset b
                    self.b = mid_point
                else:
                    # Zero in right interval => Reset a, Keep b
                    self.a = mid_point

                iteration += 1

            print(f"No zero found after {self.max_iterations} iterations")
            return False

    def plot_bisection(self):
        iterations = range(len(self.mid_points))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot for f(mid_point) over iterations to show to which y values
        # the function is heading (-> 0, if it converges)
        ax1.plot(iterations, self.mid_points_y_values, 'bo-', label='y = f(mid_point)')
        ax1.axhline(0, color='red', linestyle='--', linewidth=1)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('y')
        ax1.set_title('Convergence of Bisection Method (y-values)')
        ax1.legend()
        ax1.grid(True)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Plot for mid_points over iterations to show to which x values
        # the function is heading (-> Zero Point, if it converges)
        ax2.plot(iterations, self.mid_points, 'go-', label='x = mid_point')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('x')
        ax2.set_title('Convergence of Bisection Method (x-values)')
        ax2.legend()
        ax2.grid(True)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        plt.show(block=False)


