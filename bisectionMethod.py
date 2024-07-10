import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def get_zeroes_with_bisection(f, a, b, tolerance=1e-6, max_iterations=1000):
    print("---- Computing Zero with bisection method ----")

    # Check, if there is even a zero in the given interval
    if f(a) * f(b) > 0:
        print("No zero in the given interval")
        return False
    else:
        iteration = 0
        mid_points = []

        while iteration < max_iterations:
            # Compute the new mid point
            mid_point = (a + b) / 2
            mid_points.append(mid_point)

            # Save values, so they do not have to be calculated over and over again
            fa, fb, fmid = f(a), f(b), f(mid_point)

            print(f"Iteration {iteration}:")
            print(f"a = {a}, f(a) = {f(a)}")
            print(f"b = {b}, f(b) = {f(b)}")
            print(f"mid_point = {mid_point}, f(mid_point) = {fmid}")

            # We found the Zero, if fmid is closer to zero than our tolerance
            if abs(fmid) < tolerance:
                print(f"Found Zero at {mid_point} after {iteration + 1} iterations")
                plot_bisection(mid_points, f)
                return True

            # fmid is not within the tolerance area. Check, if the zero is within the left our within the right interval
            if fa * fmid < 0:
                # Zero in left interval => Keep a, Reset b
                b = mid_point
            else:
                # Zero in right interval => Reset a, Keep b
                a = mid_point

            iteration += 1

        print(f"No zero found after {max_iterations} iterations")
        plot_bisection(mid_points, f)
        return False


def plot_bisection(mid_points, f):
    iterations = range(len(mid_points))
    f_values = [f(mid) for mid in mid_points]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for f(mid_point) over iterations to show to which y values
    # the function is heading (-> 0, if it converges)
    ax1.plot(iterations, f_values, 'bo-', label='y = f(mid_point)')
    ax1.axhline(0, color='red', linestyle='--', linewidth=1)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('y')
    ax1.set_title('Convergence of Bisection Method (y-values)')
    ax1.legend()
    ax1.grid(True)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))


    # Plot for mid_points over iterations to show to which x values
    # the function is heading (-> Zero Point, if it converges)
    ax2.plot(iterations, mid_points, 'go-', label='x = mid_point')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('x')
    ax2.set_title('Convergence of Bisection Method (x-values)')
    ax2.legend()
    ax2.grid(True)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))


    plt.tight_layout()
    plt.show(block=False)

