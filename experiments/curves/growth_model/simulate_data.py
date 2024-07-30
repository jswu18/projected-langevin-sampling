from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def simulate_data(N: int, sigma: float, T) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.linspace(0, T, N + 1)
    pop_size = np.zeros(N + 1)
    noise = np.random.normal(0, 1, N)
    Y = np.zeros(N)

    pop_size[0] = 0.1

    for n in range(1, N + 1):
        dt = time[n] - time[n - 1]
        pop_size[n] = (
            pop_size[n - 1]
            + dt * pop_size[n - 1] * (1 - pop_size[n - 1])
            + dt * sigma * noise[n - 1]
        )
        Y[n - 1] = (pop_size[n] - pop_size[n - 1]) / dt

    return Y, time[1:], pop_size


def make_plot(time: np.ndarray, growth: np.ndarray) -> None:
    # Step 2: Create the Plot
    plt.figure(figsize=(10, 6))  # Set the figure size

    plt.plot(
        time, growth, marker="o", linestyle="-", color="b"
    )  # Plot with line and markers

    # Step 3: Add Labels and Title
    plt.xlabel("Time")  # X-axis label
    plt.ylabel("Growth")  # Y-axis label
    plt.title("Growth Over Time")  # Plot title

    # Optional: Add a grid
    plt.grid(True)

    # Step 4: Show the Plot
    plt.show()


if __name__ == "__main__":
    Y, time, pop_size = simulate_data(N=501, sigma=0.0025, T=5)
    make_plot(time, pop_size[1:])
    make_plot(time, Y)
