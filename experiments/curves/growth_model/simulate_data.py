import matplotlib.pyplot as plt
import numpy as np

from experiments.curves.growth_model.main import get_experiment_data


def make_plot(time: np.ndarray, growth: np.ndarray) -> None:
    # Step 2: Create the Plot
    plt.figure(figsize=(10, 6))

    plt.plot(time, growth, marker="o", linestyle="-", color="b")
    # Step 3: Add Labels and Title
    plt.xlabel("Time")  # X-axis label
    plt.ylabel("Growth")  # Y-axis label
    plt.title("Growth Over Time")  # Plot title

    # Optional: Add a grid
    plt.grid(True)

    # Step 4: Show the Plot
    plt.show()


if __name__ == "__main__":
    experiment_data = get_experiment_data(
        seed=0,
        number_of_data_points=501,
        observation_noise=0.0025,
        end_time=5,
        train_data_percentage=0.7,
        validation_data_percentage=0.1,
    )
    make_plot(experiment_data.full.x, experiment_data.full.y_untransformed)
    make_plot(experiment_data.full.x, experiment_data.full.y)
