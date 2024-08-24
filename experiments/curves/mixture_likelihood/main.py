import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


# Define the sine function that repeats k times in intervall x_min, x_max
def sine(x, k, x_min, x_max):
    # Scale x to [0, 2π * k]
    scaled_x = (x - x_min) * (2 * np.pi * k) / (x_max - x_min)
    return np.sin(scaled_x)


# This function models the relationship between temperature and the revenue swimming pool
def f(x):
    res = -0.1 * (x - 28) ** 2 + 26
    res += 2 * sine(x, 5, 18, 40)

    return res


def plot_function(f, lower=15, upper=40, N=100):
    temp = np.linspace(lower, upper, N)

    # Plot the function
    plt.scatter(temp, f(temp))
    plt.xlabel("temperature in celsius")
    plt.ylabel("f(x)")
    plt.title("The mean function on weekends")
    plt.legend()
    plt.grid(True)
    plt.show()


def simulate_temperature(mu=20, sigma=8, lower=15, upper=40, N=1000):
    # Generate the temperature range
    temperature_values = np.arange(lower, upper + 1)

    # Create the truncated normal distribution
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    truncated_normal = stats.truncnorm(a, b, loc=mu, scale=sigma)

    # Calculate probabilities
    probabilities = truncated_normal.pdf(temperature_values)
    probabilities /= probabilities.sum()  # Normalize to sum to 1

    # Pair temperatures with their corresponding probabilities
    # temperature_distribution = dict(zip(temperature_values, probabilities))
    # print(temperature_distribution)

    samples = np.random.choice(temperature_values, size=N, p=probabilities)

    return samples


def simulate_observations(f, x, c=-10, sigma=1):
    # Y = f(x) + c * U + sigma* epsilon
    # U is bernoulli with prob 5/7

    N = len(x)

    U = np.random.binomial(1, p=5.0 / 7, size=N)
    eps = np.random.normal(loc=0, scale=1, size=N)

    Y = f(x) + c * U + sigma * eps

    return Y


if __name__ == "__main__":
    # This plots the mean function that is the true underlying relationship between temp and revenue on weekends
    # We assume that the weekday relationship is the same but with an offest c, r.g. c=-10
    plot_function(f)

    # Simulates temperatures with a somewhat realistic distribution
    # Its a discrete distribution on [lower,lower+1,...,upper]
    # It stems from truncating a normal distribution and then discretizing it
    temp = simulate_temperature(mu=20, sigma=8, lower=15, upper=40, N=1000)

    # Generates observations Y. We essentially assume a mixture distribution with mean-function f(x)
    # Mixture weights are 5/7 and 2/7 corresponding to weekends and week days

    y_values = simulate_observations(f, temp)

    # Plot the function
    plt.scatter(temp, y_values)
    plt.xlabel("temperature in celsius")
    plt.ylabel("y")
    plt.title("Revenue of the swimming pool")
    plt.legend()
    plt.grid(True)
    plt.show()
