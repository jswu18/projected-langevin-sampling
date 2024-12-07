import torch
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(
            X, dtype=torch.float32
        )  # Assuming X and Y are numpy arrays or lists
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.length = len(self.X)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def custom_dataloader(X, Y, batch_size=32, shuffle=True):
    dataset = CustomDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def check_convergence(cost_list, patience, min_delta):
    if all(
        abs(cost_list[-i] - cost_list[-i - 1]) < min_delta
        for i in range(1, patience + 1)
    ):
        print(f"Stopping early due to lack of improvement.")
        return True

    return False


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """
        Compute the mean and std to be used for later scaling.
        Parameters:
        X : torch.Tensor, shape (n_samples, n_features)
            The data used to compute the mean and standard deviation.
        """
        self.mean_ = torch.mean(X, dim=0)
        self.scale_ = torch.std(X, dim=0)

    def transform(self, X):
        """
        Perform standardization by centering and scaling.
        Parameters:
        X : torch.Tensor, shape (n_samples, n_features)
            The data that needs to be transformed.
        Returns:
        X_scaled : torch.Tensor, shape (n_samples, n_features)
            The transformed data.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("The scaler has not been fitted yet. Call 'fit' first.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        Parameters:
        X : torch.Tensor, shape (n_samples, n_features)
            The data to fit and transform.
        Returns:
        X_scaled : torch.Tensor, shape (n_samples, n_features)
            The transformed data.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        """
        Undo the standardization of X.
        Parameters:
        X_scaled : torch.Tensor, shape (n_samples, n_features)
            The data that needs to be inversely transformed.
        Returns:
        X : torch.Tensor, shape (n_samples, n_features)
            The original data before scaling.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("The scaler has not been fitted yet. Call 'fit' first.")
        return X_scaled * self.scale_ + self.mean_


def median_heuristic(X):
    """
    Computes the median heuristic for the RBF kernel lengthscale.

    Args:
    X (torch.Tensor): A tensor of shape (n_samples, n_features), where each row is a data point.

    Returns:
    float: The median pairwise distance.
    """
    # Compute pairwise squared distances
    pairwise_dists = torch.cdist(X, X, p=2)  # Euclidean distance between all pairs

    # Get the upper triangular part of the distance matrix (since it's symmetric)
    # Sets diagonal and lower triangular to zero
    pairwise_dists_flat = pairwise_dists.triu(diagonal=1).flatten()

    # Remove zero entries (corresponding to the diagonal)
    non_zero_dists = pairwise_dists_flat[pairwise_dists_flat > 0]

    # Return the median of the non-zero distances
    lengthscale = torch.median(non_zero_dists)
    return lengthscale.item()


# Example usage:
X = torch.randn(1000, 10)  # 100 samples with 10 features
lengthscale = median_heuristic(X)
print(f"Estimated lengthscale: {lengthscale}")
