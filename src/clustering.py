"""
Clustering module for hate speech detection project.
Implements K-Means and DBSCAN clustering algorithms.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ClusteringModel:
    """Base class for clustering models."""

    def __init__(self, normalize: bool = False):
        """
        Initialize clustering model.

        Args:
            normalize (bool): Whether to normalize features before clustering
        """
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.model = None
        self.labels_ = None

    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit the clustering model.

        Args:
            embeddings (np.ndarray): Input embeddings (n_samples, n_features)

        Returns:
            np.ndarray: Cluster labels
        """
        raise NotImplementedError

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Args:
            embeddings (np.ndarray): Input embeddings

        Returns:
            np.ndarray: Predicted cluster labels
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        if self.normalize and self.scaler is not None:
            embeddings = self.scaler.transform(embeddings)

        return self.model.predict(embeddings)

    def _prepare_data(self, embeddings: np.ndarray) -> np.ndarray:
        """Prepare data by normalizing if needed."""
        if self.normalize:
            if self.scaler.mean_ is None:  # Not fitted yet
                embeddings = self.scaler.fit_transform(embeddings)
            else:
                embeddings = self.scaler.transform(embeddings)
        return embeddings


class KMeansClustering(ClusteringModel):
    """K-Means clustering implementation."""

    def __init__(
        self,
        n_clusters: int = 2,
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300,
        normalize: bool = False,
    ):
        """
        Initialize K-Means clustering.

        Args:
            n_clusters (int): Number of clusters (k=2 for hate vs non-hate)
            random_state (int): Random seed for reproducibility
            n_init (int): Number of times k-means will be run with different seeds
            max_iter (int): Maximum number of iterations
            normalize (bool): Whether to normalize features
        """
        super().__init__(normalize)
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter

        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )

    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit K-Means clustering.

        Args:
            embeddings (np.ndarray): Input embeddings

        Returns:
            np.ndarray: Cluster labels
        """
        print(f"Fitting K-Means with k={self.n_clusters}")
        print(f"Input shape: {embeddings.shape}")

        embeddings = self._prepare_data(embeddings)
        self.labels_ = self.model.fit_predict(embeddings)

        print(f"Clustering complete. Inertia: {self.model.inertia_:.2f}")
        print(f"Cluster distribution: {np.bincount(self.labels_)}")

        return self.labels_

    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers."""
        if self.model is None or not hasattr(self.model, "cluster_centers_"):
            raise ValueError("Model not fitted yet.")
        return self.model.cluster_centers_


class DBSCANClustering(ClusteringModel):
    """DBSCAN clustering implementation."""

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        normalize: bool = False,
    ):
        """
        Initialize DBSCAN clustering.

        Args:
            eps (float): Maximum distance between two samples for them to be neighbors
            min_samples (int): Minimum samples in a neighborhood for a point to be a core point
            metric (str): Distance metric to use
            normalize (bool): Whether to normalize features
        """
        super().__init__(normalize)
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

        self.model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)

    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit DBSCAN clustering.

        Args:
            embeddings (np.ndarray): Input embeddings

        Returns:
            np.ndarray: Cluster labels (-1 for noise/outliers)
        """
        print(f"Fitting DBSCAN with eps={self.eps}, min_samples={self.min_samples}")
        print(f"Input shape: {embeddings.shape}")

        embeddings = self._prepare_data(embeddings)
        self.labels_ = self.model.fit_predict(embeddings)

        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = list(self.labels_).count(-1)

        print(f"Clustering complete. Found {n_clusters} clusters")
        print(f"Noise points: {n_noise} ({n_noise/len(self.labels_)*100:.1f}%)")

        if n_clusters > 0:
            cluster_counts = np.bincount(self.labels_[self.labels_ >= 0])
            print(f"Cluster distribution: {cluster_counts}")

        return self.labels_

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        DBSCAN doesn't have a predict method by default.
        This assigns new points to nearest cluster or -1 for outliers.
        """
        raise NotImplementedError(
            "DBSCAN doesn't support prediction on new data. "
            "Use fit_predict on the entire dataset instead."
        )


def find_optimal_k(
    embeddings: np.ndarray, k_range: range = range(2, 11), random_state: int = 42
) -> Dict[str, Any]:
    """
    Find optimal number of clusters using elbow method.

    Args:
        embeddings (np.ndarray): Input embeddings
        k_range (range): Range of k values to test
        random_state (int): Random seed

    Returns:
        dict: Dictionary with inertias and silhouette scores
    """
    from sklearn.metrics import silhouette_score

    inertias = []
    silhouette_scores = []

    print("Finding optimal k...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        inertias.append(kmeans.inertia_)

        # Calculate silhouette score (skip if k=1 or only 1 cluster formed)
        if k > 1 and len(set(labels)) > 1:
            score = silhouette_score(embeddings, labels)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(0)

        print(
            f"k={k}: inertia={kmeans.inertia_:.2f}, silhouette={silhouette_scores[-1]:.3f}"
        )

    return {
        "k_values": list(k_range),
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
    }


def plot_elbow_curve(results: Dict[str, Any], save_path: Optional[str] = None):
    """
    Plot elbow curve for k-means clustering.

    Args:
        results (dict): Results from find_optimal_k
        save_path (str): Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Inertia plot
    ax1.plot(results["k_values"], results["inertias"], "bo-")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method: Inertia vs k")
    ax1.grid(True)

    # Silhouette score plot
    ax2.plot(results["k_values"], results["silhouette_scores"], "ro-")
    ax2.set_xlabel("Number of clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score vs k")
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved elbow curve to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Example usage
    print("=== Clustering Module Example ===\n")

    # Generate sample embeddings
    np.random.seed(42)
    embeddings = np.random.randn(100, 384)  # 100 samples, 384-dim embeddings

    # Test K-Means
    print("Testing K-Means...")
    kmeans = KMeansClustering(n_clusters=2)
    labels = kmeans.fit(embeddings)
    print(f"Unique labels: {np.unique(labels)}\n")

    # Test DBSCAN
    print("Testing DBSCAN...")
    dbscan = DBSCANClustering(eps=5.0, min_samples=5)
    labels = dbscan.fit(embeddings)
    print(f"Unique labels: {np.unique(labels)}")
