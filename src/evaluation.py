"""
Evaluation module for hate speech detection project.
Implements various metrics for clustering evaluation including ARI.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ClusteringEvaluator:
    """Class to evaluate clustering results."""

    def __init__(self):
        """Initialize evaluator."""
        self.results = {}

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
        label_names: Optional[Dict[int, str]] = None,
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of clustering results.

        Args:
            y_true (np.ndarray): Ground truth labels
            y_pred (np.ndarray): Predicted cluster labels
            embeddings (np.ndarray): Optional embeddings for internal metrics
            label_names (dict): Optional mapping of label indices to names

        Returns:
            dict: Dictionary of evaluation metrics
        """
        results = {}

        # Extrinsic metrics (require ground truth)
        print("=== Extrinsic Evaluation (with ground truth) ===")
        results["adjusted_rand_index"] = adjusted_rand_score(y_true, y_pred)
        results["normalized_mutual_info"] = normalized_mutual_info_score(y_true, y_pred)
        results["fowlkes_mallows"] = fowlkes_mallows_score(y_true, y_pred)
        results["homogeneity"] = homogeneity_score(y_true, y_pred)
        results["completeness"] = completeness_score(y_true, y_pred)
        results["v_measure"] = v_measure_score(y_true, y_pred)

        print(f"Adjusted Rand Index (ARI): {results['adjusted_rand_index']:.4f}")
        print(f"Normalized Mutual Info:    {results['normalized_mutual_info']:.4f}")
        print(f"Fowlkes-Mallows Score:     {results['fowlkes_mallows']:.4f}")
        print(f"Homogeneity:               {results['homogeneity']:.4f}")
        print(f"Completeness:              {results['completeness']:.4f}")
        print(f"V-Measure:                 {results['v_measure']:.4f}")

        # Intrinsic metrics (don't require ground truth)
        if embeddings is not None:
            print("\n=== Intrinsic Evaluation (without ground truth) ===")

            # Filter out noise points for DBSCAN (-1 labels)
            valid_mask = y_pred >= 0
            if valid_mask.sum() > 1 and len(set(y_pred[valid_mask])) > 1:
                results["silhouette"] = silhouette_score(
                    embeddings[valid_mask], y_pred[valid_mask]
                )
                results["calinski_harabasz"] = calinski_harabasz_score(
                    embeddings[valid_mask], y_pred[valid_mask]
                )
                results["davies_bouldin"] = davies_bouldin_score(
                    embeddings[valid_mask], y_pred[valid_mask]
                )

                print(f"Silhouette Score:          {results['silhouette']:.4f}")
                print(f"Calinski-Harabasz Index:   {results['calinski_harabasz']:.2f}")
                print(f"Davies-Bouldin Index:      {results['davies_bouldin']:.4f}")
            else:
                print("Not enough valid clusters for intrinsic metrics")

        # Cluster statistics
        print("\n=== Cluster Statistics ===")
        n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
        n_noise = list(y_pred).count(-1) if -1 in y_pred else 0

        results["n_clusters"] = n_clusters
        results["n_noise"] = n_noise

        print(f"Number of clusters found:  {n_clusters}")
        print(f"Number of noise points:    {n_noise}")

        self.results = results
        return results

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ):
        """
        Plot confusion matrix for clustering results.

        Args:
            y_true (np.ndarray): Ground truth labels
            y_pred (np.ndarray): Predicted cluster labels
            label_names (List[str]): Names for labels
            save_path (str): Path to save plot
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))

        if label_names is None:
            label_names = [f"Cluster {i}" for i in range(len(cm))]

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_names,
            yticklabels=label_names,
        )
        plt.xlabel("Predicted Cluster")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved confusion matrix to {save_path}")

        plt.show()

    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a classification report.

        Args:
            y_true (np.ndarray): Ground truth labels
            y_pred (np.ndarray): Predicted labels
            target_names (List[str]): Names for classes

        Returns:
            str: Classification report
        """
        report = classification_report(y_true, y_pred, target_names=target_names)
        print("\n=== Classification Report ===")
        print(report)
        return report

    def save_results(self, filepath: str):
        """
        Save evaluation results to CSV.

        Args:
            filepath (str): Path to save results
        """
        if not self.results:
            print("No results to save. Run evaluate() first.")
            return

        df = pd.DataFrame([self.results])

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(filepath, index=False)
        print(f"Saved results to {filepath}")


def compare_clustering_methods(
    results_dict: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    """
    Compare multiple clustering methods.

    Args:
        results_dict (dict): Dictionary mapping method names to their results
        metrics (List[str]): Metrics to compare (None for all)
        save_path (str): Path to save comparison plot
    """
    if metrics is None:
        metrics = [
            "adjusted_rand_index",
            "normalized_mutual_info",
            "homogeneity",
            "completeness",
            "v_measure",
        ]

    # Extract data for plotting
    methods = list(results_dict.keys())
    data = {metric: [] for metric in metrics}

    for method in methods:
        for metric in metrics:
            value = results_dict[method].get(metric, 0)
            data[metric].append(value)

    # Create comparison plot
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))

    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        axes[idx].bar(methods, data[metric])
        axes[idx].set_ylabel("Score")
        axes[idx].set_title(metric.replace("_", " ").title())
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot to {save_path}")

    plt.show()


def map_clusters_to_labels(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[int, int]:
    """
    Map cluster IDs to true labels using majority voting.

    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted cluster labels

    Returns:
        dict: Mapping from cluster ID to most common true label
    """
    mapping = {}

    for cluster_id in set(y_pred):
        if cluster_id == -1:  # Skip noise
            continue

        # Find most common true label in this cluster
        mask = y_pred == cluster_id
        labels_in_cluster = y_true[mask]

        if len(labels_in_cluster) > 0:
            most_common = np.bincount(labels_in_cluster).argmax()
            mapping[cluster_id] = most_common

    return mapping


if __name__ == "__main__":
    # Example usage
    print("=== Evaluation Module Example ===\n")

    # Generate sample data
    np.random.seed(42)
    n_samples = 100

    # Simulate ground truth (0: non-hate, 1: hate)
    y_true = np.random.randint(0, 2, n_samples)

    # Simulate clustering results (slightly correlated with truth)
    y_pred = y_true.copy()
    # Add some noise (20% error rate)
    flip_mask = np.random.random(n_samples) < 0.2
    y_pred[flip_mask] = 1 - y_pred[flip_mask]

    # Evaluate
    evaluator = ClusteringEvaluator()
    results = evaluator.evaluate(y_true, y_pred)

    print("\n=== Example Results ===")
    for metric, value in results.items():
        print(f"{metric}: {value}")
