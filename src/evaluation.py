"""
Evaluation module for hate speech detection.
Provides metrics for both supervised and unsupervised evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    silhouette_score,
    normalized_mutual_info_score
)
from typing import Dict, Optional, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list] = None
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        labels (list, optional): List of label names
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    # Filter out samples with -1 predictions (unlabeled)
    valid_mask = (y_pred != -1) & (y_true != -1)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    if len(y_true_valid) == 0:
        return {
            'accuracy': 0.0,
            'macro_precision': 0.0,
            'macro_recall': 0.0,
            'macro_f1': 0.0,
            'weighted_precision': 0.0,
            'weighted_recall': 0.0,
            'weighted_f1': 0.0
        }
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_valid, y_pred_valid)
    
    # Macro-averaged metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_valid, y_pred_valid, average='macro', zero_division=0
    )
    
    # Weighted metrics
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_valid, y_pred_valid, average='weighted', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'macro_precision': precision_macro,
        'macro_recall': recall_macro,
        'macro_f1': f1_macro,
        'weighted_precision': precision_weighted,
        'weighted_recall': recall_weighted,
        'weighted_f1': f1_weighted,
        'n_samples': len(y_true_valid)
    }
    
    return metrics


def calculate_cluster_purity(
    cluster_labels: np.ndarray,
    true_labels: np.ndarray
) -> float:
    """
    Calculate cluster purity.
    
    Purity measures how "pure" each cluster is with respect to true labels.
    Higher is better.
    
    Args:
        cluster_labels (np.ndarray): Cluster assignments
        true_labels (np.ndarray): True labels
        
    Returns:
        float: Purity score [0, 1]
    """
    # Remove noise points
    valid_mask = cluster_labels != -1
    cluster_labels = cluster_labels[valid_mask]
    true_labels = true_labels[valid_mask]
    
    if len(cluster_labels) == 0:
        return 0.0
    
    # Calculate purity
    total_correct = 0
    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in unique_clusters:
        # Get true labels for samples in this cluster
        cluster_mask = cluster_labels == cluster_id
        labels_in_cluster = true_labels[cluster_mask]
        
        # Count most common label
        if len(labels_in_cluster) > 0:
            unique, counts = np.unique(labels_in_cluster, return_counts=True)
            most_common_count = counts.max()
            total_correct += most_common_count
    
    purity = total_correct / len(cluster_labels)
    return purity


def calculate_silhouette_score(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    metric: str = 'euclidean',
    sample_size: Optional[int] = None
) -> float:
    """
    Calculate silhouette score for clustering.
    
    The silhouette score measures how similar an object is to its own cluster
    compared to other clusters. Higher values indicate better clustering.
    
    Range: [-1, 1]
    - +1: Sample is far from neighboring clusters (ideal)
    - 0: Sample is on or very close to decision boundary
    - -1: Sample is assigned to wrong cluster
    
    Args:
        embeddings (np.ndarray): Feature vectors/embeddings (n_samples, n_features)
        cluster_labels (np.ndarray): Cluster assignments for each sample
        metric (str): Distance metric ('euclidean', 'cosine', etc.)
        sample_size (int, optional): Subsample for speed. None = use all samples
        
    Returns:
        float: Silhouette score [-1, 1]
    """
    # Remove noise points (-1 labels)
    valid_mask = cluster_labels != -1
    embeddings_valid = embeddings[valid_mask]
    labels_valid = cluster_labels[valid_mask]
    
    # Check if we have enough samples and clusters
    n_samples = len(labels_valid)
    n_clusters = len(np.unique(labels_valid))
    
    if n_samples < 2:
        print("Warning: Need at least 2 samples to calculate silhouette score")
        return 0.0
    
    if n_clusters < 2:
        print("Warning: Need at least 2 clusters to calculate silhouette score")
        return 0.0
    
    # Subsample if requested (for large datasets)
    if sample_size is not None and n_samples > sample_size:
        indices = np.random.choice(n_samples, sample_size, replace=False)
        embeddings_valid = embeddings_valid[indices]
        labels_valid = labels_valid[indices]
    
    try:
        score = silhouette_score(
            embeddings_valid,
            labels_valid,
            metric=metric
        )
        return float(score)
    except Exception as e:
        print(f"Warning: Could not calculate silhouette score: {e}")
        return 0.0


def calculate_normalized_mutual_info(
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    average_method: str = 'arithmetic'
) -> float:
    """
    Calculate Normalized Mutual Information (NMI) between cluster and true labels.
    
    NMI measures the mutual dependence between two labelings.
    It is normalized to [0, 1] where:
    - 1: Perfect correlation (clusters perfectly match true labels)
    - 0: No correlation (independent labelings)
    
    Args:
        cluster_labels (np.ndarray): Cluster assignments
        true_labels (np.ndarray): Ground truth labels
        average_method (str): How to average MI ('arithmetic', 'geometric', 'min', 'max')
        
    Returns:
        float: NMI score [0, 1]
    """
    # Remove noise points
    valid_mask = cluster_labels != -1
    cluster_labels_valid = cluster_labels[valid_mask]
    true_labels_valid = true_labels[valid_mask]
    
    if len(cluster_labels_valid) == 0:
        return 0.0
    
    try:
        nmi = normalized_mutual_info_score(
            true_labels_valid,
            cluster_labels_valid,
            average_method=average_method
        )
        return float(nmi)
    except Exception as e:
        print(f"Warning: Could not calculate NMI: {e}")
        return 0.0


def evaluate_clustering(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    metric: str = 'euclidean',
    verbose: bool = True
) -> Dict[str, float]:
    """
    Comprehensive clustering evaluation using multiple metrics.
    
    Args:
        embeddings (np.ndarray): Feature vectors (n_samples, n_features)
        cluster_labels (np.ndarray): Cluster assignments
        true_labels (np.ndarray): Ground truth labels
        metric (str): Distance metric for silhouette score
        verbose (bool): Whether to print results
        
    Returns:
        Dict[str, float]: Dictionary containing:
            - silhouette_score: Measures cluster separation [-1, 1]
            - cluster_purity: Measures cluster homogeneity [0, 1]
            - nmi: Normalized Mutual Information [0, 1]
            - n_clusters: Number of clusters found
            - n_samples: Number of samples evaluated
    """
    # Calculate metrics
    silhouette = calculate_silhouette_score(embeddings, cluster_labels, metric=metric)
    purity = calculate_cluster_purity(cluster_labels, true_labels)
    nmi = calculate_normalized_mutual_info(cluster_labels, true_labels)
    
    # Count clusters and samples
    valid_mask = cluster_labels != -1
    n_samples = valid_mask.sum()
    n_clusters = len(np.unique(cluster_labels[valid_mask]))
    
    metrics = {
        'silhouette_score': silhouette,
        'cluster_purity': purity,
        'nmi': nmi,
        'n_clusters': n_clusters,
        'n_samples': n_samples
    }
    
    if verbose:
        print("\n" + "=" * 70)
        print("CLUSTERING EVALUATION")
        print("=" * 70)
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of samples: {n_samples}")
        print(f"\nMetrics:")
        print(f"  Silhouette Score: {silhouette:.4f}  (range: [-1, 1], higher is better)")
        print(f"  Cluster Purity:   {purity:.4f}  (range: [0, 1], higher is better)")
        print(f"  NMI Score:        {nmi:.4f}  (range: [0, 1], higher is better)")
        print("=" * 70)
    
    return metrics


def compare_clustering_methods(
    embeddings: np.ndarray,
    clustering_results: Dict[str, np.ndarray],
    true_labels: np.ndarray,
    metric: str = 'euclidean'
) -> pd.DataFrame:
    """
    Compare multiple clustering methods using all metrics.
    
    Args:
        embeddings (np.ndarray): Feature vectors
        clustering_results (Dict[str, np.ndarray]): Dictionary mapping method names to cluster labels
        true_labels (np.ndarray): Ground truth labels
        metric (str): Distance metric for silhouette score
        
    Returns:
        pd.DataFrame: Comparison table with all metrics
    """
    results = []
    
    for method_name, cluster_labels in clustering_results.items():
        metrics = evaluate_clustering(
            embeddings,
            cluster_labels,
            true_labels,
            metric=metric,
            verbose=False
        )
        metrics['method'] = method_name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    # Reorder columns
    column_order = ['method', 'n_clusters', 'silhouette_score', 'cluster_purity', 'nmi', 'n_samples']
    df = df[column_order]
    
    # Sort by a composite score (average of normalized metrics)
    df['composite_score'] = (
        (df['silhouette_score'] + 1) / 2 +  # Normalize from [-1,1] to [0,1]
        df['cluster_purity'] +
        df['nmi']
    ) / 3
    
    df = df.sort_values('composite_score', ascending=False)
    
    return df


def plot_clustering_metrics(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
    title: str = "Clustering Metrics Comparison"
):
    """
    Plot comparison of clustering metrics across different methods.
    
    Args:
        metrics_dict (Dict[str, Dict[str, float]]): Dictionary mapping method names to metrics
        save_path (Path, optional): Path to save figure
        title (str): Plot title
    """
    methods = list(metrics_dict.keys())
    silhouette_scores = [metrics_dict[m]['silhouette_score'] for m in methods]
    purity_scores = [metrics_dict[m]['cluster_purity'] for m in methods]
    nmi_scores = [metrics_dict[m]['nmi'] for m in methods]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Silhouette Score
    ax = axes[0]
    bars = ax.bar(methods, silhouette_scores, color='skyblue', edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Silhouette Score', fontweight='bold')
    ax.set_title('Silhouette Score\n(Higher = Better Separation)', fontsize=11)
    ax.set_ylim([-1, 1])
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontweight='bold', fontsize=10)
    
    # Cluster Purity
    ax = axes[1]
    bars = ax.bar(methods, purity_scores, color='lightgreen', edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Cluster Purity', fontweight='bold')
    ax.set_title('Cluster Purity\n(Higher = More Homogeneous)', fontsize=11)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # NMI Score
    ax = axes[2]
    bars = ax.bar(methods, nmi_scores, color='salmon', edgecolor='black', linewidth=1.5)
    ax.set_ylabel('NMI Score', fontweight='bold')
    ax.set_title('Normalized Mutual Information\n(Higher = Better Match)', fontsize=11)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Rotate x-axis labels if needed
    for ax in axes:
        if len(methods) > 3:
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved clustering metrics plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list] = None
):
    """
    Print detailed classification report.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        target_names (list, optional): Names for labels
    """
    # Filter valid predictions
    valid_mask = (y_pred != -1) & (y_true != -1)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    if len(y_true_valid) == 0:
        print("No valid predictions to evaluate")
        return
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(
        y_true_valid,
        y_pred_valid,
        target_names=target_names,
        zero_division=0
    ))


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list] = None,
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix"
):
    """
    Plot confusion matrix.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        target_names (list, optional): Names for labels
        save_path (Path, optional): Path to save figure
        title (str): Plot title
    """
    # Filter valid predictions
    valid_mask = (y_pred != -1) & (y_true != -1)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    if len(y_true_valid) == 0:
        print("No valid predictions to plot")
        return
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_valid, y_pred_valid)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_self_training_progress(
    history_df: pd.DataFrame,
    save_path: Optional[Path] = None
):
    """
    Plot self-training progress over iterations.
    
    Args:
        history_df (pd.DataFrame): History from SelfTrainingClassifier
        save_path (Path, optional): Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Number of labeled samples
    ax = axes[0, 0]
    ax.plot(history_df['iteration'], history_df['n_labeled'], marker='o', label='Labeled')
    ax.plot(history_df['iteration'], history_df['n_unlabeled'], marker='s', label='Unlabeled')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Labeled vs Unlabeled Samples')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Pseudo-labeled samples per iteration
    ax = axes[0, 1]
    ax.bar(history_df['iteration'], history_df['n_pseudo_labeled'], alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Pseudo-labeled Samples')
    ax.set_title('Samples Added per Iteration')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Average confidence
    ax = axes[1, 0]
    ax.plot(history_df['iteration'], history_df['avg_confidence'], marker='o', color='green')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Confidence')
    ax.set_title('Pseudo-label Confidence')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 4: Clustering quality metrics
    ax = axes[1, 1]
    ax.plot(history_df['iteration'], history_df['silhouette_score'], marker='o', label='Silhouette')
    ax.plot(history_df['iteration'], history_df['cluster_purity'], marker='s', label='Purity')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score')
    ax.set_title('Clustering Quality Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved progress plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_semi_supervised(
    y_true_train: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    pseudo_labels: Optional[np.ndarray] = None,
    target_names: Optional[list] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, float]:
    """
    Comprehensive evaluation for semi-supervised learning.
    
    Args:
        y_true_train (np.ndarray): True labels for training set
        y_true_test (np.ndarray): True labels for test set
        y_pred_test (np.ndarray): Predicted labels for test set
        pseudo_labels (np.ndarray, optional): Pseudo-labels assigned to unlabeled data
        target_names (list, optional): Names for labels
        output_dir (Path, optional): Directory to save results
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    print("\n" + "=" * 70)
    print("SEMI-SUPERVISED EVALUATION")
    print("=" * 70)
    
    # Test set metrics
    test_metrics = calculate_metrics(y_true_test, y_pred_test)
    
    print("\nTest Set Performance:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
    
    # Print detailed report
    print_classification_report(y_true_test, y_pred_test, target_names)
    
    # Plot confusion matrix
    if output_dir:
        output_dir = Path(output_dir)
        plot_confusion_matrix(
            y_true_test,
            y_pred_test,
            target_names,
            save_path=output_dir / "confusion_matrix.png"
        )
    
    # Pseudo-label analysis
    if pseudo_labels is not None:
        n_pseudo = (pseudo_labels != -1).sum()
        print(f"\nPseudo-labeling Statistics:")
        print(f"  Total pseudo-labeled: {n_pseudo}")
        
        if n_pseudo > 0:
            unique, counts = np.unique(pseudo_labels[pseudo_labels != -1], return_counts=True)
            print(f"  Pseudo-label distribution:")
            for label, count in zip(unique, counts):
                label_name = target_names[int(label)] if target_names else f"Class {label}"
                print(f"    {label_name}: {count}")
    
    return test_metrics