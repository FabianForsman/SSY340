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
    classification_report
)
from typing import Dict, Optional, Tuple
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