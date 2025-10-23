"""
Run comprehensive experiments with limited labeled data.

This script compares different approaches:
1. Supervised baseline (limited labels only)
2. Semi-supervised with clustering
3. Semi-supervised with model-based pseudo-labeling
4. Fully supervised (upper bound)

Goal: Show that we can achieve good results with very limited labeled data.
"""

import os
# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from fine_tune_model import HateSpeechFineTuner
from sentence_transformers import SentenceTransformer
from data_loader import HateSpeechDataset, create_dataloaders
from preprocessing import create_transform
from data_augmentation import data_agumentation, augment_data_to_target_count
from evaluation import plot_confusion_matrix


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def prepare_augmented_dataset(data_path: str, config: dict):
    """
    Load, preprocess, augment, and balance the dataset.
    This creates the full augmented dataset that all experiments will use.
    
    Args:
        data_path: Path to labeled data CSV
        config: Configuration dictionary
        
    Returns:
        HateSpeechDataset: Fully prepared dataset with augmentation and balancing
    """
    print("\n" + "="*80)
    print("PREPARING AUGMENTED DATASET (SHARED BY ALL EXPERIMENTS)")
    print("="*80)
    
    # Step 1: Load raw dataset
    print("\n--- Step 1: Loading raw data ---")
    dataset_raw = HateSpeechDataset(
        root=str(Path(data_path).parent),
        file=Path(data_path).name,
        text_column=config["data"]["text_column"],
        label_column=config["data"]["label_column"],
        transform=None
    )
    print(f"Loaded {len(dataset_raw)} samples")
    
    # Step 2: Apply preprocessing
    print("\n--- Step 2: Applying preprocessing ---")
    text_transform = create_transform(**config["preprocessing"])
    dataset_raw.transform = text_transform
    print("Preprocessing applied")
    
    # Step 3: Apply data augmentation
    print("\n--- Step 3: Applying data augmentation ---")
    methods = config.get("data_augmentation", {}).get("methods", {})
    
    if methods.get("synonym_replacement", {}).get("enabled", False):
        num_replacements = methods.get("synonym_replacement", {}).get("n", 1)
        print(f"Applying synonym replacement (n={num_replacements})")
        
        hate_speech_data = dataset_raw.df[dataset_raw.df['label'] == 0].copy()
        augmented_texts = []
        
        for idx, row in tqdm(hate_speech_data.iterrows(), 
                            total=len(hate_speech_data), 
                            desc="Augmenting hate speech samples"):
            text = row[dataset_raw.text_column]
            label = row[dataset_raw.label_column]
            label_desc = row.get('label_desc', None)
            
            augmented_versions = data_agumentation(text, num_replacements)
            for aug_text in augmented_versions:
                augmented_texts.append((aug_text, label, label_desc))
        
        if augmented_texts:
            df_augmented = pd.DataFrame(
                augmented_texts, 
                columns=[dataset_raw.text_column, dataset_raw.label_column, 'label_desc']
            )
            print(f"Generated {len(df_augmented)} augmented samples")
            df_combined = pd.concat([dataset_raw.df, df_augmented], ignore_index=True)
            dataset_raw.update_dataframe(df_combined)
    
    # Step 4: Balance dataset
    print("\n--- Step 4: Balancing dataset ---")
    class_counts = dataset_raw.df['label'].value_counts()
    print(f"Class distribution before balancing:\n{class_counts}")
    
    max_count = class_counts.max()
    balanced_dfs = []
    for label, count in class_counts.items():
        df_class = dataset_raw.df[dataset_raw.df['label'] == label]
        df_balanced = augment_data_to_target_count(df_class, max_count)
        balanced_dfs.append(df_balanced)
    
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    print(f"Class distribution after balancing:\n{df_balanced['label'].value_counts()}")
    dataset_raw.update_dataframe(df_balanced)
    
    # Step 5: Shuffle
    print("\n--- Step 5: Shuffling dataset ---")
    dataset_raw.df = dataset_raw.df.sample(frac=1, random_state=42).reset_index(drop=True)
    dataset_raw.update_dataframe(dataset_raw.df)
    
    print(f"\n✓ Final dataset prepared: {len(dataset_raw)} samples")
    return dataset_raw


def create_train_test_splits(
    dataset: HateSpeechDataset,
    config: dict,
    output_dir: str = "data/loaders"
) -> Tuple:
    """
    Create train/val/test splits once and return dataloaders.
    All experiments will use these same dataloaders.
    
    Args:
        dataset: Prepared dataset
        config: Configuration dictionary
        output_dir: Directory to save split CSVs (for reference)
        
    Returns:
        train_loader, val_loader, test_loader
    """
    print("\n" + "="*80)
    print("CREATING TRAIN/VAL/TEST SPLITS (SHARED BY ALL EXPERIMENTS)")
    print("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n--- Creating dataloaders with fixed random seed ---")
    
    # Create dataloaders using existing function
    # The split is deterministic due to random_state=42 in train_dev_test_split
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=config["embedding"]["batch_size"],
        shuffle=False,  # Don't shuffle to ensure reproducibility across experiments
        num_workers=2,
        train_dev_test_split_sizes=(0.7, 0.2, 0.1)
    )
    
    # Save splits to CSV for reference
    train_path = output_path / "train_split.csv"
    val_path = output_path / "val_split.csv"
    test_path = output_path / "test_split.csv"
    
    train_df = _loader_to_dataframe(train_loader, config)
    val_df = _loader_to_dataframe(val_loader, config)
    test_df = _loader_to_dataframe(test_loader, config)
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")
    print(f"\nTrain class distribution:")
    print(train_df['label'].value_counts().sort_index())
    print(f"\nVal class distribution:")
    print(val_df['label'].value_counts().sort_index())
    print(f"\nTest class distribution:")
    print(test_df['label'].value_counts().sort_index())
    
    print(f"\n✓ Splits saved to:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"  - {test_path}")
    
    return train_loader, val_loader, test_loader


def run_experiment_1_supervised_limited(
    train_loader,
    val_loader,
    test_loader,
    label_fraction: float = 0.1,
    output_dir: str = "outputs/experiments",
    config_path: str = "config.yaml"
):
    """
    Experiment 1: Supervised learning with limited labels.
    
    Train a classifier using only a small fraction of labeled data.
    
    Args:
        train_loader: Training data loader (shared across experiments)
        val_loader: Validation data loader (shared across experiments)
        test_loader: Test data loader (shared across experiments)
        label_fraction: Fraction of training data to use (e.g., 0.1 = 10%)
        output_dir: Directory to save results
        config_path: Path to config file
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: SUPERVISED BASELINE WITH LIMITED LABELS")
    print("="*80)
    print(f"Using only {label_fraction*100:.0f}% of training labels")
    print("="*80)
    
    # Load config
    config = load_config(config_path)
    
    # Convert loaders to DataFrames (only for fine-tuner compatibility)
    train_df = _loader_to_dataframe(train_loader, config)
    val_df = _loader_to_dataframe(val_loader, config)
    test_df = _loader_to_dataframe(test_loader, config)
    
    print(f"\nTrain samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")
    
    # Sample only label_fraction of training data
    train_df_limited, _ = train_test_split(
        train_df,
        train_size=label_fraction,
        random_state=42,
        stratify=train_df['label']
    )
    
    print(f"\nUsing {len(train_df_limited)} labeled samples (out of {len(train_df)} total)")
    print(f"Class distribution in limited training set:")
    for label in sorted(train_df_limited['label'].unique()):
        count = (train_df_limited['label'] == label).sum()
        print(f"  {label}: {count} samples")
    
    # Initialize fine-tuner
    fine_tuner = HateSpeechFineTuner(
        base_model="all-MiniLM-L6-v2",
        num_classes=3,
        config_path=config_path
    )
    
    # Prepare training data
    train_examples, val_examples = fine_tuner.prepare_training_data(
        train_df_limited, val_df
    )
    
    # Create and train model
    fine_tuner.create_model_with_classifier()
    
    model_output = Path(output_dir) / "exp1_supervised_limited"
    fine_tuner.train(
        train_examples=train_examples,
        val_examples=val_examples,
        output_path=str(model_output),
        epochs=10,
        batch_size=64,
        learning_rate=2e-5
    )
    
    # Evaluate
    metrics = fine_tuner.evaluate(test_df)
    
    # Generate and save confusion matrix
    y_true = metrics.pop('true_labels')
    y_pred = metrics.pop('predictions')
    save_confusion_matrix(y_true, y_pred, 'exp1_supervised_limited', output_dir="outputs/figures")
    
    # Save embeddings
    embeddings = metrics.pop('embeddings')
    full_labels = metrics.pop('full_labels')
    save_embeddings(embeddings, full_labels, 'exp1_supervised_limited', output_dir="data/embeddings")
    
    # Save results
    results = {
        'experiment': 'supervised_limited',
        'label_fraction': label_fraction,
        'train_samples': len(train_df_limited),
        'test_samples': len(test_df),
        **metrics
    }
    
    results_df = pd.DataFrame([results])
    results_path = Path(output_dir) / "exp1_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Save model to models/ directory
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = models_dir / "exp1_supervised_limited"
    
    # Save model directly
    fine_tuner.model.save(str(final_model_path))
    print(f"✓ Model saved to: {final_model_path}")
    
    return metrics, model_output


def _loader_to_dataframe(loader, config):
    """Convert DataLoader to DataFrame for compatibility with fine-tuner.
    
    Note: The fine-tuner expects columns named 'text' and 'label',
    so we standardize the column names here.
    """
    texts = []
    labels = []
    for batch in loader:
        batch_texts, batch_labels = batch
        texts.extend(batch_texts)
        labels.extend([label.item() if torch.is_tensor(label) else label for label in batch_labels])
    
    return pd.DataFrame({
        'text': texts,  # Standardize column name for fine-tuner
        'label': labels
    })


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    experiment_name: str,
    output_dir: str = "outputs/figures"
):
    """
    Generate and save confusion matrix for an experiment.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        experiment_name: Name of the experiment (e.g., 'exp1_supervised_limited')
        output_dir: Directory to save confusion matrix
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    target_names = ['hate_speech', 'offensive_language', 'neither']
    save_path = output_path / f"{experiment_name}_confusion_matrix.png"
    
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        target_names=target_names,
        save_path=save_path,
        title=f"Confusion Matrix - {experiment_name.replace('_', ' ').title()}"
    )


def save_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    experiment_name: str,
    output_dir: str = "data/embeddings"
):
    """
    Save embeddings and labels to disk.
    
    Args:
        embeddings: Embedding vectors (N x D array)
        labels: Corresponding labels (N array)
        experiment_name: Name of the experiment (e.g., 'exp1_supervised_limited')
        output_dir: Directory to save embeddings
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings as numpy array
    embeddings_path = output_path / f"{experiment_name}_embeddings.npy"
    np.save(embeddings_path, embeddings)
    
    # Save labels as numpy array
    labels_path = output_path / f"{experiment_name}_labels.npy"
    np.save(labels_path, labels)
    
    print(f"✓ Embeddings saved to: {embeddings_path}")
    print(f"✓ Labels saved to: {labels_path}")


def visualize_clusters_tsne(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    pseudo_labels: np.ndarray = None,
    cluster_labels: np.ndarray = None,
    experiment_name: str = "clusters",
    output_dir: str = "outputs/figures",
    perplexity: int = 30,
    random_state: int = 42
):
    """
    Visualize embeddings using t-SNE, showing true labels, pseudo-labels, and clusters.
    
    Args:
        embeddings: Embedding vectors (N x D array)
        true_labels: True labels (N array)
        pseudo_labels: Pseudo-labels assigned by the algorithm (N array), optional
        cluster_labels: Cluster assignments (N array), optional (for DBSCAN)
        experiment_name: Name of the experiment (e.g., 'exp2_semi_supervised_knn')
        output_dir: Directory to save visualizations
        perplexity: t-SNE perplexity parameter
        random_state: Random state for reproducibility
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating t-SNE visualization for {experiment_name}...")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, n_jobs=-1)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Define label names and colors
    label_names = ['hate_speech', 'offensive_language', 'neither']
    colors = ['#e74c3c', '#f39c12', '#3498db']  # red, orange, blue
    
    # Create figure with subplots
    n_plots = 2 if pseudo_labels is not None else 1
    if cluster_labels is not None:
        n_plots += 1
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    # Plot 1: True labels
    ax = axes[0]
    for label_idx, (label_name, color) in enumerate(zip(label_names, colors)):
        mask = true_labels == label_idx
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=color,
            label=label_name,
            alpha=0.6,
            s=20,
            edgecolors='none'
        )
    ax.set_title('True Labels', fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Pseudo-labels (if provided)
    if pseudo_labels is not None:
        ax = axes[1]
        for label_idx, (label_name, color) in enumerate(zip(label_names, colors)):
            mask = pseudo_labels == label_idx
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=color,
                label=label_name,
                alpha=0.6,
                s=20,
                edgecolors='none'
            )
        ax.set_title('Pseudo-Labels', fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE dimension 1')
        ax.set_ylabel('t-SNE dimension 2')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Clusters (if provided, for DBSCAN)
    if cluster_labels is not None:
        ax = axes[-1]
        
        # Get unique clusters
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        
        # Use colormap for clusters
        cmap = plt.cm.get_cmap('tab20', n_clusters)
        
        # Plot noise points (cluster -1) in gray
        if -1 in unique_clusters:
            noise_mask = cluster_labels == -1
            ax.scatter(
                embeddings_2d[noise_mask, 0],
                embeddings_2d[noise_mask, 1],
                c='gray',
                label='Noise',
                alpha=0.3,
                s=20,
                edgecolors='none'
            )
        
        # Plot each cluster
        for i, cluster_id in enumerate(unique_clusters):
            if cluster_id == -1:
                continue
            mask = cluster_labels == cluster_id
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[cmap(i)],
                label=f'Cluster {cluster_id}',
                alpha=0.6,
                s=20,
                edgecolors='none'
            )
        
        ax.set_title(f'DBSCAN Clusters (n={n_clusters})', fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE dimension 1')
        ax.set_ylabel('t-SNE dimension 2')
        
        # Only show legend if not too many clusters
        if n_clusters <= 10:
            ax.legend(loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = output_path / f"{experiment_name}_tsne_clusters.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ t-SNE visualization saved to: {save_path}")


def run_experiment_2_semi_supervised_model_based(
    train_loader,
    val_loader,
    test_loader,
    label_fraction: float = 0.1,
    confidence_threshold: float = 0.65,
    output_dir: str = "outputs/experiments",
    config_path: str = "config.yaml",
    k_neighbors: int = 5
):
    """
    Experiment 2: Semi-supervised with model-based pseudo-labeling.
    
    1. Train initial model on limited labels
    2. Use model to predict labels on unlabeled data (pseudo-labeling)
    3. Keep high-confidence predictions
    4. Retrain on labeled + pseudo-labeled data
    
    Args:
        train_loader: Training data loader (shared across experiments)
        val_loader: Validation data loader (shared across experiments)
        test_loader: Test data loader (shared across experiments)
        label_fraction: Fraction of training data to use as labeled
        confidence_threshold: Confidence threshold for pseudo-labels (0-1)
        output_dir: Directory to save results
        config_path: Path to config file
        k_neighbors: Number of neighbors for k-NN pseudo-labeling
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: SEMI-SUPERVISED WITH MODEL-BASED PSEUDO-LABELING")
    print("="*80)
    print(f"Initial labels: {label_fraction*100:.0f}% of training data")
    print(f"Confidence threshold: {confidence_threshold}")
    print("="*80)
    
    # Load config
    config = load_config(config_path)
    
    # Convert loaders to DataFrames (only for fine-tuner compatibility)
    train_df = _loader_to_dataframe(train_loader, config)
    val_df = _loader_to_dataframe(val_loader, config)
    test_df = _loader_to_dataframe(test_loader, config)
    
    print(f"\nTrain samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")
    
    # Split into labeled and unlabeled
    train_df_labeled, train_df_unlabeled = train_test_split(
        train_df,
        train_size=label_fraction,
        random_state=42,
        stratify=train_df['label']
    )
    
    print(f"\nLabeled samples: {len(train_df_labeled)}")
    print(f"Unlabeled samples: {len(train_df_unlabeled)}")
    
    # Initialize fine-tuner
    fine_tuner = HateSpeechFineTuner(
        base_model="all-MiniLM-L6-v2",
        num_classes=3,
        config_path=config_path
    )
    
    # Step 1: Train initial model on labeled data
    print("\n--- Step 1: Training initial model on labeled data ---")
    train_examples, val_examples = fine_tuner.prepare_training_data(
        train_df_labeled, val_df
    )
    
    fine_tuner.create_model_with_classifier()
    
    initial_model_path = Path(output_dir) / "exp2_initial_model"
    fine_tuner.train(
        train_examples=train_examples,
        val_examples=val_examples,
        output_path=str(initial_model_path),
        epochs=10,
        batch_size=64,
        learning_rate=2e-5
    )
    
    # Generate embeddings from INITIAL model for fair comparison with DBSCAN
    print("\n--- Generating embeddings from initial model for visualization ---")
    unlabeled_embeddings_initial = fine_tuner.model.encode(
        train_df_unlabeled['text'].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    labeled_embeddings_initial = fine_tuner.model.encode(
        train_df_labeled['text'].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Generate pseudo-labels using initial model for visualization
    from sklearn.neighbors import KNeighborsClassifier
    knn_vis = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn_vis.fit(labeled_embeddings_initial, train_df_labeled['label'].values)
    pseudo_labels_vis_initial = knn_vis.predict(unlabeled_embeddings_initial)
    
    # Visualize clusters with t-SNE (using INITIAL model embeddings)
    visualize_clusters_tsne(
        embeddings=unlabeled_embeddings_initial,
        true_labels=train_df_unlabeled['label'].values,
        pseudo_labels=pseudo_labels_vis_initial,
        cluster_labels=None,  # k-NN doesn't produce clusters
        experiment_name='exp2_semi_supervised_knn',
        output_dir="outputs/figures"
    )
    
    # Step 2: Use iterative pseudo-labeling with k-NN
    print("\n--- Step 2: Iterative Pseudo-labeling ---")
    model_after_iterations, combined_train_df, total_pseudo_added = iterative_pseudo_labeling(
        initial_model=fine_tuner.model,
        train_df_labeled=train_df_labeled,
        train_df_unlabeled=train_df_unlabeled,
        val_df=val_df,
        config_path=config_path,
        confidence_threshold=confidence_threshold,
        iterations=4,  # Increased from 3
        max_pseudo_per_iteration=10000,
        k_neighbors=k_neighbors
    )
    
    print(f"\nCombined training set: {len(combined_train_df)} samples")
    print(f"Total pseudo-labels added: {total_pseudo_added}")
    print(f"Final class distribution:")
    for label in sorted(combined_train_df['label'].unique()):
        count = (combined_train_df['label'] == label).sum()
        print(f"  {label}: {count} samples")
    
    # Step 4: Retrain on combined data
    print("\n--- Step 3: Retraining on labeled + pseudo-labeled data ---")
    combined_train_examples, val_examples = fine_tuner.prepare_training_data(
        combined_train_df, val_df
    )
    
    # Create fresh model
    fine_tuner.create_model_with_classifier()
    
    final_model_path = Path(output_dir) / "exp2_semi_supervised_model"
    fine_tuner.train(
        train_examples=combined_train_examples,
        val_examples=val_examples,
        output_path=str(final_model_path),
        epochs=10,
        batch_size=64,
        learning_rate=2e-5
    )
    
    # Evaluate
    metrics = fine_tuner.evaluate(test_df)
    
    # Generate and save confusion matrix
    y_true = metrics.pop('true_labels')
    y_pred = metrics.pop('predictions')
    save_confusion_matrix(y_true, y_pred, 'exp2_semi_supervised_knn', output_dir="outputs/figures")
    
    # Save embeddings
    embeddings = metrics.pop('embeddings')
    full_labels = metrics.pop('full_labels')
    save_embeddings(embeddings, full_labels, 'exp2_semi_supervised_knn', output_dir="data/embeddings")
    
    # Save results
    results = {
        'experiment': 'semi_supervised_model_based',
        'label_fraction': label_fraction,
        'initial_labeled': len(train_df_labeled),
        'pseudo_labeled': total_pseudo_added,
        'total_training': len(combined_train_df),
        'confidence_threshold': confidence_threshold,
        'test_samples': len(test_df),
        **metrics
    }
    
    results_df = pd.DataFrame([results])
    results_path = Path(output_dir) / "exp2_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Save model to models/ directory
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    final_model_save_path = models_dir / "exp2_semi_supervised_knn"
    
    # Save model directly
    fine_tuner.model.save(str(final_model_save_path))
    print(f"✓ Model saved to: {final_model_save_path}")
    
    return metrics, final_model_path

def run_experiment_3_semi_supervised_model_based(
    train_loader,
    val_loader,
    test_loader,
    label_fraction: float = 0.1,
    confidence_threshold: float = 0.65,
    output_dir: str = "outputs/experiments",
    config_path: str = "config.yaml",
    dbscan_eps: float = 0.3,
    dbscan_min_samples: int = 5
):
    """
    Experiment 3: Semi-supervised with DBSCAN-based pseudo-labeling.
    
    1. Train initial model on limited labels
    2. Use DBSCAN clustering + k-NN to predict labels on unlabeled data
    3. Keep high-confidence predictions
    4. Retrain on labeled + pseudo-labeled data
    
    Args:
        train_loader: Training data loader (shared across experiments)
        val_loader: Validation data loader (shared across experiments)
        test_loader: Test data loader (shared across experiments)
        label_fraction: Fraction of training data to use as labeled
        confidence_threshold: Confidence threshold for pseudo-labels (0-1)
        output_dir: Directory to save results
        config_path: Path to config file
        dbscan_eps: DBSCAN epsilon parameter
        dbscan_min_samples: DBSCAN min_samples parameter
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: SEMI-SUPERVISED WITH DBSCAN-BASED PSEUDO-LABELING")
    print("="*80)
    print(f"Initial labels: {label_fraction*100:.0f}% of training data")
    print(f"DBSCAN eps: {dbscan_eps}, min_samples: {dbscan_min_samples}")
    print(f"Confidence threshold: {confidence_threshold}")
    print("="*80)
    
    # Load config
    config = load_config(config_path)
    
    # Convert loaders to DataFrames (only for fine-tuner compatibility)
    train_df = _loader_to_dataframe(train_loader, config)
    val_df = _loader_to_dataframe(val_loader, config)
    test_df = _loader_to_dataframe(test_loader, config)
    
    print(f"\nTrain samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")
    
    # Split into labeled and unlabeled
    train_df_labeled, train_df_unlabeled = train_test_split(
        train_df,
        train_size=label_fraction,
        random_state=42,
        stratify=train_df['label']
    )
    
    print(f"\nLabeled samples: {len(train_df_labeled)}")
    print(f"Unlabeled samples: {len(train_df_unlabeled)}")
    
    # Initialize fine-tuner
    fine_tuner = HateSpeechFineTuner(
        base_model="all-MiniLM-L6-v2",
        num_classes=3,
        config_path=config_path
    )
    
    # Step 1: Train initial model on labeled data
    print("\n--- Step 1: Training initial model on labeled data ---")
    train_examples, val_examples = fine_tuner.prepare_training_data(
        train_df_labeled, val_df
    )
    
    fine_tuner.create_model_with_classifier()
    
    initial_model_path = Path(output_dir) / "exp3_initial_model"
    fine_tuner.train(
        train_examples=train_examples,
        val_examples=val_examples,
        output_path=str(initial_model_path),
        epochs=10,
        batch_size=64,
        learning_rate=2e-5
    )
    
    # Step 2: Generate pseudo-labels for unlabeled data using DBSCAN (single pass for analysis)
    print("\n--- Step 2: Generating pseudo-labels with DBSCAN ---")
    pseudo_labels_initial, confidences_initial, cluster_stats, unlabeled_embeddings_dbscan, cluster_assignments = generate_pseudo_labels_with_dbscan(
        model=fine_tuner.model,
        texts=train_df_unlabeled['text'].tolist(),  # Use standardized column name
        labeled_texts=train_df_labeled['text'].tolist(),  # Use standardized column name
        labeled_labels=train_df_labeled['label'].values,
        confidence_threshold=confidence_threshold,
        eps=dbscan_eps,
        min_samples=dbscan_min_samples
    )
    
    # Filter high-confidence predictions with diverse sampling
    high_conf_mask = np.array(confidences_initial) >= confidence_threshold
    pseudo_labeled_df_temp = train_df_unlabeled.copy()
    pseudo_labeled_df_temp['label'] = pseudo_labels_initial
    pseudo_labeled_df_temp['confidence'] = confidences_initial
    pseudo_labeled_df_temp = pseudo_labeled_df_temp[high_conf_mask]
    
    # Select diverse pseudo-labels (allow natural imbalance with soft cap)
    if len(pseudo_labeled_df_temp) > 0:
        pseudo_labeled_df_temp = select_diverse_pseudo_labels(
            pseudo_labeled_df_temp, 
            max_total=10000,  # Increased from implicit 5000
            allow_imbalance=True
        )
        pseudo_labeled_df = pseudo_labeled_df_temp.drop('confidence', axis=1)
    else:
        pseudo_labeled_df = pd.DataFrame()
    
    print(f"High-confidence pseudo-labels: {len(pseudo_labeled_df)} / {len(train_df_unlabeled)}")
    
    if len(pseudo_labeled_df) > 0:
        # Calculate stats only if we have pseudo-labels
        print(f"Average confidence: {pseudo_labeled_df_temp['confidence'].mean():.4f}")
        print(f"Pseudo-label distribution:")
        for label in sorted(pseudo_labeled_df['label'].unique()):
            count = (pseudo_labeled_df['label'] == label).sum()
            print(f"  {label}: {count} samples")
    else:
        print(f"Average confidence: N/A")
        print(f"Pseudo-label distribution: None")
    
    # Visualize clusters with t-SNE (for DBSCAN experiment)
    if len(pseudo_labels_initial) > 0 and len(confidences_initial) > 0:
        visualize_clusters_tsne(
            embeddings=unlabeled_embeddings_dbscan,
            true_labels=train_df_unlabeled['label'].values,
            pseudo_labels=np.array(pseudo_labels_initial),
            cluster_labels=cluster_assignments,  # DBSCAN cluster assignments
            experiment_name='exp3_semi_supervised_dbscan',
            output_dir="outputs/figures"
        )
    
    # Step 3: Combine labeled + pseudo-labeled data
    combined_train_df = pd.concat([train_df_labeled, pseudo_labeled_df], ignore_index=True)
    print(f"\nCombined training set: {len(combined_train_df)} samples")
    total_pseudo_added = len(pseudo_labeled_df)
    
    # Step 4: Retrain on combined data
    print("\n--- Step 3: Retraining on labeled + pseudo-labeled data ---")
    combined_train_examples, val_examples = fine_tuner.prepare_training_data(
        combined_train_df, val_df
    )
    
    # Create fresh model
    fine_tuner.create_model_with_classifier()
    
    final_model_path = Path(output_dir) / "exp3_semi_supervised_dbscan"
    fine_tuner.train(
        train_examples=combined_train_examples,
        val_examples=val_examples,
        output_path=str(final_model_path),
        epochs=10,
        batch_size=64,
        learning_rate=2e-5
    )
    
    # Evaluate
    metrics = fine_tuner.evaluate(test_df)
    
    # Generate and save confusion matrix
    y_true = metrics.pop('true_labels')
    y_pred = metrics.pop('predictions')
    save_confusion_matrix(y_true, y_pred, 'exp3_semi_supervised_dbscan', output_dir="outputs/figures")
    
    # Save embeddings
    embeddings = metrics.pop('embeddings')
    full_labels = metrics.pop('full_labels')
    save_embeddings(embeddings, full_labels, 'exp3_semi_supervised_dbscan', output_dir="data/embeddings")
    
    # Save results
    results = {
        'experiment': 'semi_supervised_dbscan',
        'label_fraction': label_fraction,
        'initial_labeled': len(train_df_labeled),
        'pseudo_labeled': total_pseudo_added,
        'total_training': len(combined_train_df),
        'dbscan_eps': dbscan_eps,
        'dbscan_min_samples': dbscan_min_samples,
        'n_clusters': cluster_stats['n_clusters'],
        'n_noise': cluster_stats['n_noise'],
        'silhouette': cluster_stats['silhouette'],
        'confidence_threshold': confidence_threshold,
        'test_samples': len(test_df),
        **metrics
    }
    
    results_df = pd.DataFrame([results])
    results_path = Path(output_dir) / "exp3_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Save model to models/ directory
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    final_model_save_path = models_dir / "exp3_semi_supervised_dbscan"
    
    # Save model directly
    fine_tuner.model.save(str(final_model_save_path))
    print(f"✓ Model saved to: {final_model_save_path}")
    
    return metrics, final_model_path


def iterative_pseudo_labeling(
    initial_model,
    train_df_labeled,
    train_df_unlabeled,
    val_df,
    config_path,
    confidence_threshold=0.7,
    iterations=5,
    max_pseudo_per_iteration=5000,
    k_neighbors=5
):
    """
    Iteratively add high-confidence pseudo-labels and retrain.
    
    Args:
        initial_model: Initial trained model
        train_df_labeled: Initial labeled data
        train_df_unlabeled: Unlabeled data pool
        val_df: Validation data
        config_path: Config file path
        confidence_threshold: Confidence threshold for pseudo-labels
        iterations: Number of pseudo-labeling iterations
        max_pseudo_per_iteration: Maximum pseudo-labels to add per iteration
        k_neighbors: Number of neighbors for k-NN
    
    Returns:
        final_model, combined_train_df, total_pseudo_added
    """
    current_labeled = train_df_labeled.copy()
    current_unlabeled = train_df_unlabeled.copy()
    model = initial_model
    total_pseudo_added = 0
    
    for iteration in range(iterations):
        print(f"\n--- Iteration {iteration + 1}/{iterations} ---")
        print(f"Current labeled: {len(current_labeled)}, Unlabeled pool: {len(current_unlabeled)}")
        print(f"Confidence threshold: {confidence_threshold:.2f}")
        
        if len(current_unlabeled) == 0:
            print("No more unlabeled data. Stopping.")
            break
        
        # Generate pseudo-labels
        pseudo_labels, confidences, _ = generate_pseudo_labels_with_model(
            model=model,
            texts=current_unlabeled['text'].tolist(),
            labeled_texts=current_labeled['text'].tolist(),
            labeled_labels=current_labeled['label'].values,
            confidence_threshold=confidence_threshold,
            k_neighbors=k_neighbors
        )
        
        # Filter by confidence
        high_conf_mask = np.array(confidences) >= confidence_threshold
        pseudo_df = current_unlabeled.copy()
        pseudo_df['label'] = pseudo_labels
        pseudo_df['confidence'] = confidences
        pseudo_df = pseudo_df[high_conf_mask]
        
        if len(pseudo_df) == 0:
            print("No high-confidence pseudo-labels found. Stopping.")
            break
        
        # Select diverse samples instead of just taking top-K
        pseudo_df = select_diverse_pseudo_labels(
            pseudo_df=pseudo_df,
            max_total=max_pseudo_per_iteration,
            allow_imbalance=True
        )
        
        print(f"Adding {len(pseudo_df)} pseudo-labels (avg confidence: {pseudo_df['confidence'].mean():.3f})")
        print(f"Confidence range: [{pseudo_df['confidence'].min():.3f}, {pseudo_df['confidence'].max():.3f}]")
        
        # Show confidence distribution
        conf_bins = [
            (0.50, 0.60, 'very low'),
            (0.60, 0.70, 'low'),
            (0.70, 0.80, 'medium'),
            (0.80, 0.90, 'high'),
            (0.90, 1.01, 'very high')
        ]
        print(f"Confidence distribution:")
        for low, high, label in conf_bins:
            count = ((pseudo_df['confidence'] >= low) & (pseudo_df['confidence'] < high)).sum()
            if count > 0:
                print(f"  {label} ({low:.2f}-{high-0.01:.2f}): {count} samples")
        
        print(f"Pseudo-label distribution: {pseudo_df['label'].value_counts().to_dict()}")
        
        # Show class balance ratio
        class_counts = pseudo_df['label'].value_counts()
        balance_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else float('inf')
        print(f"Class balance ratio (max/min): {balance_ratio:.2f}x")
        
        # Store the indices to remove (in current_unlabeled's index space)
        indices_to_remove = pseudo_df.index
        
        # Remove confidence column before concatenating
        pseudo_df = pseudo_df.drop('confidence', axis=1)
        
        # Update labeled/unlabeled sets
        current_labeled = pd.concat([current_labeled, pseudo_df], ignore_index=True)
        # Use loc to create a mask for rows NOT in the pseudo-labeled set
        current_unlabeled = current_unlabeled.loc[~current_unlabeled.index.isin(indices_to_remove)].reset_index(drop=True)
        total_pseudo_added += len(pseudo_df)
        
        # Only retrain if not the last iteration
        if iteration < iterations - 1 and len(current_unlabeled) > 0:
            print(f"Retraining model with {len(current_labeled)} samples...")
            fine_tuner = HateSpeechFineTuner(
                base_model="all-MiniLM-L6-v2",
                num_classes=3,
                config_path=config_path
            )
            
            train_examples, val_examples = fine_tuner.prepare_training_data(
                current_labeled, val_df
            )
            
            fine_tuner.create_model_with_classifier()
            fine_tuner.train(
                train_examples=train_examples,
                val_examples=val_examples,
                output_path=f"temp_model_iter_{iteration}",
                epochs=10, 
                batch_size=64,
                learning_rate=2e-5
            )
            
            model = fine_tuner.model
    
    print(f"\n✓ Iterative pseudo-labeling complete: added {total_pseudo_added} total pseudo-labels")
    return model, current_labeled, total_pseudo_added


def select_diverse_pseudo_labels(pseudo_df, max_total=5000, allow_imbalance=True):
    """
    Select diverse pseudo-labels based on confidence, allowing natural class distribution.
    Uses a mix of high-confidence and medium-confidence (uncertainty) samples.
    
    Args:
        pseudo_df: DataFrame with pseudo-labels and confidence scores
        max_total: Maximum total samples to select
        allow_imbalance: If True, allow natural class imbalance (with soft cap)
                        If False, force equal class distribution
    
    Returns:
        Selected DataFrame with diverse pseudo-labels
    """
    if len(pseudo_df) <= max_total:
        return pseudo_df
    
    # Sort by confidence
    pseudo_df_sorted = pseudo_df.sort_values('confidence', ascending=False)
    
    # New behavior: Mix of high-confidence and uncertainty samples
    # Take 70% high-confidence + 30% medium-confidence (near decision boundary)
    n_high_conf = int(max_total * 0.70)
    n_medium_conf = max_total - n_high_conf
    
    # High-confidence samples (top N)
    high_conf_samples = pseudo_df_sorted.head(n_high_conf)
    
    # Medium-confidence samples (between 0.6-0.85 range for uncertainty)
    remaining = pseudo_df_sorted.iloc[n_high_conf:]
    medium_conf_mask = (remaining['confidence'] >= 0.55) & (remaining['confidence'] <= 0.85)
    medium_conf_pool = remaining[medium_conf_mask].copy()
    
    if len(medium_conf_pool) > 0:
        # Select from medium confidence pool
        # Prioritize samples closer to decision boundary (confidence ~ 0.7)
        medium_conf_pool['boundary_score'] = 1.0 - np.abs(medium_conf_pool['confidence'] - 0.70)
        medium_conf_pool = medium_conf_pool.sort_values('boundary_score', ascending=False)
        medium_conf_samples = medium_conf_pool.head(min(n_medium_conf, len(medium_conf_pool)))
        medium_conf_samples = medium_conf_samples.drop('boundary_score', axis=1)
        
        selected = pd.concat([high_conf_samples, medium_conf_samples], ignore_index=True)
    else:
        # No medium confidence samples, just take more high confidence
        selected = pseudo_df_sorted.head(max_total)
    
    # Check for extreme imbalance (>3x difference)
    class_counts = selected['label'].value_counts()
    if len(class_counts) > 1:
        max_count = class_counts.max()
        min_count = class_counts.min()
        
        # If imbalance is too extreme, apply soft cap
        if max_count > 3 * min_count:
            balanced_dfs = []
            # Allow up to 2.5x the minimum class size
            cap = int(min_count * 2.5)
            
            for label in sorted(pseudo_df['label'].unique()):
                df_class = selected[selected['label'] == label]
                if len(df_class) > cap:
                    # Keep highest confidence samples
                    df_class = df_class.nlargest(cap, 'confidence')
                balanced_dfs.append(df_class)
            
            selected = pd.concat(balanced_dfs, ignore_index=True)
            # Re-sort by confidence
            selected = selected.sort_values('confidence', ascending=False)
    
    return selected


def generate_pseudo_labels_with_model(
    model: SentenceTransformer,
    texts: list,
    labeled_texts: list,
    labeled_labels: np.ndarray,
    confidence_threshold: float = 0.6,
    k_neighbors: int = 5
) -> tuple:
    """
    Generate pseudo-labels using k-NN on embeddings from trained model.
    
    This is a valid semi-supervised approach: use the fine-tuned model's embeddings
    with k-NN to predict labels on unlabeled data.
    
    Args:
        model: Trained SentenceTransformer model
        texts: Unlabeled texts to generate pseudo-labels for
        labeled_texts: Labeled texts to train k-NN on
        labeled_labels: Labels for the labeled texts
        confidence_threshold: Minimum confidence for pseudo-labels
        k_neighbors: Number of neighbors for k-NN
        
    Returns:
        predictions: Predicted labels
        confidences: Confidence scores (proportion of matching neighbors)
        unlabeled_embeddings: Embeddings for unlabeled data (for visualization)
    """
    from sklearn.neighbors import KNeighborsClassifier
    
    print(f"Using k-NN (k={k_neighbors}) on fine-tuned embeddings for pseudo-labeling...")
    
    # Get embeddings for labeled data
    print("Encoding labeled data...")
    labeled_embeddings = model.encode(
        labeled_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Get embeddings for unlabeled data
    print("Encoding unlabeled data...")
    unlabeled_embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Train k-NN classifier on labeled embeddings
    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(labeled_embeddings, labeled_labels)
    
    # Predict on unlabeled data
    predictions = knn.predict(unlabeled_embeddings)
    
    # Get confidence scores (probability of predicted class)
    # For k-NN, confidence = proportion of k neighbors with predicted label
    proba = knn.predict_proba(unlabeled_embeddings)
    confidences = np.max(proba, axis=1)
    
    print(f"Predictions generated for {len(predictions)} samples")
    print(f"Confidence range: [{confidences.min():.3f}, {confidences.max():.3f}]")
    print(f"Mean confidence: {confidences.mean():.3f}")
    
    return predictions, confidences, unlabeled_embeddings


def generate_pseudo_labels_with_dbscan(
    model: SentenceTransformer,
    texts: list,
    labeled_texts: list,
    labeled_labels: np.ndarray,
    confidence_threshold: float = 0.6,
    eps: float = 0.3,
    min_samples: int = 5
) -> tuple:
    """
    Generate pseudo-labels using DBSCAN clustering on embeddings from trained model.
    
    Strategy:
    1. Get embeddings for labeled and unlabeled data
    2. Cluster unlabeled embeddings using DBSCAN
    3. For each cluster, assign label based on nearest labeled neighbors using k-NN
    4. Confidence = proportion of nearest neighbors with same label
    5. Boost confidence for points in cohesive clusters
    6. Penalize noise points
    
    Args:
        model: Trained SentenceTransformer model
        texts: Unlabeled texts to generate pseudo-labels for
        labeled_texts: Labeled texts to train k-NN on
        labeled_labels: Labels for the labeled texts
        confidence_threshold: Minimum confidence for pseudo-labels
        eps: DBSCAN epsilon parameter (max distance between neighbors)
        min_samples: DBSCAN min_samples parameter (min points to form cluster)
        
    Returns:
        predictions: Predicted labels
        confidences: Confidence scores (proportion of matching neighbors)
        cluster_stats: Dictionary with clustering statistics
        unlabeled_embeddings: Embeddings for unlabeled data (for visualization)
        cluster_labels: DBSCAN cluster assignments (for visualization)
    """
    from sklearn.neighbors import KNeighborsClassifier
    
    print(f"Using DBSCAN (eps={eps}, min_samples={min_samples}) + k-NN for pseudo-labeling...")
    
    # Get embeddings for labeled data
    print("Encoding labeled data...")
    labeled_embeddings = model.encode(
        labeled_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Get embeddings for unlabeled data
    print("Encoding unlabeled data...")
    unlabeled_embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Apply DBSCAN clustering to unlabeled embeddings
    print("Applying DBSCAN clustering...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1)
    cluster_labels = dbscan.fit_predict(unlabeled_embeddings)
    
    # Get cluster statistics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"Found {n_clusters} clusters")
    print(f"Noise points: {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")
    
    # Calculate silhouette score (exclude noise points)
    if n_clusters > 1:
        non_noise_mask = cluster_labels != -1
        if non_noise_mask.sum() > 0:
            silhouette = silhouette_score(
                unlabeled_embeddings[non_noise_mask],
                cluster_labels[non_noise_mask],
                metric='cosine'
            )
        else:
            silhouette = 0.0
    else:
        silhouette = 0.0
    
    print(f"Silhouette score: {silhouette:.4f}")
    
    # Train k-NN classifier on labeled embeddings to assign labels
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(labeled_embeddings, labeled_labels)
    
    # Predict labels for all unlabeled points
    predictions = knn.predict(unlabeled_embeddings)
    
    # Get confidence scores (probability of predicted class)
    # For k-NN, confidence = proportion of k neighbors with predicted label
    proba = knn.predict_proba(unlabeled_embeddings)
    confidences = np.max(proba, axis=1)
    
    # Boost confidence for points in good clusters
    # Points in the same cluster should have similar labels
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        if cluster_mask.sum() > 0:
            # Get majority label in cluster
            cluster_predictions = predictions[cluster_mask]
            unique, counts = np.unique(cluster_predictions, return_counts=True)
            majority_label = unique[np.argmax(counts)]
            cluster_purity = counts.max() / cluster_mask.sum()
            
            # Boost confidence for points with majority label
            majority_mask = (predictions == majority_label) & cluster_mask
            confidences[majority_mask] = confidences[majority_mask] * 0.8 + cluster_purity * 0.2

    
    # Penalize noise points (label -1)
    noise_mask = cluster_labels == -1
    confidences[noise_mask] = confidences[noise_mask] * 0.8  # Smaller penalty
    
    print(f"Predictions generated for {len(predictions)} samples")
    print(f"Confidence range: [{confidences.min():.3f}, {confidences.max():.3f}]")
    print(f"Mean confidence: {confidences.mean():.3f}")
    
    cluster_stats = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': silhouette
    }
    
    return predictions, confidences, cluster_stats, unlabeled_embeddings, cluster_labels


def run_experiment_4_fully_supervised(
    train_loader,
    val_loader,
    test_loader,
    output_dir: str = "outputs/experiments",
    config_path: str = "config.yaml"
):
    """
    Experiment 4: Fully supervised learning (upper bound).
    
    Train on 100% of labeled data.
    
    Args:
        train_loader: Training data loader (shared across experiments)
        val_loader: Validation data loader (shared across experiments)
        test_loader: Test data loader (shared across experiments)
        output_dir: Directory to save results
        config_path: Path to config file
    """
    print("\n" + "="*80)
    print("EXPERIMENT 4: FULLY SUPERVISED (UPPER BOUND)")
    print("="*80)
    print("Using 100% of training labels")
    print("="*80)
    
    # Load config
    config = load_config(config_path)
    
    # Convert loaders to DataFrames (only for fine-tuner compatibility)
    train_df = _loader_to_dataframe(train_loader, config)
    val_df = _loader_to_dataframe(val_loader, config)
    test_df = _loader_to_dataframe(test_loader, config)
    
    print(f"\nUsing all {len(train_df)} training samples")
    
    # Initialize fine-tuner
    fine_tuner = HateSpeechFineTuner(
        base_model="all-MiniLM-L6-v2",
        num_classes=3,
        config_path=config_path
    )
    
    # Prepare training data
    train_examples, val_examples = fine_tuner.prepare_training_data(
        train_df, val_df
    )
    
    # Create and train model
    fine_tuner.create_model_with_classifier()
    
    model_output = Path(output_dir) / "exp4_fully_supervised"
    fine_tuner.train(
        train_examples=train_examples,
        val_examples=val_examples,
        output_path=str(model_output),
        epochs=10,
        batch_size=64,
        learning_rate=2e-5
    )
    
    # Evaluate
    metrics = fine_tuner.evaluate(test_df)
    
    # Generate and save confusion matrix
    y_true = metrics.pop('true_labels')
    y_pred = metrics.pop('predictions')
    save_confusion_matrix(y_true, y_pred, 'exp4_fully_supervised', output_dir="outputs/figures")
    
    # Save embeddings
    embeddings = metrics.pop('embeddings')
    full_labels = metrics.pop('full_labels')
    save_embeddings(embeddings, full_labels, 'exp4_fully_supervised', output_dir="data/embeddings")
    
    # Save results
    results = {
        'experiment': 'fully_supervised',
        'label_fraction': 1.0,
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        **metrics
    }
    
    results_df = pd.DataFrame([results])
    results_path = Path(output_dir) / "exp4_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Save model to models/ directory
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    final_model_save_path = models_dir / "exp4_fully_supervised"
    
    # Save model directly
    fine_tuner.model.save(str(final_model_save_path))
    print(f"✓ Model saved to: {final_model_save_path}")

    return metrics, model_output



def generate_comparison_report(output_dir: str):
    """Generate final comparison report of all experiments."""
    print("\n" + "="*80)
    print("FINAL COMPARISON REPORT")
    print("="*80)
    
    output_path = Path(output_dir)
    
    # Load all results
    results = []
    for exp_file in ['exp1_results.csv', 'exp2_results.csv', 'exp3_results.csv', 'exp4_results.csv']:
        exp_path = output_path / exp_file
        if exp_path.exists():
            df = pd.read_csv(exp_path)
            results.append(df)
    
    if not results:
        print("No experiment results found!")
        return
    
    # Combine results
    all_results = pd.concat(results, ignore_index=True)
    
    # Display comparison
    print("\n" + all_results.to_string(index=False))
    print("\n" + "="*80)
    
    # Save combined results
    combined_path = output_path / "all_experiments_comparison.csv"
    all_results.to_csv(combined_path, index=False)
    print(f"\n✓ Combined results saved to: {combined_path}")
    
    # Calculate improvements
    if len(all_results) >= 2:
        baseline_acc = all_results.iloc[0]['accuracy']
        semi_knn_acc = all_results.iloc[1]['accuracy'] if len(all_results) > 1 else None
        semi_dbscan_acc = all_results.iloc[2]['accuracy'] if len(all_results) > 2 else None
        full_acc = all_results.iloc[3]['accuracy'] if len(all_results) > 3 else None
        
        print("\nKEY FINDINGS:")
        print(f"  Baseline (limited labels): {baseline_acc:.2%}")
        
        if semi_knn_acc:
            improvement_knn = (semi_knn_acc - baseline_acc) / baseline_acc * 100
            print(f"  Semi-supervised (k-NN): {semi_knn_acc:.2%} ({improvement_knn:+.1f}% vs baseline)")
        
        if semi_dbscan_acc:
            improvement_dbscan = (semi_dbscan_acc - baseline_acc) / baseline_acc * 100
            print(f"  Semi-supervised (DBSCAN): {semi_dbscan_acc:.2%} ({improvement_dbscan:+.1f}% vs baseline)")
        
        if full_acc:
            print(f"  Fully supervised: {full_acc:.2%} (upper bound)")
            
            if semi_knn_acc:
                gap_knn = (full_acc - semi_knn_acc) / full_acc * 100
                print(f"    → k-NN reaches {(semi_knn_acc/full_acc)*100:.1f}% of fully supervised performance")
            
            if semi_dbscan_acc:
                gap_dbscan = (full_acc - semi_dbscan_acc) / full_acc * 100
                print(f"    → DBSCAN reaches {(semi_dbscan_acc/full_acc)*100:.1f}% of fully supervised performance")
        
        # Compare k-NN vs DBSCAN if both exist
        if semi_knn_acc and semi_dbscan_acc:
            print("\n  Comparison (k-NN vs DBSCAN):")
            if semi_dbscan_acc > semi_knn_acc:
                improvement = (semi_dbscan_acc - semi_knn_acc) / semi_knn_acc * 100
                print(f"    DBSCAN outperforms k-NN by {improvement:+.1f}%")
            elif semi_knn_acc > semi_dbscan_acc:
                improvement = (semi_knn_acc - semi_dbscan_acc) / semi_dbscan_acc * 100
                print(f"    k-NN outperforms DBSCAN by {improvement:+.1f}%")
            else:
                print(f"    k-NN and DBSCAN perform equally")


def main():
    parser = argparse.ArgumentParser(
        description="Run limited label experiments"
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/raw/labeled_data.csv',
        help='Path to labeled data CSV'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/experiments',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--label-fraction',
        type=float,
        default=0.1,
        help='Fraction of training data to use as labeled (e.g., 0.1 = 10%%)'
    )
    
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.55,
        help='Confidence threshold for pseudo-labeling'
    )
    
    parser.add_argument(
        '--k-neighbors',
        type=int,
        default=5,
        help='Number of neighbors for k-NN pseudo-labeling'
    )
    
    parser.add_argument(
        '--dbscan-eps',
        type=float,
        default=0.3,
        help='DBSCAN epsilon parameter (max distance between neighbors, lower=tighter clusters)'
    )
    
    parser.add_argument(
        '--dbscan-min-samples',
        type=int,
        default=5,
        help='DBSCAN min_samples parameter (min points to form cluster, lower=more clusters)'
    )
    
    parser.add_argument(
        '--experiments',
        nargs='+',
        choices=['exp1', 'exp2', 'exp3', 'exp4', 'all'],
        default=['all'],
        help='Which experiments to run'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare augmented dataset (shared by all experiments)
    dataset = prepare_augmented_dataset(args.data, config)
    
    # Create train/val/test splits ONCE (shared by all experiments)
    # Save splits to data/loaders directory
    train_loader, val_loader, test_loader = create_train_test_splits(
        dataset=dataset,
        config=config,
        output_dir="data/loaders"
    )
    
    run_experiments = args.experiments
    if 'all' in run_experiments:
        run_experiments = ['exp1', 'exp2', 'exp3', 'exp4']
    
    # Run experiments (all using the SAME dataloaders/test set)
    if 'exp1' in run_experiments:
        model1 = run_experiment_1_supervised_limited(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            label_fraction=args.label_fraction,
            output_dir=args.output_dir,
            config_path=args.config
        )
    
    if 'exp2' in run_experiments:
        model2 = run_experiment_2_semi_supervised_model_based(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            label_fraction=args.label_fraction,
            confidence_threshold=args.confidence_threshold,
            output_dir=args.output_dir,
            config_path=args.config,
            k_neighbors=args.k_neighbors
        )

    if 'exp3' in run_experiments:
        model3 = run_experiment_3_semi_supervised_model_based(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            label_fraction=args.label_fraction,
            confidence_threshold=args.confidence_threshold,
            output_dir=args.output_dir,
            config_path=args.config,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples
        )
    
    if 'exp4' in run_experiments:
        model4 = run_experiment_4_fully_supervised(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            output_dir=args.output_dir,
            config_path=args.config
        )
    
    # Generate comparison report
    generate_comparison_report(args.output_dir)
    
    print("\n" + "="*80)
    print("✓ ALL EXPERIMENTS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
