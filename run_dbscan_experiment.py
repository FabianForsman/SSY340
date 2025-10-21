"""
Experiment 4: Semi-supervised with DBSCAN clustering for pseudo-labeling.

This experiment uses DBSCAN instead of k-NN to generate pseudo-labels.
DBSCAN advantages:
- Automatically determines number of clusters
- Identifies outliers/noise points
- Works well with varying density clusters

Key parameters:
- eps: Maximum distance between two samples to be considered neighbors
- min_samples: Minimum samples in a neighborhood to form a cluster
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from src.fine_tune_model import HateSpeechFineTuner


def run_experiment_4_semi_supervised_dbscan(
    data_path: str,
    label_fraction: float = 0.1,
    eps: float = 0.5,
    min_samples: int = 5,
    confidence_threshold: float = 0.6,
    output_dir: str = "outputs/experiments",
    config_path: str = "config.yaml"
):
    """
    Experiment 4: Semi-supervised with DBSCAN clustering.
    
    1. Train initial model on limited labels
    2. Use DBSCAN to cluster unlabeled data based on embeddings
    3. Assign labels based on nearest labeled neighbors for each cluster
    4. Keep high-confidence predictions
    5. Retrain on labeled + pseudo-labeled data
    
    Args:
        data_path: Path to labeled data CSV
        label_fraction: Fraction of training data to use as labeled
        eps: DBSCAN epsilon (max distance between neighbors)
        min_samples: DBSCAN min_samples (min samples per cluster)
        confidence_threshold: Confidence threshold for pseudo-labels
        output_dir: Directory to save results
        config_path: Path to config file
    """
    print("\n" + "="*80)
    print("EXPERIMENT 4: SEMI-SUPERVISED WITH DBSCAN CLUSTERING")
    print("="*80)
    print(f"Initial labels: {label_fraction*100:.0f}% of training data")
    print(f"DBSCAN eps: {eps}")
    print(f"DBSCAN min_samples: {min_samples}")
    print(f"Confidence threshold: {confidence_threshold}")
    print("="*80)
    
    # Initialize fine-tuner
    fine_tuner = HateSpeechFineTuner(
        base_model="all-MiniLM-L6-v2",
        num_classes=3,
        config_path=config_path
    )
    
    # Load full dataset
    train_df, val_df, test_df = fine_tuner.load_and_preprocess_data(
        data_path=data_path,
        test_size=0.2,
        val_size=0.1,
        apply_preprocessing=True,
        balance_classes=True
    )
    
    # Split into labeled and unlabeled
    train_df_labeled, train_df_unlabeled = train_test_split(
        train_df,
        train_size=label_fraction,
        random_state=42,
        stratify=train_df['label']
    )
    
    print(f"\nLabeled samples: {len(train_df_labeled)}")
    print(f"Unlabeled samples: {len(train_df_unlabeled)}")
    
    # Step 1: Train initial model on labeled data
    print("\n--- Step 1: Training initial model on labeled data ---")
    train_examples, val_examples = fine_tuner.prepare_training_data(
        train_df_labeled, val_df
    )
    
    fine_tuner.create_model_with_classifier()
    
    initial_model_path = Path(output_dir) / "exp4_initial_model"
    fine_tuner.train(
        train_examples=train_examples,
        val_examples=val_examples,
        output_path=str(initial_model_path),
        epochs=6,
        batch_size=32,
        learning_rate=2e-5
    )
    
    # Step 2: Generate pseudo-labels using DBSCAN
    print("\n--- Step 2: Generating pseudo-labels with DBSCAN ---")
    pseudo_labels, confidences, cluster_info = generate_pseudo_labels_with_dbscan(
        model=fine_tuner.model,
        unlabeled_texts=train_df_unlabeled['text'].tolist(),
        labeled_texts=train_df_labeled['text'].tolist(),
        labeled_labels=train_df_labeled['label'].values,
        eps=eps,
        min_samples=min_samples,
        confidence_threshold=confidence_threshold
    )
    
    # Check if pseudo-labeling was successful
    if len(pseudo_labels) == 0 or len(confidences) == 0:
        print("Warning: DBSCAN pseudo-labeling failed. No labels generated.")
        print("Try adjusting eps or min_samples parameters.")
        pseudo_labeled_df = pd.DataFrame()
    else:
        # Filter high-confidence predictions
        high_conf_mask = np.array(confidences) >= confidence_threshold
        pseudo_labeled_df = train_df_unlabeled.copy()
        pseudo_labeled_df['label'] = pseudo_labels
        pseudo_labeled_df = pseudo_labeled_df[high_conf_mask].reset_index(drop=True)
    
    print(f"High-confidence pseudo-labels: {len(pseudo_labeled_df)} / {len(train_df_unlabeled)}")
    
    if len(pseudo_labeled_df) > 0:
        high_conf_mask = np.array(confidences) >= confidence_threshold
        if high_conf_mask.sum() > 0:
            print(f"Average confidence: {np.mean(confidences[high_conf_mask]):.4f}")
        print(f"Pseudo-label distribution:")
        for label in sorted(pseudo_labeled_df['label'].unique()):
            count = (pseudo_labeled_df['label'] == label).sum()
            print(f"  {label}: {count} samples")
        print(f"\nCluster statistics:")
        print(f"  Number of clusters: {cluster_info['n_clusters']}")
        print(f"  Noise points: {cluster_info['n_noise']}")
        print(f"  Silhouette score: {cluster_info['silhouette']:.4f}")
    else:
        print(f"Average confidence: N/A")
        print(f"Pseudo-label distribution: None")
    
    # Step 3: Combine labeled + pseudo-labeled data
    combined_train_df = pd.concat([train_df_labeled, pseudo_labeled_df], ignore_index=True)
    print(f"\nCombined training set: {len(combined_train_df)} samples")
    
    # Step 4: Retrain on combined data
    print("\n--- Step 3: Retraining on labeled + pseudo-labeled data ---")
    combined_train_examples, val_examples = fine_tuner.prepare_training_data(
        combined_train_df, val_df
    )
    
    # Create fresh model
    fine_tuner.create_model_with_classifier()
    
    final_model_path = Path(output_dir) / "exp4_semi_supervised_dbscan"
    fine_tuner.train(
        train_examples=combined_train_examples,
        val_examples=val_examples,
        output_path=str(final_model_path),
        epochs=6,
        batch_size=32,
        learning_rate=2e-5
    )
    
    # Evaluate
    metrics = fine_tuner.evaluate(test_df)
    
    # Save results
    results = {
        'experiment': 'semi_supervised_dbscan',
        'label_fraction': label_fraction,
        'initial_labeled': len(train_df_labeled),
        'pseudo_labeled': len(pseudo_labeled_df),
        'total_training': len(combined_train_df),
        'eps': eps,
        'min_samples': min_samples,
        'n_clusters': cluster_info['n_clusters'],
        'n_noise': cluster_info['n_noise'],
        'confidence_threshold': confidence_threshold,
        'test_samples': len(test_df),
        **metrics
    }
    
    results_df = pd.DataFrame([results])
    results_path = Path(output_dir) / "exp4_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    
    return metrics, final_model_path


def generate_pseudo_labels_with_dbscan(
    model,
    unlabeled_texts: list,
    labeled_texts: list,
    labeled_labels: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    confidence_threshold: float = 0.6
) -> tuple:
    """
    Generate pseudo-labels using DBSCAN clustering on embeddings.
    
    Strategy:
    1. Get embeddings for labeled and unlabeled data
    2. Cluster unlabeled embeddings using DBSCAN
    3. For each cluster, assign label based on nearest labeled neighbors
    4. Confidence = proportion of nearest neighbors with same label
    
    Args:
        model: Trained SentenceTransformer model
        unlabeled_texts: Texts to generate pseudo-labels for
        labeled_texts: Labeled texts for reference
        labeled_labels: Labels for the labeled texts
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        confidence_threshold: Minimum confidence for pseudo-labels
        
    Returns:
        predictions: Predicted labels
        confidences: Confidence scores
        cluster_info: Dictionary with clustering statistics
    """
    from sklearn.neighbors import KNeighborsClassifier
    
    print(f"Using DBSCAN (eps={eps}, min_samples={min_samples}) for clustering...")
    
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
        unlabeled_texts,
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
    
    # Train k-NN classifier on labeled embeddings to assign labels to clusters
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(labeled_embeddings, labeled_labels)
    
    # Predict labels for all unlabeled points
    predictions = knn.predict(unlabeled_embeddings)
    
    # Get confidence scores (probability of predicted class)
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
            confidences[majority_mask] *= (0.5 + 0.5 * cluster_purity)
    
    # Penalize noise points (label -1)
    noise_mask = cluster_labels == -1
    confidences[noise_mask] *= 0.5
    
    print(f"Predictions generated for {len(predictions)} samples")
    print(f"Confidence range: [{confidences.min():.3f}, {confidences.max():.3f}]")
    print(f"Mean confidence: {confidences.mean():.3f}")
    
    cluster_info = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': silhouette
    }
    
    return predictions, confidences, cluster_info


def main():
    parser = argparse.ArgumentParser(
        description="Run DBSCAN-based semi-supervised experiment"
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
        default='outputs/exp_dbscan',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--label-fraction',
        type=float,
        default=0.1,
        help='Fraction of training data to use as labeled'
    )
    
    parser.add_argument(
        '--eps',
        type=float,
        default=0.5,
        help='DBSCAN epsilon parameter (max distance between neighbors)'
    )
    
    parser.add_argument(
        '--min-samples',
        type=int,
        default=5,
        help='DBSCAN min_samples parameter (min samples per cluster)'
    )
    
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.6,
        help='Confidence threshold for pseudo-labeling'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run experiment
    run_experiment_4_semi_supervised_dbscan(
        data_path=args.data,
        label_fraction=args.label_fraction,
        eps=args.eps,
        min_samples=args.min_samples,
        confidence_threshold=args.confidence_threshold,
        output_dir=args.output_dir,
        config_path=args.config
    )
    
    print("\n" + "="*80)
    print("✓ DBSCAN EXPERIMENT COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
