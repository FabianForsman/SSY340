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
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
from tqdm import tqdm

from fine_tune_model import HateSpeechFineTuner
from sentence_transformers import SentenceTransformer
from data_loader import HateSpeechDataset, create_dataloaders
from preprocessing import create_transform
from data_augmentation import data_agumentation, augment_data_to_target_count


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
    output_dir: str = "outputs/experiments"
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
        num_workers=4
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
        epochs=2,
        batch_size=64,
        learning_rate=2e-5
    )
    
    # Evaluate
    metrics = fine_tuner.evaluate(test_df)
    
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
    
    # Copy model to models/ directory for easy access
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


def run_experiment_2_semi_supervised_model_based(
    train_loader,
    val_loader,
    test_loader,
    label_fraction: float = 0.1,
    confidence_threshold: float = 0.6,
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
        epochs=2,
        batch_size=64,
        learning_rate=2e-5
    )
    
    # Step 2: Generate pseudo-labels for unlabeled data using k-NN on embeddings
    print("\n--- Step 2: Generating pseudo-labels ---")
    pseudo_labels, confidences = generate_pseudo_labels_with_model(
        model=fine_tuner.model,
        texts=train_df_unlabeled[config["data"]["text_column"]].tolist(),
        labeled_texts=train_df_labeled[config["data"]["text_column"]].tolist(),
        labeled_labels=train_df_labeled['label'].values,
        confidence_threshold=confidence_threshold,
        k_neighbors=k_neighbors
    )
    
    # Check if pseudo-labeling was successful
    if len(pseudo_labels) == 0 or len(confidences) == 0:
        print("Warning: Pseudo-labeling failed. No labels generated.")
        print("Continuing with only labeled data (equivalent to Experiment 1).")
        pseudo_labeled_df = pd.DataFrame()
    else:
        # Filter high-confidence predictions
        high_conf_mask = np.array(confidences) >= confidence_threshold
        pseudo_labeled_df = train_df_unlabeled.copy()
        pseudo_labeled_df['label'] = pseudo_labels
        pseudo_labeled_df = pseudo_labeled_df[high_conf_mask].reset_index(drop=True)
    
    print(f"High-confidence pseudo-labels: {len(pseudo_labeled_df)} / {len(train_df_unlabeled)}")
    
    if len(pseudo_labeled_df) > 0:
        # Calculate stats only if we have pseudo-labels
        high_conf_mask = np.array(confidences) >= confidence_threshold
        if high_conf_mask.sum() > 0:
            print(f"Average confidence: {np.mean(confidences[high_conf_mask]):.4f}")
        print(f"Pseudo-label distribution:")
        for label in sorted(pseudo_labeled_df['label'].unique()):
            count = (pseudo_labeled_df['label'] == label).sum()
            print(f"  {label}: {count} samples")
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
    
    final_model_path = Path(output_dir) / "exp2_semi_supervised_model"
    fine_tuner.train(
        train_examples=combined_train_examples,
        val_examples=val_examples,
        output_path=str(final_model_path),
        epochs=2,
        batch_size=64,
        learning_rate=2e-5
    )
    
    # Evaluate
    metrics = fine_tuner.evaluate(test_df)
    
    # Save results
    results = {
        'experiment': 'semi_supervised_model_based',
        'label_fraction': label_fraction,
        'initial_labeled': len(train_df_labeled),
        'pseudo_labeled': len(pseudo_labeled_df),
        'total_training': len(combined_train_df),
        'confidence_threshold': confidence_threshold,
        'test_samples': len(test_df),
        **metrics
    }
    
    results_df = pd.DataFrame([results])
    results_path = Path(output_dir) / "exp2_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Copy model to models/ directory for easy access
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    final_model_save_path = models_dir / "exp2_semi_supervised_model_based"
    
    # Save model directly
    fine_tuner.model.save(str(final_model_save_path))
    print(f"✓ Model saved to: {final_model_save_path}")
    
    return metrics, final_model_path


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
    
    return predictions, confidences


def run_experiment_3_fully_supervised(
    train_loader,
    val_loader,
    test_loader,
    output_dir: str = "outputs/experiments",
    config_path: str = "config.yaml"
):
    """
    Experiment 3: Fully supervised learning (upper bound).
    
    Train on 100% of labeled data.
    
    Args:
        train_loader: Training data loader (shared across experiments)
        val_loader: Validation data loader (shared across experiments)
        test_loader: Test data loader (shared across experiments)
        output_dir: Directory to save results
        config_path: Path to config file
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: FULLY SUPERVISED (UPPER BOUND)")
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
    
    model_output = Path(output_dir) / "exp3_fully_supervised"
    fine_tuner.train(
        train_examples=train_examples,
        val_examples=val_examples,
        output_path=str(model_output),
        epochs=2,
        batch_size=64,
        learning_rate=2e-5
    )
    
    # Evaluate
    metrics = fine_tuner.evaluate(test_df)
    
    # Save results
    results = {
        'experiment': 'fully_supervised',
        'label_fraction': 1.0,
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        **metrics
    }
    
    results_df = pd.DataFrame([results])
    results_path = Path(output_dir) / "exp3_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Copy model to models/ directory for easy access
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    final_model_save_path = models_dir / "exp3_fully_supervised"
    
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
    for exp_file in ['exp1_results.csv', 'exp2_results.csv', 'exp3_results.csv']:
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
        semi_acc = all_results.iloc[1]['accuracy'] if len(all_results) > 1 else None
        full_acc = all_results.iloc[2]['accuracy'] if len(all_results) > 2 else None
        
        print("\nKEY FINDINGS:")
        print(f"  Baseline (limited labels): {baseline_acc:.2%}")
        if semi_acc:
            improvement = (semi_acc - baseline_acc) / baseline_acc * 100
            print(f"  Semi-supervised: {semi_acc:.2%} ({improvement:+.1f}% vs baseline)")
        if full_acc:
            print(f"  Fully supervised: {full_acc:.2%} (upper bound)")
            if semi_acc:
                gap = (full_acc - semi_acc) / full_acc * 100
                print(f"  Semi-supervised reaches {(semi_acc/full_acc)*100:.1f}% of fully supervised performance")


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
        default=0.6,
        help='Confidence threshold for pseudo-labeling'
    )
    
    parser.add_argument(
        '--k-neighbors',
        type=int,
        default=5,
        help='Number of neighbors for k-NN pseudo-labeling'
    )
    
    parser.add_argument(
        '--experiments',
        nargs='+',
        choices=['exp1', 'exp2', 'exp3', 'all'],
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
    train_loader, val_loader, test_loader = create_train_test_splits(
        dataset=dataset,
        config=config,
        output_dir=args.output_dir
    )
    
    run_experiments = args.experiments
    if 'all' in run_experiments:
        run_experiments = ['exp1', 'exp2', 'exp3']
    
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
        model3 = run_experiment_3_fully_supervised(
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
