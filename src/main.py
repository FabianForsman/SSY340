"""
Main training script for hate speech detection project.
Orchestrates the entire pipeline: data loading, preprocessing,
embedding generation, clustering, and evaluation.
"""

import os
# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_loader import HateSpeechDataset, create_dataloaders
from preprocessing import create_transform
from embeddings import EmbeddingGenerator
from visualization import visualize_class_dist
from data_augmentation import data_agumentation, augment_data_to_target_count
from semi_supervised import run_self_training, SelfTrainingClassifier
from evaluation import evaluate_semi_supervised, plot_self_training_progress
from clustering import KMeansClustering, evaluate_clustering


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_output_directories(config):
    """
    Create output directories for embeddings, results, and figures.
    
    Args:
        config (dict): Configuration dictionary
    """
    Path(config["paths"]["embeddings_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["figures_dir"]).mkdir(parents=True, exist_ok=True)


def load_and_create_dataset(config):
    """
    Load raw data and create dataset without preprocessing.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        HateSpeechDataset: Raw dataset without transformations
    """
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA AND CREATING DATASET")
    print("=" * 70)

    if config["data"]["download_from_kaggle"]:
        print("Note: To download from Kaggle, ensure kaggle.json is in ~/.kaggle/")
        print("      Kaggle download not implemented in this version")

    # Create dataset (without transform first to get raw data info)
    dataset_raw = HateSpeechDataset(
        root=config["paths"]["raw_data_dir"],
        file=config["data"]["data_file"],
        text_column=config["data"]["text_column"],
        label_column=config["data"]["label_column"],
        transform=None  # No transform to see raw data
    )
    
    print("\n--- Raw Dataset Information ---")
    dataset_raw.get_dataset_info()

    return dataset_raw


def apply_preprocessing(config, dataset_raw):
    """
    Apply preprocessing transformations to the dataset.
    
    Args:
        config (dict): Configuration dictionary
        dataset_raw (HateSpeechDataset): Raw dataset
        
    Returns:
        tuple: (dataset, text_transform) - Preprocessed dataset and transform
    """
    print("\n" + "=" * 70)
    print("STEP 2: APPLYING PREPROCESSING TO DATASET")
    print("=" * 70)
    
    # Create text transform for preprocessing
    text_transform = create_transform(**config["preprocessing"])
    
    print(f"Applying preprocessing transform:")
    print(f"  - Lowercase: {config['preprocessing'].get('lowercase', True)}")
    print(f"  - Remove URLs: {config['preprocessing'].get('remove_urls', True)}")
    print(f"  - Remove mentions: {config['preprocessing'].get('remove_mentions', True)}")
    print(f"  - Remove hashtags: {config['preprocessing'].get('remove_hashtags', True)}")
    print(f"  - Remove numbers: {config['preprocessing'].get('remove_numbers', True)}")
    print(f"  - Remove stopwords: {config['preprocessing'].get('remove_stopwords', False)}")
    print(f"  - Remove quotes: {config['preprocessing'].get('remove_quotes', True)}")
    
    # Update the raw dataset with the transform
    dataset_raw.transform = text_transform
    
    print(f"\nPreprocessing applied to {len(dataset_raw)} samples")
    
    return dataset_raw, text_transform

def apply_data_augmentation(config, dataset):
    """
    Apply data augmentation to the dataset.
    
    Args:
        config (dict): Configuration dictionary
        dataset (HateSpeechDataset): Preprocessed dataset
        
    Returns:
        HateSpeechDataset: Augmented dataset
    """
    data_augmentation_enabled = config.get("data_augmentation", {}).get("enabled", False)
    if not data_augmentation_enabled:
        print("Data augmentation not enabled. Skipping this step.")
        return dataset
    
    print("\n" + "=" * 70)
    print("STEP 3: APPLYING DATA AUGMENTATION")
    print("=" * 70)
    
    methods = config.get("data_augmentation", {}).get("methods", {})
    
    # Synonym replacement
    if methods.get("synonym_replacement", {}).get("enabled", False):
        num_replacements = methods.get("synonym_replacement", {}).get("n", 1)
        print(f"Synonym Replacement enabled:")
        print(f"  - Number of words to replace: {num_replacements}")

    hate_speech_data = dataset.df[dataset.df['label'] == 0].copy()

    augmented_texts = []
    for idx, row in tqdm(hate_speech_data.iterrows(), total=len(hate_speech_data), desc="Augmenting hate speech samples"):
        text = row[dataset.text_column]
        label = row[dataset.label_column]
        label_desc = row.get('label_desc', None)  # Get label_desc if it exists
        
        # Apply synonym replacement
        if methods.get("synonym_replacement", {}).get("enabled", False):
            augmented_versions = data_agumentation(text, num_replacements)
            for aug_text in augmented_versions:
                augmented_texts.append((aug_text, label, label_desc))

    # Create new DataFrame for augmented data
    if augmented_texts:
        df_augmented = pd.DataFrame(augmented_texts, columns=[dataset.text_column, dataset.label_column, 'label_desc'])
        print(f"Generated {len(df_augmented)} augmented samples.")
        
        # Combine with original dataset
        df_combined = pd.concat([dataset.df, df_augmented], ignore_index=True)
        print(f"Combined dataset size after augmentation: {len(df_combined)} samples.")
        # Update the dataset's DataFrame
        dataset.update_dataframe(df_combined)
        
    else:
        print("No augmented samples were generated.")

    return dataset

def balance_dataset(dataset):
    """
    Balance the dataset by augmenting minority classes to match the majority class count.

    Args:
        dataset (HateSpeechDataset): Dataset to balance
    Returns:
        HateSpeechDataset: Balanced dataset
    """
    print("\n" + "=" * 70)
    print("STEP 4: BALANCING DATASET")
    print("=" * 70)

    class_counts = dataset.df['label'].value_counts()
    max_count = class_counts.max()
    print(f"Class distribution before balancing:\n{class_counts}")

    balanced_dfs = []
    for label, count in class_counts.items():
        df_class = dataset.df[dataset.df['label'] == label]
        df_balanced = augment_data_to_target_count(df_class, max_count)
        balanced_dfs.append(df_balanced)

    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    print(f"Class distribution after balancing:\n{df_balanced['label'].value_counts()}")

    # Update the dataset's DataFrame
    dataset.update_dataframe(df_balanced)

    return dataset

def save_processed_data(dataloaders, csv_names, config):
    """
    Save processed dataset splits to CSV files.
    
    Args:
        dataloaders (tuple): Tuple of (train_loader, dev_loader, test_loader)
        csv_names (tuple): Tuple of (train_csv, dev_csv, test_csv)
        config (dict): Configuration dictionary
    """

    print("\n" + "=" * 70)
    print("STEP 5: SAVING PROCESSED DATA TO CSV FILES")
    print("=" * 70)

    train_loader, dev_loader, test_loader = dataloaders
    train_csv, dev_csv, test_csv = csv_names
    
    def save_split(loader, filepath):
        texts = []
        labels = []
        for batch in loader:
            batch_texts, batch_labels = batch
            texts.extend(batch_texts)
            labels.extend([label.item() if torch.is_tensor(label) else label for label in batch_labels])
        df = pd.DataFrame({
            config["data"]["text_column"]: texts,
            config["data"]["label_column"]: labels
        })
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} samples to {filepath}")
    
    save_split(train_loader, Path(config["paths"]["loaders_csvs"]) / train_csv)
    save_split(dev_loader, Path(config["paths"]["loaders_csvs"]) / dev_csv)
    save_split(test_loader, Path(config["paths"]["loaders_csvs"]) / test_csv)

    print(f"Processed data saved to {config['paths']['loaders_csvs']}")


def calculate_similarities(embeddings, all_texts, all_labels, generator, embeddings_path, num_samples=3):
    """
    Calculate and display embedding similarities for sample texts.
    
    Args:
        embeddings (np.ndarray): Generated embeddings
        all_texts (list): List of all text samples
        all_labels (list): List of all labels
        generator (EmbeddingGenerator): Embedding generator instance
        embeddings_path (Path): Path to saved embeddings
        num_samples (int): Number of samples to display similarities for
    """
    print("\n" + "=" * 70)
    print(f"CALCULATING EMBEDDING SIMILARITIES (FIRST {num_samples} SAMPLES)")
    print("=" * 70)
    
    embeddings = generator.load_embeddings(embeddings_path)
    similarities = generator.get_similarities(embeddings[:num_samples], embeddings[:num_samples])
    print(f"Similarity matrix (first {num_samples} samples):")
    print(similarities)
    for text, label in zip(all_texts[:num_samples], all_labels[:num_samples]):
        print(f"Text: {text} | Label: {label}")


def run_pipeline(config):
    """
    Run the complete hate speech detection pipeline.

    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (embeddings, dataloaders, all_labels) - Generated embeddings, dataloaders, and labels
    """
    print("=" * 70)
    print("HATE SPEECH DETECTION - UNSUPERVISED LEARNING")
    print("=" * 70)

    # Create output directories
    create_output_directories(config)

    # Step 1: Load Data and Create Dataset
    dataset_raw = load_and_create_dataset(config)

    # Visualize class distribution
    #visualize_class_dist(dataset_raw)
    
    # Step 2: Apply Preprocessing
    dataset, text_transform = apply_preprocessing(config, dataset_raw)
    
    # Step 3: Apply data augmentation
    dataset = apply_data_augmentation(config, dataset)

    # Visualize class distribution after augmentation
    #visualize_class_dist(dataset)

    # Step 4: Balance the dataset
    dataset = balance_dataset(dataset)
    
    # Step 5: Create DataLoaders
    # Shuffle before splitting
    dataset.df = dataset.df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Visualize class distribution after balancing
    #visualize_class_dist(dataset)

    # Create DataLoaders for train, dev, test splits
    dataloaders = create_dataloaders(
        dataset,
        batch_size=config["embedding"]["batch_size"],
        shuffle=True,
        num_workers=4
    )

    # Step 8: Save processed data
    save_processed_data(dataloaders, ("train.csv", "dev.csv", "test.csv"), config)

    # Step 6: Generate Embeddings
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING EMBEDDINGS")
    print("=" * 70)
    generator = EmbeddingGenerator(model_name=config["embedding"]["model"])
    all_texts = [dataset[i][0] for i in range(len(dataset))]
    all_labels = dataset.df[dataset.label_column].tolist()
    embeddings = generator.encode(
        all_texts,
        batch_size=32,
        show_progress=True,
        normalize=True
    )
    embeddings_path = Path(config["paths"]["embeddings_dir"]) / "embeddings.npy"
    generator.save_embeddings(embeddings, embeddings_path)

    # Step 7: Calculate Similarities
    calculate_similarities(embeddings, all_texts, all_labels, generator, embeddings_path)

    return embeddings, dataloaders, np.array(all_labels)


def run_semi_supervised_pipeline(config, embeddings, all_labels, dataloaders):
    """
    Run semi-supervised self-training pipeline.
    
    Args:
        config (dict): Configuration dictionary
        embeddings (np.ndarray): All embeddings
        all_labels (np.ndarray): All labels
        dataloaders (tuple): (train_loader, dev_loader, test_loader)
        
    Returns:
        tuple: (classifier, test_predictions, test_labels)
    """
    if not config.get('semi_supervised', {}).get('enabled', False):
        print("\nSemi-supervised learning not enabled. Skipping.")
        return None, None, None
    
    print("\n" + "=" * 70)
    print("STEP 8: SEMI-SUPERVISED SELF-TRAINING")
    print("=" * 70)
    
    train_loader, dev_loader, test_loader = dataloaders
    
    # Get train/dev/test splits from embeddings
    # We need to extract them in the same order as the dataloaders
    train_size = len(train_loader.dataset)
    dev_size = len(dev_loader.dataset)
    test_size = len(test_loader.dataset)
    
    # Split embeddings and labels
    embeddings_train = embeddings[:train_size]
    labels_train = all_labels[:train_size]
    
    embeddings_dev = embeddings[train_size:train_size + dev_size]
    labels_dev = all_labels[train_size:train_size + dev_size]
    
    embeddings_test = embeddings[train_size + dev_size:]
    labels_test = all_labels[train_size + dev_size:]
    
    # Simulate semi-supervised scenario:
    # Use only a fraction of training labels, treat rest as unlabeled
    train_split = config.get('semi_supervised', {}).get('train_split', 0.5)
    n_labeled = int(len(labels_train) * train_split)
    
    print(f"\nSimulating semi-supervised scenario:")
    print(f"  Using {n_labeled}/{len(labels_train)} training labels ({train_split*100:.0f}%)")
    print(f"  Treating {len(labels_train) - n_labeled} samples as unlabeled")
    
    # Split training data into labeled and unlabeled
    # Shuffle indices for random split
    np.random.seed(config.get('random_seed', 42))
    indices = np.random.permutation(len(labels_train))
    labeled_indices = indices[:n_labeled]
    unlabeled_indices = indices[n_labeled:]
    
    embeddings_labeled = embeddings_train[labeled_indices]
    labels_labeled = labels_train[labeled_indices]
    
    embeddings_unlabeled = embeddings_train[unlabeled_indices]
    labels_unlabeled_true = labels_train[unlabeled_indices]  # Keep for evaluation
    
    # Run self-training
    classifier, pseudo_labels = run_self_training(
        embeddings_labeled,
        labels_labeled,
        embeddings_unlabeled,
        config,
        verbose=True
    )
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("EVALUATING ON TEST SET")
    print("=" * 70)
    
    # Predict on test set
    test_predictions = classifier.predict(embeddings_test)
    
    # Define target names
    target_names = ['Hate Speech', 'Offensive Language', 'Neither']
    
    # Comprehensive evaluation
    test_metrics = evaluate_semi_supervised(
        labels_labeled,  # Original labeled training data
        labels_test,
        test_predictions,
        pseudo_labels,
        target_names=target_names,
        output_dir=Path(config["paths"]["results_dir"])
    )
    
    # Save results
    results_dir = Path(config["paths"]["results_dir"])
    classifier.save_results(results_dir)
    
    # Plot training progress
    history_df = classifier.get_history_dataframe()
    if len(history_df) > 0:
        plot_self_training_progress(
            history_df,
            save_path=results_dir / "self_training_progress.png"
        )
    
    # Save test predictions
    test_results_df = pd.DataFrame({
        'true_label': labels_test,
        'predicted_label': test_predictions,
        'true_label_name': [target_names[int(l)] if l >= 0 else 'Unknown' for l in labels_test],
        'predicted_label_name': [target_names[int(l)] if l >= 0 and l < len(target_names) else 'Unknown' for l in test_predictions]
    })
    test_results_path = results_dir / "test_predictions.csv"
    test_results_df.to_csv(test_results_path, index=False)
    print(f"\nSaved test predictions to {test_results_path}")
    
    # Save evaluation metrics
    metrics_df = pd.DataFrame([test_metrics])
    metrics_path = results_dir / "test_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved test metrics to {metrics_path}")
    
    return classifier, test_predictions, labels_test


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Hate Speech Detection - Unsupervised Learning"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Run main pipeline - returns embeddings, dataloaders, and labels from balanced dataset
    embeddings, dataloaders, all_labels = run_pipeline(config)
    
    # Run semi-supervised pipeline with the same dataloaders and embeddings
    classifier, test_predictions, test_labels = run_semi_supervised_pipeline(
        config,
        embeddings,
        all_labels,
        dataloaders
    )

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    if classifier is not None:
        print(f"Semi-supervised training completed:")
        print(f"  Iterations: {len(classifier.history['iteration'])}")
        print(f"  Final labeled samples: {classifier.history['n_labeled'][-1] if classifier.history['n_labeled'] else 0}")
        
        if test_predictions is not None and test_labels is not None:
            from evaluation import calculate_metrics
            final_metrics = calculate_metrics(test_labels, test_predictions)
            print(f"  Test accuracy: {final_metrics['accuracy']:.4f}")
            print(f"  Test macro F1: {final_metrics['macro_f1']:.4f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
