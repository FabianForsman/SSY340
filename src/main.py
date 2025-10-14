"""
Main training script for hate speech detection project.
Orchestrates the entire pipeline: data loading, preprocessing,
embedding generation, clustering, and evaluation.
"""

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

from data_loader import HateSpeechDataset, create_dataloader
from preprocessing import create_transform
from embeddings import EmbeddingGenerator
from visualization import visualize_class_dist
from data_augmentation import data_agumentation, augment_data_to_target_count


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

def save_processed_data(config, dataset, dataset_raw):
    """
    Save processed data to CSV files.
    
    Args:
        config (dict): Configuration dictionary
        dataset (HateSpeechDataset): Preprocessed dataset
        dataset_raw (HateSpeechDataset): Raw dataset
    """
    if not config.get("save_processed_data", True):
        return
        
    print("\nSaving processed data...")
    processed_data = []
    
    # Use DataLoader for efficient batch processing
    save_dataloader = create_dataloader(
        dataset, 
        batch_size=config["embedding"]["batch_size"],
        shuffle=False,
        num_workers=4
    )
    
    for texts, labels in tqdm(save_dataloader, desc="Processing batches for saving"):
        for text, label in zip(texts, labels):
            processed_data.append({
                'cleaned_text': text,
                'label': label.item() if torch.is_tensor(label) else label
            })
    
    df_processed = pd.DataFrame(processed_data)
    processed_path = Path(config["paths"]["processed_data_dir"]) / "processed_data.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(processed_path, index=False)
    print(f"Saved processed data to {processed_path}")
    
    # Also save with original data
    df_full = dataset_raw.df.copy()
    df_full['cleaned_text'] = df_processed['cleaned_text']
    df_full_path = Path(config["paths"]["processed_data_dir"]) / "full_processed_data.csv"
    df_full.to_csv(df_full_path, index=False)
    print(f"Saved full processed data (with original columns) to {df_full_path}")


def generate_embeddings(config, dataset):
    """
    Generate embeddings for the dataset.
    
    Args:
        config (dict): Configuration dictionary
        dataset (HateSpeechDataset): Preprocessed dataset
        
    Returns:
        tuple: (embeddings, all_texts, all_labels, generator, embeddings_path)
    """
    print("\n" + "=" * 70)
    print("STEP 3: GENERATING EMBEDDINGS")
    print("=" * 70)

    embedding_model = config["embedding"]["model"]
    print(f"Using embedding model: {embedding_model}")

    generator = EmbeddingGenerator(embedding_model)
    
    # Extract all cleaned texts from dataset using DataLoader
    print(f"Extracting texts from dataset...")
    all_texts = []
    all_labels = []
    
    # Create DataLoader for efficient batch processing
    embedding_dataloader = create_dataloader(
        dataset,
        batch_size=config["embedding"]["batch_size"],
        shuffle=False,  # Keep original order
        num_workers=4
    )
    
    for texts, labels in tqdm(embedding_dataloader, desc="Extracting samples"):
        all_texts.extend(texts)
        all_labels.extend([label.item() if torch.is_tensor(label) else label for label in labels])
    
    # Generate embeddings
    print(f"Generating embeddings for {len(all_texts)} texts...")
    embeddings = generator.encode(
        all_texts,
        batch_size=config["embedding"]["batch_size"],
        normalize=config["embedding"]["normalize"],
        show_progress=True,
    )

    # Save embeddings
    embeddings_path = (
        Path(config["paths"]["embeddings_dir"]) / f"embeddings_{embedding_model}.npy"
    )
    generator.save_embeddings(embeddings, embeddings_path)
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    return embeddings, all_texts, all_labels, generator, embeddings_path


def load_saved_embeddings(config, dataset=None):
    """
    Load previously saved embeddings from disk.
    
    Args:
        config (dict): Configuration dictionary
        dataset (HateSpeechDataset, optional): Preprocessed dataset to extract texts and labels.
                                               If None, only embeddings will be loaded.
        
    Returns:
        tuple: (embeddings, all_texts, all_labels, generator, embeddings_path)
               If dataset is None, all_texts and all_labels will be empty lists.
    """
    print("\n" + "=" * 70)
    print("STEP 3: LOADING SAVED EMBEDDINGS")
    print("=" * 70)

    embedding_model = config["embedding"]["model"]
    embeddings_path = (
        Path(config["paths"]["embeddings_dir"]) / f"embeddings_{embedding_model}.npy"
    )
    
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found at {embeddings_path}. "
            f"Please run generate_embeddings first."
        )
    
    print(f"Loading embeddings from: {embeddings_path}")
    generator = EmbeddingGenerator(embedding_model)
    embeddings = generator.load_embeddings(embeddings_path)
    print(f"Loaded embeddings shape: {embeddings.shape}")
    
    # Extract texts and labels if dataset is provided
    all_texts = []
    all_labels = []
    
    if dataset is not None:
        print(f"Extracting texts and labels from dataset...")
        embedding_dataloader = create_dataloader(
            dataset,
            batch_size=config["embedding"]["batch_size"],
            shuffle=False,
            num_workers=4
        )
        
        for texts, labels in tqdm(embedding_dataloader, desc="Extracting samples"):
            all_texts.extend(texts)
            all_labels.extend([label.item() if torch.is_tensor(label) else label for label in labels])
    
    return embeddings, all_texts, all_labels, generator, embeddings_path


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
    print(f"STEP 4: CALCULATING EMBEDDING SIMILARITIES (FIRST {num_samples} SAMPLES)")
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
        np.ndarray: Generated embeddings
    """
    print("=" * 70)
    print("HATE SPEECH DETECTION - UNSUPERVISED LEARNING")
    print("=" * 70)

    # Create output directories
    create_output_directories(config)

    # Step 1: Load Data and Create Dataset
    dataset_raw = load_and_create_dataset(config)

    # Visualize class distribution
    visualize_class_dist(dataset_raw)
    
    # Step 2: Apply Preprocessing
    dataset, text_transform = apply_preprocessing(config, dataset_raw)
    
    # Step 3: Apply data augmentation
    dataset = apply_data_augmentation(config, dataset)

    # Visualize class distribution after augmentation
    visualize_class_dist(dataset)

    # Step 4: Balance the dataset
    dataset = balance_dataset(dataset)

    # Visualize class distribution after augmentation
    visualize_class_dist(dataset)

    # Step 5: Generate or Load Embeddings
    # Uncomment ONE of the following options:
    
    # Option 1: Generate new embeddings
    embeddings, all_texts, all_labels, generator, embeddings_path = generate_embeddings(config, dataset)
    
    # Option 2: Load previously saved embeddings
    # embeddings, all_texts, all_labels, generator, embeddings_path = load_saved_embeddings(config, dataset)

    # Step 6: Calculate the embedding similarities
    calculate_similarities(embeddings, all_texts, all_labels, generator, embeddings_path)
    
    # Step 7: Save processed data
    save_processed_data(config, dataset, dataset_raw)
    return embeddings


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

    # Run pipeline
    embeddings = run_pipeline(config)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
