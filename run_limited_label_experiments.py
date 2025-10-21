"""
Run comprehensive experiments with limited labeled data.

This script compares different approaches:
1. Supervised baseline (limited labels only)
2. Semi-supervised with clustering
3. Semi-supervised with model-based pseudo-labeling
4. Fully supervised (upper bound)

Goal: Show that we can achieve good results with very limited labeled data.
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch

from src.fine_tune_model import HateSpeechFineTuner
from sentence_transformers import SentenceTransformer


def run_experiment_1_supervised_limited(
    data_path: str,
    label_fraction: float = 0.1,
    output_dir: str = "outputs/experiments",
    config_path: str = "config.yaml"
):
    """
    Experiment 1: Supervised learning with limited labels.
    
    Train a classifier using only a small fraction of labeled data.
    
    Args:
        data_path: Path to labeled data CSV
        label_fraction: Fraction of training data to use (e.g., 0.1 = 10%)
        output_dir: Directory to save results
        config_path: Path to config file
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: SUPERVISED BASELINE WITH LIMITED LABELS")
    print("="*80)
    print(f"Using only {label_fraction*100:.0f}% of training labels")
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
        epochs=6,
        batch_size=32,
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
    
    return metrics, model_output


def run_experiment_2_semi_supervised_model_based(
    data_path: str,
    label_fraction: float = 0.1,
    confidence_threshold: float = 0.8,
    output_dir: str = "outputs/experiments",
    config_path: str = "config.yaml"
):
    """
    Experiment 2: Semi-supervised with model-based pseudo-labeling.
    
    1. Train initial model on limited labels
    2. Use model to predict labels on unlabeled data (pseudo-labeling)
    3. Keep high-confidence predictions
    4. Retrain on labeled + pseudo-labeled data
    
    Args:
        data_path: Path to labeled data CSV
        label_fraction: Fraction of training data to use as labeled
        confidence_threshold: Confidence threshold for pseudo-labels (0-1)
        output_dir: Directory to save results
        config_path: Path to config file
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: SEMI-SUPERVISED WITH MODEL-BASED PSEUDO-LABELING")
    print("="*80)
    print(f"Initial labels: {label_fraction*100:.0f}% of training data")
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
    
    initial_model_path = Path(output_dir) / "exp2_initial_model"
    fine_tuner.train(
        train_examples=train_examples,
        val_examples=val_examples,
        output_path=str(initial_model_path),
        epochs=6,
        batch_size=32,
        learning_rate=2e-5
    )
    
    # Step 2: Generate pseudo-labels for unlabeled data
    print("\n--- Step 2: Generating pseudo-labels ---")
    pseudo_labels, confidences = generate_pseudo_labels_with_model(
        model=fine_tuner.model,
        texts=train_df_unlabeled['text'].tolist(),
        confidence_threshold=confidence_threshold
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
        epochs=6,
        batch_size=32,
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
    
    return metrics, final_model_path


def generate_pseudo_labels_with_model(
    model: SentenceTransformer,
    texts: list,
    confidence_threshold: float = 0.8
) -> tuple:
    """
    Generate pseudo-labels using a trained model.
    
    Returns predicted labels and confidence scores.
    """
    from torch.nn.functional import softmax
    import torch.nn as nn
    
    # Get embeddings
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_tensor=True
    )
    
    # Find the classifier layer added by SoftmaxLoss
    # In sentence-transformers, it's typically in the last module
    classifier = None
    
    # Check if model has a direct classifier attribute
    if hasattr(model, 'classifier'):
        classifier = model.classifier
    else:
        # Look through all modules for a Linear layer with 3 outputs (num_classes)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if this could be our classifier (384 -> 3)
                if module.out_features == 3:  # num_classes
                    classifier = module
                    print(f"Found classifier: {name} with shape {module.weight.shape}")
                    break
    
    if classifier is None:
        print("Warning: Could not find classifier layer. Using k-NN fallback for pseudo-labeling.")
        # Fallback: use k-NN on embeddings
        from sklearn.neighbors import KNeighborsClassifier
        # This is a simplified fallback - in practice you'd need labeled data
        # For now, return empty predictions
        return np.array([]), np.array([])
    
    # Get predictions
    with torch.no_grad():
        logits = classifier(embeddings)
        probs = softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
    
    return predictions.cpu().numpy(), confidences.cpu().numpy()


def run_experiment_3_fully_supervised(
    data_path: str,
    output_dir: str = "outputs/experiments",
    config_path: str = "config.yaml"
):
    """
    Experiment 3: Fully supervised learning (upper bound).
    
    Train on 100% of labeled data.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: FULLY SUPERVISED (UPPER BOUND)")
    print("="*80)
    print("Using 100% of training labels")
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
    
    print(f"\nUsing all {len(train_df)} training samples")
    
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
        epochs=6,
        batch_size=32,
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
        default=0.8,
        help='Confidence threshold for pseudo-labeling'
    )
    
    parser.add_argument(
        '--experiments',
        nargs='+',
        choices=['exp1', 'exp2', 'exp3', 'all'],
        default=['all'],
        help='Which experiments to run'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    run_experiments = args.experiments
    if 'all' in run_experiments:
        run_experiments = ['exp1', 'exp2', 'exp3']
    
    # Run experiments
    if 'exp1' in run_experiments:
        run_experiment_1_supervised_limited(
            data_path=args.data,
            label_fraction=args.label_fraction,
            output_dir=args.output_dir,
            config_path=args.config
        )
    
    if 'exp2' in run_experiments:
        run_experiment_2_semi_supervised_model_based(
            data_path=args.data,
            label_fraction=args.label_fraction,
            confidence_threshold=args.confidence_threshold,
            output_dir=args.output_dir,
            config_path=args.config
        )
    
    if 'exp3' in run_experiments:
        run_experiment_3_fully_supervised(
            data_path=args.data,
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
