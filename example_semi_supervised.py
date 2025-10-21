"""
Example script demonstrating semi-supervised self-training.
Simulates a scenario with limited labeled data.
"""

import os
# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from embeddings import EmbeddingGenerator
from semi_supervised import SelfTrainingClassifier
from evaluation import calculate_metrics, plot_confusion_matrix


def main():
    """Run a simple semi-supervised example."""
    
    print("=" * 70)
    print("SEMI-SUPERVISED SELF-TRAINING EXAMPLE")
    print("=" * 70)
    
    # Sample data: hate speech detection examples
    texts = [
        # Class 0: Hate speech (labeled)
        "I hate those people they should all die",
        "Kill all the idiots",
        # Class 1: Offensive language (labeled)
        "This is fucking stupid",
        "What a dumb ass idea",
        # Class 2: Neither (labeled)
        "The weather is nice today",
        "I love this movie",
        
        # Unlabeled samples (will be pseudo-labeled)
        "Those people are disgusting trash",  # Likely class 0
        "Damn this traffic is annoying",       # Likely class 1
        "Great job on the project",            # Likely class 2
        "I despise them all",                  # Likely class 0
        "This shit is crazy",                  # Likely class 1
        "Beautiful sunset tonight",            # Likely class 2
        "Get rid of those scum",               # Likely class 0
        "Hell yeah that rocks",                # Likely class 1
        "Thanks for your help",                # Likely class 2
    ]
    
    # Labels: first 6 are labeled, rest are unlabeled (-1)
    labels = np.array([0, 0, 1, 1, 2, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    labeled_mask = labels != -1
    
    # True labels for evaluation (normally unknown)
    true_labels = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    
    print(f"\nDataset:")
    print(f"  Total samples: {len(texts)}")
    print(f"  Labeled: {labeled_mask.sum()}")
    print(f"  Unlabeled: {(~labeled_mask).sum()}")
    
    # Step 1: Generate embeddings
    print("\n" + "-" * 70)
    print("Step 1: Generating SBERT embeddings...")
    print("-" * 70)
    
    generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    embeddings = generator.encode(texts, show_progress=False, normalize=True)
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Step 2: Initialize self-training classifier
    print("\n" + "-" * 70)
    print("Step 2: Initializing self-training classifier...")
    print("-" * 70)
    
    classifier = SelfTrainingClassifier(
        clustering_method='kmeans',
        n_clusters=3,
        confidence_threshold=0.7,  # Lower threshold for small dataset
        max_iterations=3,
        min_samples_per_iteration=1,
        use_silhouette_for_confidence=True,
        random_state=42
    )
    
    # Step 3: Fit classifier
    print("\n" + "-" * 70)
    print("Step 3: Running self-training...")
    print("-" * 70)
    
    classifier.fit(
        embeddings=embeddings,
        initial_labels=labels,
        labeled_mask=labeled_mask,
        texts=texts,
        verbose=True
    )
    
    # Step 4: Evaluate results
    print("\n" + "-" * 70)
    print("Step 4: Evaluating results...")
    print("-" * 70)
    
    # Get final predictions
    final_predictions = classifier.final_labels
    
    # Compare with true labels
    unlabeled_indices = np.where(~labeled_mask)[0]
    pseudo_labels = final_predictions[unlabeled_indices]
    true_unlabeled = true_labels[unlabeled_indices]
    
    print(f"\nPseudo-labeling results:")
    print(f"{'Text':<40} {'Pseudo':<10} {'True':<10} {'Correct':<10}")
    print("-" * 70)
    
    label_names = {0: 'Hate', 1: 'Offensive', 2: 'Neither'}
    
    correct = 0
    for i, idx in enumerate(unlabeled_indices):
        pseudo = pseudo_labels[i]
        true = true_unlabeled[i]
        is_correct = pseudo == true
        if is_correct:
            correct += 1
        
        print(f"{texts[idx]:<40} {label_names.get(pseudo, 'Unknown'):<10} "
              f"{label_names[true]:<10} {'✓' if is_correct else '✗':<10}")
    
    accuracy = correct / len(unlabeled_indices) if len(unlabeled_indices) > 0 else 0
    print(f"\nPseudo-label accuracy: {accuracy:.2%} ({correct}/{len(unlabeled_indices)})")
    
    # Overall metrics
    metrics = calculate_metrics(true_labels, final_predictions)
    print(f"\nOverall metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    
    # Show training history
    if len(classifier.history['iteration']) > 0:
        print(f"\nTraining history:")
        history_df = classifier.get_history_dataframe()
        print(history_df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
