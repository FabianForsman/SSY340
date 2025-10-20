# Semi-Supervised Self-Training Implementation

## Overview

This implementation adds **semi-supervised self-training** using clustering and pseudo-labeling on SBERT sentence embeddings. The algorithm iteratively assigns confident pseudo-labels to unlabeled data and re-clusters to improve predictions.

## Architecture

### New Files

1. **`src/semi_supervised.py`**: Core self-training implementation
   - `SelfTrainingClassifier`: Main classifier with iterative pseudo-labeling
   - `run_self_training()`: Convenience function to run the pipeline

2. **`src/clustering.py`**: Clustering algorithms
   - `KMeansClustering`: K-Means wrapper
   - `DBSCANClustering`: DBSCAN wrapper  
   - `HierarchicalClustering`: Hierarchical clustering wrapper
   - `evaluate_clustering()`: Clustering quality metrics
   - `select_optimal_k()`: Automatic K selection

3. **`src/evaluation.py`**: Evaluation metrics and visualization
   - `calculate_metrics()`: Classification metrics (accuracy, F1, etc.)
   - `calculate_cluster_purity()`: Cluster purity score
   - `plot_confusion_matrix()`: Confusion matrix visualization
   - `plot_self_training_progress()`: Training progress plots
   - `evaluate_semi_supervised()`: Comprehensive evaluation

### Updated Files

1. **`src/main.py`**: 
   - Added `run_semi_supervised_pipeline()` function
   - Integrated semi-supervised training into main flow
   - Splits training data into labeled/unlabeled subsets

2. **`config.yaml`**:
   - Added `semi_supervised` section with hyperparameters

## Algorithm Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Load labeled training data (small subset)           │
│    + unlabeled data (majority of training set)         │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Generate SBERT embeddings for all data              │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
        ╔═════════════════════════╗
        ║  SELF-TRAINING LOOP     ║
        ╚═════════════════════════╝
                  │
    ┌─────────────┴─────────────┐
    │                           │
    ▼                           │
┌─────────────────────────────────────────────────────────┐
│ 3. Cluster all data (labeled + unlabeled)              │
│    using K-Means/DBSCAN on embeddings                  │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Map clusters to labels via majority voting          │
│    using labeled samples in each cluster               │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 5. Calculate confidence scores for unlabeled samples   │
│    - Option A: Distance to cluster center              │
│    - Option B: Silhouette coefficient                  │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 6. Select high-confidence predictions                  │
│    (confidence > threshold)                             │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 7. Add pseudo-labels to labeled set                    │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
          ┌───────┴────────┐
          │  Converged?    │
          │  or Max iters? │
          └───┬────────┬───┘
              │ No     │ Yes
              │        │
              └────────┤
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 8. Evaluate on test set                                │
│    - Confusion matrix                                   │
│    - Classification metrics                             │
│    - Training progress plots                            │
└─────────────────────────────────────────────────────────┘
```

## Configuration

Edit `config.yaml` to enable and configure semi-supervised learning:

```yaml
semi_supervised:
  enabled: true  # Set to true to enable
  clustering_method: "kmeans"  # 'kmeans', 'dbscan', or 'hierarchical'
  n_clusters: 3  # Number of clusters (for kmeans/hierarchical)
  confidence_threshold: 0.8  # Confidence threshold for pseudo-labeling [0-1]
  max_iterations: 5  # Maximum self-training iterations
  min_samples_per_iteration: 10  # Minimum samples to add per iteration
  use_silhouette_for_confidence: true  # Use silhouette score for confidence
  train_split: 0.5  # Fraction of training data to use as labeled
```

### Key Hyperparameters

- **`confidence_threshold`**: Higher values (e.g., 0.9) = more conservative, fewer but higher quality pseudo-labels
- **`train_split`**: Fraction of labeled data (0.1 = 10% labeled, 90% unlabeled)
- **`max_iterations`**: Maximum self-training iterations before stopping
- **`clustering_method`**: 
  - `kmeans`: Fast, assumes spherical clusters
  - `dbscan`: Handles noise, finds arbitrary-shaped clusters
  - `hierarchical`: Creates cluster hierarchy

## Usage

### Basic Usage

```bash
# Run with semi-supervised learning enabled
python src/main.py --config config.yaml
```

### Programmatic Usage

```python
from semi_supervised import SelfTrainingClassifier
from embeddings import EmbeddingGenerator

# Generate embeddings
generator = EmbeddingGenerator("all-MiniLM-L6-v2")
embeddings = generator.encode(texts)

# Initialize classifier
classifier = SelfTrainingClassifier(
    clustering_method='kmeans',
    n_clusters=3,
    confidence_threshold=0.8,
    max_iterations=5
)

# Fit with labeled and unlabeled data
classifier.fit(
    embeddings=all_embeddings,
    initial_labels=all_labels,  # -1 for unlabeled
    labeled_mask=labeled_mask,  # Boolean mask
    verbose=True
)

# Predict on test set
predictions = classifier.predict(test_embeddings)
```

## Output Files

The pipeline generates the following files in `outputs/results/`:

1. **`self_training_history.csv`**: Training progress per iteration
   - Number of labeled/unlabeled samples
   - Pseudo-labeled samples added
   - Average confidence scores
   - Clustering quality metrics (silhouette, purity)

2. **`cluster_label_mapping.csv`**: Mapping from cluster IDs to labels

3. **`test_predictions.csv`**: Test set predictions with true labels

4. **`test_metrics.csv`**: Test set performance metrics

5. **`confusion_matrix.png`**: Confusion matrix visualization

6. **`self_training_progress.png`**: 4-panel training progress plot
   - Labeled vs unlabeled samples over time
   - Pseudo-labeled samples per iteration
   - Average confidence scores
   - Clustering quality metrics

## Confidence Scoring Methods

### Method 1: Distance to Cluster Center (Default if `use_silhouette_for_confidence: false`)

```python
confidence = (cosine_similarity + 1) / 2  # Map [-1, 1] to [0, 1]
```

**Pros**: Fast, intuitive
**Cons**: Doesn't consider cluster separation

### Method 2: Silhouette Coefficient (Default if `use_silhouette_for_confidence: true`)

```python
a = avg_distance_to_same_cluster
b = min_avg_distance_to_other_clusters
silhouette = (a - b) / max(a, b)
confidence = (silhouette + 1) / 2  # Map [-1, 1] to [0, 1]
```

**Pros**: Considers both cohesion and separation
**Cons**: Slower to compute

## Evaluation Metrics

### Clustering Metrics (Unsupervised)
- **Silhouette Score**: [-1, 1], higher is better
- **Calinski-Harabasz Index**: Higher is better
- **Davies-Bouldin Index**: Lower is better
- **Cluster Purity**: [0, 1], higher is better

### Classification Metrics (Supervised on Test Set)
- **Accuracy**: Overall correctness
- **Macro F1**: Unweighted average F1 across classes
- **Weighted F1**: Weighted by class support
- **Per-class Precision/Recall/F1**

## Example Results

After running with `train_split: 0.3` (30% labeled):

```
=== SEMI-SUPERVISED SELF-TRAINING ===
Initial setup:
  Total samples: 15000
  Labeled samples: 4500
  Unlabeled samples: 10500

--- Iteration 1/5 ---
High-confidence pseudo-labels: 2345
Average confidence: 0.8723
Cluster purity: 0.8156

--- Iteration 2/5 ---
High-confidence pseudo-labels: 1823
Average confidence: 0.8591
...

Test Set Performance:
  Accuracy: 0.8234
  Macro F1: 0.7956
  Weighted F1: 0.8187
```

## Tips for Best Results

1. **Start with higher `confidence_threshold`** (0.85-0.95) to ensure quality pseudo-labels
2. **Use `kmeans` for balanced datasets**, `dbscan` for imbalanced/noisy data
3. **Experiment with `train_split`**: 10-30% labeled is typical for semi-supervised learning
4. **Monitor clustering metrics**: High silhouette + high purity = good clustering
5. **Check training progress plots**: Confidence should remain stable/increase over iterations

## Troubleshooting

**Problem**: Few/no pseudo-labels added
- **Solution**: Lower `confidence_threshold` or increase `n_clusters`

**Problem**: Poor test performance
- **Solution**: Increase `train_split` (use more labeled data) or try different `clustering_method`

**Problem**: Clustering takes too long
- **Solution**: Use `kmeans` instead of `dbscan`, or reduce embedding dimension

**Problem**: All samples get same pseudo-label
- **Solution**: Check cluster-label mapping, ensure labeled data represents all classes

## Future Enhancements

- [ ] Active learning: Select most informative unlabeled samples to label
- [ ] Co-training with multiple embeddings models
- [ ] Ensemble of different clustering algorithms
- [ ] Adaptive confidence thresholds per iteration
- [ ] GPU acceleration for large-scale clustering
