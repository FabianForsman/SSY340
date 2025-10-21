# Semi-Supervised Self-Training for Hate Speech Detection
## Experimental Results Report

**Date**: October 20, 2025  
**Algorithm**: K-Means Clustering with Self-Training  
**Task**: Hate Speech Detection (3-class classification)

---

## Executive Summary

This report presents the results of applying semi-supervised self-training to hate speech detection using SBERT embeddings and K-Means clustering. The experiment aimed to leverage unlabeled data to improve classification performance in a simulated low-resource scenario.

**Key Result**: The semi-supervised approach achieved **32.95% accuracy**, which is only marginally better than random chance (33.3%) and significantly underperforms compared to supervised baselines. Analysis reveals that hate speech classes are not naturally separable in the embedding space, making unsupervised clustering ineffective for this task.

---

## 1. Experimental Setup

### 1.1 Dataset
- **Source**: Twitter hate speech dataset
- **Total samples**: 24,783 tweets
- **Classes**: 
  - Hate Speech (label 0): 1,430 samples
  - Offensive Language (label 1): 19,190 samples  
  - Neither (label 2): 4,163 samples
- **Preprocessing**: Class balancing → 57,570 total samples (19,190 per class)
- **Split**: 70% train / 20% dev / 10% test

### 1.2 Semi-Supervised Configuration
- **Labeled samples**: 28,208 (70% of training set)
- **Unlabeled samples**: 12,090 (30% of training set)
- **Embedding model**: SBERT (all-MiniLM-L6-v2, 384 dimensions)
- **Clustering algorithm**: K-Means with k=12 clusters
- **Confidence threshold**: 0.65 (for pseudo-labeling)
- **Maximum iterations**: 2

### 1.3 Algorithm Overview
1. Generate SBERT embeddings for all samples
2. Use labeled samples to initialize cluster-to-label mapping via majority voting
3. Cluster unlabeled samples using K-Means
4. Assign pseudo-labels to high-confidence unlabeled samples
5. Add pseudo-labeled samples to training set
6. Repeat until convergence or maximum iterations

---

## 2. Results

### 2.1 Overall Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 32.95% |
| **Macro F1-Score** | 0.2558 |
| **Weighted F1-Score** | 0.2575 |
| **Cluster Purity** | 34.31% |
| **Silhouette Score** | 0.0360 |

### 2.2 Training Progression

| Iteration | Labeled Samples | Pseudo-Labels Added | Avg Confidence |
|-----------|----------------|---------------------|----------------|
| 1 | 39,796 | 11,588 | 0.7644 |
| 2 | 39,796 | 0 | N/A |

**Early Stopping**: The algorithm terminated after iteration 2 because no unlabeled samples met the confidence threshold.

### 2.3 Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Hate Speech** | 0.32 | 0.22 | 0.26 | 1,874 |
| **Offensive Language** | 0.32 | 0.03 | 0.05 | 1,916 |
| **Neither** | 0.33 | 0.73 | 0.46 | 1,967 |

**Key Observation**: The model is heavily biased toward predicting "Neither" class (73% recall), while "Offensive Language" is severely underdetected (3% recall).

### 2.4 Cluster-to-Label Mapping

After iteration 1, the 12 clusters were mapped to labels as follows:

| Label | Number of Clusters | Percentage |
|-------|-------------------|------------|
| Neither (2) | 8 | 66.7% |
| Hate Speech (0) | 2 | 16.7% |
| Offensive Language (1) | 2 | 16.7% |

**Analysis**: The strong bias toward "Neither" in the cluster mapping (8 out of 12 clusters) directly causes the prediction bias observed in the test results.

---

## 3. Analysis & Discussion

### 3.1 Why Did Semi-Supervised Learning Fail?

**Root Cause**: Hate speech detection is a **semantically nuanced task** where classes overlap heavily in embedding space. Unlike tasks with natural clusters (e.g., topic classification), hate speech, offensive language, and neutral content often use similar vocabulary and sentence structures.

**Evidence**:
1. **Low Cluster Purity (34.31%)**: Indicates that clusters contain mixed labels
2. **Low Silhouette Score (0.036)**: Suggests poor cluster separation
3. **Skewed Cluster Mapping**: 66.7% of clusters map to a single class

### 3.2 Pseudo-Labeling Quality

Distribution of pseudo-labels added in iteration 1:
- **Neither**: 6,480 samples (56%)
- **Hate Speech**: 3,955 samples (34%)
- **Offensive Language**: 1,153 samples (10%)

The pseudo-labels amplified the existing bias, making the problem worse. The model learned to over-predict "Neither" because that's what the majority of clusters suggested.

### 3.3 Comparison to Baseline

| Approach | Expected Accuracy |
|----------|------------------|
| Random Guessing | 33.3% |
| **Semi-Supervised (This Work)** | **32.95%** |
| Supervised (Logistic Regression on embeddings) | ~60-70% (typical) |
| Fine-tuned Transformer | ~75-85% (state-of-the-art) |

**Conclusion**: The semi-supervised approach performed **worse than random**, indicating negative transfer from pseudo-labels.

---

## 4. Visualizations

The following publication-quality figures are available in `outputs/report/`:

1. **`multipanel_results.png`**: 4-panel overview showing training progression, confidence, clustering quality, and prediction distribution
2. **`cluster_analysis.png`**: Cluster-to-label mapping and pseudo-labeling efficiency
3. **`confusion_matrix_enhanced.png`**: Detailed confusion matrix with counts and percentages
4. **`performance_table.png`**: Professional metrics summary table
5. **`pseudo_label_analysis.png`**: Confidence distribution and labeling efficiency

See `outputs/report/README.md` for detailed descriptions of each visualization.

---

## 5. Lessons Learned

### 5.1 When Semi-Supervised Clustering Works
✅ **Good fit**:
- Natural topic clusters (e.g., news articles)
- Well-separated classes in embedding space
- Large unlabeled data, minimal labeled data (<10%)

### 5.2 When It Doesn't Work
❌ **Poor fit**:
- Semantically nuanced tasks (sentiment, hate speech, sarcasm)
- Classes with overlapping vocabulary
- Already have substantial labeled data (>50%)

### 5.3 Our Case
We have **28,208 labeled samples (70%)** - this is **more than sufficient** for supervised learning! The complexity of semi-supervised learning added no value.

---

## 6. Recommendations

### 6.1 Immediate: Switch to Supervised Learning
Given the amount of labeled data, use:
- **Logistic Regression** on SBERT embeddings (baseline)
- **SVM** with RBF kernel
- **Neural network** with dropout for regularization

**Expected improvement**: 60-70% accuracy (2x better)

### 6.2 Medium-Term: Fine-Tune SBERT
Domain adaptation can help:
```python
from sentence_transformers import SentenceTransformer, losses

model = SentenceTransformer('all-MiniLM-L6-v2')
# Fine-tune on hate speech triplets (anchor, positive, negative)
```

**Expected improvement**: 70-75% accuracy

### 6.3 Long-Term: Transformer Classifiers
Use pre-trained language models:
- **BERT/RoBERTa** fine-tuned for hate speech
- **HateBERT** (domain-specific model)
- **DeBERTa** (state-of-the-art NLU)

**Expected improvement**: 75-85% accuracy

---

## 7. Conclusion

This experiment demonstrated that **semi-supervised self-training with clustering is not effective** for hate speech detection. The key findings are:

1. **Hate speech classes overlap in embedding space**, preventing effective clustering (purity: 34%)
2. **Cluster-to-label mapping is heavily skewed**, causing prediction bias
3. **Pseudo-labeling provides negative transfer**, degrading performance below random chance
4. **We already have sufficient labeled data** (28,208 samples) for supervised learning

**Recommendation**: Abandon the semi-supervised approach and implement a supervised classifier, which will likely achieve 2-3x better accuracy with simpler implementation.

---

## 8. Files & Reproducibility

### 8.1 Code Files
- `src/semi_supervised.py`: Self-training implementation
- `src/clustering.py`: K-Means, DBSCAN, Hierarchical clustering
- `src/evaluation.py`: Metrics and visualization
- `src/report_visualization.py`: Report generation

### 8.2 Documentation
- `SEMI_SUPERVISED_README.md`: Complete implementation guide
- `IMPLEMENTATION_SUMMARY.md`: Quick reference
- `BUGFIX_SUMMARY.md`: Issues encountered and fixed

### 8.3 Configuration
```yaml
semi_supervised:
  enabled: true
  clustering_method: kmeans
  n_clusters: 12
  confidence_threshold: 0.65
  max_iterations: 2
  train_split: 0.7
```

### 8.4 Reproduce Results
```bash
# Run the pipeline
python src/main.py --config config.yaml

# Generate report visualizations
python src/report_visualization.py

# View results
open outputs/report/multipanel_results.png
```

---

## Appendix: Data Flow Diagram

```
Raw Data (24,783 tweets)
         ↓
Preprocessing & Balancing
         ↓
Balanced Dataset (57,570 tweets)
         ↓
Train/Dev/Test Split (70/20/10)
         ↓
SBERT Embeddings (384-dim)
         ↓
Semi-Supervised Setup:
  - 70% labeled (28,208)
  - 30% unlabeled (12,090)
         ↓
K-Means Clustering (k=12)
         ↓
Cluster-to-Label Mapping (majority vote)
         ↓
Pseudo-Labeling (confidence > 0.65)
         ↓
Add to Training Set
         ↓
Repeat → Early Stopping
         ↓
Test Evaluation: 32.95% accuracy ❌
```

---

**For questions or further analysis, see the implementation files or contact the development team.**
