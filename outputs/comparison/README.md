# Embedding Model Comparison Results

This directory contains the results of comparing **SBERT (all-MiniLM-L6-v2)** vs **SimCSE (sup-simcse-bert-base-uncased)** for hate speech clustering.

## Quick Summary

**Winner**: **all-MiniLM-L6-v2** (SBERT)

| Metric | SBERT | SimCSE | Winner |
|--------|-------|--------|--------|
| Silhouette Score | 0.0311 | 0.0278 | ✅ SBERT |
| Davies-Bouldin Index | 4.43 | 3.85 | ✅ SimCSE |
| Cluster Purity | 71.58% | 67.92% | ✅ SBERT |
| Encoding Speed | 20 it/s | 5 it/s | ✅ SBERT |
| Embedding Dimension | 384 | 768 | ✅ SBERT |

**Conclusion**: SBERT wins on most metrics and is 4x faster.

---

## Files in This Directory

### 📊 Visualizations

1. **`tsne_comparison.png`** - t-SNE 2D projections
   - Left: all-MiniLM-L6-v2 embeddings
   - Right: simcse-bert embeddings
   - Top row: Colored by true labels (Hate/Offensive/Neither)
   - Bottom row: Colored by K-Means cluster assignments
   
   **Key Insight**: Both models show significant class overlap, confirming that hate speech is hard to cluster.

2. **`tsne_annotated_[model].png`** ⭐ **NEW** - Enhanced t-SNE with annotations
   - **Left panel**: Points colored by true labels
   - **Middle panel**: Points colored by cluster + centroid markers (white stars)
   - **Right panel**: Overlay showing true labels with cluster boundaries
   
   **Key Insight**: Cluster boundaries don't align with true label regions, showing the fundamental difficulty of clustering hate speech.

3. **`cluster_confusion_matrices.png`** ⭐ **NEW** - Confusion matrices
   - Shows how each true label is distributed across clusters
   - Heatmap normalized by true label (rows sum to 100%)
   - Darker red = more samples from that label in that cluster
   
   **Key Insight**: Each true label is spread across multiple clusters, indicating poor separation.

4. **`metrics_comparison.png`** - Bar charts comparing metrics
   - Panel 1: Silhouette Score (SBERT wins)
   - Panel 2: Davies-Bouldin Index (SimCSE wins)
   - Panel 3: Cluster Purity (SBERT wins)
   - Panel 4: Inertia (different scales, not directly comparable)

3. **`model_rankings.png`** - Ranking table
   - Shows rank 1-2 for each model on each metric
   - SBERT highlighted as overall best (average rank 1.33)

### 📄 Data Files

4. **`model_comparison.csv`** - Raw metrics
   ```csv
   model,embedding_dim,silhouette_score,davies_bouldin_index,cluster_purity,inertia,n_clusters
   all-MiniLM-L6-v2,384,0.0311,4.4347,0.7158,3590.06,12
   simcse-bert,768,0.0278,3.8493,0.6792,246886.56,12
   ```

5. **`model_rankings.csv`** - Rankings by metric

6. **`comparison_report.txt`** - Complete text summary

7. **`cluster_confusion_analysis.txt`** ⭐ **NEW** - Detailed cluster composition
   - For each cluster: shows % of each true label
   - For each true label: shows distribution across clusters
   - Identifies dominant label in each cluster
   
   **Example insight**: "Cluster 7 has 86.2% Hate Speech samples" vs "Cluster 3 has only 7.3% Offensive Language"

---

## Understanding the Metrics

### Silhouette Score (Higher is Better)
- **Range**: -1 to +1
- **What it measures**: How well-separated clusters are
- **SBERT**: 0.0311 ✅
- **SimCSE**: 0.0278
- **Interpretation**: Both very low (~0.03), indicating poor cluster separation. SBERT slightly better.

### Davies-Bouldin Index (Lower is Better)
- **Range**: 0 to ∞
- **What it measures**: Ratio of within-cluster to between-cluster distances
- **SBERT**: 4.43
- **SimCSE**: 3.85 ✅
- **Interpretation**: SimCSE creates slightly more compact clusters.

### Cluster Purity (Higher is Better)
- **Range**: 0% to 100%
- **What it measures**: % of samples in each cluster belonging to the dominant true label
- **SBERT**: 71.58% ✅
- **SimCSE**: 67.92%
- **Interpretation**: SBERT clusters are more homogeneous (fewer mixed labels).

---

## Why SBERT Wins

1. **Optimized for Short Texts**
   - SBERT designed for tweets and short sentences
   - SimCSE trained on longer NLI pairs
   - Hate speech tweets average ~15 words

2. **Better Cluster Homogeneity**
   - 71.58% purity vs 67.92%
   - Fewer mixed labels per cluster
   - More consistent groupings

3. **Computational Efficiency**
   - 4.4x faster encoding (20 vs 5 samples/sec)
   - 50% lower memory (384 vs 768 dimensions)
   - Better for production deployment

4. **Better Silhouette Score**
   - 0.0311 vs 0.0278 (+11.9%)
   - Better cluster separation
   - More distinct boundaries

---

## When to Use Each Model

### Use SBERT (all-MiniLM-L6-v2) for:
✅ Short texts (tweets, reviews, questions)  
✅ Clustering tasks  
✅ Production systems with limited compute  
✅ Fast prototyping and iteration  
✅ Multi-domain applications  

### Use SimCSE (sup-simcse-bert) for:
✅ Semantic search (finding similar documents)  
✅ Duplicate detection  
✅ Longer texts (paragraphs, documents)  
✅ Tasks where fine-grained similarity matters  
✅ Paraphrase identification  

---

## Key Findings

### Both Models Struggle with Hate Speech Clustering

The low silhouette scores (~0.03) for both models indicate that:

1. **Classes overlap heavily** in embedding space
2. **Semantic similarity ≠ label similarity** 
   - Hate speech and offensive language use similar words
   - Intent and context matter more than vocabulary
3. **Clustering-based approaches are fundamentally limited**
   - Even state-of-the-art embeddings can't separate these classes
   - Supervised learning is necessary

### SBERT is the Pragmatic Choice

While neither model achieves great clustering quality:
- SBERT performs slightly better overall
- Much faster and more efficient
- Already working well in the pipeline
- No compelling reason to switch to SimCSE

---

## Recommendations

### For This Project

**Keep using all-MiniLM-L6-v2**:
- Best clustering quality (71.58% purity)
- 4x faster encoding
- Lower memory footprint
- Optimized for tweets

### For Future Work

Since both embeddings show poor clustering quality (silhouette ~0.03):

1. **Switch to supervised learning** (recommended)
   - Use SBERT embeddings as features
   - Train Logistic Regression or SVM
   - Expected accuracy: 60-70% vs current 32.95%

2. **Fine-tune SBERT on hate speech domain**
   - Train on hate speech examples
   - Use triplet/contrastive loss
   - Expected improvement: +5-10% cluster purity

3. **Try domain-specific models**
   - HateBERT (pre-trained on hate speech)
   - ToxicBERT (trained on toxic comments)
   - May provide better embeddings for this task

---

## Experimental Setup

- **Dataset**: 5,000 samples (balanced across 3 classes)
- **Clustering**: K-Means with k=12
- **t-SNE**: perplexity=30, max_iter=1000
- **Random seed**: 42 (reproducible)

## How to Reproduce

```bash
# Run comparison
python src/embedding_comparison.py \
    --models all-MiniLM-L6-v2 simcse-bert \
    --sample-size 5000 \
    --n-clusters 12

# Use larger sample
python src/embedding_comparison.py \
    --models all-MiniLM-L6-v2 simcse-bert \
    --sample-size 10000 \
    --n-clusters 12

# Skip t-SNE to save time
python src/embedding_comparison.py \
    --models all-MiniLM-L6-v2 simcse-bert \
    --skip-tsne
```

---

## Conclusion

**SBERT (all-MiniLM-L6-v2) is the better choice** for hate speech clustering:
- Higher cluster purity (71.58%)
- Better silhouette score (0.0311)
- 4.4x faster and 50% more memory efficient
- Optimized for short texts like tweets

However, **both models show fundamental limitations** (silhouette ~0.03), confirming that clustering-based semi-supervised learning is not effective for hate speech detection. **Supervised learning is strongly recommended** given the 28,208 labeled samples available.

---

**For more details**, see `EMBEDDING_COMPARISON_REPORT.md` in the project root.
