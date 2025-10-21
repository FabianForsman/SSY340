# DBSCAN Experiment Guide

## What is DBSCAN?

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that:
- **Automatically finds the number of clusters** (unlike k-NN which requires k)
- **Identifies outliers/noise** (points that don't belong to any cluster)
- **Works well with varying density** clusters

## Key Parameters

### **1. eps (epsilon)**
- Maximum distance between two samples to be neighbors
- **Smaller eps** → More, tighter clusters + more noise
- **Larger eps** → Fewer, looser clusters + less noise
- **Recommended starting values**: 0.3, 0.5, 0.7

### **2. min_samples**
- Minimum samples in a neighborhood to form a cluster
- **Smaller min_samples** → More clusters
- **Larger min_samples** → Fewer, denser clusters
- **Recommended starting values**: 3, 5, 10

## Running DBSCAN Experiments

### Pull latest code:
```bash
cd ~/SSY340
git pull origin SemiSupervisedLearning
```

### **Test 1: Default Parameters**
```bash
python run_dbscan_experiment.py \
    --data data/raw/labeled_data.csv \
    --label-fraction 0.1 \
    --eps 0.5 \
    --min-samples 5 \
    --confidence-threshold 0.6 \
    --output-dir outputs/dbscan_default
```

### **Test 2: Tighter Clusters (lower eps)**
```bash
python run_dbscan_experiment.py \
    --eps 0.3 \
    --min-samples 5 \
    --confidence-threshold 0.6 \
    --output-dir outputs/dbscan_tight
```

### **Test 3: Looser Clusters (higher eps)**
```bash
python run_dbscan_experiment.py \
    --eps 0.7 \
    --min-samples 5 \
    --confidence-threshold 0.6 \
    --output-dir outputs/dbscan_loose
```

### **Test 4: More Clusters (lower min_samples)**
```bash
python run_dbscan_experiment.py \
    --eps 0.5 \
    --min-samples 3 \
    --confidence-threshold 0.6 \
    --output-dir outputs/dbscan_min3
```

### **Test 5: Fewer Clusters (higher min_samples)**
```bash
python run_dbscan_experiment.py \
    --eps 0.5 \
    --min-samples 10 \
    --confidence-threshold 0.6 \
    --output-dir outputs/dbscan_min10
```

### **Test 6: Combined Best (if patterns emerge)**
```bash
python run_dbscan_experiment.py \
    --eps 0.4 \
    --min-samples 5 \
    --confidence-threshold 0.6 \
    --output-dir outputs/dbscan_optimal
```

## What to Look For

### **Good DBSCAN Settings:**
- ✅ **3-10 clusters** (meaningful groupings)
- ✅ **Low noise** (<20% of points)
- ✅ **High silhouette score** (>0.3)
- ✅ **High pseudo-labeling** (>60% of unlabeled data)
- ✅ **Accuracy improvement** over baseline

### **Bad DBSCAN Settings:**
- ❌ Too many clusters (>20) → eps too small
- ❌ Too few clusters (1-2) → eps too large
- ❌ High noise (>50%) → eps too small or min_samples too large
- ❌ Low silhouette (<0.2) → Poor cluster separation

## Expected Results

| Configuration | Clusters | Noise | Pseudo-labels | Accuracy |
|--------------|----------|-------|---------------|----------|
| eps=0.3, min=5 | 15-20 | 30-40% | 10,000-15,000 | ~84% |
| **eps=0.5, min=5** | **5-10** | **10-20%** | **20,000-30,000** | **~85%** |
| eps=0.7, min=5 | 2-5 | 5-10% | 30,000-35,000 | ~85-86% |

## Grid Search (Optional)

Run all combinations to find optimal parameters:

```bash
# Create a script to test all combinations
for eps in 0.3 0.4 0.5 0.6 0.7; do
    for min_s in 3 5 10; do
        echo "Testing eps=$eps, min_samples=$min_s"
        python run_dbscan_experiment.py \
            --eps $eps \
            --min-samples $min_s \
            --output-dir outputs/dbscan_grid/eps${eps}_min${min_s}
    done
done
```

## Comparison with k-NN

After finding optimal DBSCAN parameters, compare:

| Method | Pseudo-labels | Accuracy | Advantage |
|--------|---------------|----------|-----------|
| **k-NN (k=5)** | 37,139 (99.5%) | **85.63%** | Simple, predictable |
| **DBSCAN** | ~25,000 (67%) | ~85%? | Finds natural clusters, removes noise |

## For Your Report

Compare three approaches:
1. **Supervised baseline** (10% labels): 83.78%
2. **Semi-supervised k-NN**: 85.63% (+1.85%)
3. **Semi-supervised DBSCAN**: ?% (+?%)

Show which clustering approach works better for hate speech detection!
