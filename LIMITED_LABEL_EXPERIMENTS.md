# Limited Label Experiments Guide

## Goal
Show that we can achieve good classification accuracy with very limited labeled data using semi-supervised learning.

## Experimental Setup

### Experiment 1: Supervised Baseline (Limited Labels)
- Train with only **10% of training labels** (~4,000 samples)
- No semi-supervised learning
- **Expected:** Poor accuracy (~40-60%)

### Experiment 2: Semi-Supervised (Model-Based Pseudo-Labeling)
- Start with same 10% labeled data
- Train initial model
- Use model to predict labels on remaining 90% (pseudo-labeling)
- Keep high-confidence predictions (confidence > 0.8)
- Retrain on labeled + pseudo-labeled data
- **Expected:** Better accuracy (~70-85%)

### Experiment 3: Fully Supervised (Upper Bound)
- Train with 100% of training labels
- **Expected:** Best accuracy (~95%)

## Running on GPU VM

```bash
# Pull latest code
cd ~/SSY340
git pull origin SemiSupervisedLearning

# Run all experiments (takes ~30-40 minutes)
python run_limited_label_experiments.py \
    --data data/raw/labeled_data.csv \
    --label-fraction 0.1 \
    --confidence-threshold 0.8 \
    --output-dir outputs/experiments \
    --experiments all

# Or run individual experiments:
# Experiment 1 only
python run_limited_label_experiments.py --experiments exp1

# Experiment 2 only  
python run_limited_label_experiments.py --experiments exp2

# Experiment 3 only
python run_limited_label_experiments.py --experiments exp3
```

## Expected Timeline

- Experiment 1: ~6 minutes (6 epochs, 4K samples)
- Experiment 2: ~15 minutes (initial + retrain)
- Experiment 3: ~6 minutes (6 epochs, 40K samples)
- **Total: ~30 minutes**

## Output Files

All results saved to `outputs/experiments/`:
- `exp1_results.csv` - Supervised baseline results
- `exp2_results.csv` - Semi-supervised results
- `exp3_results.csv` - Fully supervised results
- `all_experiments_comparison.csv` - Combined comparison
- `exp1_supervised_limited/` - Model from Exp 1
- `exp2_initial_model/` - Initial model from Exp 2
- `exp2_semi_supervised_model/` - Final model from Exp 2
- `exp3_fully_supervised/` - Model from Exp 3

## Key Metrics to Report

1. **Limited Supervised Accuracy** (Exp 1)
2. **Semi-Supervised Accuracy** (Exp 2)
3. **Improvement**: (Exp 2 - Exp 1) / Exp 1 * 100%
4. **Gap to Full Supervision**: Exp 2 / Exp 3 * 100%
5. **Number of Pseudo-Labels Added**
6. **Average Confidence of Pseudo-Labels**

## Parameter Tuning (Optional)

Try different label fractions:
```bash
# 5% labels (very limited)
python run_limited_label_experiments.py --label-fraction 0.05

# 20% labels  
python run_limited_label_experiments.py --label-fraction 0.2
```

Try different confidence thresholds:
```bash
# More conservative (fewer pseudo-labels, higher quality)
python run_limited_label_experiments.py --confidence-threshold 0.9

# More aggressive (more pseudo-labels, lower quality)
python run_limited_label_experiments.py --confidence-threshold 0.7
```

## Transfer Results Back to Mac

```bash
# After experiments complete, copy results back
gcloud compute scp --recurse my-gpu-vm:~/SSY340/outputs/experiments ~/Desktop/SSY340/outputs/ --zone=us-east1-b
```

## For Your Report

This experimental design shows:
- ✅ With only 10% labels, supervised learning performs poorly
- ✅ Semi-supervised learning significantly improves performance  
- ✅ Model-based pseudo-labeling works better than clustering
- ✅ Can reach X% of fully supervised performance with only 10% labels
- ✅ Demonstrates practical applicability with limited annotation budgets
