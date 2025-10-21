# Semi-Supervised Improvement Strategies

## Current Results (Baseline)
- 10% labels, confidence=0.8, k=5
- **Accuracy: 83.68%**

## Quick Improvement Tests

Pull latest code:
```bash
cd ~/SSY340
git pull origin SemiSupervisedLearning
```

### **Strategy 1: Lower Confidence Threshold** (⏱️ ~15 min)
Add more pseudo-labels by accepting lower confidence:

```bash
# Confidence 0.7 (more pseudo-labels)
python run_limited_label_experiments.py \
    --experiments exp2 \
    --label-fraction 0.1 \
    --confidence-threshold 0.7 \
    --k-neighbors 5 \
    --output-dir outputs/exp_conf_0.7

# Confidence 0.6 (even more)
python run_limited_label_experiments.py \
    --experiments exp2 \
    --label-fraction 0.1 \
    --confidence-threshold 0.6 \
    --k-neighbors 5 \
    --output-dir outputs/exp_conf_0.6
```

**Expected:** More pseudo-labels but possibly lower quality → accuracy might improve to ~84-85%

---

### **Strategy 2: Higher k for k-NN** (⏱️ ~15 min)
Use more neighbors for more robust predictions:

```bash
# k=10
python run_limited_label_experiments.py \
    --experiments exp2 \
    --label-fraction 0.1 \
    --confidence-threshold 0.8 \
    --k-neighbors 10 \
    --output-dir outputs/exp_k_10

# k=15
python run_limited_label_experiments.py \
    --experiments exp2 \
    --label-fraction 0.1 \
    --confidence-threshold 0.8 \
    --k-neighbors 15 \
    --output-dir outputs/exp_k_15
```

**Expected:** More stable predictions → accuracy might improve to ~84-85%

---

### **Strategy 3: More Initial Labels** (⏱️ ~20 min)
Use 20% or 30% instead of 10%:

```bash
# 20% labels
python run_limited_label_experiments.py \
    --experiments exp1 exp2 \
    --label-fraction 0.2 \
    --confidence-threshold 0.8 \
    --k-neighbors 5 \
    --output-dir outputs/exp_20pct

# 30% labels
python run_limited_label_experiments.py \
    --experiments exp1 exp2 \
    --label-fraction 0.3 \
    --confidence-threshold 0.8 \
    --k-neighbors 5 \
    --output-dir outputs/exp_30pct
```

**Expected:** 
- 20% labels: baseline ~88%, semi-supervised ~90-91%
- 30% labels: baseline ~91%, semi-supervised ~92-93%

---

### **Strategy 4: Combined Best Settings** (⏱️ ~20 min)
Combine multiple improvements:

```bash
# Best combination
python run_limited_label_experiments.py \
    --experiments exp1 exp2 \
    --label-fraction 0.2 \
    --confidence-threshold 0.7 \
    --k-neighbors 10 \
    --output-dir outputs/exp_combined_best
```

**Expected:** ~90-92% accuracy (close to fully supervised!)

---

## Quick Test (5 minutes)

Just test one improvement quickly:

```bash
# Lower confidence to 0.7
python run_limited_label_experiments.py \
    --experiments exp2 \
    --label-fraction 0.1 \
    --confidence-threshold 0.7 \
    --k-neighbors 5 \
    --output-dir outputs/exp_quick_test
```

---

## Expected Improvements Summary

| Strategy | Params | Expected Accuracy | Time |
|----------|--------|-------------------|------|
| **Current** | 10%, conf=0.8, k=5 | 83.68% | - |
| Lower confidence | 10%, conf=0.7, k=5 | ~84-85% | 15 min |
| Higher k | 10%, conf=0.8, k=10 | ~84-85% | 15 min |
| More labels | 20%, conf=0.8, k=5 | ~90% | 20 min |
| Combined | 20%, conf=0.7, k=10 | ~90-92% | 20 min |

---

## Recommendation

**For quick improvement:** Test Strategy 1 (conf=0.7) - fastest, good ROI

**For best results:** Test Strategy 4 (combined) - might reach 90%+

**For report diversity:** Test all 4 strategies to show different approaches
