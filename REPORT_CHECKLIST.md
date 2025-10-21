# Report Generation Checklist

## 🎯 Quick Start

**Option 1: Run Everything at Once (Recommended)**
```bash
./run_full_analysis.sh
```

**Option 2: Step-by-Step (More Control)**
Follow the steps below ⬇️

---

## ✅ Step-by-Step Checklist

### Step 1: Fine-Tune the Model (30-60 min)

```bash
python src/fine_tune_model.py --balance-classes --epochs 6 --compare
```

- [ ] Fine-tuning started
- [ ] Training completes without errors
- [ ] Model saved to `models/fine_tuned/`
- [ ] Comparison results shown
- [ ] Expected improvement: +20-70% accuracy

---

### Step 2: Compare All Models (15 min)

```bash
python src/embedding_comparison.py \
  --models all-MiniLM-L6-v2 models/fine_tuned simcse-bert \
  --sample-size 3000 \
  --n-clusters 12
```

- [ ] Comparison completes
- [ ] Check `outputs/comparison/tsne_annotated_*.png` (3 files)
- [ ] Check `outputs/comparison/metrics_comparison.png`
- [ ] Check `outputs/comparison/model_comparison.csv`
- [ ] Review silhouette, purity, NMI scores

**Expected:**
- Silhouette: Fine-tuned > Base > SimCSE
- Purity: Fine-tuned (75-85%) > Base (72%) > SimCSE (68%)

---

### Step 3: Run Semi-Supervised Learning (20 min)

**First, update config.yaml:**
```yaml
embedding:
  model: "models/fine_tuned"  # Use fine-tuned model
```

**Then run:**
```bash
python src/main.py --config config.yaml
```

- [ ] Semi-supervised learning completes
- [ ] Check `outputs/results/test_metrics.csv`
- [ ] Check `outputs/results/self_training_history.csv`
- [ ] Review final test accuracy

**Expected:**
- Test accuracy: 40-55% (up from 32.95%)
- F1 score improves

---

### Step 4: Generate Report Visualizations (5 min)

```bash
python src/report_visualization.py
```

- [ ] Visualization script completes
- [ ] Check `outputs/report/multipanel_results.png`
- [ ] Check `outputs/report/cluster_analysis.png`
- [ ] Check `outputs/report/confusion_matrix_enhanced.png`
- [ ] Check `outputs/report/latex_tables.tex`
- [ ] Check `outputs/report/executive_summary.txt`

---

## 📊 Verify Results

### Model Comparison Results
- [ ] `outputs/comparison/model_comparison.csv` exists
- [ ] Contains metrics for all 3 models
- [ ] Shows improvement from fine-tuning

### Semi-Supervised Results
- [ ] `outputs/results/test_metrics.csv` exists
- [ ] Test accuracy > 40%
- [ ] F1 scores calculated

### Visualizations
- [ ] At least 6 figures in `outputs/report/`
- [ ] t-SNE plots show clear clusters
- [ ] Confusion matrix readable
- [ ] LaTeX tables formatted correctly

---

## 📈 Key Metrics to Report

### Table 1: Model Comparison

| Metric | Base | Fine-tuned | SimCSE |
|--------|------|------------|--------|
| Silhouette | _____ | _____ | _____ |
| Purity | _____ | _____ | _____ |
| NMI | _____ | _____ | _____ |

**Fill from:** `outputs/comparison/model_comparison.csv`

### Table 2: Semi-Supervised Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | _____% |
| F1 (Macro) | _____ |
| F1 (Weighted) | _____ |

**Fill from:** `outputs/results/test_metrics.csv`

### Table 3: Improvement Summary

| Aspect | Before | After | Improvement |
|--------|--------|-------|-------------|
| Clustering Purity | 71.58% | _____% | +_____% |
| Test Accuracy | 32.95% | _____% | +_____% |

---

## 🎨 Figures for Report

### Must-Include Figures:

1. **Figure 1: t-SNE Visualization (Fine-tuned)**
   - [ ] `outputs/comparison/tsne_annotated_fine_tuned.png`
   - Shows class separation in embedding space

2. **Figure 2: Model Comparison**
   - [ ] `outputs/comparison/metrics_comparison.png`
   - Shows all clustering metrics side-by-side

3. **Figure 3: Confusion Matrix**
   - [ ] `outputs/report/confusion_matrix_enhanced.png`
   - Shows classification performance

4. **Figure 4: Self-Training Progress**
   - [ ] `outputs/report/self_training_progress.png`
   - Shows improvement over iterations

5. **Figure 5: Cluster Analysis**
   - [ ] `outputs/report/cluster_analysis.png`
   - Shows silhouette, purity, NMI

6. **Figure 6: Complete Overview**
   - [ ] `outputs/report/multipanel_results.png`
   - 4-panel summary figure

---

## ⏱️ Time Estimates

- [ ] **Fine-tuning:** 30-60 min (GPU) or 1-2 hours (CPU)
- [ ] **Model comparison:** 15 min
- [ ] **Semi-supervised:** 20 min
- [ ] **Visualizations:** 5 min
- [ ] **Total:** ~1-2 hours (GPU) or 2-3 hours (CPU)

---

## 🐛 Troubleshooting

### If fine-tuning fails:
```bash
# Try with smaller batch size
python src/fine_tune_model.py --batch-size 8 --epochs 4
```

### If out of memory:
```bash
# Use CPU only
CUDA_VISIBLE_DEVICES="" python src/fine_tune_model.py --batch-size 8
```

### If comparison fails:
```bash
# Use smaller sample
python src/embedding_comparison.py --sample-size 2000 --models all-MiniLM-L6-v2
```

### If visualization fails:
```bash
# Check if results exist
ls outputs/results/
ls outputs/comparison/
```

---

## 📝 Report Writing Order

1. **Methods Section**
   - [ ] Describe fine-tuning approach
   - [ ] Explain semi-supervised method
   - [ ] Detail evaluation metrics

2. **Results Section**
   - [ ] Present model comparison table
   - [ ] Show t-SNE visualizations
   - [ ] Report semi-supervised accuracy
   - [ ] Include confusion matrix

3. **Discussion Section**
   - [ ] Analyze why fine-tuning helps
   - [ ] Discuss class imbalance impact
   - [ ] Compare with baseline

4. **Conclusion**
   - [ ] Summarize improvements
   - [ ] Discuss limitations
   - [ ] Suggest future work

---

## ✨ Final Checks

- [ ] All scripts ran successfully
- [ ] All expected files generated
- [ ] Metrics make sense (accuracy 30-60%, not 0% or 100%)
- [ ] Figures are publication quality
- [ ] LaTeX tables are properly formatted
- [ ] Results are reproducible

---

## 🚀 You're Ready!

Once all boxes are checked, you have everything needed for your report!

**Summary of outputs:**
- ✅ Fine-tuned model
- ✅ Comparison results (3 models)
- ✅ Semi-supervised results
- ✅ 6+ publication-ready figures
- ✅ LaTeX tables
- ✅ Summary statistics

Good luck with your report! 🎉
