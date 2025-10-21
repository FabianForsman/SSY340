# Semi-Supervised Self-Training Results - Report Package

This directory contains publication-quality visualizations and summaries of the semi-supervised self-training experiment for hate speech detection.

## Generated Files

### 📊 Visualizations (PNG Format)

#### 1. `multipanel_results.png` - **Main Results Figure**
A comprehensive 2×2 panel figure showing:
- **Panel A**: Self-training progression (labeled samples over iterations)
- **Panel B**: Pseudo-label confidence scores vs. threshold
- **Panel C**: Clustering quality metrics (purity & silhouette score)
- **Panel D**: Test set prediction distribution across classes

**Use in report**: This is your primary results figure. Shows the complete story of the self-training process in one view.

#### 2. `cluster_analysis.png` - **Cluster Mapping Analysis**
Two-panel figure analyzing cluster-to-label assignments:
- **Left**: Number of clusters assigned to each label (shows class imbalance in mapping)
- **Right**: Pseudo-labels added per iteration (shows labeling efficiency)

**Use in report**: Explains why the model is biased toward "Neither" class - 8 out of 12 clusters map to it.

#### 3. `confusion_matrix_enhanced.png` - **Detailed Confusion Matrix**
Enhanced confusion matrix with:
- Row-normalized percentages (by true label)
- Absolute counts in each cell
- Color gradient (green = correct, red = incorrect)

**Use in report**: Shows per-class performance. Reveals that "Offensive Language" has only 3% recall.

#### 4. `performance_table.png` - **Metrics Summary Table**
Professional table displaying:
- Test accuracy, F1-scores
- Clustering quality metrics
- Training statistics (iterations, samples, pseudo-labels)

**Use in report**: Use as a supplementary table or convert to LaTeX using the provided .tex file.

#### 5. `pseudo_label_analysis.png` - **Pseudo-Labeling Analysis**
Two-panel analysis of the pseudo-labeling process:
- **Left**: Confidence score distribution across all predictions
- **Right**: Number of pseudo-labels added per iteration (with success/failure colors)

**Use in report**: Shows that most predictions have low confidence, explaining early stopping.

### 📄 Data Files

#### 6. `performance_summary.csv`
Machine-readable version of the performance table. Contains:
```
Metric,Value
Test Accuracy,0.3295
Macro F1-Score,0.2558
...
```

**Use**: Import into Excel/Google Sheets for custom visualizations or reporting.

#### 7. `latex_tables.tex`
LaTeX-formatted tables for academic papers. Contains:
- Performance metrics table
- Training progression table

**Use**: Copy-paste directly into your LaTeX document.

#### 8. `executive_summary.txt`
Comprehensive text summary including:
- Experimental setup
- Training results
- Performance metrics
- Key observations
- Recommendations

**Use**: Great for the "Results" section of your report or as a reference.

## How to Use in Your Report

### For Academic Papers

1. **Main Results Section**: Use `multipanel_results.png` as Figure 1
2. **Method Validation**: Use `cluster_analysis.png` to explain the clustering approach
3. **Performance Details**: Include `confusion_matrix_enhanced.png` to show per-class results
4. **Tables**: Copy tables from `latex_tables.tex` into your paper

### For Presentations

1. Start with `multipanel_results.png` - tells the complete story
2. Use `performance_table.png` for a clean metrics overview
3. Show `confusion_matrix_enhanced.png` to discuss challenges
4. Reference `executive_summary.txt` for talking points

### For Technical Reports

Include all visualizations in sequence:
1. Setup & methodology context
2. `cluster_analysis.png` - explains cluster-label mapping
3. `multipanel_results.png` - shows training progression
4. `confusion_matrix_enhanced.png` - detailed performance
5. `pseudo_label_analysis.png` - explains pseudo-labeling behavior
6. Conclusions from `executive_summary.txt`

## Key Findings to Highlight

Based on these visualizations, your report should emphasize:

1. **Challenge**: Hate speech classes overlap heavily in embedding space
   - Evidence: Low cluster purity (34.31%) and silhouette score (0.036)

2. **Class Imbalance**: Cluster mapping strongly favors "Neither" class
   - Evidence: 8/12 clusters → "Neither", only 2 clusters each for Hate/Offensive

3. **Early Stopping**: Limited benefit from pseudo-labeling
   - Evidence: Only 1 effective iteration before confidence threshold blocks further progress

4. **Performance**: Below expectations for a 70/30 labeled/unlabeled split
   - Evidence: 32.95% accuracy (barely better than random 33.3%)

5. **Class-Specific Issues**: "Offensive Language" particularly problematic
   - Evidence: Only 3% recall for offensive class in confusion matrix

## Recommendations for Your Report

Based on these results, conclude with:

1. **Semi-supervised clustering is not optimal** for this hate speech task due to semantic overlap
2. **Supervised learning recommended** given you have 28,208+ labeled samples
3. **Future work**: Fine-tune SBERT on hate speech domain or explore transformer-based classifiers

## Regenerating Visualizations

To regenerate these visualizations with updated data:

```bash
python src/report_visualization.py
```

Options:
```bash
python src/report_visualization.py --results-dir outputs/results --output-dir outputs/report
```

---

**Questions?** Check `SEMI_SUPERVISED_README.md` for implementation details or `IMPLEMENTATION_SUMMARY.md` for a quick reference.
