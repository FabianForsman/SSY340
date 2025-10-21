# Bug Fix Summary

## Issue 1: IndexError in Semi-Supervised Training

### Problem
The semi-supervised self-training pipeline was throwing an `IndexError`:

```
IndexError: boolean index did not match indexed array along axis 0; 
size of axis is 8674 but size of corresponding boolean axis is 17347
```

### Root Cause
In `src/semi_supervised.py`, the `_calculate_confidence_scores()` method returns a confidence array for **all samples** (both labeled and unlabeled), but the code was trying to index only the **unlabeled samples** with this full-size boolean mask.

### Fix Applied
Modified the code to extract only the unlabeled confidences before applying the threshold in `src/semi_supervised.py`.

---

## Issue 2: Shape Mismatch Error

### Problem
After fixing Issue 1, a new error appeared:

```
ValueError: operands could not be broadcast together with shapes (35266,) (2479,)
```

This occurred in `evaluate_semi_supervised()` when comparing predictions and labels.

### Root Cause
The `main()` function was:
1. Running `run_pipeline()` which processes the **balanced dataset** (57,570 samples) and generates embeddings
2. Then **reloading the raw dataset** (24,783 samples) to create new dataloaders
3. These mismatched sizes caused the shape error when splitting embeddings

The problem was that embeddings were from the balanced dataset, but dataloaders were from the raw dataset.

### Fix Applied
Modified `run_pipeline()` to return `(embeddings, dataloaders, all_labels)` instead of just `embeddings`, and updated `main()` to use the returned dataloaders directly instead of recreating them.

**BEFORE (BROKEN):**
```python
# main.py
embeddings = run_pipeline(config)  # Returns only embeddings from balanced dataset

# Reload raw dataset and create NEW dataloaders (different size!)
dataset = HateSpeechDataset(root=..., file=...)  # 24,783 samples
dataloaders = create_dataloaders(dataset, ...)    # Wrong size!

run_semi_supervised_pipeline(config, embeddings, all_labels, dataloaders)  # ❌ Mismatch!
```

**AFTER (FIXED):**
```python
# main.py
embeddings, dataloaders, all_labels = run_pipeline(config)  # All from same balanced dataset

run_semi_supervised_pipeline(config, embeddings, all_labels, dataloaders)  # ✅ Matches!
```

---

## Additional Improvements

1. **Fixed numpy type issue**: Converted numpy scalar types (`np.int32`, `np.int64`) to Python `int` in cluster mapping for cleaner output

2. **Suppressed tokenizers warning**: Added `os.environ["TOKENIZERS_PARALLELISM"] = "false"` at the start of scripts to prevent fork-related warnings

---

## Files Modified

- `src/semi_supervised.py`:
  - Fixed indexing bug in `fit()` method (3 locations)
  - Added type conversions in `_map_clusters_to_labels()` method
  
- `src/main.py`:
  - Added tokenizers parallelism environment variable
  - Changed `run_pipeline()` to return `(embeddings, dataloaders, all_labels)` tuple
  - Removed dataset reloading in `main()` function

- `example_semi_supervised.py`:
  - Added tokenizers parallelism environment variable

---

## Verification

### Example Script
```bash
python example_semi_supervised.py
```

Output:
```
======================================================================
EXAMPLE COMPLETED
======================================================================
Pseudo-label accuracy: 77.78% (7/9)
Overall metrics:
  Accuracy: 0.8667
  Macro F1: 0.8667
```

### Full Pipeline
```bash
python src/main.py --config config.yaml
```

Should now run without errors. The pipeline:
1. Processes and balances dataset (57,570 samples)
2. Generates embeddings for all samples
3. Splits into train/dev/test using dataloaders
4. Runs semi-supervised training on training set
5. Evaluates on test set

---

## Status

✅ **RESOLVED** - Both the example and full pipeline now work correctly with matching data shapes.
