# Fine-Tuning SBERT for Hate Speech Classification

This guide explains how to fine-tune the `all-MiniLM-L6-v2` model on your hate speech classification dataset.

## Overview

The `fine_tune_model.py` script fine-tunes a pre-trained Sentence-BERT model using:
- **Softmax Classification Loss**: Trains the model directly for 3-class classification (hate speech, offensive language, neither)
- **Label-aware embeddings**: The fine-tuned model produces embeddings optimized for your specific task
- **Better performance**: Typically improves accuracy by 10-25% compared to zero-shot embeddings

## Quick Start

### 1. Basic Fine-Tuning

```bash
python src/fine_tune_model.py --config config.yaml
```

This will:
- Load data from `data/raw/labeled_data.csv`
- Fine-tune `all-MiniLM-L6-v2` for 4 epochs
- Save the model to `models/fine_tuned/`
- Evaluate on test set

### 2. Custom Configuration

```bash
python src/fine_tune_model.py \
  --data data/raw/labeled_data.csv \
  --base-model all-MiniLM-L6-v2 \
  --output models/fine_tuned_custom \
  --epochs 6 \
  --batch-size 32 \
  --learning-rate 2e-5 \
  --balance-classes
```

### 3. Compare Base vs Fine-Tuned

```bash
python src/fine_tune_model.py \
  --config config.yaml \
  --compare
```

This will train the model and then compare it against the base model.

### 4. Evaluate Only (No Training)

```bash
python src/fine_tune_model.py \
  --evaluate-only models/fine_tuned \
  --data data/raw/labeled_data.csv \
  --compare
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config.yaml` | Path to configuration file |
| `--data` | `data/raw/labeled_data.csv` | Path to labeled data CSV |
| `--base-model` | `all-MiniLM-L6-v2` | Base model to fine-tune |
| `--output` | `models/fine_tuned` | Output directory for fine-tuned model |
| `--epochs` | `4` | Number of training epochs |
| `--batch-size` | `16` | Batch size for training |
| `--learning-rate` | `2e-5` | Learning rate |
| `--test-size` | `0.2` | Fraction of data for test set |
| `--val-size` | `0.1` | Fraction of training data for validation |
| `--balance-classes` | `False` | Balance class distribution via oversampling |
| `--no-preprocessing` | `False` | Skip text preprocessing |
| `--evaluate-only` | `None` | Path to model for evaluation only (skip training) |
| `--compare` | `False` | Compare base model vs fine-tuned model |

## How It Works

### 1. Data Preparation
- Loads labeled data from CSV
- Applies text preprocessing (same as in `preprocessing.py`)
- Splits into train (70%), validation (9%), test (21%)
- Optionally balances classes via oversampling

### 2. Fine-Tuning Process
- Uses **Softmax Classification Loss** to train for 3-class classification
- Creates a classification head on top of SBERT embeddings
- Trains for 4 epochs (adjustable)
- Validates every 500 steps
- Saves best model based on validation accuracy

### 3. Evaluation
- Tests on held-out test set
- Reports accuracy using k-NN classifier on embeddings
- Optionally compares against base model

## Output Files

After training, you'll find in `models/fine_tuned/`:

```
models/fine_tuned/
├── config.json              # Model configuration
├── config_sentence_transformers.json  # Sentence-transformers config
├── modules.json             # Model architecture
├── pytorch_model.bin        # Fine-tuned weights
├── sentence_bert_config.json
├── tokenizer_config.json
├── vocab.txt
├── special_tokens_map.json
└── training_info.yaml       # Training metadata
```

## Using the Fine-Tuned Model

### In Python Code

```python
from sentence_transformers import SentenceTransformer

# Load fine-tuned model
model = SentenceTransformer('models/fine_tuned')

# Generate embeddings
texts = ["Example tweet 1", "Example tweet 2"]
embeddings = model.encode(texts)

# Use embeddings for clustering, classification, etc.
```

### Update config.yaml

To use the fine-tuned model in your main pipeline:

```yaml
embedding:
  # Use fine-tuned model instead of base model
  model: "models/fine_tuned"  # Path to fine-tuned model
  batch_size: 64
  normalize: false
```

Then run your pipeline as usual:
```bash
python src/main.py --config config.yaml
```

## Expected Results

### Before Fine-Tuning (Base Model)
- Clustering purity: ~65-72%
- Silhouette score: ~0.02-0.03
- Semi-supervised accuracy: ~33%

### After Fine-Tuning
- Clustering purity: **75-85%** (expected improvement)
- Silhouette score: **0.05-0.10** (better separation)
- Semi-supervised accuracy: **40-55%** (expected improvement)

## Training Tips

### 1. Hyperparameter Tuning

**Epochs**: More epochs = better performance, but risk overfitting
- Start with 4 epochs
- Try 6-8 for potentially better results
- Monitor validation accuracy to prevent overfitting

**Batch Size**: Affects memory usage and training speed
- Larger = faster training, but needs more GPU memory
- Smaller = more stable gradients
- Try: 16 (default), 32, or 64

**Learning Rate**: Critical for convergence
- Default `2e-5` works well for most cases
- Too high: unstable training
- Too low: slow convergence
- Try: `1e-5`, `2e-5`, `3e-5`

### 2. Class Balancing

Your dataset is imbalanced:
- Hate speech: ~1,400 (5%)
- Offensive language: ~19,200 (77%)
- Neither: ~4,150 (17%)

Options:
- **No balancing** (default): Model learns natural distribution
- **`--balance-classes`**: Oversample minority classes to equal counts
  - Better for hate speech detection
  - May reduce overall accuracy slightly

### 3. Text Preprocessing

Preprocessing is applied by default using settings from `config.yaml`:
- Removes URLs, mentions, hashtags
- Lowercases text
- Removes stopwords, numbers, quotes

To skip preprocessing:
```bash
python src/fine_tune_model.py --no-preprocessing
```

## Troubleshooting

### Out of Memory Error

Reduce batch size:
```bash
python src/fine_tune_model.py --batch-size 8
```

Or use CPU (slower):
```bash
CUDA_VISIBLE_DEVICES="" python src/fine_tune_model.py
```

### Poor Performance

Try:
1. More epochs: `--epochs 8`
2. Balance classes: `--balance-classes`
3. Adjust learning rate: `--learning-rate 1e-5`
4. Different base model: `--base-model all-mpnet-base-v2`

### Model Not Improving

Check validation accuracy during training:
- Should increase over time
- If decreasing: overfitting, reduce epochs
- If flat: learning rate too low, increase it

## Advanced Usage

### Fine-Tune Different Model

```bash
# Use larger, more powerful model
python src/fine_tune_model.py \
  --base-model all-mpnet-base-v2 \
  --output models/fine_tuned_mpnet
```

### Grid Search for Best Hyperparameters

```bash
# Create a script to try different combinations
for lr in 1e-5 2e-5 3e-5; do
  for bs in 16 32; do
    python src/fine_tune_model.py \
      --learning-rate $lr \
      --batch-size $bs \
      --output models/fine_tuned_lr${lr}_bs${bs}
  done
done
```

### Cross-Validation

For more robust evaluation, modify the script to use k-fold cross-validation instead of a single train-test split.

## Integration with Existing Pipeline

### Option 1: Replace Embeddings

Update `src/embeddings.py`:

```python
AVAILABLE_MODELS = {
    'all-MiniLM-L6-v2': 'all-MiniLM-L6-v2',
    'fine-tuned': 'models/fine_tuned',  # Add fine-tuned model
    # ... other models
}
```

Then in `config.yaml`:
```yaml
embedding:
  model: "fine-tuned"
```

### Option 2: Generate New Embeddings

```bash
# Generate embeddings with fine-tuned model
python src/embeddings.py --model models/fine_tuned
```

### Option 3: Compare in Embedding Comparison Tool

```bash
python src/embedding_comparison.py \
  --models all-MiniLM-L6-v2 models/fine_tuned \
  --sample-size 3000 \
  --n-clusters 12
```

## References

- [Sentence-Transformers Documentation](https://www.sbert.net/)
- [Fine-Tuning Guide](https://www.sbert.net/docs/training/overview.html)
- [Softmax Loss](https://www.sbert.net/docs/package_reference/losses.html#softmaxloss)

## Questions?

If you encounter issues:
1. Check that all dependencies are installed: `pip install -r requirements.txt`
2. Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
3. Review training logs for errors
4. Try simpler configuration first (fewer epochs, smaller batch size)
