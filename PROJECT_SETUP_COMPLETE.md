# Project Setup Complete! 🎉

## What Has Been Created

A complete machine learning project for **Hate Speech Detection using Unsupervised Learning** has been set up based on your Planning Report Group 13.

### ✅ Directory Structure

```
SSY340/
├── config.yaml                 # Configuration file with all settings
├── requirements.txt            # Python dependencies
├── setup.py                    # Setup verification script
├── README.md                   # Comprehensive documentation
│
├── data/                       # Data directory
│   ├── raw/                    # Raw datasets (download here)
│   ├── processed/              # Preprocessed data
│   └── embeddings/             # Saved embeddings (.npy files)
│
├── models/                     # Saved models
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb    # Data analysis
│   └── 02_model_training.ipynb      # Model training & evaluation
│
├── outputs/                    # Results and visualizations
│   ├── figures/                # Plots (confusion matrices, elbow curves)
│   └── results/                # CSV files with metrics
│
├── src/                        # Source code modules
│   ├── __init__.py            # Package initialization
│   ├── data_loader.py         # Data loading from Kaggle
│   ├── preprocessing.py       # Text preprocessing
│   ├── embeddings.py          # SBERT embeddings generation
│   ├── clustering.py          # K-Means & DBSCAN
│   ├── evaluation.py          # ARI, NMI, metrics
│   └── main.py                # Main pipeline script
│
├── tests/                      # Unit tests (ready for implementation)
│
└── venv/                       # Virtual environment (✅ CREATED & ACTIVATED)
```

### 📦 Created Modules

#### 1. **data_loader.py**

- Downloads datasets from Kaggle (with API)
- Loads CSV files automatically
- Provides dataset statistics

#### 2. **preprocessing.py**

- Text cleaning (URLs, mentions, hashtags)
- Stopword removal (optional)
- Case normalization
- Text statistics

#### 3. **embeddings.py**

- Three SBERT models supported:
  - `all-MiniLM-L6-v2` (384-dim, fast)
  - `paraphrase-MiniLM-L6-v2` (384-dim, semantic similarity)
  - `all-mpnet-base-v2` (768-dim, high accuracy)
- Batch encoding
- Save/load embeddings

#### 4. **clustering.py**

- K-Means clustering
- DBSCAN clustering
- Optimal k finder (elbow method)
- Visualization tools

#### 5. **evaluation.py**

- **Adjusted Rand Index (ARI)** - primary metric
- Normalized Mutual Information (NMI)
- Homogeneity, Completeness, V-Measure
- Silhouette Score
- Confusion matrices
- Method comparison plots

#### 6. **main.py**

- Complete pipeline orchestration
- Configurable via `config.yaml`
- Automated workflow

### 📓 Jupyter Notebooks

Two ready-to-use notebooks:

1. **01_data_exploration.ipynb** - Data analysis, visualization, preprocessing
2. **02_model_training.ipynb** - Model training, evaluation, results

## Next Steps

### Step 1: Install Dependencies

```powershell
pip install -r requirements.txt
```

This will install:

- `sentence-transformers` for embeddings
- `scikit-learn` for clustering
- `pandas`, `numpy` for data
- `matplotlib`, `seaborn` for visualization
- `nltk` for text processing
- And more...

### Step 2: Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Step 3: Get the Dataset

**Option A: Download from Kaggle**

1. Create Kaggle account
2. Go to: https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset
3. Download CSV file
4. Place in `data/raw/`

**Option B: Use Kaggle API**

1. Get API credentials from Kaggle settings
2. Place `kaggle.json` in `~/.kaggle/`
3. Run in Python:

```python
from src.data_loader import DataLoader
loader = DataLoader()
loader.download_kaggle_dataset('mrmorj/hate-speech-and-offensive-language-dataset')
```

### Step 4: Run the Project

**Option 1: Full Pipeline**

```powershell
python src/main.py --config config.yaml
```

**Option 2: Jupyter Notebooks**

```powershell
jupyter notebook
```

Then open notebooks in order.

**Option 3: Python Code**

```python
from src import DataLoader, TextPreprocessor, EmbeddingGenerator
from src import KMeansClustering, ClusteringEvaluator

# Load data
loader = DataLoader('data/raw')
df = loader.load_hate_speech_dataset()

# Preprocess
preprocessor = TextPreprocessor(lowercase=True, remove_urls=True)
df = preprocessor.preprocess_dataframe(df, 'tweet')

# Generate embeddings
generator = EmbeddingGenerator('all-MiniLM-L6-v2')
embeddings = generator.encode_dataframe(df, 'cleaned_text')

# Cluster
kmeans = KMeansClustering(n_clusters=2)
labels = kmeans.fit(embeddings)

# Evaluate
evaluator = ClusteringEvaluator()
results = evaluator.evaluate(df['class'].values, labels, embeddings)
print(f"Adjusted Rand Index: {results['adjusted_rand_index']:.4f}")
```

## Configuration

Edit `config.yaml` to customize:

- Data paths and column names
- Preprocessing options (stopwords, case, URLs, etc.)
- Embedding model (`all-MiniLM-L6-v2`, `paraphrase-MiniLM-L6-v2`, `all-mpnet-base-v2`)
- Clustering parameters (k, eps, min_samples)
- Evaluation settings

## Project Features

### ✅ Following Planning Report Requirements

1. **Dataset**: Hate Speech and Offensive Language Dataset (~25,000 tweets)
2. **Embeddings**: SBERT models (all-MiniLM-L6-v2, paraphrase-MiniLM-L6-v2, all-mpnet-base-v2)
3. **Clustering**: K-Means and DBSCAN
4. **Evaluation**: Adjusted Rand Index (ARI) as primary metric
5. **Unsupervised**: Models trained without labels
6. **Extrinsic Evaluation**: Compare clusters to ground truth

### 🎯 Key Capabilities

- **Modular Design**: Each component is independent and reusable
- **Configurable**: All settings in one YAML file
- **Documented**: Comprehensive README and docstrings
- **Reproducible**: Random seeds and saved configurations
- **Extensible**: Easy to add new models or methods
- **Production-Ready**: Proper error handling and logging

## Timeline Alignment

- **Week 40 (Oct 9)**: ✅ Planning report + Project setup
- **Week 41**: Ready for data preprocessing and initial model training
- **Week 42**: Ready for testing different approaches
- **Week 43 (Oct 23)**: Ready for project poster
- **Week 43 (Oct 27)**: Ready for final project report

## Tips

1. **Start with notebooks** for exploration and understanding
2. **Use the config file** to experiment with different settings
3. **Try different embedding models** to compare performance
4. **Adjust clustering parameters** (k, eps) based on elbow curves
5. **Save your embeddings** to avoid re-computing (they're large)
6. **Document your experiments** in the notebooks

## Troubleshooting

### Import errors?

```powershell
pip install -r requirements.txt
```

### Kaggle API not working?

- Check `~/.kaggle/kaggle.json` exists
- Verify JSON format
- Try manual download instead

### Out of memory?

- Reduce batch size in config
- Use smaller embedding model
- Process data in chunks

### CUDA errors?

- Set `device: 'cpu'` in embeddings config
- Or install PyTorch with CUDA support

## Resources

- **Planning Report**: `Planning_Report_Group13.pdf`
- **Documentation**: All modules have detailed docstrings
- **Examples**: Check notebooks and `__main__` blocks in modules

---

**Good luck with your project! 🚀**

_The project structure follows ML engineering best practices and is ready for immediate use._
