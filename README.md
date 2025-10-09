# Hate Speech Detection - Unsupervised Learning

**Course:** SSY340 Deep Machine Learning  
**Group:** 13  
**Members:** Fabian Forsman, Thea Granström, Angelica Hörngren, Conrad Olsson  
**Institution:** Chalmers University of Technology

## Project Overview

This project investigates whether **unsupervised and semi-supervised methods** can effectively classify toxic and hate speech content on social media platforms compared to supervised approaches. The goal is to evaluate the possibility of reducing annotation costs while maintaining good performance.

### Problem Statement

Social media platforms face constant challenges with toxic and hate speech content. While supervised models like fine-tuned BERT classifiers show strong performance, they require large annotated datasets that are costly and time-consuming to produce. This project explores alternatives that could:

- Reduce annotation effort
- Maintain competitive performance
- Potentially detect new types of hate speech not present in labeled data

## Dataset

**Primary Dataset:** [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)

- ~25,000 labeled tweets
- Categories: hate speech, offensive language, neither

**Additional Datasets (optional):**

- Toxic Comment Classification Challenge
- TwitterHate Dataset

## Methodology

### 1. Sentence Embeddings (SBERT)

We use pre-trained Sentence-BERT models to generate dense vector representations:

- **all-MiniLM-L6-v2**: Fast and lightweight (384-dim), good for short texts
- **paraphrase-MiniLM-L6-v2**: Optimized for semantic similarity (384-dim)
- **all-mpnet-base-v2**: Higher accuracy but more computationally expensive (768-dim)

### 2. Clustering Methods

- **K-Means**: Efficient clustering with predetermined number of clusters (k=2 for hate vs non-hate)
- **DBSCAN**: Density-based clustering that can identify outliers and arbitrary-shaped clusters

### 3. Evaluation

**Extrinsic Evaluation** (with ground truth labels):

- **Adjusted Rand Index (ARI)**: Primary metric measuring agreement between clusters and true labels
- Normalized Mutual Information (NMI)
- Homogeneity, Completeness, V-Measure
- Fowlkes-Mallows Score

**Intrinsic Evaluation** (without labels):

- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index

## Project Structure

```
SSY340/
├── data/
│   ├── raw/              # Raw datasets
│   ├── processed/        # Preprocessed data
│   └── embeddings/       # Saved embeddings
├── models/               # Saved models
├── notebooks/            # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── outputs/
│   ├── figures/          # Plots and visualizations
│   └── results/          # Evaluation results
├── src/                  # Source code modules
│   ├── data_loader.py    # Data loading utilities
│   ├── preprocessing.py  # Text preprocessing
│   ├── embeddings.py     # Embedding generation
│   ├── clustering.py     # Clustering algorithms
│   ├── evaluation.py     # Evaluation metrics
│   └── main.py          # Main training script
├── tests/                # Unit tests
├── config.yaml           # Configuration file
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
```

**Activate the environment:**

- Windows: `.\venv\Scripts\Activate.ps1`
- Linux/Mac: `source venv/bin/activate`

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data (if needed)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 4. Configure Kaggle API (optional)

To download datasets from Kaggle:

1. Create a Kaggle account
2. Go to Account Settings → API → Create New API Token
3. Place `kaggle.json` in:
   - Windows: `C:\Users\<Username>\.kaggle\`
   - Linux/Mac: `~/.kaggle/`

## Usage

### Option 1: Run Complete Pipeline

```bash
python src/main.py --config config.yaml
```

### Option 2: Use Jupyter Notebooks

```bash
jupyter notebook
```

Then open:

1. `notebooks/01_data_exploration.ipynb` - Data analysis and preprocessing
2. `notebooks/02_model_training.ipynb` - Model training and evaluation

### Option 3: Use Python Modules Directly

```python
from src.data_loader import DataLoader
from src.preprocessing import TextPreprocessor
from src.embeddings import EmbeddingGenerator
from src.clustering import KMeansClustering
from src.evaluation import ClusteringEvaluator

# Load and preprocess data
loader = DataLoader('data/raw')
df = loader.load_hate_speech_dataset()

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
```

## Configuration

Edit `config.yaml` to customize:

- Data paths and column names
- Preprocessing options
- Embedding model selection
- Clustering parameters
- Evaluation settings

## Results

Results will be saved to:

- `outputs/results/` - CSV files with metrics
- `outputs/figures/` - Plots and visualizations

## Timeline

- **Week 40 (Oct 9)**: Planning report ✓
- **Week 41**: Data preprocessing and baseline models
- **Week 42**: Testing different approaches (clustering methods, embedding models)
- **Week 43 (Oct 23)**: Project poster
- **Week 43 (Oct 27)**: Final project report

## References

1. Davidson et al. (2017) - Hate Speech and Offensive Language Dataset
2. Sentence-BERT (Reimers & Gurevych, 2019)
3. Vaswani et al. (2017) - Attention is All You Need
4. Goodfellow et al. (2016) - Deep Learning

## License

This is an academic project for SSY340 Deep Machine Learning at Chalmers University of Technology.

## Contact

For questions or issues, please contact the project group members.
