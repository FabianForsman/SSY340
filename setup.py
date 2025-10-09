"""
Quick setup script for the hate speech detection project.
Run this after creating the virtual environment and installing requirements.
"""

import sys
from pathlib import Path


def setup_project():
    """Set up the project environment."""
    print("=" * 70)
    print("HATE SPEECH DETECTION PROJECT - SETUP")
    print("=" * 70)

    # Check Python version
    print(f"\nPython version: {sys.version}")

    # Try importing key dependencies
    print("\nChecking dependencies...")
    dependencies = [
        "numpy",
        "pandas",
        "sklearn",
        "sentence_transformers",
        "torch",
        "matplotlib",
        "seaborn",
        "nltk",
        "yaml",
    ]

    missing = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep} (missing)")
            missing.append(dep)

    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False

    # Download NLTK data
    print("\nDownloading NLTK data...")
    try:
        import nltk

        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        print("  ✓ NLTK data downloaded")
    except Exception as e:
        print(f"  ⚠ Could not download NLTK data: {e}")

    # Verify directory structure
    print("\nVerifying directory structure...")
    required_dirs = [
        "data/raw",
        "data/processed",
        "data/embeddings",
        "models",
        "notebooks",
        "outputs/figures",
        "outputs/results",
        "src",
    ]

    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (missing)")

    # Check for config file
    print("\nChecking configuration...")
    if Path("config.yaml").exists():
        print("  ✓ config.yaml found")
    else:
        print("  ✗ config.yaml not found")

    print("\n" + "=" * 70)
    print("SETUP COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Download the dataset from Kaggle:")
    print(
        "   https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset"
    )
    print("2. Place the CSV file in data/raw/")
    print("3. Run notebooks or use: python src/main.py --config config.yaml")
    print("\nFor Jupyter notebooks:")
    print("  jupyter notebook")
    print("  Then open notebooks/01_data_exploration.ipynb")

    return True


if __name__ == "__main__":
    setup_project()
