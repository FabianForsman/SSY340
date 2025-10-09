"""
Data loading module for hate speech detection project.
Handles downloading and loading the Hate Speech and Offensive Language Dataset.
"""

import os
import pandas as pd
import kaggle
from pathlib import Path


class DataLoader:
    """Class to handle data loading operations."""

    def __init__(self, data_dir="data/raw"):
        """
        Initialize DataLoader.

        Args:
            data_dir (str): Directory to store raw data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_kaggle_dataset(self, dataset_name, force_download=False):
        """
        Download dataset from Kaggle.

        Args:
            dataset_name (str): Kaggle dataset identifier (e.g., 'mrmorj/hate-speech-and-offensive-language-dataset')
            force_download (bool): Whether to re-download if data already exists

        Returns:
            Path: Path to downloaded data directory
        """
        dataset_path = self.data_dir / dataset_name.split("/")[-1]

        if dataset_path.exists() and not force_download:
            print(f"Dataset already exists at {dataset_path}")
            return dataset_path

        print(f"Downloading dataset: {dataset_name}")
        kaggle.api.dataset_download_files(dataset_name, path=self.data_dir, unzip=True)

        return dataset_path

    def load_hate_speech_dataset(self, file_path=None):
        """
        Load the Hate Speech and Offensive Language Dataset.

        Args:
            file_path (str): Path to the CSV file. If None, looks in data_dir

        Returns:
            pd.DataFrame: Loaded dataset
        """
        if file_path is None:
            # Try common filenames
            possible_files = [
                self.data_dir / "labeled_data.csv",
                self.data_dir / "hate_speech.csv",
                self.data_dir / "data.csv",
            ]

            for path in possible_files:
                if path.exists():
                    file_path = path
                    break

            if file_path is None:
                raise FileNotFoundError(
                    f"Could not find dataset file in {self.data_dir}. "
                    "Please specify file_path explicitly."
                )

        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)

        print(f"Loaded {len(df)} samples")
        print(f"Columns: {df.columns.tolist()}")

        return df

    def get_dataset_info(self, df):
        """
        Print information about the dataset.

        Args:
            df (pd.DataFrame): Dataset to analyze
        """
        print("\n=== Dataset Information ===")
        print(f"Total samples: {len(df)}")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nData types:\n{df.dtypes}")
        print(f"\nMissing values:\n{df.isnull().sum()}")

        # Check for label columns
        label_columns = [
            col
            for col in df.columns
            if "class" in col.lower() or "label" in col.lower()
        ]
        if label_columns:
            print(f"\n=== Label Distribution ===")
            for col in label_columns:
                print(f"\n{col}:")
                print(df[col].value_counts())

        return df.describe()


def load_unlabeled_tweets(file_path):
    """
    Load unlabeled tweet data for unsupervised pre-training.

    Args:
        file_path (str): Path to unlabeled data file

    Returns:
        pd.DataFrame: Unlabeled tweets
    """
    print(f"Loading unlabeled data from {file_path}")
    df = pd.read_csv(file_path)
    return df


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()

    print("To download from Kaggle, you'll need to:")
    print("1. Set up Kaggle API credentials (~/.kaggle/kaggle.json)")
    print("2. Find the dataset identifier on Kaggle")
    print("3. Use: loader.download_kaggle_dataset('username/dataset-name')")
    print("\nFor the hate speech dataset, try:")
    print(
        "loader.download_kaggle_dataset('mrmorj/hate-speech-and-offensive-language-dataset')"
    )
