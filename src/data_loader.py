"""
Data loading module for hate speech detection project.
Handles loading the Hate Speech and Offensive Language Dataset.
"""

import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


class HateSpeechDataset(Dataset):
    """PyTorch Dataset for hate speech detection."""
    
    def __init__(self, root, file="labeled_data.csv", transform=None, text_column="tweet", label_column="label"):
        """Constructor
        
        Args:
            root (Path/str): Filepath to the data root, e.g. './data/raw'
            file (str): Name of the CSV file containing the data
            transform (callable, optional): Optional transform to be applied on text samples
            text_column (str): Name of the column containing text data
            label_column (str): Name of the column containing labels
        """
        root = Path(root)
        if not (root.exists() and root.is_dir()):
            raise ValueError(f"Data root '{root}' is invalid")
        
        self.root = root
        self.transform = transform
        self.text_column = text_column
        self.label_column = label_column
        
        # Validate file parameter
        if file is None:
            raise ValueError("File parameter cannot be None")
        
        # Load the dataset
        file_path = root / file
        if not file_path.exists():
            raise ValueError(f"Data file '{file_path}' does not exist")
        
        print(f"Loading data from path: {file_path}")
        self.df = pd.read_csv(file_path)
        
        # Drop uneccessary columns if they exist
        self.df = self.df.drop(columns=['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither'])

        # Rename 'class' column for clarity
        self.df.rename(columns={'class': 'label'}, inplace=True)

        # Map numeric labels to descriptive labels
        label_mapping = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
        self.df['label_desc'] = self.df['label'].map(label_mapping)

        # Store samples as list of (text, label) tuples
        self._samples = self._collect_samples()
    
    def __getitem__(self, index):
        """Get sample by index
        
        Args:
            index (int): Index of the sample
            
        Returns:
            tuple: (text, label) where text is the tweet text and label is the class
        """
        text, label = self._samples[index]
        
        # Apply transforms if any
        if self.transform is not None:
            text = self.transform(text)
        
        return text, label
    
    def __len__(self):
        """Total number of samples"""
        return len(self._samples)
    
    def _collect_samples(self):
        """Collect all text samples and labels
        
        Helper method for the constructor
        
        Returns:
            list: List of (text, label) tuples
        """
        samples = []
        for idx, row in self.df.iterrows():
            text = row[self.text_column]
            label = row[self.label_column]
            samples.append((text, label))
        return samples
    
    def get_dataset_info(self):
        """Print information about the dataset."""
        print("\n=== Dataset Information ===")
        print(f"Total samples: {len(self)}")
        print(f"\nColumns: {self.df.columns.tolist()}")
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        
        # Label distribution
        print(f"\n=== Label Distribution ===")
        print(f"\n{self.label_column}:")
        print(self.df[self.label_column].value_counts())
        
        return self.df.describe()
    
    def get_sample_by_index(self, idx):
        """Get sample by DataFrame index
        
        Convenience method for exploration.
        
        Args:
            idx (int): Index in the original DataFrame
            
        Returns:
            tuple: (text, label)
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        return self[idx]

    def update_dataframe(self, new_dataframe):
        """Get the underlying DataFrame
        
        Returns:
            pd.DataFrame: The dataset as a pandas DataFrame
        """
        self.df = new_dataframe
        self._samples = self._collect_samples()

class UnlabeledTweetDataset(Dataset):
    """PyTorch Dataset for unlabeled tweets (for unsupervised pre-training)."""
    
    def __init__(self, root, file, transform=None, text_column="tweet"):
        """Constructor
        
        Args:
            root (Path/str): Filepath to the data root
            file (str): Name of the CSV file containing the data
            transform (callable, optional): Optional transform to be applied on text samples
            text_column (str): Name of the column containing text data
        """
        root = Path(root)
        if not (root.exists() and root.is_dir()):
            raise ValueError(f"Data root '{root}' is invalid")
        
        self.root = root
        self.transform = transform
        self.text_column = text_column
        
        # Load the dataset
        file_path = root / file
        if not file_path.exists():
            raise ValueError(f"Data file '{file_path}' does not exist")
        
        print(f"Loading unlabeled data from {file_path}")
        self.df = pd.read_csv(file_path)
        
        # Validate column exists
        if text_column not in self.df.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataset. Available columns: {self.df.columns.tolist()}")
        
        # Store samples as list of text
        self._samples = list(self.df[text_column])
    
    def __getitem__(self, index):
        """Get sample by index
        
        Args:
            index (int): Index of the sample
            
        Returns:
            str: Tweet text
        """
        text = self._samples[index]
        
        # Apply transforms if any
        if self.transform is not None:
            text = self.transform(text)
        
        return text
    
    def __len__(self):
        """Total number of samples"""
        return len(self._samples)


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0, **kwargs):
    """Create a PyTorch DataLoader from a dataset.
    
    Args:
        dataset (Dataset): PyTorch dataset
        batch_size (int): Number of samples per batch
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of subprocesses for data loading
        **kwargs: Additional arguments to pass to DataLoader
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    dataset = HateSpeechDataset(root="data/raw", file="labeled_data.csv")
    dataset.get_dataset_info()
    
    # Create a DataLoader
    dataloader = create_dataloader(dataset, batch_size=32, shuffle=True)
    
    # Test iteration
    print("\n=== Testing DataLoader ===")
    for i, (texts, labels) in enumerate(dataloader):
        print(f"Batch {i+1}: {len(texts)} samples")
        print(f"First text: {texts[0][:50]}...")
        print(f"First label: {labels[0]}")
        if i == 0:  # Only print first batch
            break