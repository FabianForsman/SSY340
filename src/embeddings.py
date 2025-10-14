"""
Embedding generation module for hate speech detection project.
Generates sentence embeddings using various SBERT models.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import torch
from pathlib import Path


class EmbeddingGenerator:
    """Class to generate embeddings from text using Sentence-BERT models."""

    # Available models as per the planning report
    AVAILABLE_MODELS = {
        "all-MiniLM-L6-v2": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "dim": 384,
            "description": "Fast and lightweight, good for short texts like tweets",
        },
        "paraphrase-MiniLM-L6-v2": {
            "name": "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "dim": 384,
            "description": "Optimized for paraphrase detection, good for semantic similarity",
        },
        "all-mpnet-base-v2": {
            "name": "sentence-transformers/all-mpnet-base-v2",
            "dim": 768,
            "description": "Higher accuracy but more computationally expensive",
        },
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Initialize EmbeddingGenerator.

        Args:
            model_name (str): Name of the model to use (key from AVAILABLE_MODELS)
            device (str): Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {model_name} not available. "
                f"Choose from: {list(self.AVAILABLE_MODELS.keys())}"
            )

        self.model_name = model_name
        self.model_info = self.AVAILABLE_MODELS[model_name]

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading model: {self.model_info['name']}")
        print(f"Description: {self.model_info['description']}")
        print(f"Embedding dimension: {self.model_info['dim']}")
        print(f"Using device: {self.device}")

        # Load model
        self.model = SentenceTransformer(self.model_info["name"], device=self.device)

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for texts. 

        Args:
            texts (str or List[str]): Text or list of texts to encode
            batch_size (int): Batch size for encoding
            show_progress (bool): Whether to show progress bar
            normalize (bool): Whether to normalize embeddings to unit length

        Returns:
            np.ndarray: Array of embeddings (num_texts, embedding_dim)
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]

        print(f"Encoding {len(texts)} texts...")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

        print(f"Generated embeddings shape: {embeddings.shape}")

        return embeddings

    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """
        Save embeddings to disk.

        Args:
            embeddings (np.ndarray): Embeddings to save
            filepath (str): Path to save file (.npy format)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        np.save(filepath, embeddings)
        print(f"Saved embeddings to {filepath}")
        print(f"Shape: {embeddings.shape}")

    @staticmethod
    def load_embeddings(filepath: str) -> np.ndarray:
        """
        Load embeddings from disk.

        Args:
            filepath (str): Path to embeddings file

        Returns:
            np.ndarray: Loaded embeddings
        """
        embeddings = np.load(filepath)
        print(f"Loaded embeddings from {filepath}")
        print(f"Shape: {embeddings.shape}")
        return embeddings

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension of the current model."""
        return self.model_info["dim"]
    
    def get_similarities(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """Get the similarity scores of the current model."""
        return self.model.similarity(emb1, emb2)
        


if __name__ == "__main__":
    # Example usage
    print("=== Embedding Generator Example ===\n")

    sample_texts = [
        "This is a hate speech example",
        "This is offensive language",
        "This is a normal tweet",
    ]

    # Test with default model
    generator = EmbeddingGenerator("all-MiniLM-L6-v2")
    embeddings = generator.encode(sample_texts, show_progress=False)

    print(f"\nGenerated {len(embeddings)} embeddings")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Sample embedding (first 5 dims): {embeddings[0][:5]}")
