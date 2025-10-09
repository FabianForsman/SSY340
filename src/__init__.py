"""
Initialize the src package.
"""

__version__ = "0.1.0"
__author__ = "Group 13 - SSY340"

from .data_loader import DataLoader
from .preprocessing import TextPreprocessor
from .embeddings import EmbeddingGenerator
from .clustering import KMeansClustering, DBSCANClustering
from .evaluation import ClusteringEvaluator

__all__ = [
    "DataLoader",
    "TextPreprocessor",
    "EmbeddingGenerator",
    "KMeansClustering",
    "DBSCANClustering",
    "ClusteringEvaluator",
]
