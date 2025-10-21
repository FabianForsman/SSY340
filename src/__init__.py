"""
Initialize the src package.
"""

__version__ = "0.1.0"
__author__ = "Group 13 - SSY340"

# Import only classes/functions that exist
try:
    from .data_loader import DataLoader
except ImportError:
    DataLoader = None

try:
    from .preprocessing import TextTransform
except ImportError:
    TextTransform = None

try:
    from .embeddings import EmbeddingGenerator
except ImportError:
    EmbeddingGenerator = None

try:
    from .clustering import KMeansClustering, DBSCANClustering
except ImportError:
    KMeansClustering = None
    DBSCANClustering = None

__all__ = [
    "DataLoader",
    "TextTransform",
    "EmbeddingGenerator",
    "KMeansClustering",
    "DBSCANClustering",
]
