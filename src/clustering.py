"""
Clustering module for hate speech detection.
Implements various clustering algorithms for unsupervised learning.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from typing import Dict, Optional, Tuple
import warnings


class KMeansClustering:
    """K-Means clustering wrapper."""
    
    def __init__(self, n_clusters=3, random_state=42, **kwargs):
        """
        Initialize K-Means clustering.
        
        Args:
            n_clusters (int): Number of clusters
            random_state (int): Random seed
            **kwargs: Additional arguments for sklearn KMeans
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            **kwargs
        )
        self.cluster_centers_ = None
        
    def fit(self, embeddings: np.ndarray) -> 'KMeansClustering':
        """
        Fit K-Means model.
        
        Args:
            embeddings (np.ndarray): Data to cluster
            
        Returns:
            self: Fitted model
        """
        self.model.fit(embeddings)
        self.cluster_centers_ = self.model.cluster_centers_
        return self
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels.
        
        Args:
            embeddings (np.ndarray): Data to predict
            
        Returns:
            np.ndarray: Cluster labels
        """
        return self.model.predict(embeddings)
    
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit model and predict cluster labels.
        
        Args:
            embeddings (np.ndarray): Data to cluster
            
        Returns:
            np.ndarray: Cluster labels
        """
        labels = self.model.fit_predict(embeddings)
        self.cluster_centers_ = self.model.cluster_centers_
        return labels
    
    def get_cluster_centers(
        self,
        embeddings: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ) -> Dict[int, np.ndarray]:
        """
        Get cluster centers.
        
        Args:
            embeddings (np.ndarray, optional): Not used for KMeans
            labels (np.ndarray, optional): Not used for KMeans
            
        Returns:
            Dict[int, np.ndarray]: Mapping from cluster_id to center
        """
        if self.cluster_centers_ is None:
            return {}
        
        return {i: center for i, center in enumerate(self.cluster_centers_)}


class DBSCANClustering:
    """DBSCAN clustering wrapper."""
    
    def __init__(self, eps=0.5, min_samples=5, **kwargs):
        """
        Initialize DBSCAN clustering.
        
        Args:
            eps (float): Maximum distance between samples
            min_samples (int): Minimum samples in a neighborhood
            **kwargs: Additional arguments for sklearn DBSCAN
        """
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            **kwargs
        )
        self.labels_ = None
        
    def fit(self, embeddings: np.ndarray) -> 'DBSCANClustering':
        """
        Fit DBSCAN model.
        
        Args:
            embeddings (np.ndarray): Data to cluster
            
        Returns:
            self: Fitted model
        """
        self.labels_ = self.model.fit_predict(embeddings)
        return self
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        DBSCAN doesn't support predict. Use fit_predict instead.
        
        Args:
            embeddings (np.ndarray): Data to predict
            
        Returns:
            np.ndarray: Cluster labels
        """
        warnings.warn(
            "DBSCAN doesn't support predict(). Using fit_predict() instead."
        )
        return self.fit_predict(embeddings)
    
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit model and predict cluster labels.
        
        Args:
            embeddings (np.ndarray): Data to cluster
            
        Returns:
            np.ndarray: Cluster labels (-1 for noise)
        """
        self.labels_ = self.model.fit_predict(embeddings)
        return self.labels_
    
    def get_cluster_centers(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Calculate cluster centers as mean of cluster members.
        
        Args:
            embeddings (np.ndarray): Original embeddings
            labels (np.ndarray): Cluster labels
            
        Returns:
            Dict[int, np.ndarray]: Mapping from cluster_id to center
        """
        centers = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            mask = labels == label
            centers[label] = embeddings[mask].mean(axis=0)
        
        return centers


class HierarchicalClustering:
    """Agglomerative (Hierarchical) clustering wrapper."""
    
    def __init__(self, n_clusters=3, linkage='ward', **kwargs):
        """
        Initialize hierarchical clustering.
        
        Args:
            n_clusters (int): Number of clusters
            linkage (str): Linkage criterion ('ward', 'complete', 'average', 'single')
            **kwargs: Additional arguments for sklearn AgglomerativeClustering
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            **kwargs
        )
        self.labels_ = None
        
    def fit(self, embeddings: np.ndarray) -> 'HierarchicalClustering':
        """
        Fit hierarchical clustering model.
        
        Args:
            embeddings (np.ndarray): Data to cluster
            
        Returns:
            self: Fitted model
        """
        self.labels_ = self.model.fit_predict(embeddings)
        return self
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Hierarchical clustering doesn't support predict. Use fit_predict instead.
        
        Args:
            embeddings (np.ndarray): Data to predict
            
        Returns:
            np.ndarray: Cluster labels
        """
        warnings.warn(
            "Hierarchical clustering doesn't support predict(). Using fit_predict() instead."
        )
        return self.fit_predict(embeddings)
    
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit model and predict cluster labels.
        
        Args:
            embeddings (np.ndarray): Data to cluster
            
        Returns:
            np.ndarray: Cluster labels
        """
        self.labels_ = self.model.fit_predict(embeddings)
        return self.labels_
    
    def get_cluster_centers(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Calculate cluster centers as mean of cluster members.
        
        Args:
            embeddings (np.ndarray): Original embeddings
            labels (np.ndarray): Cluster labels
            
        Returns:
            Dict[int, np.ndarray]: Mapping from cluster_id to center
        """
        centers = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            centers[label] = embeddings[mask].mean(axis=0)
        
        return centers


def evaluate_clustering(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate clustering quality.
    
    Args:
        embeddings (np.ndarray): Original embeddings
        cluster_labels (np.ndarray): Predicted cluster labels
        true_labels (np.ndarray, optional): Ground truth labels for supervised metrics
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Remove noise points for metrics calculation
    valid_mask = cluster_labels != -1
    valid_embeddings = embeddings[valid_mask]
    valid_clusters = cluster_labels[valid_mask]
    
    # Check if we have enough clusters
    n_clusters = len(np.unique(valid_clusters))
    
    if n_clusters > 1 and len(valid_embeddings) > n_clusters:
        try:
            # Silhouette Score: [-1, 1], higher is better
            metrics['silhouette_score'] = silhouette_score(
                valid_embeddings,
                valid_clusters
            )
        except:
            metrics['silhouette_score'] = 0.0
        
        try:
            # Calinski-Harabasz Index: higher is better
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(
                valid_embeddings,
                valid_clusters
            )
        except:
            metrics['calinski_harabasz_score'] = 0.0
        
        try:
            # Davies-Bouldin Index: lower is better
            metrics['davies_bouldin_score'] = davies_bouldin_score(
                valid_embeddings,
                valid_clusters
            )
        except:
            metrics['davies_bouldin_score'] = float('inf')
    else:
        metrics['silhouette_score'] = 0.0
        metrics['calinski_harabasz_score'] = 0.0
        metrics['davies_bouldin_score'] = float('inf')
    
    # Number of clusters
    metrics['n_clusters'] = n_clusters
    metrics['n_noise_points'] = (cluster_labels == -1).sum()
    
    # If true labels are provided, calculate supervised metrics
    if true_labels is not None:
        from evaluation import calculate_cluster_purity, calculate_metrics
        
        # Purity
        metrics['purity'] = calculate_cluster_purity(
            cluster_labels[valid_mask],
            true_labels[valid_mask]
        )
        
        # Standard classification metrics (if clusters map to labels)
        # This requires a mapping from clusters to labels
        # For now, we'll skip this as it needs a mapping strategy
    
    return metrics


def select_optimal_k(
    embeddings: np.ndarray,
    k_range: Tuple[int, int] = (2, 10),
    random_state: int = 42
) -> Tuple[int, Dict[int, Dict[str, float]]]:
    """
    Select optimal number of clusters using elbow method and silhouette analysis.
    
    Args:
        embeddings (np.ndarray): Data to cluster
        k_range (Tuple[int, int]): Range of k values to try (min, max)
        random_state (int): Random seed
        
    Returns:
        Tuple[int, Dict]: Optimal k and metrics for all k values
    """
    results = {}
    
    for k in range(k_range[0], k_range[1] + 1):
        clusterer = KMeansClustering(n_clusters=k, random_state=random_state)
        labels = clusterer.fit_predict(embeddings)
        
        metrics = evaluate_clustering(embeddings, labels)
        metrics['inertia'] = clusterer.model.inertia_
        
        results[k] = metrics
    
    # Select k with highest silhouette score
    optimal_k = max(
        results.keys(),
        key=lambda k: results[k]['silhouette_score']
    )
    
    return optimal_k, results