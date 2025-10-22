"""
Semi-supervised self-training module for hate speech detection.
Implements iterative pseudo-labeling using clustering on sentence embeddings.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from pathlib import Path
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings

from clustering import KMeansClustering, DBSCANClustering, evaluate_clustering
from evaluation import calculate_metrics, calculate_cluster_purity


class SelfTrainingClassifier:
    """
    Semi-supervised self-training classifier using clustering and pseudo-labeling.
    
    The algorithm:
    1. Start with labeled data (train set)
    2. Cluster all data (labeled + unlabeled) using embeddings
    3. Assign pseudo-labels to unlabeled samples based on cluster assignments
    4. Select high-confidence predictions (based on distance to cluster center)
    5. Add confident pseudo-labeled samples to labeled set
    6. Repeat until convergence or max iterations
    """
    
    def __init__(
        self,
        clustering_method='kmeans',
        n_clusters=3,
        confidence_threshold=0.6,
        max_iterations=5,
        min_samples_per_iteration=10,
        use_silhouette_for_confidence=True,
        random_state=42
    ):
        """
        Initialize SelfTrainingClassifier.
        
        Args:
            clustering_method (str): 'kmeans' or 'dbscan'
            n_clusters (int): Number of clusters (for kmeans)
            confidence_threshold (float): Threshold for pseudo-label confidence (0-1)
            max_iterations (int): Maximum number of self-training iterations
            min_samples_per_iteration (int): Minimum samples to add per iteration
            use_silhouette_for_confidence (bool): Use silhouette score for confidence
            random_state (int): Random seed
        """
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.min_samples_per_iteration = min_samples_per_iteration
        self.use_silhouette_for_confidence = use_silhouette_for_confidence
        self.random_state = random_state
        self.threshold_decay = 0.95 # Decay factor for confidence threshold per iteration
        
        # Initialize clustering model
        if clustering_method == 'kmeans':
            self.clusterer = KMeansClustering(
                n_clusters=n_clusters,
                random_state=random_state
            )
        elif clustering_method == 'dbscan':
            self.clusterer = DBSCANClustering(
                eps=0.5,
                min_samples=5
            )
        else:
            raise ValueError(f"Unknown clustering method: {clustering_method}")
        
        # Training history
        self.history = {
            'iteration': [],
            'n_labeled': [],
            'n_pseudo_labeled': [],
            'n_unlabeled': [],
            'avg_confidence': [],
            'silhouette_score': [],
            'cluster_purity': []
        }
        
        self.cluster_to_label_mapping = None
        self.final_labels = None
        
    def fit(
        self,
        embeddings: np.ndarray,
        initial_labels: np.ndarray,
        labeled_mask: np.ndarray,
        texts: Optional[List[str]] = None,
        verbose: bool = True
    ) -> 'SelfTrainingClassifier':
        """
        Fit the self-training classifier.
        
        Args:
            embeddings (np.ndarray): Sentence embeddings (n_samples, embedding_dim)
            initial_labels (np.ndarray): Initial labels (n_samples,)
                - Use actual labels for labeled samples
                - Use -1 or any placeholder for unlabeled samples
            labeled_mask (np.ndarray): Boolean mask indicating labeled samples
            texts (List[str], optional): Original text samples for debugging
            verbose (bool): Print progress information
            
        Returns:
            self: Fitted classifier
        """
        if verbose:
            print("\n" + "=" * 70)
            print("SEMI-SUPERVISED SELF-TRAINING")
            print("=" * 70)
        
        n_samples = len(embeddings)
        n_labeled = labeled_mask.sum()
        n_unlabeled = (~labeled_mask).sum()
        
        if verbose:
            print(f"Initial setup:")
            print(f"  Total samples: {n_samples}")
            print(f"  Labeled samples: {n_labeled}")
            print(f"  Unlabeled samples: {n_unlabeled}")
            print(f"  Clustering method: {self.clustering_method}")
            print(f"  Confidence threshold: {self.confidence_threshold}")
        
        # Current labels (will be updated during iterations)
        current_labels = initial_labels.copy()
        current_labeled_mask = labeled_mask.copy()
        
        # Keep track of original labels for evaluation
        original_labeled_mask = labeled_mask.copy()
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")
            
            # Step 1: Cluster all data
            cluster_labels = self.clusterer.fit_predict(embeddings)
            
            # Step 2: Map clusters to true labels using labeled samples
            self.cluster_to_label_mapping = self._map_clusters_to_labels(
                cluster_labels,
                current_labels,
                current_labeled_mask
            )
            
            if verbose:
                print(f"Cluster to label mapping: {self.cluster_to_label_mapping}")
            
            # Step 3: Calculate confidence scores for unlabeled samples
            confidences = self._calculate_confidence_scores(
                embeddings,
                cluster_labels,
                current_labeled_mask
            )
            
            # Step 4: Select high-confidence pseudo-labels
            unlabeled_indices = np.where(~current_labeled_mask)[0]
            # Get confidences only for unlabeled samples
            unlabeled_confidences = confidences[unlabeled_indices]
            dynamic_threshold = self.confidence_threshold * (self.threshold_decay ** iteration)
            high_confidence_mask = unlabeled_confidences > dynamic_threshold
            pseudo_labeled_indices = unlabeled_indices[high_confidence_mask]
            
            n_pseudo_labeled = len(pseudo_labeled_indices)
            
            if verbose:
                if n_pseudo_labeled > 0:
                    print(f"High-confidence pseudo-labels: {n_pseudo_labeled}")
                    print(f"Average confidence: {unlabeled_confidences[high_confidence_mask].mean():.4f}")
                else:
                    print("No high-confidence pseudo-labels found")
            
            # Step 5: Add pseudo-labeled samples to labeled set
            if n_pseudo_labeled >= self.min_samples_per_iteration:
                for idx in pseudo_labeled_indices:
                    cluster_id = cluster_labels[idx]
                    pseudo_label = self.cluster_to_label_mapping.get(cluster_id, -1)
                    if pseudo_label != -1:
                        current_labels[idx] = pseudo_label
                        current_labeled_mask[idx] = True
                
                # Evaluate current iteration
                if original_labeled_mask.sum() > 0:
                    # Calculate purity on originally labeled samples
                    purity = calculate_cluster_purity(
                        cluster_labels[original_labeled_mask],
                        initial_labels[original_labeled_mask]
                    )
                    
                    # Calculate silhouette score
                    if len(np.unique(cluster_labels[cluster_labels != -1])) > 1:
                        valid_mask = cluster_labels != -1
                        if valid_mask.sum() > 1:
                            silhouette = silhouette_score(
                                embeddings[valid_mask],
                                cluster_labels[valid_mask]
                            )
                        else:
                            silhouette = 0.0
                    else:
                        silhouette = 0.0
                    
                    if verbose:
                        print(f"Cluster purity: {purity:.4f}")
                        print(f"Silhouette score: {silhouette:.4f}")
                else:
                    purity = 0.0
                    silhouette = 0.0
                
                # Record history
                self.history['iteration'].append(iteration + 1)
                self.history['n_labeled'].append(current_labeled_mask.sum())
                self.history['n_pseudo_labeled'].append(n_pseudo_labeled)
                self.history['n_unlabeled'].append((~current_labeled_mask).sum())
                self.history['avg_confidence'].append(
                    unlabeled_confidences[high_confidence_mask].mean() if n_pseudo_labeled > 0 else 0.0
                )
                self.history['silhouette_score'].append(silhouette)
                self.history['cluster_purity'].append(purity)
                
            else:
                if verbose:
                    print(f"Stopping early: Only {n_pseudo_labeled} samples meet confidence threshold")
                break
            
            # Check convergence
            if (~current_labeled_mask).sum() == 0:
                if verbose:
                    print("All samples labeled. Stopping.")
                break
        
        # Final clustering with all pseudo-labels
        self.final_labels = current_labels.copy()
        
        if verbose:
            print(f"\n{'=' * 70}")
            print("SELF-TRAINING COMPLETED")
            print(f"{'=' * 70}")
            print(f"Final labeled samples: {current_labeled_mask.sum()}/{n_samples}")
            print(f"Unlabeled samples remaining: {(~current_labeled_mask).sum()}")
        
        return self
    
    def _map_clusters_to_labels(
        self,
        cluster_labels: np.ndarray,
        true_labels: np.ndarray,
        labeled_mask: np.ndarray
    ) -> dict:
        """
        Map cluster IDs to true labels using majority voting on labeled samples.
        
        Args:
            cluster_labels (np.ndarray): Cluster assignments
            true_labels (np.ndarray): True labels
            labeled_mask (np.ndarray): Mask indicating labeled samples
            
        Returns:
            dict: Mapping from cluster_id to label
        """
        mapping = {}
        unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
        
        for cluster_id in unique_clusters:
            # Get labeled samples in this cluster
            in_cluster = (cluster_labels == cluster_id) & labeled_mask
            
            if in_cluster.sum() > 0:
                # Majority vote
                labels_in_cluster = true_labels[in_cluster]
                unique, counts = np.unique(labels_in_cluster, return_counts=True)
                majority_label = unique[np.argmax(counts)]
                # Convert numpy types to Python int for cleaner output
                mapping[int(cluster_id)] = int(majority_label)
            else:
                # No labeled samples in this cluster
                mapping[int(cluster_id)] = -1
        
        return mapping
    
    def _calculate_confidence_scores(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        labeled_mask: np.ndarray
    ) -> np.ndarray:
        """
        Calculate confidence scores for unlabeled samples.
        
        Uses distance to cluster center or silhouette coefficient.
        
        Args:
            embeddings (np.ndarray): Sentence embeddings
            cluster_labels (np.ndarray): Cluster assignments
            labeled_mask (np.ndarray): Mask indicating labeled samples
            
        Returns:
            np.ndarray: Confidence scores for all samples (0 for labeled)
        """
        n_samples = len(embeddings)
        confidences = np.zeros(n_samples)
        unlabeled_indices = np.where(~labeled_mask)[0]
        
        if len(unlabeled_indices) == 0:
            return confidences
        
        if self.use_silhouette_for_confidence:
            # Use silhouette coefficient as confidence measure
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for idx in unlabeled_indices:
                    cluster_id = cluster_labels[idx]
                    if cluster_id == -1:  # Noise point in DBSCAN
                        confidences[idx] = 0.0
                        continue
                    
                    # Calculate silhouette for this sample
                    same_cluster = (cluster_labels == cluster_id)
                    if same_cluster.sum() <= 1:
                        confidences[idx] = 0.0
                        continue
                    
                    # Distance to same cluster samples
                    same_cluster_emb = embeddings[same_cluster]
                    a = np.mean(cosine_similarity(
                        embeddings[idx].reshape(1, -1),
                        same_cluster_emb
                    ))
                    
                    # Distance to nearest different cluster
                    other_clusters = np.unique(cluster_labels[cluster_labels != cluster_id])
                    if len(other_clusters) == 0:
                        confidences[idx] = a
                        continue
                    
                    b_values = []
                    for other_cluster_id in other_clusters:
                        if other_cluster_id == -1:
                            continue
                        other_cluster = (cluster_labels == other_cluster_id)
                        if other_cluster.sum() > 0:
                            other_cluster_emb = embeddings[other_cluster]
                            b_val = np.mean(cosine_similarity(
                                embeddings[idx].reshape(1, -1),
                                other_cluster_emb
                            ))
                            b_values.append(b_val)
                    
                    if len(b_values) > 0:
                        b = min(b_values)
                        # Normalize silhouette to [0, 1]
                        confidences[idx] = (a - b) / max(a, b) if max(a, b) > 0 else 0.0
                        confidences[idx] = (confidences[idx] + 1) / 2  # Map [-1, 1] to [0, 1]
                    else:
                        confidences[idx] = a
        else:
            # Use distance to cluster center
            cluster_centers = self.clusterer.get_cluster_centers(embeddings, cluster_labels)
            
            for idx in unlabeled_indices:
                cluster_id = cluster_labels[idx]
                if cluster_id == -1 or cluster_id not in cluster_centers:
                    confidences[idx] = 0.0
                    continue
                
                center = cluster_centers[cluster_id]
                # Cosine similarity (higher is better)
                similarity = cosine_similarity(
                    embeddings[idx].reshape(1, -1),
                    center.reshape(1, -1)
                )[0, 0]
                
                # Map to [0, 1] where 1 is most confident
                confidences[idx] = (similarity + 1) / 2
        
        return confidences
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict labels for new samples.
        
        Args:
            embeddings (np.ndarray): Sentence embeddings
            
        Returns:
            np.ndarray: Predicted labels
        """
        if self.cluster_to_label_mapping is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Predict clusters
        cluster_labels = self.clusterer.predict(embeddings)
        
        # Map to labels
        predictions = np.array([
            self.cluster_to_label_mapping.get(c, -1)
            for c in cluster_labels
        ])
        
        return predictions
    
    def get_history_dataframe(self) -> pd.DataFrame:
        """Get training history as DataFrame."""
        return pd.DataFrame(self.history)
    
    def save_results(self, output_dir: Path, prefix: str = ""):
        """
        Save self-training results to disk.
        
        Args:
            output_dir (Path): Output directory
            prefix (str): Prefix for filenames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save history
        history_df = self.get_history_dataframe()
        history_path = output_dir / f"{prefix}self_training_history.csv"
        history_df.to_csv(history_path, index=False)
        print(f"Saved history to {history_path}")
        
        # Save cluster mapping
        mapping_df = pd.DataFrame([
            {'cluster_id': k, 'label': v}
            for k, v in self.cluster_to_label_mapping.items()
        ])
        mapping_path = output_dir / f"{prefix}cluster_label_mapping.csv"
        mapping_df.to_csv(mapping_path, index=False)
        print(f"Saved cluster-label mapping to {mapping_path}")


def run_self_training(
    embeddings_train: np.ndarray,
    labels_train: np.ndarray,
    embeddings_unlabeled: np.ndarray,
    config: dict,
    texts_train: Optional[List[str]] = None,
    texts_unlabeled: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[SelfTrainingClassifier, np.ndarray]:
    """
    Run semi-supervised self-training.
    
    Args:
        embeddings_train (np.ndarray): Embeddings for labeled training data
        labels_train (np.ndarray): Labels for training data
        embeddings_unlabeled (np.ndarray): Embeddings for unlabeled data
        config (dict): Configuration dictionary
        texts_train (List[str], optional): Training texts
        texts_unlabeled (List[str], optional): Unlabeled texts
        verbose (bool): Print progress
        
    Returns:
        Tuple[SelfTrainingClassifier, np.ndarray]: Fitted classifier and pseudo-labels
    """
    # Combine labeled and unlabeled data
    all_embeddings = np.vstack([embeddings_train, embeddings_unlabeled])
    all_labels = np.concatenate([
        labels_train,
        np.full(len(embeddings_unlabeled), -1)  # Placeholder for unlabeled
    ])
    labeled_mask = np.concatenate([
        np.ones(len(embeddings_train), dtype=bool),
        np.zeros(len(embeddings_unlabeled), dtype=bool)
    ])
    
    all_texts = None
    if texts_train is not None and texts_unlabeled is not None:
        all_texts = texts_train + texts_unlabeled
    
    # Get self-training config
    st_config = config.get('semi_supervised', {})
    
    # Initialize classifier
    classifier = SelfTrainingClassifier(
        clustering_method=st_config.get('clustering_method', 'kmeans'),
        n_clusters=st_config.get('n_clusters', 3),
        confidence_threshold=st_config.get('confidence_threshold', 0.8),
        max_iterations=st_config.get('max_iterations', 5),
        min_samples_per_iteration=st_config.get('min_samples_per_iteration', 10),
        use_silhouette_for_confidence=st_config.get('use_silhouette_for_confidence', True),
        random_state=config.get('random_seed', 42)
    )
    
    # Fit classifier
    classifier.fit(
        all_embeddings,
        all_labels,
        labeled_mask,
        texts=all_texts,
        verbose=verbose
    )
    
    # Get pseudo-labels for unlabeled data
    pseudo_labels = classifier.final_labels[~labeled_mask]
    
    return classifier, pseudo_labels
