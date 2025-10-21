"""
Embedding Model Comparison Tool
================================
Compares different embedding models for clustering quality using:
- Silhouette Score
- t-SNE visualization
- Cluster purity
- Davies-Bouldin Index
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, confusion_matrix
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches

from embeddings import EmbeddingGenerator
from clustering import KMeansClustering
from evaluation import calculate_cluster_purity


class EmbeddingComparison:
    """Compare different embedding models for clustering tasks."""
    
    def __init__(self, output_dir='outputs/comparison'):
        """
        Initialize the comparison tool.
        
        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        
        self.results = {}
        
    def compare_models(
        self,
        texts: List[str],
        labels: np.ndarray,
        model_names: List[str],
        n_clusters: int = 12,
        batch_size: int = 32
    ) -> pd.DataFrame:
        """
        Compare multiple embedding models.
        
        Args:
            texts: List of text samples
            labels: True labels for the samples
            model_names: List of model names to compare
            n_clusters: Number of clusters for K-Means
            batch_size: Batch size for encoding
            
        Returns:
            DataFrame with comparison results
        """
        print("=" * 70)
        print("EMBEDDING MODEL COMPARISON")
        print("=" * 70)
        print(f"Number of samples: {len(texts)}")
        print(f"Number of models: {len(model_names)}")
        print(f"Number of clusters: {n_clusters}")
        print("=" * 70)
        
        results_list = []
        
        for model_name in model_names:
            print(f"\n{'='*70}")
            print(f"Evaluating model: {model_name}")
            print(f"{'='*70}")
            
            # Generate embeddings
            generator = EmbeddingGenerator(model_name)
            embeddings = generator.encode(texts, batch_size=batch_size, show_progress=True)
            
            # Perform clustering
            print("\nClustering...")
            clusterer = KMeansClustering(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(embeddings)
            
            # Calculate metrics
            print("Calculating metrics...")
            silhouette = silhouette_score(embeddings, cluster_labels)
            davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
            purity = calculate_cluster_purity(cluster_labels, labels)
            
            # Calculate inertia (within-cluster sum of squares)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(embeddings)
            inertia = kmeans.inertia_
            
            # Store results
            result = {
                'model': model_name,
                'embedding_dim': generator.get_embedding_dim(),
                'silhouette_score': silhouette,
                'davies_bouldin_index': davies_bouldin,
                'cluster_purity': purity,
                'inertia': inertia,
                'n_clusters': n_clusters
            }
            
            results_list.append(result)
            
            # Store for visualization
            self.results[model_name] = {
                'embeddings': embeddings,
                'cluster_labels': cluster_labels,
                'true_labels': labels,
                'metrics': result
            }
            
            print(f"\nResults for {model_name}:")
            print(f"  Silhouette Score: {silhouette:.4f}")
            print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}")
            print(f"  Cluster Purity: {purity:.4f}")
            print(f"  Inertia: {inertia:.2f}")
        
        # Create DataFrame
        df_results = pd.DataFrame(results_list)
        
        # Save results
        csv_path = self.output_dir / 'model_comparison.csv'
        df_results.to_csv(csv_path, index=False)
        print(f"\n{'='*70}")
        print(f"Saved comparison results to: {csv_path}")
        print(f"{'='*70}")
        
        return df_results
    
    def create_tsne_visualization(
        self,
        model_names: List[str] = None,
        perplexity: int = 30,
        n_iter: int = 1000,
        random_state: int = 42
    ):
        """
        Create t-SNE visualizations for each model.
        
        Args:
            model_names: List of model names to visualize (None = all)
            perplexity: t-SNE perplexity parameter
            n_iter: Number of iterations for t-SNE
            random_state: Random state for reproducibility
        """
        if model_names is None:
            model_names = list(self.results.keys())
        
        n_models = len(model_names)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 12))
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('t-SNE Visualization: Embedding Comparison', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        class_names = {0: 'Hate', 1: 'Offensive', 2: 'Neither'}
        colors = ['#E63946', '#F18F01', '#06A77D']
        
        for idx, model_name in enumerate(model_names):
            if model_name not in self.results:
                print(f"Warning: No results found for {model_name}")
                continue
            
            data = self.results[model_name]
            embeddings = data['embeddings']
            cluster_labels = data['cluster_labels']
            true_labels = data['true_labels']
            
            print(f"\nRunning t-SNE for {model_name}...")
            
            # Apply t-SNE
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                max_iter=n_iter,
                random_state=random_state,
                verbose=1
            )
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # Plot 1: Colored by true labels
            ax1 = axes[0, idx]
            for label, color in zip([0, 1, 2], colors):
                mask = true_labels == label
                ax1.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c=color,
                    label=class_names[label],
                    alpha=0.6,
                    s=20,
                    edgecolors='black',
                    linewidths=0.5
                )
            
            ax1.set_title(f'{model_name}\n(Colored by True Labels)', 
                         fontsize=12, fontweight='bold')
            ax1.set_xlabel('t-SNE Component 1', fontsize=10)
            ax1.set_ylabel('t-SNE Component 2', fontsize=10)
            ax1.legend(loc='best', frameon=True, shadow=True)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Colored by cluster assignments
            ax2 = axes[1, idx]
            
            # Use different colors for clusters
            n_clusters = len(np.unique(cluster_labels))
            cluster_colors = sns.color_palette("husl", n_clusters)
            
            for cluster_id in range(n_clusters):
                mask = cluster_labels == cluster_id
                ax2.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c=[cluster_colors[cluster_id]],
                    label=f'Cluster {cluster_id}',
                    alpha=0.6,
                    s=20,
                    edgecolors='black',
                    linewidths=0.5
                )
            
            ax2.set_title(f'{model_name}\n(Colored by Clusters)', 
                         fontsize=12, fontweight='bold')
            ax2.set_xlabel('t-SNE Component 1', fontsize=10)
            ax2.set_ylabel('t-SNE Component 2', fontsize=10)
            
            # Only show legend if few clusters
            if n_clusters <= 12:
                ax2.legend(loc='best', frameon=True, shadow=True, 
                          ncol=2, fontsize=8)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        tsne_path = self.output_dir / 'tsne_comparison.png'
        plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved t-SNE visualization to: {tsne_path}")
        plt.close()
    
    def create_annotated_tsne_visualization(
        self,
        model_names: List[str] = None,
        perplexity: int = 30,
        n_iter: int = 1000,
        random_state: int = 42,
        annotate_centroids: bool = True
    ):
        """
        Create enhanced t-SNE visualizations with cluster centroid annotations.
        
        Args:
            model_names: List of model names to visualize (None = all)
            perplexity: t-SNE perplexity parameter
            n_iter: Number of iterations for t-SNE
            random_state: Random state for reproducibility
            annotate_centroids: Whether to annotate cluster centroids
        """
        if model_names is None:
            model_names = list(self.results.keys())
        
        for model_name in model_names:
            if model_name not in self.results:
                print(f"Warning: No results found for {model_name}")
                continue
            
            data = self.results[model_name]
            embeddings = data['embeddings']
            cluster_labels = data['cluster_labels']
            true_labels = data['true_labels']
            
            print(f"\nCreating annotated t-SNE for {model_name}...")
            
            # Apply t-SNE
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                max_iter=n_iter,
                random_state=random_state,
                verbose=0
            )
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # Create figure with 3 subplots
            fig = plt.figure(figsize=(20, 6))
            
            class_names = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
            class_colors = ['#E63946', '#F18F01', '#06A77D']
            
            # Subplot 1: Colored by true labels
            ax1 = plt.subplot(131)
            for label, color in zip([0, 1, 2], class_colors):
                mask = true_labels == label
                ax1.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c=color,
                    label=class_names[label],
                    alpha=0.5,
                    s=30,
                    edgecolors='black',
                    linewidths=0.3
                )
            
            ax1.set_title(f'{model_name}\nTrue Labels', 
                         fontsize=13, fontweight='bold', pad=10)
            ax1.set_xlabel('t-SNE Component 1', fontsize=11, fontweight='bold')
            ax1.set_ylabel('t-SNE Component 2', fontsize=11, fontweight='bold')
            ax1.legend(loc='best', frameon=True, shadow=True, fontsize=10)
            ax1.grid(True, alpha=0.2, linestyle='--')
            
            # Subplot 2: Colored by clusters with centroids
            ax2 = plt.subplot(132)
            n_clusters = len(np.unique(cluster_labels))
            cluster_colors = sns.color_palette("husl", n_clusters)
            
            for cluster_id in range(n_clusters):
                mask = cluster_labels == cluster_id
                cluster_points = embeddings_2d[mask]
                
                ax2.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    c=[cluster_colors[cluster_id]],
                    label=f'C{cluster_id}',
                    alpha=0.5,
                    s=30,
                    edgecolors='black',
                    linewidths=0.3
                )
                
                # Calculate and plot centroid
                if annotate_centroids and len(cluster_points) > 0:
                    centroid = cluster_points.mean(axis=0)
                    ax2.scatter(
                        centroid[0], centroid[1],
                        c='white',
                        s=200,
                        marker='*',
                        edgecolors='black',
                        linewidths=2,
                        zorder=100
                    )
                    ax2.annotate(
                        f'C{cluster_id}',
                        xy=(centroid[0], centroid[1]),
                        fontsize=10,
                        fontweight='bold',
                        ha='center',
                        va='center',
                        zorder=101
                    )
            
            ax2.set_title(f'{model_name}\nCluster Assignments (with centroids)', 
                         fontsize=13, fontweight='bold', pad=10)
            ax2.set_xlabel('t-SNE Component 1', fontsize=11, fontweight='bold')
            ax2.set_ylabel('t-SNE Component 2', fontsize=11, fontweight='bold')
            if n_clusters <= 12:
                ax2.legend(loc='best', frameon=True, shadow=True, 
                          ncol=3, fontsize=8)
            ax2.grid(True, alpha=0.2, linestyle='--')
            
            # Subplot 3: Mixed view - cluster boundaries with true label colors
            ax3 = plt.subplot(133)
            
            # Plot points colored by true labels
            for label, color in zip([0, 1, 2], class_colors):
                mask = true_labels == label
                ax3.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c=color,
                    alpha=0.4,
                    s=30,
                    edgecolors='none'
                )
            
            # Add cluster boundaries using convex hulls
            from scipy.spatial import ConvexHull
            for cluster_id in range(n_clusters):
                mask = cluster_labels == cluster_id
                cluster_points = embeddings_2d[mask]
                
                if len(cluster_points) >= 3:  # Need at least 3 points for hull
                    try:
                        hull = ConvexHull(cluster_points)
                        for simplex in hull.simplices:
                            ax3.plot(
                                cluster_points[simplex, 0],
                                cluster_points[simplex, 1],
                                'k-',
                                linewidth=1.5,
                                alpha=0.3
                            )
                    except:
                        pass  # Skip if hull fails
            
            # Create custom legend
            legend_elements = [
                mpatches.Patch(facecolor=class_colors[0], label='Hate Speech', alpha=0.6),
                mpatches.Patch(facecolor=class_colors[1], label='Offensive Language', alpha=0.6),
                mpatches.Patch(facecolor=class_colors[2], label='Neither', alpha=0.6),
                mpatches.Patch(facecolor='white', edgecolor='black', 
                              label='Cluster Boundaries', alpha=0.3)
            ]
            
            ax3.set_title(f'{model_name}\nOverlay: True Labels + Cluster Boundaries', 
                         fontsize=13, fontweight='bold', pad=10)
            ax3.set_xlabel('t-SNE Component 1', fontsize=11, fontweight='bold')
            ax3.set_ylabel('t-SNE Component 2', fontsize=11, fontweight='bold')
            ax3.legend(handles=legend_elements, loc='best', frameon=True, 
                      shadow=True, fontsize=10)
            ax3.grid(True, alpha=0.2, linestyle='--')
            
            plt.tight_layout()
            
            # Save figure
            tsne_annotated_path = self.output_dir / f'tsne_annotated_{model_name}.png'
            plt.savefig(tsne_annotated_path, dpi=300, bbox_inches='tight')
            print(f"Saved annotated t-SNE to: {tsne_annotated_path}")
            plt.close()
    
    def create_cluster_confusion_matrices(self, model_names: List[str] = None):
        """
        Create confusion matrices comparing cluster assignments to ground truth labels.
        
        Args:
            model_names: List of model names to analyze (None = all)
        """
        if model_names is None:
            model_names = list(self.results.keys())
        
        n_models = len(model_names)
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 7))
        
        # Ensure axes is iterable
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle('Cluster-to-Label Confusion Matrices', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        for idx, model_name in enumerate(model_names):
            if model_name not in self.results:
                print(f"Warning: No results found for {model_name}")
                continue
            
            data = self.results[model_name]
            cluster_labels = data['cluster_labels']
            true_labels = data['true_labels']
            
            # Create confusion matrix
            cm = confusion_matrix(true_labels, cluster_labels)
            
            # Get dimensions
            n_true_labels = cm.shape[0]
            n_clusters = cm.shape[1]
            
            # Normalize by true label (rows)
            # Handle case where a row might be all zeros
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            cm_normalized = cm.astype('float') / row_sums
            
            # Plot using seaborn heatmap for better handling
            ax = axes[idx]
            
            # Set labels
            class_names = ['Hate Speech', 'Offensive', 'Neither']
            
            # Use seaborn heatmap
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt='.2f',
                cmap='YlOrRd',
                vmin=0,
                vmax=1,
                xticklabels=[f'C{i}' for i in range(n_clusters)],
                yticklabels=class_names[:n_true_labels],
                cbar_kws={'label': 'Proportion of True Label'},
                ax=ax,
                linewidths=0.5,
                linecolor='gray'
            )
            
            # Rotate x labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            ax.set_xlabel('Cluster Assignment', fontsize=11, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
            ax.set_title(f'{model_name}\n(Normalized by True Label)', 
                        fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        confusion_path = self.output_dir / 'cluster_confusion_matrices.png'
        plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved cluster confusion matrices to: {confusion_path}")
        plt.close()
        
        # Also create detailed analysis
        self._create_confusion_analysis(model_names)
    
    def _create_confusion_analysis(self, model_names: List[str]):
        """Create detailed text analysis of cluster-to-label mappings."""
        analysis = """
======================================================================
CLUSTER-TO-LABEL CONFUSION ANALYSIS
======================================================================
"""
        
        class_names = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
        
        for model_name in model_names:
            if model_name not in self.results:
                continue
            
            data = self.results[model_name]
            cluster_labels = data['cluster_labels']
            true_labels = data['true_labels']
            
            analysis += f"\n{'='*70}\n"
            analysis += f"MODEL: {model_name}\n"
            analysis += f"{'='*70}\n\n"
            
            # Create confusion matrix
            cm = confusion_matrix(true_labels, cluster_labels)
            n_clusters = cm.shape[1]
            
            # Analyze each cluster
            analysis += "CLUSTER COMPOSITION:\n"
            analysis += "-" * 70 + "\n"
            
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_size = cluster_mask.sum()
                
                if cluster_size == 0:
                    continue
                
                analysis += f"\nCluster {cluster_id} ({cluster_size} samples):\n"
                
                # Count each true label in this cluster
                for true_label in [0, 1, 2]:
                    count = ((cluster_labels == cluster_id) & (true_labels == true_label)).sum()
                    pct = (count / cluster_size) * 100
                    analysis += f"  {class_names[true_label]}: {count} ({pct:.1f}%)\n"
                
                # Determine dominant label
                cluster_true_labels = true_labels[cluster_mask]
                dominant_label = np.bincount(cluster_true_labels).argmax()
                dominant_pct = (cluster_true_labels == dominant_label).sum() / cluster_size * 100
                analysis += f"  → Dominant: {class_names[dominant_label]} ({dominant_pct:.1f}%)\n"
            
            # Overall statistics
            analysis += f"\n{'='*70}\n"
            analysis += "LABEL DISTRIBUTION ACROSS CLUSTERS:\n"
            analysis += "-" * 70 + "\n"
            
            for true_label in [0, 1, 2]:
                label_mask = true_labels == true_label
                label_count = label_mask.sum()
                
                analysis += f"\n{class_names[true_label]} ({label_count} samples):\n"
                
                # Which clusters contain this label?
                clusters_with_label = []
                for cluster_id in range(n_clusters):
                    count = ((cluster_labels == cluster_id) & (true_labels == true_label)).sum()
                    if count > 0:
                        pct = (count / label_count) * 100
                        clusters_with_label.append((cluster_id, count, pct))
                
                # Sort by count
                clusters_with_label.sort(key=lambda x: x[1], reverse=True)
                
                for cluster_id, count, pct in clusters_with_label:
                    analysis += f"  Cluster {cluster_id}: {count} samples ({pct:.1f}%)\n"
            
            analysis += "\n"
        
        # Save analysis
        analysis_path = self.output_dir / 'cluster_confusion_analysis.txt'
        with open(analysis_path, 'w') as f:
            f.write(analysis)
        
        print(f"Saved cluster confusion analysis to: {analysis_path}")
        print(analysis)
    
    def create_metrics_comparison_plot(self, results_df: pd.DataFrame):
        """
        Create bar plots comparing metrics across models.
        
        Args:
            results_df: DataFrame with comparison results
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Embedding Model Metrics Comparison', 
                     fontsize=16, fontweight='bold')
        
        models = results_df['model'].values
        colors = sns.color_palette("husl", len(models))
        
        # Silhouette Score (higher is better)
        ax1 = axes[0, 0]
        bars1 = ax1.bar(models, results_df['silhouette_score'], 
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
        ax1.set_title('Silhouette Score (Higher = Better)', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Davies-Bouldin Index (lower is better)
        ax2 = axes[0, 1]
        bars2 = ax2.bar(models, results_df['davies_bouldin_index'], 
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Davies-Bouldin Index', fontsize=11, fontweight='bold')
        ax2.set_title('Davies-Bouldin Index (Lower = Better)', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Cluster Purity (higher is better)
        ax3 = axes[1, 0]
        bars3 = ax3.bar(models, results_df['cluster_purity'], 
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Cluster Purity', fontsize=11, fontweight='bold')
        ax3.set_title('Cluster Purity (Higher = Better)', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Inertia (lower is generally better, but depends on scale)
        ax4 = axes[1, 1]
        bars4 = ax4.bar(models, results_df['inertia'], 
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Inertia', fontsize=11, fontweight='bold')
        ax4.set_title('Within-Cluster Sum of Squares', fontsize=12, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add value labels
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        
        # Save figure
        metrics_path = self.output_dir / 'metrics_comparison.png'
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics comparison to: {metrics_path}")
        plt.close()
    
    def create_ranking_table(self, results_df: pd.DataFrame):
        """
        Create a ranking table showing best models for each metric.
        
        Args:
            results_df: DataFrame with comparison results
        """
        # Create ranking DataFrame
        rankings = pd.DataFrame()
        
        # Silhouette (higher is better)
        rankings['Silhouette Score'] = results_df.sort_values(
            'silhouette_score', ascending=False
        )['model'].reset_index(drop=True)
        
        # Davies-Bouldin (lower is better)
        rankings['Davies-Bouldin'] = results_df.sort_values(
            'davies_bouldin_index', ascending=True
        )['model'].reset_index(drop=True)
        
        # Cluster Purity (higher is better)
        rankings['Cluster Purity'] = results_df.sort_values(
            'cluster_purity', ascending=False
        )['model'].reset_index(drop=True)
        
        # Add rank numbers
        rankings.index = [f'Rank {i+1}' for i in range(len(rankings))]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=rankings.values,
                        colLabels=rankings.columns,
                        rowLabels=rankings.index,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.3, 0.3, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(len(rankings.columns)):
            cell = table[(0, i)]
            cell.set_facecolor('#2E86AB')
            cell.set_text_props(weight='bold', color='white')
        
        # Style row labels
        for i in range(len(rankings)):
            cell = table[(i+1, -1)]
            cell.set_facecolor('#A23B72')
            cell.set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(rankings) + 1):
            for j in range(len(rankings.columns)):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#F0F0F0')
                else:
                    cell.set_facecolor('white')
                
                # Highlight best (rank 1)
                if i == 1:
                    cell.set_facecolor('#90EE90')
                    cell.set_text_props(weight='bold')
        
        plt.title('Model Rankings by Metric\n(Rank 1 = Best)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        ranking_path = self.output_dir / 'model_rankings.png'
        plt.savefig(ranking_path, dpi=300, bbox_inches='tight')
        print(f"Saved model rankings to: {ranking_path}")
        plt.close()
        
        # Also save as CSV
        csv_path = self.output_dir / 'model_rankings.csv'
        rankings.to_csv(csv_path)
        print(f"Saved rankings CSV to: {csv_path}")
    
    def generate_comparison_report(self, results_df: pd.DataFrame):
        """
        Generate a comprehensive comparison report.
        
        Args:
            results_df: DataFrame with comparison results
        """
        report = f"""
{'='*70}
EMBEDDING MODEL COMPARISON REPORT
{'='*70}

MODELS EVALUATED
----------------
"""
        for _, row in results_df.iterrows():
            report += f"- {row['model']} (dim: {row['embedding_dim']})\n"
        
        report += f"""
CLUSTERING CONFIGURATION
------------------------
- Number of clusters: {results_df['n_clusters'].iloc[0]}
- Clustering algorithm: K-Means

RESULTS SUMMARY
---------------
"""
        
        # Add metrics table
        report += "\n" + results_df.to_string(index=False) + "\n"
        
        # Find best models
        report += f"""
BEST MODELS BY METRIC
---------------------
"""
        
        best_silhouette = results_df.loc[results_df['silhouette_score'].idxmax()]
        report += f"Silhouette Score: {best_silhouette['model']} ({best_silhouette['silhouette_score']:.4f})\n"
        
        best_db = results_df.loc[results_df['davies_bouldin_index'].idxmin()]
        report += f"Davies-Bouldin Index: {best_db['model']} ({best_db['davies_bouldin_index']:.4f})\n"
        
        best_purity = results_df.loc[results_df['cluster_purity'].idxmax()]
        report += f"Cluster Purity: {best_purity['model']} ({best_purity['cluster_purity']:.4f})\n"
        
        report += f"""
INTERPRETATION
--------------
- Silhouette Score: Measures how similar samples are to their own cluster
  compared to other clusters. Range: [-1, 1]. Higher is better.
  
- Davies-Bouldin Index: Ratio of within-cluster to between-cluster distances.
  Lower values indicate better clustering.
  
- Cluster Purity: Percentage of samples in each cluster that belong to the
  most common true label. Higher is better.

RECOMMENDATIONS
---------------
"""
        
        # Determine overall best model
        # Normalize metrics and compute average rank
        df_copy = results_df.copy()
        df_copy['silhouette_rank'] = df_copy['silhouette_score'].rank(ascending=False)
        df_copy['db_rank'] = df_copy['davies_bouldin_index'].rank(ascending=True)
        df_copy['purity_rank'] = df_copy['cluster_purity'].rank(ascending=False)
        df_copy['avg_rank'] = (df_copy['silhouette_rank'] + df_copy['db_rank'] + df_copy['purity_rank']) / 3
        
        best_overall = df_copy.loc[df_copy['avg_rank'].idxmin()]
        
        report += f"Overall Best Model: {best_overall['model']}\n"
        report += f"  - Average Rank: {best_overall['avg_rank']:.2f}\n"
        report += f"  - Silhouette Score: {best_overall['silhouette_score']:.4f}\n"
        report += f"  - Davies-Bouldin Index: {best_overall['davies_bouldin_index']:.4f}\n"
        report += f"  - Cluster Purity: {best_overall['cluster_purity']:.4f}\n"
        
        report += f"""
{'='*70}
"""
        
        # Save report
        report_path = self.output_dir / 'comparison_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"Saved comparison report to: {report_path}")


def main():
    """Main function to run embedding comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare embedding models for clustering')
    parser.add_argument('--data-path', type=str, default='data/loaders/train.csv',
                       help='Path to data file')
    parser.add_argument('--sample-size', type=int, default=5000,
                       help='Number of samples to use (for speed)')
    parser.add_argument('--n-clusters', type=int, default=12,
                       help='Number of clusters')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['all-MiniLM-L6-v2', 'simcse-bert'],
                       help='Models to compare')
    parser.add_argument('--output-dir', type=str, default='outputs/comparison',
                       help='Output directory')
    parser.add_argument('--skip-tsne', action='store_true',
                       help='Skip t-SNE visualization (saves time)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    
    # Sample data if needed
    if len(df) > args.sample_size:
        print(f"Sampling {args.sample_size} samples from {len(df)} total...")
        df = df.sample(n=args.sample_size, random_state=42)
    
    texts = df['tweet'].tolist()
    labels = df['label'].values
    
    print(f"Using {len(texts)} samples")
    print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    # Create comparison tool
    comparison = EmbeddingComparison(output_dir=args.output_dir)
    
    # Compare models
    results_df = comparison.compare_models(
        texts=texts,
        labels=labels,
        model_names=args.models,
        n_clusters=args.n_clusters
    )
    
    # Create visualizations
    print("\nCreating visualizations...")
    comparison.create_metrics_comparison_plot(results_df)
    comparison.create_ranking_table(results_df)
    
    # Create cluster confusion matrices
    print("\nCreating cluster confusion matrices...")
    comparison.create_cluster_confusion_matrices(model_names=args.models)
    
    if not args.skip_tsne:
        comparison.create_tsne_visualization(model_names=args.models)
        # Create annotated t-SNE visualizations
        print("\nCreating annotated t-SNE visualizations...")
        comparison.create_annotated_tsne_visualization(model_names=args.models)
    else:
        print("Skipping t-SNE visualization (use --skip-tsne=False to enable)")
    
    # Generate report
    comparison.generate_comparison_report(results_df)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print(f"Results saved to: {args.output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
