"""
Report Visualization Module
============================
Creates publication-quality visualizations and tables for academic reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


class ReportGenerator:
    """Generate comprehensive visualizations and tables for research reports."""
    
    def __init__(self, output_dir='outputs/report'):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory to save report figures and tables
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set publication-quality style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        
    def create_comprehensive_report(self, results_dir='outputs/results'):
        """
        Create a comprehensive report with all visualizations and tables.
        
        Args:
            results_dir: Directory containing result files
        """
        results_dir = Path(results_dir)
        
        print("=" * 70)
        print("GENERATING COMPREHENSIVE REPORT")
        print("=" * 70)
        
        # Load data
        history = pd.read_csv(results_dir / 'self_training_history.csv')
        test_metrics = pd.read_csv(results_dir / 'test_metrics.csv')
        test_predictions = pd.read_csv(results_dir / 'test_predictions.csv')
        cluster_mapping = pd.read_csv(results_dir / 'cluster_label_mapping.csv')
        
        # Generate visualizations
        print("\n1. Creating multi-panel results figure...")
        self.create_multipanel_figure(history, test_predictions)
        
        print("2. Creating cluster analysis visualization...")
        self.create_cluster_analysis_figure(cluster_mapping, history)
        
        print("3. Creating performance comparison table...")
        self.create_performance_table(test_metrics, history)
        
        print("4. Creating confusion matrix heatmap...")
        self.create_enhanced_confusion_matrix(test_predictions)
        
        print("5. Creating pseudo-labeling analysis...")
        self.create_pseudo_label_analysis(history, test_predictions)
        
        print("6. Generating LaTeX tables...")
        self.generate_latex_tables(test_metrics, history)
        
        print("7. Creating executive summary...")
        self.create_executive_summary(test_metrics, history, test_predictions)
        
        print("\n" + "=" * 70)
        print(f"REPORT GENERATED SUCCESSFULLY")
        print(f"Output directory: {self.output_dir}")
        print("=" * 70)
        
    def create_multipanel_figure(self, history, predictions):
        """Create a 2x2 multi-panel figure showing key results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Semi-Supervised Self-Training Results', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Panel A: Training progression
        ax1 = axes[0, 0]
        iterations = history['iteration'].values
        labeled = history['n_labeled'].values
        ax1.plot(iterations, labeled, marker='o', linewidth=2, 
                markersize=8, color='#2E86AB')
        ax1.fill_between(iterations, labeled, alpha=0.3, color='#2E86AB')
        ax1.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Labeled Samples', fontsize=11, fontweight='bold')
        ax1.set_title('A) Self-Training Progression', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xticks(iterations)
        
        # Panel B: Confidence scores
        ax2 = axes[0, 1]
        confidence = history['avg_confidence'].values
        ax2.plot(iterations, confidence, marker='s', linewidth=2, 
                markersize=8, color='#A23B72')
        ax2.axhline(y=0.65, color='red', linestyle='--', 
                   label='Threshold (0.65)', linewidth=2)
        ax2.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Average Confidence', fontsize=11, fontweight='bold')
        ax2.set_title('B) Pseudo-Label Confidence', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', frameon=True, shadow=True)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xticks(iterations)
        ax2.set_ylim([0, 1])
        
        # Panel C: Clustering quality
        ax3 = axes[1, 0]
        purity = history['cluster_purity'].values
        silhouette = history['silhouette_score'].values
        
        ax3_twin = ax3.twinx()
        l1 = ax3.plot(iterations, purity, marker='o', linewidth=2, 
                     markersize=8, color='#F18F01', label='Cluster Purity')
        l2 = ax3_twin.plot(iterations, silhouette, marker='^', linewidth=2, 
                          markersize=8, color='#06A77D', label='Silhouette Score')
        
        ax3.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Cluster Purity', fontsize=11, fontweight='bold', color='#F18F01')
        ax3_twin.set_ylabel('Silhouette Score', fontsize=11, fontweight='bold', color='#06A77D')
        ax3.set_title('C) Clustering Quality Metrics', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='y', labelcolor='#F18F01')
        ax3_twin.tick_params(axis='y', labelcolor='#06A77D')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_xticks(iterations)
        
        # Combine legends
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax3.legend(lns, labs, loc='best', frameon=True, shadow=True)
        
        # Panel D: Class distribution in predictions
        ax4 = axes[1, 1]
        class_counts = predictions['predicted_label'].value_counts().sort_index()
        class_names = ['Hate Speech', 'Offensive', 'Neither']
        colors = ['#E63946', '#F18F01', '#06A77D']
        
        bars = ax4.bar(range(len(class_counts)), class_counts.values, 
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax4.set_xlabel('Class', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Number of Predictions', fontsize=11, fontweight='bold')
        ax4.set_title('D) Test Set Prediction Distribution', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(len(class_names)))
        ax4.set_xticklabels(class_names, rotation=15, ha='right')
        ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / 'multipanel_results.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   Saved to: {output_path}")
        plt.close()
        
    def create_cluster_analysis_figure(self, cluster_mapping, history):
        """Create detailed cluster analysis visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Cluster-to-Label Mapping Analysis', 
                     fontsize=14, fontweight='bold')
        
        # Panel 1: Cluster mapping distribution
        ax1 = axes[0]
        label_names = {0: 'Hate', 1: 'Offensive', 2: 'Neither'}
        
        # Get mapping (no iterations in this file)
        mapped_labels = cluster_mapping['label'].values
        label_counts = pd.Series(mapped_labels).value_counts().sort_index()
        
        colors_map = {0: '#E63946', 1: '#F18F01', 2: '#06A77D'}
        colors = [colors_map[i] for i in label_counts.index]
        
        bars = ax1.bar([label_names[i] for i in label_counts.index], 
                      label_counts.values, 
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Assigned Label', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Number of Clusters', fontsize=11, fontweight='bold')
        ax1.set_title('Clusters per Label Category', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Panel 2: Pseudo-label distribution over iterations
        ax2 = axes[1]
        
        if len(history) > 1:
            # Get pseudo-label distributions
            iter_data = []
            for idx, row in history.iterrows():
                if row['n_pseudo_labeled'] > 0:
                    # Parse the pseudo_label_dist if it exists
                    iter_data.append({
                        'iteration': row['iteration'],
                        'count': row['n_pseudo_labeled']
                    })
            
            if iter_data:
                df_iters = pd.DataFrame(iter_data)
                ax2.plot(df_iters['iteration'], df_iters['count'], 
                        marker='o', linewidth=2, markersize=8, color='#2E86AB')
                ax2.fill_between(df_iters['iteration'], df_iters['count'], 
                                alpha=0.3, color='#2E86AB')
                ax2.set_xlabel('Iteration', fontsize=11, fontweight='bold')
                ax2.set_ylabel('Pseudo-Labels Added', fontsize=11, fontweight='bold')
                ax2.set_title('Pseudo-Labeling Progress', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3, linestyle='--')
                ax2.set_xticks(df_iters['iteration'])
        
        plt.tight_layout()
        output_path = self.output_dir / 'cluster_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   Saved to: {output_path}")
        plt.close()
        
    def create_performance_table(self, metrics, history):
        """Create a formatted performance summary table."""
        # Extract key metrics
        final_iter = history.iloc[-1]
        
        data = {
            'Metric': [
                'Test Accuracy',
                'Macro F1-Score',
                'Weighted F1-Score',
                'Cluster Purity',
                'Silhouette Score',
                'Total Iterations',
                'Final Labeled Samples',
                'Total Pseudo-Labels',
                'Average Confidence'
            ],
            'Value': [
                f"{metrics['accuracy'].values[0]:.4f}",
                f"{metrics['macro_f1'].values[0]:.4f}",
                f"{metrics['weighted_f1'].values[0]:.4f}",
                f"{final_iter['cluster_purity']:.4f}",
                f"{final_iter['silhouette_score']:.4f}",
                f"{final_iter['iteration']:.0f}",
                f"{final_iter['n_labeled']:.0f}",
                f"{history['n_pseudo_labeled'].sum():.0f}",
                f"{final_iter['avg_confidence']:.4f}"
            ]
        }
        
        df = pd.DataFrame(data)
        
        # Create styled table
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values,
                        colLabels=df.columns,
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.7, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(len(df.columns)):
            cell = table[(0, i)]
            cell.set_facecolor('#2E86AB')
            cell.set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#F0F0F0')
                else:
                    cell.set_facecolor('white')
                    
        plt.title('Semi-Supervised Learning Performance Summary', 
                 fontsize=14, fontweight='bold', pad=20)
        
        output_path = self.output_dir / 'performance_table.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   Saved to: {output_path}")
        plt.close()
        
        # Also save as CSV
        csv_path = self.output_dir / 'performance_summary.csv'
        df.to_csv(csv_path, index=False)
        print(f"   Saved CSV to: {csv_path}")
        
    def create_enhanced_confusion_matrix(self, predictions):
        """Create an enhanced confusion matrix with percentages."""
        from sklearn.metrics import confusion_matrix
        
        y_true = predictions['true_label'].values
        y_pred = predictions['predicted_label'].values
        
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm_normalized, annot=False, fmt='.2%', 
                   cmap='RdYlGn', cbar_kws={'label': 'Proportion'},
                   linewidths=2, linecolor='black', ax=ax,
                   vmin=0, vmax=1)
        
        # Add text annotations with counts and percentages
        class_names = ['Hate Speech', 'Offensive', 'Neither']
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = cm_normalized[i, j]
                text = f'{count}\n({pct:.1%})'
                color = 'white' if pct > 0.5 else 'black'
                ax.text(j + 0.5, i + 0.5, text,
                       ha='center', va='center',
                       fontsize=12, fontweight='bold', color=color)
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix (Normalized by True Label)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names, rotation=0)
        
        plt.tight_layout()
        output_path = self.output_dir / 'confusion_matrix_enhanced.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   Saved to: {output_path}")
        plt.close()
        
    def create_pseudo_label_analysis(self, history, predictions):
        """Analyze pseudo-labeling quality and distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Pseudo-Labeling Analysis', fontsize=14, fontweight='bold')
        
        # Panel 1: Confidence distribution
        ax1 = axes[0]
        if 'confidence' in predictions.columns:
            confidences = predictions['confidence'].values
            ax1.hist(confidences, bins=50, color='#2E86AB', alpha=0.7, 
                    edgecolor='black', linewidth=1.2)
            ax1.axvline(x=0.65, color='red', linestyle='--', 
                       linewidth=2, label='Threshold')
            ax1.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax1.set_title('Confidence Score Distribution', fontsize=12, fontweight='bold')
            ax1.legend(frameon=True, shadow=True)
            ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Panel 2: Labeling efficiency
        ax2 = axes[1]
        iterations = history['iteration'].values
        pseudo_per_iter = history['n_pseudo_labeled'].values
        
        colors = ['#06A77D' if p > 0 else '#E63946' for p in pseudo_per_iter]
        bars = ax2.bar(iterations, pseudo_per_iter, color=colors, 
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Pseudo-Labels Added', fontsize=11, fontweight='bold')
        ax2.set_title('Labeling Efficiency per Iteration', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax2.set_xticks(iterations)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / 'pseudo_label_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   Saved to: {output_path}")
        plt.close()
        
    def generate_latex_tables(self, metrics, history):
        """Generate LaTeX-formatted tables for academic papers."""
        # Performance metrics table
        latex_performance = r"""\begin{table}[h]
\centering
\caption{Semi-Supervised Self-Training Performance}
\label{tab:performance}
\begin{tabular}{lc}
\hline
\textbf{Metric} & \textbf{Value} \\
\hline
Test Accuracy & """ + f"{metrics['accuracy'].values[0]:.4f}" + r""" \\
Macro F1-Score & """ + f"{metrics['macro_f1'].values[0]:.4f}" + r""" \\
Weighted F1-Score & """ + f"{metrics['weighted_f1'].values[0]:.4f}" + r""" \\
Cluster Purity & """ + f"{history.iloc[-1]['cluster_purity']:.4f}" + r""" \\
Silhouette Score & """ + f"{history.iloc[-1]['silhouette_score']:.4f}" + r""" \\
\hline
\end{tabular}
\end{table}"""
        
        # Training progression table
        latex_progression = r"""\begin{table}[h]
\centering
\caption{Self-Training Progression}
\label{tab:progression}
\begin{tabular}{cccc}
\hline
\textbf{Iteration} & \textbf{Labeled} & \textbf{Pseudo-Labels} & \textbf{Confidence} \\
\hline
"""
        for _, row in history.iterrows():
            latex_progression += f"{int(row['iteration'])} & {int(row['n_labeled'])} & {int(row['n_pseudo_labeled'])} & {row['avg_confidence']:.4f} \\\\\n"
        
        latex_progression += r"""\hline
\end{tabular}
\end{table}"""
        
        # Save to file
        latex_path = self.output_dir / 'latex_tables.tex'
        with open(latex_path, 'w') as f:
            f.write("% Performance Table\n")
            f.write(latex_performance)
            f.write("\n\n% Progression Table\n")
            f.write(latex_progression)
        
        print(f"   Saved to: {latex_path}")
        
    def create_executive_summary(self, metrics, history, predictions):
        """Create a text summary of results for the report."""
        final_iter = history.iloc[-1]
        
        summary = f"""
{'='*70}
SEMI-SUPERVISED SELF-TRAINING - EXECUTIVE SUMMARY
{'='*70}

EXPERIMENTAL SETUP
------------------
- Algorithm: K-Means Clustering with Self-Training
- Number of Clusters: 12
- Confidence Threshold: 0.65
- Initial Labeled Samples: {history.iloc[0]['n_labeled']:.0f}
- Initial Unlabeled Samples: {history.iloc[0]['n_unlabeled']:.0f}

TRAINING RESULTS
----------------
- Total Iterations Completed: {final_iter['iteration']:.0f}
- Final Labeled Samples: {final_iter['n_labeled']:.0f}
- Total Pseudo-Labels Added: {history['n_pseudo_labeled'].sum():.0f}
- Unlabeled Samples Remaining: {final_iter['n_unlabeled']:.0f}

CLUSTERING QUALITY
------------------
- Final Cluster Purity: {final_iter['cluster_purity']:.4f}
- Final Silhouette Score: {final_iter['silhouette_score']:.4f}
- Average Confidence (final): {final_iter['avg_confidence']:.4f}

TEST SET PERFORMANCE
--------------------
- Accuracy: {metrics['accuracy'].values[0]:.4f} ({metrics['accuracy'].values[0]*100:.2f}%)
- Macro F1-Score: {metrics['macro_f1'].values[0]:.4f}
- Weighted F1-Score: {metrics['weighted_f1'].values[0]:.4f}

CLASS-WISE PERFORMANCE
----------------------
"""
        
        # Add per-class metrics
        class_names = ['Hate Speech', 'Offensive Language', 'Neither']
        for i, name in enumerate(class_names):
            col_prefix = name.lower().replace(' ', '_')
            if f'{col_prefix}_precision' in metrics.columns:
                prec = metrics[f'{col_prefix}_precision'].values[0]
                rec = metrics[f'{col_prefix}_recall'].values[0]
                f1 = metrics[f'{col_prefix}_f1'].values[0]
                summary += f"{name}:\n"
                summary += f"  Precision: {prec:.4f}\n"
                summary += f"  Recall: {rec:.4f}\n"
                summary += f"  F1-Score: {f1:.4f}\n\n"
        
        summary += f"""
KEY OBSERVATIONS
----------------
1. Low clustering quality (purity: {final_iter['cluster_purity']:.2%}) indicates
   that hate speech classes are not naturally separable in embedding space.

2. Early stopping after {final_iter['iteration']:.0f} iterations suggests limited
   benefit from additional pseudo-labeling.

3. The model shows bias toward predicting "Neither" class, likely due to
   cluster-to-label mapping skewness.

RECOMMENDATIONS
---------------
1. Consider supervised learning approaches given the substantial amount
   of labeled data ({history.iloc[0]['n_labeled']:.0f} samples).

2. Explore alternative embedding models or fine-tuning SBERT on the
   hate speech domain.

3. Investigate ensemble methods combining multiple clustering algorithms.

{'='*70}
"""
        
        # Save to file
        summary_path = self.output_dir / 'executive_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        print(f"   Saved to: {summary_path}")
        print("\n" + summary)


def main():
    """Main function to generate comprehensive report."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate comprehensive report visualizations')
    parser.add_argument('--results-dir', type=str, default='outputs/results',
                       help='Directory containing result files')
    parser.add_argument('--output-dir', type=str, default='outputs/report',
                       help='Directory to save report files')
    
    args = parser.parse_args()
    
    # Create report generator
    generator = ReportGenerator(output_dir=args.output_dir)
    
    # Generate comprehensive report
    generator.create_comprehensive_report(results_dir=args.results_dir)
    
    print("\nReport generation complete!")
    print(f"\nView your results in: {args.output_dir}")
    print("\nFiles generated:")
    print("  - multipanel_results.png (4-panel overview)")
    print("  - cluster_analysis.png (cluster mapping analysis)")
    print("  - performance_table.png (metrics summary table)")
    print("  - confusion_matrix_enhanced.png (detailed confusion matrix)")
    print("  - pseudo_label_analysis.png (pseudo-labeling analysis)")
    print("  - performance_summary.csv (metrics in CSV format)")
    print("  - latex_tables.tex (LaTeX tables for papers)")
    print("  - executive_summary.txt (text summary)")


if __name__ == '__main__':
    main()
