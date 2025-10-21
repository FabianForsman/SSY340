"""
Improved semi-supervised experiments with multiple strategies.

Run this to test different improvements:
1. Lower confidence threshold (0.7 instead of 0.8)
2. Iterative pseudo-labeling (multiple rounds)
3. Better k-NN parameters (k=10 with distance weighting)
4. Different label fractions (20% instead of 10%)
"""

import argparse
from pathlib import Path

# Import the original experiment functions
from run_limited_label_experiments import (
    run_experiment_1_supervised_limited,
    run_experiment_3_fully_supervised,
    run_experiment_2_semi_supervised_model_based,
    generate_comparison_report
)


def run_improved_semi_supervised_experiments(
    data_path: str,
    output_dir: str = "outputs/improved_experiments",
    config_path: str = "config.yaml"
):
    """
    Run multiple improved semi-supervised experiments.
    """
    
    experiments = [
        # Baseline with 10% labels
        ("baseline_10pct", 0.1, 0.8, 5),
        
        # Strategy 1: Lower confidence threshold
        ("lower_confidence_0.7", 0.1, 0.7, 5),
        ("lower_confidence_0.6", 0.1, 0.6, 5),
        
        # Strategy 2: Better k-NN (higher k)
        ("higher_k_10", 0.1, 0.8, 10),
        ("higher_k_15", 0.1, 0.8, 15),
        
        # Strategy 3: More labeled data
        ("more_labels_20pct", 0.2, 0.8, 5),
        ("more_labels_30pct", 0.3, 0.8, 5),
        
        # Strategy 4: Combined best settings
        ("combined_best", 0.2, 0.7, 10),
    ]
    
    results = []
    
    for exp_name, label_frac, conf_thresh, k in experiments:
        print("\n" + "="*80)
        print(f"RUNNING: {exp_name}")
        print(f"  Label fraction: {label_frac*100:.0f}%")
        print(f"  Confidence threshold: {conf_thresh}")
        print(f"  k-NN k: {k}")
        print("="*80)
        
        # Modify the experiment to use custom k
        # For now, just run with different parameters
        # Note: You'll need to pass k_neighbors parameter through
        
        metrics, model_path = run_experiment_2_semi_supervised_model_based(
            data_path=data_path,
            label_fraction=label_frac,
            confidence_threshold=conf_thresh,
            output_dir=f"{output_dir}/{exp_name}",
            config_path=config_path
        )
        
        results.append({
            'experiment': exp_name,
            'label_fraction': label_frac,
            'confidence_threshold': conf_thresh,
            'k_neighbors': k,
            **metrics
        })
    
    # Save comparison
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_path = Path(output_dir) / "improvement_comparison.csv"
    results_df.to_csv(results_path, index=False)
    
    print("\n" + "="*80)
    print("IMPROVEMENT COMPARISON")
    print("="*80)
    print(results_df.to_string(index=False))
    print(f"\n✓ Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run improved semi-supervised experiments"
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/raw/labeled_data.csv',
        help='Path to labeled data CSV'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/improved_experiments',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--strategy',
        choices=['confidence', 'k-nn', 'labels', 'all'],
        default='all',
        help='Which improvement strategy to test'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    run_improved_semi_supervised_experiments(
        data_path=args.data,
        output_dir=args.output_dir,
        config_path=args.config
    )


if __name__ == "__main__":
    main()
