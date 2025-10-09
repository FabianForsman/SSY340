"""
Main training script for hate speech detection project.
Orchestrates the entire pipeline: data loading, preprocessing,
embedding generation, clustering, and evaluation.
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_loader import DataLoader
from preprocessing import TextPreprocessor
from embeddings import EmbeddingGenerator
from clustering import (
    KMeansClustering,
    DBSCANClustering,
    find_optimal_k,
    plot_elbow_curve,
)
from evaluation import ClusteringEvaluator, compare_clustering_methods


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_pipeline(config):
    """
    Run the complete hate speech detection pipeline.

    Args:
        config (dict): Configuration dictionary
    """
    print("=" * 70)
    print("HATE SPEECH DETECTION - UNSUPERVISED LEARNING")
    print("=" * 70)

    # Create output directories
    Path(config["paths"]["embeddings_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["figures_dir"]).mkdir(parents=True, exist_ok=True)

    # Step 1: Load Data
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)

    loader = DataLoader(config["paths"]["raw_data_dir"])

    if config["data"]["download_from_kaggle"]:
        print("Note: To download from Kaggle, ensure kaggle.json is in ~/.kaggle/")
        # loader.download_kaggle_dataset(config['data']['kaggle_dataset'])

    df = loader.load_hate_speech_dataset(config["data"]["data_file"])
    loader.get_dataset_info(df)

    # Step 2: Preprocess Text
    print("\n" + "=" * 70)
    print("STEP 2: PREPROCESSING TEXT")
    print("=" * 70)

    preprocessor = TextPreprocessor(**config["preprocessing"])
    df = preprocessor.preprocess_dataframe(
        df, text_column=config["data"]["text_column"], output_column="cleaned_text"
    )

    # Save processed data
    processed_path = Path(config["paths"]["processed_data_dir"]) / "processed_data.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Saved processed data to {processed_path}")

    # Step 3: Generate Embeddings
    print("\n" + "=" * 70)
    print("STEP 3: GENERATING EMBEDDINGS")
    print("=" * 70)

    embedding_model = config["embedding"]["model"]
    print(f"Using embedding model: {embedding_model}")

    generator = EmbeddingGenerator(embedding_model)
    embeddings = generator.encode_dataframe(
        df,
        text_column="cleaned_text",
        batch_size=config["embedding"]["batch_size"],
        normalize=config["embedding"]["normalize"],
    )

    # Save embeddings
    embeddings_path = (
        Path(config["paths"]["embeddings_dir"]) / f"embeddings_{embedding_model}.npy"
    )
    generator.save_embeddings(embeddings, embeddings_path)

    # Step 4: Clustering
    print("\n" + "=" * 70)
    print("STEP 4: CLUSTERING")
    print("=" * 70)

    results_all = {}

    # K-Means Clustering
    if config["clustering"]["use_kmeans"]:
        print("\n--- K-Means Clustering ---")

        if config["clustering"]["find_optimal_k"]:
            print("\nFinding optimal k...")
            k_results = find_optimal_k(
                embeddings, k_range=range(2, config["clustering"]["max_k"] + 1)
            )

            # Plot elbow curve
            elbow_path = Path(config["paths"]["figures_dir"]) / "elbow_curve.png"
            plot_elbow_curve(k_results, save_path=elbow_path)

        n_clusters = config["clustering"]["kmeans_n_clusters"]
        kmeans = KMeansClustering(
            n_clusters=n_clusters, random_state=config["random_seed"]
        )
        kmeans_labels = kmeans.fit(embeddings)

        # Add labels to dataframe
        df["kmeans_cluster"] = kmeans_labels

        results_all["kmeans"] = kmeans_labels

    # DBSCAN Clustering
    if config["clustering"]["use_dbscan"]:
        print("\n--- DBSCAN Clustering ---")

        dbscan = DBSCANClustering(
            eps=config["clustering"]["dbscan_eps"],
            min_samples=config["clustering"]["dbscan_min_samples"],
        )
        dbscan_labels = dbscan.fit(embeddings)

        # Add labels to dataframe
        df["dbscan_cluster"] = dbscan_labels

        results_all["dbscan"] = dbscan_labels

    # Step 5: Evaluation
    print("\n" + "=" * 70)
    print("STEP 5: EVALUATION")
    print("=" * 70)

    # Get ground truth labels
    y_true = df[config["data"]["label_column"]].values

    evaluation_results = {}

    for method_name, y_pred in results_all.items():
        print(f"\n{'='*70}")
        print(f"EVALUATING: {method_name.upper()}")
        print(f"{'='*70}")

        evaluator = ClusteringEvaluator()
        results = evaluator.evaluate(y_true, y_pred, embeddings=embeddings)

        evaluation_results[method_name] = results

        # Save individual results
        results_path = (
            Path(config["paths"]["results_dir"]) / f"{method_name}_results.csv"
        )
        evaluator.save_results(results_path)

        # Plot confusion matrix
        cm_path = (
            Path(config["paths"]["figures_dir"]) / f"{method_name}_confusion_matrix.png"
        )
        evaluator.plot_confusion_matrix(y_true, y_pred, save_path=cm_path)

    # Compare methods
    if len(evaluation_results) > 1:
        print("\n" + "=" * 70)
        print("COMPARING METHODS")
        print("=" * 70)

        comparison_path = Path(config["paths"]["figures_dir"]) / "method_comparison.png"
        compare_clustering_methods(evaluation_results, save_path=comparison_path)

    # Save final dataframe with all predictions
    final_path = Path(config["paths"]["results_dir"]) / "final_results.csv"
    df.to_csv(final_path, index=False)
    print(f"\nSaved final results with all predictions to {final_path}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)

    return df, embeddings, evaluation_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Hate Speech Detection - Unsupervised Learning"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Run pipeline
    df, embeddings, results = run_pipeline(config)

    print("\n=== Summary ===")
    for method, metrics in results.items():
        print(f"\n{method.upper()}:")
        print(f"  ARI: {metrics.get('adjusted_rand_index', 0):.4f}")
        print(f"  NMI: {metrics.get('normalized_mutual_info', 0):.4f}")
        print(f"  V-Measure: {metrics.get('v_measure', 0):.4f}")


if __name__ == "__main__":
    main()
