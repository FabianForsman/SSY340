"""
Fine-tune SBERT model (all-MiniLM-L6-v2) on hate speech classification task.

This script fine-tunes a pre-trained sentence transformer model using:
1. Softmax Classification Loss (primary) - Direct classification training
2. Optional: Contrastive Learning - Learn to group similar classes together

The fine-tuned model produces better embeddings for hate speech detection.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.datasets import SentenceLabelDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing import TextTransform, download_nltk_data


class HateSpeechFineTuner:
    """Fine-tune sentence transformer models for hate speech classification."""
    
    def __init__(
        self,
        base_model: str = "all-MiniLM-L6-v2",
        num_classes: int = 3,
        config_path: Optional[str] = None
    ):
        """
        Initialize fine-tuner.
        
        Args:
            base_model: Pre-trained model name
            num_classes: Number of classification labels (3 for hate/offensive/neither)
            config_path: Path to config.yaml file
        """
        self.base_model_name = base_model
        self.num_classes = num_classes
        self.model = None
        self.label_mapping = {
            0: "hate_speech",
            1: "offensive_language", 
            2: "neither"
        }
        
        # Load config if provided
        self.config = {}
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
    def load_and_preprocess_data(
        self,
        data_path: str,
        text_column: str = "tweet",
        label_column: str = "class",
        test_size: float = 0.2,
        val_size: float = 0.1,
        apply_preprocessing: bool = True,
        balance_classes: bool = False,
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess the dataset.
        
        Args:
            data_path: Path to CSV file
            text_column: Column name for text
            label_column: Column name for labels
            test_size: Fraction for test set
            val_size: Fraction for validation set (from training set)
            apply_preprocessing: Whether to apply text preprocessing
            balance_classes: Whether to balance class distribution
            random_seed: Random seed for reproducibility
            
        Returns:
            train_df, val_df, test_df
        """
        print(f"\nLoading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Check required columns
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataset")
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset")
        
        # Keep only necessary columns
        df = df[[text_column, label_column]].copy()
        df.columns = ['text', 'label']
        
        # Remove missing values
        df = df.dropna()
        
        print(f"Total samples: {len(df)}")
        print(f"\nClass distribution:")
        for label, name in self.label_mapping.items():
            count = (df['label'] == label).sum()
            print(f"  {label} ({name}): {count} ({count/len(df)*100:.1f}%)")
        
        # Apply text preprocessing
        if apply_preprocessing:
            print("\nApplying text preprocessing...")
            preprocessing_config = self.config.get('preprocessing', {})
            
            # Download NLTK data if needed
            download_nltk_data()
            
            # Create preprocessor
            text_transform = TextTransform(**preprocessing_config)
            
            # Apply preprocessing
            df['text'] = df['text'].apply(text_transform)
        
        # Balance classes if requested
        if balance_classes:
            print("\nBalancing classes...")
            df = self._balance_classes(df, random_seed)
            print(f"Balanced dataset size: {len(df)}")
        
        # Split into train, val, test
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_seed, stratify=df['label']
        )
        
        train_df, val_df = train_test_split(
            train_df, test_size=val_size, random_state=random_seed, stratify=train_df['label']
        )
        
        print(f"\nDataset splits:")
        print(f"  Training:   {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")
        print(f"  Test:       {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def _balance_classes(self, df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
        """Balance class distribution using oversampling."""
        class_counts = df['label'].value_counts()
        max_count = class_counts.max()
        
        balanced_dfs = []
        for label in df['label'].unique():
            class_df = df[df['label'] == label]
            
            if len(class_df) < max_count:
                # Oversample minority class
                oversampled = class_df.sample(
                    n=max_count,
                    replace=True,
                    random_state=random_seed
                )
                balanced_dfs.append(oversampled)
            else:
                balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        return balanced_df
    
    def prepare_training_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data loaders for training.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            
        Returns:
            train_dataloader, val_dataloader
        """
        print("\nPreparing training data...")
        
        # Create InputExample objects for classification
        train_examples = [
            InputExample(texts=[row['text']], label=int(row['label']))
            for _, row in train_df.iterrows()
        ]
        
        val_examples = [
            InputExample(texts=[row['text']], label=int(row['label']))
            for _, row in val_df.iterrows()
        ]
        
        print(f"Created {len(train_examples)} training examples")
        print(f"Created {len(val_examples)} validation examples")
        
        return train_examples, val_examples
    
    def create_model_with_classifier(self):
        """Create sentence transformer model with classification head."""
        print(f"\nLoading base model: {self.base_model_name}")
        
        # Load pre-trained model
        model = SentenceTransformer(self.base_model_name, device=self.device)
        
        print(f"Model embedding dimension: {model.get_sentence_embedding_dimension()}")
        print(f"Number of classes: {self.num_classes}")
        
        self.model = model
        return model
    
    def train(
        self,
        train_examples: List[InputExample],
        val_examples: List[InputExample],
        output_path: str,
        epochs: int = 4,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
        evaluation_steps: int = 500,
        save_best_model: bool = True
    ):
        """
        Fine-tune the model.
        
        Args:
            train_examples: Training examples
            val_examples: Validation examples
            output_path: Directory to save fine-tuned model
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            evaluation_steps: Evaluate every N steps
            save_best_model: Save model with best validation score
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call create_model_with_classifier() first.")
        
        print("\n" + "="*70)
        print("STARTING FINE-TUNING")
        print("="*70)
        print(f"Base model: {self.base_model_name}")
        print(f"Training samples: {len(train_examples)}")
        print(f"Validation samples: {len(val_examples)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print("="*70)
        
        # Create data loader
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size
        )
        
        # Define loss function - Softmax loss for classification
        train_loss = losses.SoftmaxLoss(
            model=self.model,
            sentence_embedding_dimension=self.model.get_sentence_embedding_dimension(),
            num_labels=self.num_classes
        )
        
        # Create evaluator for validation
        evaluator = self._create_evaluator(val_examples)
        
        # Calculate training steps
        steps_per_epoch = len(train_dataloader)
        total_steps = steps_per_epoch * epochs
        
        print(f"\nTraining configuration:")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Evaluation steps: {evaluation_steps}")
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Train the model
        print("\nStarting training...")
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            warmup_steps=warmup_steps,
            evaluation_steps=evaluation_steps,
            output_path=str(output_dir),
            save_best_model=save_best_model,
            show_progress_bar=True,
            optimizer_params={'lr': learning_rate}
        )
        
        print(f"\n✓ Fine-tuning complete! Model saved to: {output_dir}")
        
        # Save training info
        info = {
            'base_model': self.base_model_name,
            'num_classes': self.num_classes,
            'label_mapping': self.label_mapping,
            'training_samples': len(train_examples),
            'validation_samples': len(val_examples),
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        info_path = output_dir / 'training_info.yaml'
        with open(info_path, 'w') as f:
            yaml.dump(info, f, default_flow_style=False)
        print(f"Training info saved to: {info_path}")
        
    def _create_evaluator(self, val_examples: List[InputExample]):
        """Create evaluator for validation during training."""
        # Prepare validation data
        sentences = [example.texts[0] for example in val_examples]
        labels = [example.label for example in val_examples]
        
        # Use LabelAccuracyEvaluator for classification
        # Updated API for sentence-transformers 3.0+
        evaluator = evaluation.LabelAccuracyEvaluator(
            sentences=sentences,
            labels=labels,
            batch_size=32,
            name="validation"
        )
        
        return evaluator
    
    def evaluate(
        self,
        test_df: pd.DataFrame,
        model_path: Optional[str] = None,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_df: Test dataframe
            model_path: Path to fine-tuned model (if different from self.model)
            batch_size: Batch size for inference
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Load model if path provided
        if model_path:
            print(f"\nLoading fine-tuned model from: {model_path}")
            model = SentenceTransformer(model_path, device=self.device)
        else:
            model = self.model
        
        if model is None:
            raise ValueError("No model available for evaluation")
        
        print("\n" + "="*70)
        print("EVALUATING MODEL")
        print("="*70)
        
        # Get embeddings
        print(f"Generating embeddings for {len(test_df)} test samples...")
        embeddings = model.encode(
            test_df['text'].tolist(),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # For fine-tuned model with softmax loss, we can use the classifier
        # Or use a simple k-NN classifier on embeddings
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Note: This is a simple evaluation. For production, you'd want to
        # extract predictions from the softmax layer directly
        print("\nTraining k-NN classifier on embeddings...")
        
        # We need some labeled data for k-NN, so we'll use cross-validation
        # or load the training embeddings. For now, simple accuracy on embeddings
        
        # Use the model's built-in evaluator if available
        sentences = test_df['text'].tolist()
        labels = test_df['label'].tolist()
        
        evaluator = evaluation.LabelAccuracyEvaluator(
            sentences=sentences,
            labels=labels,
            name="test_evaluation",
            batch_size=batch_size
        )
        
        accuracy = evaluator(model)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        # For more detailed metrics, we'd need to get predictions
        # This is a simplified evaluation
        metrics = {
            'accuracy': accuracy,
            'test_samples': len(test_df)
        }
        
        print("="*70)
        
        return metrics
    
    def compare_models(
        self,
        test_df: pd.DataFrame,
        base_model_path: str,
        fine_tuned_model_path: str,
        output_path: Optional[str] = None
    ):
        """
        Compare base model vs fine-tuned model.
        
        Args:
            test_df: Test dataframe
            base_model_path: Path or name of base model
            fine_tuned_model_path: Path to fine-tuned model
            output_path: Path to save comparison results
        """
        print("\n" + "="*70)
        print("COMPARING BASE MODEL VS FINE-TUNED MODEL")
        print("="*70)
        
        # Evaluate base model
        print("\n1. Evaluating BASE model...")
        base_model = SentenceTransformer(base_model_path, device=self.device)
        base_metrics = self._evaluate_with_knn(base_model, test_df)
        
        # Evaluate fine-tuned model
        print("\n2. Evaluating FINE-TUNED model...")
        ft_model = SentenceTransformer(fine_tuned_model_path, device=self.device)
        ft_metrics = self._evaluate_with_knn(ft_model, test_df)
        
        # Create comparison
        comparison = pd.DataFrame({
            'Model': ['Base (all-MiniLM-L6-v2)', 'Fine-tuned'],
            'Accuracy': [base_metrics['accuracy'], ft_metrics['accuracy']],
            'F1 (Macro)': [base_metrics['f1_macro'], ft_metrics['f1_macro']],
            'F1 (Weighted)': [base_metrics['f1_weighted'], ft_metrics['f1_weighted']]
        })
        
        print("\n" + "="*70)
        print("COMPARISON RESULTS")
        print("="*70)
        print(comparison.to_string(index=False))
        print("="*70)
        
        improvement = (ft_metrics['accuracy'] - base_metrics['accuracy']) / base_metrics['accuracy'] * 100
        print(f"\nImprovement: {improvement:+.2f}%")
        
        if output_path:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            comparison.to_csv(output_dir / 'model_comparison.csv', index=False)
            print(f"\nComparison saved to: {output_dir / 'model_comparison.csv'}")
        
        return comparison
    
    def _evaluate_with_knn(
        self,
        model: SentenceTransformer,
        test_df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None,
        k: int = 5
    ) -> Dict[str, float]:
        """Evaluate model using k-NN classifier on embeddings."""
        # Generate embeddings
        test_embeddings = model.encode(
            test_df['text'].tolist(),
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # For proper evaluation, we need training data
        # For now, use simple train-test split from test set
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            test_embeddings,
            test_df['label'].values,
            test_size=0.5,
            random_state=42,
            stratify=test_df['label'].values
        )
        
        # Train k-NN classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        # Predict
        y_pred = knn.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (Macro): {f1_macro:.4f}")
        print(f"  F1 (Weighted): {f1_weighted:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }


def main():
    """Main function to run fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune SBERT model for hate speech classification"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/raw/labeled_data.csv',
        help='Path to labeled data CSV'
    )
    
    parser.add_argument(
        '--base-model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Base model to fine-tune'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='models/fine_tuned',
        help='Output directory for fine-tuned model'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=4,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data for test set'
    )
    
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.1,
        help='Fraction of training data for validation'
    )
    
    parser.add_argument(
        '--balance-classes',
        action='store_true',
        help='Balance class distribution via oversampling'
    )
    
    parser.add_argument(
        '--no-preprocessing',
        action='store_true',
        help='Skip text preprocessing'
    )
    
    parser.add_argument(
        '--evaluate-only',
        type=str,
        default=None,
        help='Path to fine-tuned model for evaluation only (skip training)'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare base model vs fine-tuned model'
    )
    
    args = parser.parse_args()
    
    # Initialize fine-tuner
    fine_tuner = HateSpeechFineTuner(
        base_model=args.base_model,
        num_classes=3,
        config_path=args.config
    )
    
    # Load and preprocess data
    train_df, val_df, test_df = fine_tuner.load_and_preprocess_data(
        data_path=args.data,
        test_size=args.test_size,
        val_size=args.val_size,
        apply_preprocessing=not args.no_preprocessing,
        balance_classes=args.balance_classes
    )
    
    if args.evaluate_only:
        # Evaluation only mode
        fine_tuner.evaluate(test_df, model_path=args.evaluate_only)
        
        if args.compare:
            fine_tuner.compare_models(
                test_df,
                base_model_path=args.base_model,
                fine_tuned_model_path=args.evaluate_only,
                output_path=Path(args.evaluate_only).parent / 'comparison'
            )
    else:
        # Training mode
        # Prepare training data
        train_examples, val_examples = fine_tuner.prepare_training_data(
            train_df, val_df
        )
        
        # Create model
        fine_tuner.create_model_with_classifier()
        
        # Train
        fine_tuner.train(
            train_examples=train_examples,
            val_examples=val_examples,
            output_path=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Evaluate on test set
        fine_tuner.evaluate(test_df)
        
        # Compare if requested
        if args.compare:
            fine_tuner.compare_models(
                test_df,
                base_model_path=args.base_model,
                fine_tuned_model_path=args.output,
                output_path=Path(args.output).parent / 'comparison'
            )
        
        print("\n✓ All done!")


if __name__ == "__main__":
    main()
