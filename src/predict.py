"""
Prediction module for hate speech detection models.

This module provides a consistent interface for making predictions with trained models.
It ensures that:
1. All models use the same preprocessing pipeline
2. Predictions are made in a consistent format
3. Models can be easily loaded from saved paths
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Dict
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from src.preprocessing import TextTransform


class HateSpeechPredictor:
    """
    Predictor for hate speech detection models.
    
    Handles preprocessing, model loading, and prediction in a consistent way.
    """
    
    def __init__(self, model_path: str, config_path: str = "config.yaml"):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model directory
            config_path: Path to config file with preprocessing settings
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize preprocessing transform
        self.preprocessor = self._create_preprocessor()
        
        # Load model
        self.model = self._load_model()
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Label mapping
        self.label_mapping = {
            0: "Hate Speech",
            1: "Offensive Language",
            2: "Neither"
        }
    
    def _create_preprocessor(self) -> TextTransform:
        """Create text preprocessing transform from config."""
        preprocessing_config = self.config.get("preprocessing", {})
        return TextTransform(**preprocessing_config)
    
    def _load_model(self) -> SentenceTransformer:
        """Load model from saved path."""
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {self.model_path}")
        
        print(f"Loading model from: {self.model_path}")
        model = SentenceTransformer(str(self.model_path))
        print(f"✓ Model loaded successfully")
        return model
    
    def preprocess_texts(self, texts: Union[str, List[str]]) -> List[str]:
        """
        Preprocess text(s) using the configured preprocessing pipeline.
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            List of preprocessed texts
        """
        if isinstance(texts, str):
            texts = [texts]
        
        return [self.preprocessor(text) for text in texts]
    
    def predict(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 64,
        return_probabilities: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions on text(s).
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for encoding
            return_probabilities: If True, also return probability scores
            
        Returns:
            predictions: Array of predicted labels (0, 1, 2)
            probabilities (optional): Array of probability scores if return_probabilities=True
        """
        # Preprocess texts
        preprocessed_texts = self.preprocess_texts(texts)
        
        # Get embeddings
        embeddings = self.model.encode(
            preprocessed_texts,
            batch_size=batch_size,
            show_progress_bar=len(preprocessed_texts) > 100,
            convert_to_numpy=True
        )
        
        # Get predictions from model's classifier
        # The fine-tuned model should have a classifier attached
        # We'll use the model's predict method if available
        if hasattr(self.model, 'predict'):
            predictions = self.model.predict(embeddings)
        else:
            # If no predict method, we need to use the classifier module
            # This is for models saved with the classifier head
            import torch
            from torch import nn
            
            # Get the classifier from the model
            classifier = None
            for module in self.model.modules():
                if isinstance(module, nn.Linear) and module.out_features <= 10:
                    classifier = module
                    break
            
            if classifier is None:
                raise ValueError("Model does not have a classifier. Cannot make predictions.")
            
            # Make predictions
            with torch.no_grad():
                embeddings_tensor = torch.FloatTensor(embeddings).to(self.device)
                logits = classifier(embeddings_tensor)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                predictions = np.argmax(probabilities, axis=1)
            
            if return_probabilities:
                return predictions, probabilities
            
        return predictions
    
    def predict_dataframe(
        self, 
        df: pd.DataFrame, 
        text_column: str = None,
        batch_size: int = 64
    ) -> pd.DataFrame:
        """
        Make predictions on a DataFrame.
        
        Args:
            df: DataFrame containing text data
            text_column: Name of column containing text (uses config default if None)
            batch_size: Batch size for encoding
            
        Returns:
            DataFrame with predictions added
        """
        if text_column is None:
            text_column = self.config["data"]["text_column"]
        
        texts = df[text_column].tolist()
        predictions = self.predict(texts, batch_size=batch_size)
        
        df_with_predictions = df.copy()
        df_with_predictions['predicted_label'] = predictions
        df_with_predictions['predicted_label_desc'] = [
            self.label_mapping[pred] for pred in predictions
        ]
        
        return df_with_predictions
    
    def evaluate(
        self, 
        df: pd.DataFrame, 
        text_column: str = None,
        label_column: str = None,
        batch_size: int = 64
    ) -> Dict:
        """
        Evaluate model on labeled data.
        
        Args:
            df: DataFrame containing text and labels
            text_column: Name of column containing text
            label_column: Name of column containing labels
            batch_size: Batch size for encoding
            
        Returns:
            Dictionary with evaluation metrics
        """
        if text_column is None:
            text_column = self.config["data"]["text_column"]
        if label_column is None:
            label_column = self.config["data"]["label_column"]
        
        texts = df[text_column].tolist()
        true_labels = df[label_column].values
        
        predictions = self.predict(texts, batch_size=batch_size)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1_macro = f1_score(true_labels, predictions, average='macro')
        f1_weighted = f1_score(true_labels, predictions, average='weighted')
        
        # Generate classification report
        report = classification_report(
            true_labels, 
            predictions,
            target_names=[self.label_mapping[i] for i in range(3)],
            digits=4
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def predict_single(self, text: str) -> Tuple[int, str]:
        """
        Make prediction on a single text.
        
        Args:
            text: Text to classify
            
        Returns:
            Tuple of (predicted_label, predicted_label_description)
        """
        prediction = self.predict(text)[0]
        return prediction, self.label_mapping[prediction]


def load_and_predict(
    model_path: str,
    data_path: str,
    config_path: str = "config.yaml",
    text_column: str = None,
    label_column: str = None,
    output_path: str = None
) -> pd.DataFrame:
    """
    Convenience function to load model and make predictions on data.
    
    Args:
        model_path: Path to saved model
        data_path: Path to CSV file with data
        config_path: Path to config file
        text_column: Name of text column (uses config default if None)
        label_column: Name of label column (uses config default if None)
        output_path: Path to save predictions (optional)
        
    Returns:
        DataFrame with predictions
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if text_column is None:
        text_column = config["data"]["text_column"]
    if label_column is None:
        label_column = config["data"]["label_column"]
    
    # Load data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Initialize predictor
    predictor = HateSpeechPredictor(model_path, config_path)
    
    # Make predictions
    print(f"Making predictions on {len(df)} samples...")
    df_with_predictions = predictor.predict_dataframe(df, text_column)
    
    # Evaluate if labels are present
    if label_column in df.columns:
        print("\n=== Evaluation Results ===")
        metrics = predictor.evaluate(df, text_column, label_column)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"\nClassification Report:")
        print(metrics['classification_report'])
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
    
    # Save predictions if output path provided
    if output_path:
        df_with_predictions.to_csv(output_path, index=False)
        print(f"\n✓ Predictions saved to: {output_path}")
    
    return df_with_predictions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Make predictions with trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model")
    parser.add_argument("--data", type=str, required=True, help="Path to data CSV")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, help="Path to save predictions")
    
    args = parser.parse_args()
    
    load_and_predict(
        model_path=args.model,
        data_path=args.data,
        config_path=args.config,
        output_path=args.output
    )
