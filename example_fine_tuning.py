"""
Example: Fine-tune and compare SBERT model for hate speech classification.

This script demonstrates the complete fine-tuning workflow:
1. Fine-tune all-MiniLM-L6-v2 on hate speech data
2. Compare base model vs fine-tuned model
3. Generate embeddings with both models
4. Show performance improvements
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.fine_tune_model import HateSpeechFineTuner


def main():
    """Run complete fine-tuning example."""
    
    print("="*80)
    print("HATE SPEECH MODEL FINE-TUNING EXAMPLE")
    print("="*80)
    
    # Configuration
    config = {
        'data_path': 'data/raw/labeled_data.csv',
        'base_model': 'all-MiniLM-L6-v2',
        'output_path': 'models/fine_tuned_example',
        'config_path': 'config.yaml',
        'epochs': 3,  # Quick example
        'batch_size': 16,
        'test_size': 0.2,
        'balance_classes': True,  # Better for hate speech detection
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize fine-tuner
    fine_tuner = HateSpeechFineTuner(
        base_model=config['base_model'],
        num_classes=3,
        config_path=config['config_path']
    )
    
    # Step 1: Load and preprocess data
    print("\n" + "="*80)
    print("STEP 1: Loading and preprocessing data")
    print("="*80)
    
    train_df, val_df, test_df = fine_tuner.load_and_preprocess_data(
        data_path=config['data_path'],
        test_size=config['test_size'],
        val_size=0.1,
        apply_preprocessing=True,
        balance_classes=config['balance_classes']
    )
    
    # Step 2: Prepare training data
    print("\n" + "="*80)
    print("STEP 2: Preparing training data")
    print("="*80)
    
    train_examples, val_examples = fine_tuner.prepare_training_data(
        train_df, val_df
    )
    
    # Step 3: Create model
    print("\n" + "="*80)
    print("STEP 3: Creating model with classification head")
    print("="*80)
    
    fine_tuner.create_model_with_classifier()
    
    # Step 4: Fine-tune
    print("\n" + "="*80)
    print("STEP 4: Fine-tuning model")
    print("="*80)
    
    fine_tuner.train(
        train_examples=train_examples,
        val_examples=val_examples,
        output_path=config['output_path'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        warmup_steps=100,
        evaluation_steps=500
    )
    
    # Step 5: Evaluate
    print("\n" + "="*80)
    print("STEP 5: Evaluating fine-tuned model")
    print("="*80)
    
    metrics = fine_tuner.evaluate(test_df)
    
    # Step 6: Compare models
    print("\n" + "="*80)
    print("STEP 6: Comparing base vs fine-tuned model")
    print("="*80)
    
    comparison = fine_tuner.compare_models(
        test_df,
        base_model_path=config['base_model'],
        fine_tuned_model_path=config['output_path'],
        output_path=Path(config['output_path']).parent / 'comparison'
    )
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Fine-tuned model saved to: {config['output_path']}")
    print(f"✓ Test accuracy: {metrics['accuracy']:.4f}")
    print("\nTo use the fine-tuned model:")
    print(f"  1. Update config.yaml: embedding.model = '{config['output_path']}'")
    print(f"  2. Run: python src/main.py --config config.yaml")
    print("\nOr generate embeddings directly:")
    print(f"  python src/embeddings.py --model {config['output_path']}")
    print("="*80)


if __name__ == "__main__":
    main()
