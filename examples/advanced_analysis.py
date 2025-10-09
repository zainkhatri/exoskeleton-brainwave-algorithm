#!/usr/bin/env python3
"""
Advanced brainwave analysis example.
Demonstrates hyperparameter tuning, ensemble methods, and custom configurations.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import BrainwaveAnalysisPipeline
from ml.models import GazeDirectionClassifier, EnsembleClassifier


def main():
    """Run advanced brainwave analysis example."""
    print("🧠 Brainwave Analysis - Advanced Example")
    print("=" * 50)
    
    # Initialize pipeline
    print("Initializing analysis pipeline...")
    pipeline = BrainwaveAnalysisPipeline()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_info = pipeline.load_data(use_sample_data=True)
    preprocessing_results = pipeline.preprocess_data(data_info)
    
    # Get features and labels
    X = preprocessing_results['combined_features']
    y = preprocessing_results['labels']
    
    print(f"Data shape: {X.shape}")
    print(f"Labels distribution: {np.bincount(y)}")
    
    # Train multiple models
    print("\n🤖 Training multiple models...")
    models = {}
    model_types = ['random_forest', 'svm', 'neural_network']
    
    for model_type in model_types:
        print(f"Training {model_type}...")
        
        # Create classifier with specific model type
        classifier = GazeDirectionClassifier(pipeline.config._config)
        classifier.model_type = model_type
        
        # Train model
        results = classifier.train(X, y)
        models[model_type] = classifier
        
        print(f"  {model_type} accuracy: {results['test_accuracy']:.3f}")
    
    # Hyperparameter tuning for best model
    print(f"\n🔧 Performing hyperparameter tuning...")
    best_model = models['random_forest']  # Use random forest for tuning
    
    tuning_results = best_model.hyperparameter_tuning(X, y)
    print(f"Best parameters: {tuning_results['best_params']}")
    print(f"Best score: {tuning_results['best_score']:.3f}")
    
    # Create ensemble
    print(f"\n🎭 Creating ensemble classifier...")
    ensemble = EnsembleClassifier(list(models.values()), voting_method='soft')
    
    # Evaluate ensemble
    ensemble_predictions = ensemble.predict(X)
    ensemble_accuracy = np.mean(ensemble_predictions == y)
    print(f"Ensemble accuracy: {ensemble_accuracy:.3f}")
    
    # Compare all models
    print(f"\n📊 Model Comparison:")
    print("-" * 40)
    for name, model in models.items():
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"{name:15}: {accuracy:.3f}")
    print(f"{'Ensemble':15}: {ensemble_accuracy:.3f}")
    
    # Feature importance analysis
    print(f"\n🔍 Feature Importance Analysis:")
    rf_model = models['random_forest']
    importance = rf_model.get_feature_importance()
    
    if importance is not None:
        top_features = importance.argsort()[-10:][::-1]
        print("Top 10 most important features:")
        for i, idx in enumerate(top_features):
            print(f"  {i+1:2d}. Feature {idx:3d}: {importance[idx]:.3f}")
    
    # Cross-validation analysis
    print(f"\n📈 Cross-validation Analysis:")
    for name, model in models.items():
        cv_scores = model.evaluate_model(X, y)['classification_report']
        print(f"{name}: {cv_scores['accuracy']:.3f}")
    
    # Save best model
    print(f"\n💾 Saving best model...")
    best_model.save_model("best_model.pkl")
    
    # Create comprehensive visualizations
    print(f"\n📊 Creating visualizations...")
    plots = pipeline.create_visualizations()
    print(f"Created {len(plots)} plots")
    
    print(f"\n✅ Advanced analysis completed successfully!")


if __name__ == "__main__":
    main()
