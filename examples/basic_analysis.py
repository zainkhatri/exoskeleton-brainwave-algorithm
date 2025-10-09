#!/usr/bin/env python3
"""
Basic brainwave analysis example.
Demonstrates the core functionality of the brainwave analysis pipeline.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import BrainwaveAnalysisPipeline


def main():
    """Run basic brainwave analysis example."""
    print("🧠 Brainwave Analysis - Basic Example")
    print("=" * 50)
    
    # Initialize pipeline
    print("Initializing analysis pipeline...")
    pipeline = BrainwaveAnalysisPipeline()
    
    # Run complete analysis with sample data
    print("Running complete analysis pipeline...")
    results = pipeline.run_full_pipeline(
        use_sample_data=True,
        hyperparameter_tuning=False,
        create_plots=True
    )
    
    # Print summary results
    print("\n📊 Analysis Results:")
    print("-" * 30)
    final_results = results['final_results']
    
    print(f"Model Type: {final_results.get('model_type', 'Unknown')}")
    print(f"Test Accuracy: {final_results.get('accuracy', 0):.3f}")
    print(f"Cross-validation Mean: {final_results.get('cv_mean', 0):.3f} ± {final_results.get('cv_std', 0):.3f}")
    print(f"Number of Features: {final_results.get('n_features', 0)}")
    print(f"Number of Samples: {final_results.get('n_samples', 0)}")
    
    # Show feature importance (top 5)
    if 'feature_importance' in final_results:
        importance = final_results['feature_importance']
        top_indices = importance.argsort()[-5:][::-1]
        print(f"\n🔍 Top 5 Most Important Features:")
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. Feature {idx}: {importance[idx]:.3f}")
    
    # Demonstrate prediction
    print(f"\n🎯 Making a prediction on new data...")
    import numpy as np
    
    # Create synthetic new data
    new_eeg_data = np.random.randn(64, 256)  # 64 channels, 256 samples
    new_eog_data = np.random.randn(2, 256)   # 2 EOG channels
    
    prediction = pipeline.predict(new_eeg_data, new_eog_data)
    print(f"Predicted gaze direction: {prediction['gaze_direction']}")
    print(f"Confidence: {prediction['confidence']:.3f}")
    
    print(f"\n✅ Analysis completed successfully!")
    print(f"📁 Results saved to: {pipeline.config.get('paths.output_dir', 'output/')}")
    print(f"📊 Plots saved to: {pipeline.config.get('paths.plots_dir', 'plots/')}")


if __name__ == "__main__":
    main()
