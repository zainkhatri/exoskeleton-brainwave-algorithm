"""
Machine learning models for EEG/EOG gaze direction prediction.
Includes multiple model types with proper evaluation and cross-validation.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class GazeDirectionClassifier:
    """Main classifier for gaze direction prediction from EEG/EOG data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize classifier.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.ml_config = self.config.get('ml', {})
        self.model_type = self.ml_config.get('model_type', 'random_forest')
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        
    def _create_model(self, model_type: Optional[str] = None) -> Any:
        """
        Create model instance based on type.
        
        Args:
            model_type: Type of model to create
            
        Returns:
            Model instance
        """
        model_type = model_type or self.model_type
        
        if model_type == 'random_forest':
            rf_config = self.ml_config.get('random_forest', {})
            model = RandomForestClassifier(
                n_estimators=rf_config.get('n_estimators', 100),
                max_depth=rf_config.get('max_depth', 10),
                min_samples_split=rf_config.get('min_samples_split', 5),
                min_samples_leaf=rf_config.get('min_samples_leaf', 2),
                random_state=self.ml_config.get('random_state', 42)
            )
        elif model_type == 'svm':
            svm_config = self.ml_config.get('svm', {})
            model = SVC(
                kernel=svm_config.get('kernel', 'rbf'),
                C=svm_config.get('C', 1.0),
                gamma=svm_config.get('gamma', 'scale'),
                random_state=self.ml_config.get('random_state', 42)
            )
        elif model_type == 'neural_network':
            nn_config = self.ml_config.get('neural_network', {})
            model = MLPClassifier(
                hidden_layer_sizes=nn_config.get('hidden_layer_sizes', [100, 50]),
                activation=nn_config.get('activation', 'relu'),
                solver=nn_config.get('solver', 'adam'),
                alpha=nn_config.get('alpha', 0.001),
                max_iter=nn_config.get('max_iter', 1000),
                random_state=self.ml_config.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Created {model_type} model")
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train the classifier.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Names of features
            
        Returns:
            Training results dictionary
        """
        if X.shape[0] != len(y):
            raise ValueError("Number of samples in X and y must match")
        
        # Store feature names
        self.feature_names = feature_names
        
        # Split data
        test_size = self.ml_config.get('test_size', 0.2)
        random_state = self.ml_config.get('random_state', 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_folds = self.ml_config.get('cross_validation_folds', 5)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_folds)
        
        # Results
        results = {
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
        
        logger.info(f"Training completed. Test accuracy: {test_accuracy:.3f}, CV mean: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance (for tree-based models).
        
        Returns:
            Feature importance array or None if not available
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            logger.warning("Feature importance not available for this model type")
            return None
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray,
                            param_grid: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using grid search.
        
        Args:
            X: Feature matrix
            y: Target labels
            param_grid: Parameter grid for grid search
            
        Returns:
            Tuning results
        """
        if param_grid is None:
            param_grid = self._get_default_param_grid()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create base model
        base_model = self._create_model()
        
        # Grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_scaled, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Hyperparameter tuning completed. Best score: {grid_search.best_score_:.3f}")
        
        return results
    
    def _get_default_param_grid(self) -> Dict[str, List]:
        """Get default parameter grid for hyperparameter tuning."""
        if self.model_type == 'random_forest':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
        elif self.model_type == 'svm':
            return {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear']
            }
        elif self.model_type == 'neural_network':
            return {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        else:
            return {}
    
    def save_model(self, filepath: str):
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from file.
        
        Args:
            filepath: Path to model file
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def evaluate_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Evaluation results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Predictions
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Metrics
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y, y_pred),
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        return results


class EnsembleClassifier:
    """Ensemble classifier combining multiple models."""
    
    def __init__(self, models: List[GazeDirectionClassifier], 
                 voting_method: str = 'soft'):
        """
        Initialize ensemble classifier.
        
        Args:
            models: List of trained classifiers
            voting_method: Voting method ('hard' or 'soft')
        """
        self.models = models
        self.voting_method = voting_method
        self.is_trained = all(model.is_trained for model in models)
        
        if not self.is_trained:
            raise ValueError("All models must be trained before creating ensemble")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Ensemble predictions
        """
        if self.voting_method == 'soft':
            # Average probabilities
            probabilities = np.mean([model.predict_proba(X) for model in self.models], axis=0)
            predictions = np.argmax(probabilities, axis=1)
        else:
            # Majority voting
            predictions = np.array([model.predict(X) for model in self.models])
            predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get ensemble prediction probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Ensemble prediction probabilities
        """
        probabilities = np.mean([model.predict_proba(X) for model in self.models], axis=0)
        return probabilities
