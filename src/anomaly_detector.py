"""
Anomaly Detection Pipeline for Refinery Sensor Data.
Uses Isolation Forest for unsupervised anomaly detection.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os


class AnomalyDetector:
    """Isolation Forest-based anomaly detector for refinery sensors."""
    
    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.feature_cols = [
            'temperature_c', 'pressure_bar', 'flow_rate_m3h', 
            'vibration_mms', 'temp_pressure_ratio'
        ]
        self.is_fitted = False
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and scale features for model input."""
        X = df[self.feature_cols].values
        return X
    
    def fit(self, df: pd.DataFrame) -> 'AnomalyDetector':
        """Fit the anomaly detection model."""
        X = self.prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        print(f"Model trained on {len(df)} samples")
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict anomalies. Returns 1 for anomaly, 0 for normal."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Isolation Forest returns -1 for anomalies, 1 for normal
        predictions = self.model.predict(X_scaled)
        # Convert to 0/1 format (1 = anomaly)
        return (predictions == -1).astype(int)
    
    def get_anomaly_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Get anomaly scores (lower = more anomalous)."""
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        return -self.model.score_samples(X_scaled)
    
    def evaluate(self, df: pd.DataFrame, true_labels: np.ndarray) -> dict:
        """Evaluate model performance against ground truth."""
        predictions = self.predict(df)
        
        cm = confusion_matrix(true_labels, predictions)
        report = classification_report(true_labels, predictions, output_dict=True)
        
        return {
            'confusion_matrix': cm,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'accuracy': report['accuracy']
        }
    
    def save(self, path: str):
        """Save model and scaler to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'contamination': self.contamination
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'AnomalyDetector':
        """Load model from disk."""
        model_data = joblib.load(path)
        detector = cls(contamination=model_data['contamination'])
        detector.model = model_data['model']
        detector.scaler = model_data['scaler']
        detector.feature_cols = model_data['feature_cols']
        detector.is_fitted = True
        return detector


def main():
    print("=" * 60)
    print("REFINERY ANOMALY DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sensor_data.csv')
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    print(f"Loaded {len(df)} samples")
    
    # Train model
    detector = AnomalyDetector(contamination=0.05)
    detector.fit(df)
    
    # Evaluate
    results = detector.evaluate(df, df['is_anomaly'].values)
    
    print("\n" + "=" * 40)
    print("MODEL PERFORMANCE")
    print("=" * 40)
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall:    {results['recall']:.3f}")
    print(f"F1-Score:  {results['f1_score']:.3f}")
    print(f"Accuracy:  {results['accuracy']:.3f}")
    
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'anomaly_detector.pkl')
    detector.save(model_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
