import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import pandas as pd

class ModelEvaluator:
    def __init__(self, real_data, synthetic_data, real_labels, synthetic_labels, test_data, test_labels):
        """
        Initialize the evaluator with real and synthetic datasets
        
        Args:
            real_data: Training data from the real dataset
            synthetic_data: Generated synthetic data
            real_labels: Labels for the real training data
            synthetic_labels: Labels for the synthetic data
            test_data: Real test data for efficacy evaluation
            test_labels: Labels for the test data
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.real_labels = real_labels
        self.synthetic_labels = synthetic_labels
        self.test_data = test_data
        self.test_labels = test_labels

    def evaluate_detection(self, n_splits=4):
        """
        Evaluate detection metric using k-fold cross validation
        Returns: Average AUC score (lower is better)
        """
        print("\n=== Detection Metric Evaluation ===")
        
        # Create binary labels for real (0) and synthetic (1) data
        detection_data = np.vstack([self.real_data, self.synthetic_data])
        detection_labels = np.array([0] * len(self.real_data) + [1] * len(self.synthetic_data))
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        auc_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(detection_data), 1):
            # Split data into train and test
            X_train, X_test = detection_data[train_idx], detection_data[test_idx]
            y_train, y_test = detection_labels[train_idx], detection_labels[test_idx]
            
            # Train Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            # Evaluate
            y_pred_proba = rf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            auc_scores.append(auc)
            
            print(f"Fold {fold} AUC: {auc:.4f}")
        
        avg_auc = np.mean(auc_scores)
        print(f"\nAverage Detection AUC: {avg_auc:.4f}")
        print("Note: Lower detection AUC indicates better synthetic data quality")
        
        return avg_auc

    def evaluate_efficacy(self):
        """
        Evaluate efficacy metric by comparing model performance
        Returns: Efficacy score (higher is better, max 1.0)
        """
        print("\n=== Efficacy Metric Evaluation ===")
        
        # Train and evaluate on real data
        rf_real = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_real.fit(self.real_data, self.real_labels)
        real_auc = roc_auc_score(self.test_labels, 
                                rf_real.predict_proba(self.test_data)[:, 1])
        print(f"Real Data AUC: {real_auc:.4f}")
        
        # Train and evaluate on synthetic data
        rf_synthetic = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_synthetic.fit(self.synthetic_data, self.synthetic_labels)
        synthetic_auc = roc_auc_score(self.test_labels, 
                                     rf_synthetic.predict_proba(self.test_data)[:, 1])
        print(f"Synthetic Data AUC: {synthetic_auc:.4f}")
        
        # Calculate efficacy score
        efficacy_score = synthetic_auc / real_auc
        print(f"\nEfficacy Score: {efficacy_score:.4f}")
        print("Note: Higher efficacy score indicates better synthetic data utility")
        
        return efficacy_score

def evaluate_model(real_data, synthetic_data, real_labels, synthetic_labels, test_data, test_labels):
    """
    Wrapper function to evaluate both metrics
    """
    evaluator = ModelEvaluator(real_data, synthetic_data, real_labels, synthetic_labels, 
                              test_data, test_labels)
    
    print("\nStarting Model Evaluation...")
    detection_score = evaluator.evaluate_detection()
    efficacy_score = evaluator.evaluate_efficacy()
    
    return {
        'detection_auc': detection_score,
        'efficacy_score': efficacy_score
    }

if __name__ == "__main__":
    # Example usage
    from main import dataset, model, synthetic_data
    
    # Get the original data splits
    train_data = dataset.df.iloc[:, :-1].values  # All columns except the last (target)
    train_labels = dataset.df.iloc[:, -1].values
    test_data = dataset.df.iloc[:, :-1].values  # Using same data for demonstration
    test_labels = dataset.df.iloc[:, -1].values
    
    # Convert synthetic data to numpy if it's a tensor
    if torch.is_tensor(synthetic_data):
        synthetic_data = synthetic_data.cpu().numpy()
    
    # Get synthetic labels (for cGAN, these should be the conditioned labels)
    synthetic_labels = np.array([0, 1] * (len(synthetic_data) // 2))  # Example labels
    
    # Run evaluation
    results = evaluate_model(
        real_data=train_data,
        synthetic_data=synthetic_data,
        real_labels=train_labels,
        synthetic_labels=synthetic_labels,
        test_data=test_data,
        test_labels=test_labels
    )
    
    print("\nFinal Results:")
    print(f"Detection AUC: {results['detection_auc']:.4f}")
    print(f"Efficacy Score: {results['efficacy_score']:.4f}")