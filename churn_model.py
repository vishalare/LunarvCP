import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictionModel:
    """
    Advanced Churn Prediction Model using ensemble methods
    """
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42, 
                class_weight='balanced',
                max_iter=1000
            )
        }
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_model = None
        self.feature_importance = None
        self.model_performance = {}
        
    def preprocess_data(self, df, target_column='churn'):
        """
        Preprocess data for churn prediction
        """
        df_processed = df.copy()
        
        # Handle missing values
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_columns:
            if col != target_column:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        # Fill categorical missing values with mode
        for col in categorical_columns:
            if col != target_column:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        
        # Encode categorical variables
        for col in categorical_columns:
            if col != target_column:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[col] = le
        
        # Ensure target column is binary
        if target_column in df_processed.columns:
            # Check if target column is categorical and convert to binary if needed
            if df_processed[target_column].dtype == 'object':
                # Convert common categorical values to binary
                df_processed[target_column] = df_processed[target_column].map({
                    'Yes': 1, 'No': 0,
                    'yes': 1, 'no': 0,
                    'YES': 1, 'NO': 0,
                    'True': 1, 'False': 0,
                    'true': 1, 'false': 0,
                    'TRUE': 1, 'FALSE': 0,
                    'Churned': 1, 'Loyal': 0,
                    'churned': 1, 'loyal': 0,
                    '1': 1, '0': 0,
                    1: 1, 0: 0
                }).fillna(0)
            
            # Now convert to int
            df_processed[target_column] = df_processed[target_column].astype(int)
        
        return df_processed
    
    def select_features(self, df, target_column='churn', min_features=5):
        """
        Select most important features for the model
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Remove target column from features
        feature_columns = [col for col in df.columns if col != target_column]
        
        if len(feature_columns) < min_features:
            return feature_columns
        
        # Use Random Forest to get feature importance
        X = df[feature_columns]
        y = df[target_column]
        
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance and select top features
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top features (at least min_features)
        selected_features = feature_importance.head(max(min_features, len(feature_columns)//2))['feature'].tolist()
        
        return selected_features
    
    def train_models(self, df, target_column='churn', test_size=0.2):
        """
        Train multiple models and select the best one
        """
        # Preprocess data
        df_processed = self.preprocess_data(df, target_column)
        
        # Select features
        feature_columns = self.select_features(df_processed, target_column)
        X = df_processed[feature_columns]
        y = df_processed[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train and evaluate each model
        for name, model in self.models.items():
            if name == 'logistic_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            auc = roc_auc_score(y_test, y_pred_proba)
            
            self.model_performance[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'feature_columns': feature_columns
            }
        
        # Select best model based on F1 score
        best_model_name = max(self.model_performance.keys(), 
                             key=lambda x: self.model_performance[x]['f1_score'])
        self.best_model = self.models[best_model_name]
        
        # Store feature importance for best model
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return self.model_performance
    
    def predict_churn(self, df, target_column='churn'):
        """
        Predict churn probabilities for new data
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Preprocess data
        df_processed = self.preprocess_data(df, target_column)
        
        # Get feature columns from training
        feature_columns = self.model_performance[list(self.model_performance.keys())[0]]['feature_columns']
        
        # Ensure all required features are present
        missing_features = set(feature_columns) - set(df_processed.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        X = df_processed[feature_columns]
        
        # Scale features if using logistic regression
        if isinstance(self.best_model, LogisticRegression):
            X_scaled = self.scaler.transform(X)
            churn_probabilities = self.best_model.predict_proba(X_scaled)[:, 1]
        else:
            churn_probabilities = self.best_model.predict_proba(X)[:, 1]
        
        # Create results dataframe
        results = df.copy()
        results['churn_probability'] = churn_probabilities
        results['churn_prediction'] = (churn_probabilities > 0.5).astype(int)
        results['risk_level'] = pd.cut(
            churn_probabilities, 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        return results
    
    def get_model_summary(self):
        """
        Get comprehensive model performance summary
        """
        if not self.model_performance:
            return "No models trained yet."
        
        summary = {
            'best_model': None,
            'performance_comparison': self.model_performance,
            'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else None
        }
        
        # Find best model
        best_model_name = max(self.model_performance.keys(), 
                             key=lambda x: self.model_performance[x]['f1_score'])
        summary['best_model'] = {
            'name': best_model_name,
            'metrics': self.model_performance[best_model_name]
        }
        
        return summary
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        if self.best_model is None:
            raise ValueError("No model to save. Please train the model first.")
        
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.model_performance[list(self.model_performance.keys())[0]]['feature_columns'],
            'model_performance': self.model_performance
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model
        """
        model_data = joblib.load(filepath)
        
        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.model_performance = model_data['model_performance']
        
        print(f"Model loaded from {filepath}")
        return self
