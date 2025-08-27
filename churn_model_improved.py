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

class ImprovedChurnPredictionModel:
    """
    Improved Churn Prediction Model with better error handling and validation
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
        self.training_features = None
        
    def validate_data(self, df, target_column='churn'):
        """
        Validate data before processing
        """
        if df.empty:
            raise ValueError("Dataset is empty!")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Check for sufficient data
        if len(df) < 50:
            raise ValueError("Dataset too small! Need at least 50 records for reliable training")
        
        # Check target column distribution
        target_counts = df[target_column].value_counts()
        if len(target_counts) < 2:
            raise ValueError("Target column must have at least 2 unique values")
        
        # Check for extreme class imbalance
        min_class_ratio = target_counts.min() / target_counts.max()
        if min_class_ratio < 0.05:
            warnings.warn(f"Warning: Extreme class imbalance detected! Ratio: {min_class_ratio:.3f}")
        
        return True
    
    def smart_churn_conversion(self, series):
        """
        Smart conversion of churn column to binary
        """
        # First, try to identify the pattern
        unique_values = series.unique()
        
        # Convert to string for consistent handling
        series_str = series.astype(str).str.strip().str.lower()
        
        # Define conversion patterns
        positive_patterns = ['yes', 'true', '1', '1.0', 'churned', 'churn', 'left', 'gone']
        negative_patterns = ['no', 'false', '0', '0.0', 'loyal', 'stay', 'retain', 'active']
        
        # Create conversion mapping
        conversion_map = {}
        
        for val in unique_values:
            val_str = str(val).strip().lower()
            if val_str in positive_patterns:
                conversion_map[val] = 1
            elif val_str in negative_patterns:
                conversion_map[val] = 0
            else:
                # Try to convert to numeric
                try:
                    num_val = float(val)
                    if num_val == 1 or num_val == 0:
                        conversion_map[val] = int(num_val)
                    else:
                        # If it's a number but not 0/1, treat as probability
                        conversion_map[val] = 1 if num_val > 0.5 else 0
                except:
                    # Unknown value - ask user or use default
                    print(f"⚠️ Unknown churn value: '{val}'. Treating as 'No' (0)")
                    conversion_map[val] = 0
        
        # Apply conversion
        converted = series.map(conversion_map)
        
        # Validate conversion
        if converted.isna().any():
            print("⚠️ Some values couldn't be converted. Filling with 0")
            converted = converted.fillna(0)
        
        return converted.astype(int)
    
    def preprocess_data(self, df, target_column='churn'):
        """
        Improved data preprocessing with better error handling
        """
        try:
            # Validate data first
            self.validate_data(df, target_column)
            
            df_processed = df.copy()
            
            # Handle missing values more intelligently
            numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
            categorical_columns = df_processed.select_dtypes(include=['object']).columns
            
            # Fill numeric missing values with median
            for col in numeric_columns:
                if col != target_column and df_processed[col].isna().any():
                    median_val = df_processed[col].median()
                    df_processed[col] = df_processed[col].fillna(median_val)
                    print(f"✅ Filled missing values in '{col}' with median: {median_val}")
            
            # Fill categorical missing values with mode
            for col in categorical_columns:
                if col != target_column and df_processed[col].isna().any():
                    mode_val = df_processed[col].mode()[0]
                    df_processed[col] = df_processed[col].fillna(mode_val)
                    print(f"✅ Filled missing values in '{col}' with mode: '{mode_val}'")
            
            # Smart churn conversion
            if target_column in df_processed.columns:
                print(f"🔄 Converting churn column '{target_column}' to binary...")
                original_values = df_processed[target_column].value_counts()
                print(f"   Original values: {dict(original_values)}")
                
                df_processed[target_column] = self.smart_churn_conversion(df_processed[target_column])
                
                converted_values = df_processed[target_column].value_counts()
                print(f"   Converted to: {dict(converted_values)}")
            
            # Encode categorical variables
            for col in categorical_columns:
                if col != target_column:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self.label_encoders[col] = le
                    print(f"✅ Encoded categorical column '{col}' ({len(le.classes_)} unique values)")
            
            print(f"✅ Data preprocessing completed! Shape: {df_processed.shape}")
            return df_processed
            
        except Exception as e:
            print(f"❌ Error in data preprocessing: {str(e)}")
            raise
    
    def select_features(self, df, target_column='churn', min_features=5):
        """
        Improved feature selection with validation
        """
        try:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")
            
            # Remove target column from features
            feature_columns = [col for col in df.columns if col != target_column]
            
            if len(feature_columns) < min_features:
                print(f"⚠️ Only {len(feature_columns)} features available, using all")
                return feature_columns
            
            # Use Random Forest to get feature importance
            X = df[feature_columns]
            y = df[target_column]
            
            # Handle potential errors in feature selection
            try:
                rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                rf.fit(X, y)
                
                # Get feature importance and select top features
                feature_importance = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': rf.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Select top features (at least min_features)
                selected_features = feature_importance.head(max(min_features, len(feature_columns)//2))['feature'].tolist()
                
                print(f"✅ Selected {len(selected_features)} features out of {len(feature_columns)}")
                print(f"   Top features: {selected_features[:3]}")
                
                return selected_features
                
            except Exception as e:
                print(f"⚠️ Feature selection failed, using all features: {str(e)}")
                return feature_columns
                
        except Exception as e:
            print(f"❌ Error in feature selection: {str(e)}")
            raise
    
    def train_models(self, df, target_column='churn', test_size=0.2):
        """
        Improved model training with comprehensive error handling
        """
        try:
            print("🚀 Starting model training...")
            
            # Preprocess data
            df_processed = self.preprocess_data(df, target_column)
            
            # Select features
            feature_columns = self.select_features(df_processed, target_column)
            self.training_features = feature_columns  # Store for later use
            
            X = df_processed[feature_columns]
            y = df_processed[target_column]
            
            print(f"📊 Training data shape: X={X.shape}, y={y.shape}")
            print(f"🎯 Target distribution: {dict(y.value_counts())}")
            
            # Split data with stratification
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                print(f"✅ Data split: Train={X_train.shape}, Test={X_test.shape}")
            except Exception as e:
                print(f"⚠️ Stratified split failed, using random split: {str(e)}")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train and evaluate each model
            for name, model in self.models.items():
                try:
                    print(f"🔄 Training {name}...")
                    
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
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    # Handle AUC calculation
                    try:
                        auc = roc_auc_score(y_test, y_pred_proba)
                    except:
                        auc = 0.5  # Default value if AUC can't be calculated
                    
                    self.model_performance[name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'auc': auc,
                        'feature_columns': feature_columns
                    }
                    
                    print(f"   ✅ {name}: Accuracy={accuracy:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
                    
                except Exception as e:
                    print(f"   ❌ {name} training failed: {str(e)}")
                    self.model_performance[name] = {
                        'accuracy': 0, 'precision': 0, 'recall': 0, 
                        'f1_score': 0, 'auc': 0.5, 'feature_columns': feature_columns
                    }
            
            # Select best model based on F1 score
            valid_models = {k: v for k, v in self.model_performance.items() if v['f1_score'] > 0}
            
            if not valid_models:
                raise ValueError("No models trained successfully!")
            
            best_model_name = max(valid_models.keys(), 
                                 key=lambda x: valid_models[x]['f1_score'])
            self.best_model = self.models[best_model_name]
            
            print(f"🏆 Best model: {best_model_name} (F1: {valid_models[best_model_name]['f1_score']:.3f})")
            
            # Store feature importance for best model
            if hasattr(self.best_model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': self.best_model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            return self.model_performance
            
        except Exception as e:
            print(f"❌ Model training failed: {str(e)}")
            raise
    
    def predict_churn(self, df, target_column='churn'):
        """
        Improved churn prediction with better error handling
        """
        try:
            if self.best_model is None:
                raise ValueError("Model not trained yet. Please train the model first.")
            
            if self.training_features is None:
                raise ValueError("Training features not available. Please retrain the model.")
            
            print("🔮 Making churn predictions...")
            
            # Preprocess data
            df_processed = self.preprocess_data(df, target_column)
            
            # Get feature columns from training
            feature_columns = self.training_features
            
            # Ensure all required features are present
            missing_features = set(feature_columns) - set(df_processed.columns)
            if missing_features:
                print(f"⚠️ Missing features: {missing_features}")
                print("   Adding missing features with default values...")
                
                for feature in missing_features:
                    if feature in self.label_encoders:
                        # For categorical features, use most common value
                        df_processed[feature] = 0
                    else:
                        # For numeric features, use 0
                        df_processed[feature] = 0
            
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
            
            # Create risk levels
            results['risk_level'] = pd.cut(
                churn_probabilities, 
                bins=[0, 0.3, 0.7, 1.0], 
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
            
            print(f"✅ Predictions completed for {len(results)} customers")
            print(f"   Risk distribution: {dict(results['risk_level'].value_counts())}")
            
            return results
            
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            raise
    
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
        valid_models = {k: v for k, v in self.model_performance.items() if v['f1_score'] > 0}
        if valid_models:
            best_model_name = max(valid_models.keys(), 
                                 key=lambda x: valid_models[x]['f1_score'])
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
            'feature_columns': self.training_features,
            'model_performance': self.model_performance
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model
        """
        model_data = joblib.load(filepath)
        
        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.model_performance = model_data['model_performance']
        self.training_features = model_data.get('feature_columns', None)
        
        print(f"✅ Model loaded from {filepath}")
        return self

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Improved Churn Model...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'tenure': np.random.randint(1, 120, n_samples),
        'monthly_charges': np.random.uniform(20, 150, n_samples),
        'total_charges': np.random.uniform(100, 5000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])  # 30% churn rate
    }
    
    df = pd.DataFrame(data)
    
    # Test the model
    model = ImprovedChurnPredictionModel()
    
    try:
        # Train models
        performance = model.train_models(df, 'churn')
        print("\n📊 Training Results:")
        for name, metrics in performance.items():
            print(f"   {name}: F1={metrics['f1_score']:.3f}, Accuracy={metrics['accuracy']:.3f}")
        
        # Make predictions
        results = model.predict_churn(df, 'churn')
        print(f"\n🔮 Prediction Results:")
        print(f"   Shape: {results.shape}")
        print(f"   Risk Levels: {dict(results['risk_level'].value_counts())}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
