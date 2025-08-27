import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

# Advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available - install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available - install with: pip install lightgbm")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available - install with: pip install catboost")

import joblib
import time
from datetime import datetime

class LunarvMLEngine:
    """
    🌙 Lunarv Advanced ML Engine for Churn Prediction
    Features: XGBoost, LightGBM, CatBoost, Advanced Feature Selection, 
    Class Imbalance Handling, Target Encoding
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoders = {}
        self.best_model = None
        self.feature_importance = None
        self.model_performance = {}
        self.training_features = None
        self.training_time = {}
        self.data_info = {}
        
        # Initialize available models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all available ML models"""
        # Random Forest (always available)
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200, 
            max_depth=15, 
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting (always available)
        from sklearn.ensemble import GradientBoostingClassifier
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=200, 
            learning_rate=0.1, 
            max_depth=8,
            random_state=42
        )
        
        # Logistic Regression (always available)
        self.models['logistic_regression'] = LogisticRegression(
            random_state=42, 
            max_iter=2000,
            solver='liblinear'
        )
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
        
        # LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
        
        # CatBoost (if available)
        if CATBOOST_AVAILABLE:
            self.models['catboost'] = cb.CatBoostClassifier(
                iterations=200,
                depth=8,
                learning_rate=0.1,
                random_seed=42,
                verbose=False
            )
    
    def estimate_training_time(self, dataset_size):
        """Estimate training time based on dataset size"""
        if dataset_size < 1000:
            return "1-2 minutes"
        elif dataset_size < 10000:
            return "2-3 minutes"
        elif dataset_size < 50000:
            return "3-5 minutes"
        else:
            return "5-10 minutes"
    
    def analyze_data_quality(self, df, target_column='churn'):
        """Comprehensive data quality analysis"""
        analysis = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data': {},
            'data_types': {},
            'unique_values': {},
            'class_balance': {},
            'recommendations': []
        }
        
        # Missing data analysis
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        for col in df.columns:
            analysis['missing_data'][col] = {
                'count': int(missing_data[col]),
                'percentage': round(missing_percent[col], 2)
            }
        
        # Data types and unique values
        for col in df.columns:
            analysis['data_types'][col] = str(df[col].dtype)
            analysis['unique_values'][col] = int(df[col].nunique())
        
        # Class balance analysis
        if target_column in df.columns:
            target_counts = df[target_column].value_counts()
            total = len(df)
            
            for value, count in target_counts.items():
                analysis['class_balance'][str(value)] = {
                    'count': int(count),
                    'percentage': round((count / total) * 100, 2)
                }
            
            # Check for class imbalance
            min_class_ratio = target_counts.min() / target_counts.max()
            if min_class_ratio < 0.2:
                analysis['recommendations'].append("⚠️ Severe class imbalance detected - consider oversampling/undersampling")
            elif min_class_ratio < 0.4:
                analysis['recommendations'].append("⚠️ Moderate class imbalance detected - consider class weights")
        
        # Missing data recommendations
        high_missing_cols = [col for col, data in analysis['missing_data'].items() 
                           if data['percentage'] > 20]
        if high_missing_cols:
            analysis['recommendations'].append(f"⚠️ High missing data in: {', '.join(high_missing_cols[:3])}")
        
        return analysis
    
    def handle_class_imbalance(self, X, y, method='auto'):
        """
        Handle class imbalance using various techniques
        methods: 'auto', 'oversample', 'undersample', 'class_weights', 'none'
        """
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        
        class_counts = np.bincount(y)
        min_class_ratio = class_counts.min() / class_counts.max()
        
        if method == 'auto':
            if min_class_ratio < 0.1:
                method = 'oversample'
            elif min_class_ratio < 0.3:
                method = 'class_weights'
            else:
                method = 'none'
        
        if method == 'oversample':
            smote = SMOTE(random_state=42, k_neighbors=min(5, class_counts.min()-1))
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print(f"✅ Applied SMOTE oversampling: {len(y)} → {len(y_resampled)} samples")
            return X_resampled, y_resampled
        
        elif method == 'undersample':
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X, y)
            print(f"✅ Applied random undersampling: {len(y)} → {len(y_resampled)} samples")
            return X_resampled, y_resampled
        
        elif method == 'class_weights':
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            weight_dict = dict(zip(np.unique(y), class_weights))
            
            # Apply weights to models that support it
            for name, model in self.models.items():
                if hasattr(model, 'class_weight'):
                    if name == 'random_forest':
                        model.class_weight = 'balanced'
                    elif name == 'logistic_regression':
                        model.class_weight = 'balanced'
                    elif name == 'xgboost' and XGBOOST_AVAILABLE:
                        model.scale_pos_weight = weight_dict[1] / weight_dict[0]
                    elif name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                        model.class_weight = weight_dict
            
            print(f"✅ Applied class weights: {weight_dict}")
            return X, y
        
        else:  # method == 'none'
            print("ℹ️ No class imbalance handling applied")
            return X, y
    
    def apply_target_encoding(self, df, categorical_columns, target_column='churn', smoothing=10):
        """
        Apply target encoding with smoothing to prevent overfitting
        smoothing: higher values = more smoothing (less overfitting)
        """
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col == target_column:
                continue
                
            # Calculate target encoding with smoothing
            global_mean = df[target_column].mean()
            agg_stats = df.groupby(col)[target_column].agg(['count', 'mean'])
            
            # Apply smoothing formula: (count * mean + smoothing * global_mean) / (count + smoothing)
            smoothed_means = (agg_stats['count'] * agg_stats['mean'] + 
                            smoothing * global_mean) / (agg_stats['count'] + smoothing)
            
            # Create mapping
            encoding_map = smoothed_means.to_dict()
            
            # Apply encoding
            df_encoded[col] = df_encoded[col].map(encoding_map)
            
            # Fill missing values with global mean
            df_encoded[col] = df_encoded[col].fillna(global_mean)
            
            # Store encoder for later use
            self.target_encoders[col] = {
                'mapping': encoding_map,
                'global_mean': global_mean,
                'smoothing': smoothing
            }
            
            print(f"✅ Applied target encoding to '{col}' (smoothing={smoothing})")
        
        return df_encoded
    
    def advanced_feature_selection(self, df, target_column='churn', method='tree_based', n_features=None):
        """
        Advanced feature selection using multiple methods
        methods: 'tree_based', 'correlation', 'mutual_info', 'recursive'
        """
        if n_features is None:
            n_features = min(20, len(df.columns) - 1)  # Default to 20 or max available
        
        feature_columns = [col for col in df.columns if col != target_column]
        
        if len(feature_columns) <= n_features:
            print(f"ℹ️ Only {len(feature_columns)} features available, using all")
            return feature_columns
        
        print(f"🔍 Selecting {n_features} best features from {len(feature_columns)} using {method} method...")
        
        if method == 'tree_based':
            # Use Random Forest for feature importance
            X = df[feature_columns]
            y = df[target_column]
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            selected_features = feature_importance.head(n_features)['feature'].tolist()
            
        elif method == 'correlation':
            # Use correlation with target
            correlations = df[feature_columns].corrwith(df[target_column]).abs()
            selected_features = correlations.nlargest(n_features).index.tolist()
            
        elif method == 'mutual_info':
            # Use mutual information
            from sklearn.feature_selection import mutual_info_classif
            X = df[feature_columns]
            y = df[target_column]
            
            mi_scores = mutual_info_classif(X, y, random_state=42)
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': mi_scores
            }).sort_values('importance', ascending=False)
            
            selected_features = feature_importance.head(n_features)['feature'].tolist()
            
        else:  # recursive
            # Use recursive feature elimination
            from sklearn.feature_selection import RFE
            from sklearn.linear_model import LogisticRegression
            
            X = df[feature_columns]
            y = df[target_column]
            
            estimator = LogisticRegression(random_state=42, max_iter=1000)
            rfe = RFE(estimator, n_features_to_select=n_features)
            rfe.fit(X, y)
            
            selected_features = [feature_columns[i] for i in range(len(feature_columns)) if rfe.support_[i]]
        
        print(f"✅ Selected features: {', '.join(selected_features[:5])}{'...' if len(selected_features) > 5 else ''}")
        return selected_features
    
    def smart_churn_conversion(self, series):
        """Smart conversion of churn column to binary"""
        unique_values = series.unique()
        series_str = series.astype(str).str.strip().str.lower()
        
        # Define conversion patterns
        positive_patterns = ['yes', 'true', '1', '1.0', 'churned', 'churn', 'left', 'gone', 'attrited']
        negative_patterns = ['no', 'false', '0', '0.0', 'loyal', 'stay', 'retain', 'active', 'current']
        
        conversion_map = {}
        
        for val in unique_values:
            val_str = str(val).strip().lower()
            if val_str in positive_patterns:
                conversion_map[val] = 1
            elif val_str in negative_patterns:
                conversion_map[val] = 0
            else:
                try:
                    num_val = float(val)
                    if num_val == 1 or num_val == 0:
                        conversion_map[val] = int(num_val)
                    else:
                        conversion_map[val] = 1 if num_val > 0.5 else 0
                except:
                    print(f"⚠️ Unknown churn value: '{val}'. Treating as 'No' (0)")
                    conversion_map[val] = 0
        
        converted = series.map(conversion_map)
        
        if converted.isna().any():
            print("⚠️ Some values couldn't be converted. Filling with 0")
            converted = converted.fillna(0)
        
        return converted.astype(int)
    
    def preprocess_data(self, df, target_column='churn', handle_missing=True, 
                       target_encode_categorical=True, class_imbalance_method='auto'):
        """Comprehensive data preprocessing"""
        print("🔄 Starting comprehensive data preprocessing...")
        
        df_processed = df.copy()
        
        # Handle missing values
        if handle_missing:
            numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
            categorical_columns = df_processed.select_dtypes(include=['object']).columns
            
            for col in numeric_columns:
                if col != target_column and df_processed[col].isna().any():
                    median_val = df_processed[col].median()
                    df_processed[col] = df_processed[col].fillna(median_val)
                    print(f"✅ Filled missing values in '{col}' with median: {median_val}")
            
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
        
        # Target encoding for categorical variables
        if target_encode_categorical:
            categorical_columns = df_processed.select_dtypes(include=['object']).columns
            categorical_columns = [col for col in categorical_columns if col != target_column]
            
            if len(categorical_columns) > 0:
                df_processed = self.apply_target_encoding(df_processed, categorical_columns, target_column)
            else:
                print("ℹ️ No categorical columns to encode")
        else:
            # Traditional label encoding
            categorical_columns = df_processed.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col != target_column:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self.label_encoders[col] = le
                    print(f"✅ Label encoded categorical column '{col}' ({len(le.classes_)} unique values)")
        
        print(f"✅ Data preprocessing completed! Shape: {df_processed.shape}")
        return df_processed
    
    def train_models(self, df, target_column='churn', test_size=0.2, 
                    feature_selection_method='tree_based', class_imbalance_method='auto'):
        """Train all available models with comprehensive evaluation"""
        start_time = time.time()
        print("🚀 Starting advanced model training...")
        
        # Data quality analysis
        self.data_info = self.analyze_data_quality(df, target_column)
        print(f"📊 Dataset: {self.data_info['total_rows']:,} rows, {self.data_info['total_columns']} columns")
        
        # Preprocess data
        df_processed = self.preprocess_data(df, target_column, 
                                         target_encode_categorical=True,
                                         class_imbalance_method=class_imbalance_method)
        
        # Feature selection
        feature_columns = self.advanced_feature_selection(df_processed, target_column, 
                                                        method=feature_selection_method)
        self.training_features = feature_columns
        
        X = df_processed[feature_columns]
        y = df_processed[target_column]
        
        print(f"📊 Training data shape: X={X.shape}, y={y.shape}")
        print(f"🎯 Target distribution: {dict(y.value_counts())}")
        
        # Handle class imbalance
        X_resampled, y_resampled = self.handle_class_imbalance(X, y, class_imbalance_method)
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=test_size, 
                random_state=42, stratify=y_resampled
            )
            print(f"✅ Data split: Train={X_train.shape}, Test={X_test.shape}")
        except Exception as e:
            print(f"⚠️ Stratified split failed, using random split: {str(e)}")
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=test_size, random_state=42
            )
        
        # Scale features for models that need it
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train and evaluate each model
        for name, model in self.models.items():
            try:
                print(f"🔄 Training {name}...")
                model_start_time = time.time()
                
                # Select appropriate data (scaled or unscaled)
                if name in ['logistic_regression']:
                    X_train_model, X_test_model = X_train_scaled, X_test_scaled
                else:
                    X_train_model, X_test_model = X_train, X_test
                
                # Train model
                model.fit(X_train_model, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_model)
                y_pred_proba = model.predict_proba(X_test_model)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    auc = 0.5
                
                # Store performance
                self.model_performance[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc': auc,
                    'feature_columns': feature_columns,
                    'training_time': time.time() - model_start_time
                }
                
                self.training_time[name] = time.time() - model_start_time
                
                print(f"   ✅ {name}: Accuracy={accuracy:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
                
            except Exception as e:
                print(f"   ❌ {name} training failed: {str(e)}")
                self.model_performance[name] = {
                    'accuracy': 0, 'precision': 0, 'recall': 0, 
                    'f1_score': 0, 'auc': 0.5, 'feature_columns': feature_columns,
                    'training_time': 0
                }
                self.training_time[name] = 0
        
        # Select best model
        valid_models = {k: v for k, v in self.model_performance.items() if v['f1_score'] > 0}
        
        if not valid_models:
            raise ValueError("No models trained successfully!")
        
        best_model_name = max(valid_models.keys(), 
                             key=lambda x: valid_models[x]['f1_score'])
        self.best_model = self.models[best_model_name]
        
        total_training_time = time.time() - start_time
        print(f"🏆 Best model: {best_model_name} (F1: {valid_models[best_model_name]['f1_score']:.3f})")
        print(f"⏱️ Total training time: {total_training_time:.2f} seconds")
        
        # Store feature importance for best model
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return self.model_performance
    
    def predict_churn(self, df, target_column='churn'):
        """Make churn predictions using the best trained model"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        if self.training_features is None:
            raise ValueError("Training features not available. Please retrain the model.")
        
        print("🔮 Making churn predictions...")
        
        # Preprocess data
        df_processed = self.preprocess_data(df, target_column, 
                                         target_encode_categorical=True)
        
        # Get feature columns from training
        feature_columns = self.training_features
        
        # Ensure all required features are present
        missing_features = set(feature_columns) - set(df_processed.columns)
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
            print("   Adding missing features with default values...")
            
            for feature in missing_features:
                if feature in self.target_encoders:
                    # For target encoded features, use global mean
                    df_processed[feature] = self.target_encoders[feature]['global_mean']
                else:
                    # For other features, use 0
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
    
    def get_model_summary(self):
        """Get comprehensive model performance summary"""
        if not self.model_performance:
            return "No models trained yet."
        
        summary = {
            'best_model': None,
            'performance_comparison': self.model_performance,
            'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else None,
            'training_times': self.training_time,
            'data_info': self.data_info
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
        """Save the trained model and all components"""
        if self.best_model is None:
            raise ValueError("No model to save. Please train the model first.")
        
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'target_encoders': self.target_encoders,
            'feature_columns': self.training_features,
            'model_performance': self.model_performance,
            'feature_importance': self.feature_importance,
            'training_time': self.training_time,
            'data_info': self.data_info
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model and all components"""
        model_data = joblib.load(filepath)
        
        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data.get('label_encoders', {})
        self.target_encoders = model_data.get('target_encoders', {})
        self.model_performance = model_data['model_performance']
        self.feature_importance = model_data.get('feature_importance', None)
        self.training_features = model_data.get('feature_columns', None)
        self.training_time = model_data.get('training_time', {})
        self.data_info = model_data.get('data_info', {})
        
        print(f"✅ Model loaded from {filepath}")
        return self

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Lunarv ML Engine...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'tenure_months': np.random.randint(1, 120, n_samples),
        'monthly_charges': np.random.uniform(20, 150, n_samples),
        'total_charges': np.random.uniform(100, 5000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
    }
    
    df = pd.DataFrame(data)
    
    # Test the engine
    engine = LunarvMLEngine()
    
    try:
        # Train models
        performance = engine.train_models(df, 'churn', 
                                        feature_selection_method='tree_based',
                                        class_imbalance_method='auto')
        
        print("\n📊 Training Results:")
        for name, metrics in performance.items():
            print(f"   {name}: F1={metrics['f1_score']:.3f}, Accuracy={metrics['accuracy']:.3f}")
        
        # Make predictions
        results = engine.predict_churn(df, 'churn')
        print(f"\n🔮 Prediction Results:")
        print(f"   Shape: {results.shape}")
        print(f"   Risk Levels: {dict(results['risk_level'].value_counts())}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
