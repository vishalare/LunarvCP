#!/usr/bin/env python3
"""
Demo script for Churn Prediction Tool
This script demonstrates the core functionality without the web interface
"""

import pandas as pd
import numpy as np
from churn_model import ChurnPredictionModel
from data_utils import DataHandler
from visualization_utils import ChurnVisualizer
import matplotlib.pyplot as plt
import time

def run_demo():
    """
    Run a complete demo of the churn prediction tool
    """
    print("🚀 Starting Churn Prediction Tool Demo")
    print("=" * 50)
    
    # Step 1: Create sample data
    print("\n📊 Step 1: Creating sample data...")
    data_handler = DataHandler()
    sample_df = data_handler.create_sample_data()
    
    print(f"✅ Generated {len(sample_df):,} customer records")
    print(f"📋 Columns: {', '.join(sample_df.columns)}")
    print(f"🎯 Churn rate: {(sample_df['churn'].sum() / len(sample_df) * 100):.1f}%")
    
    # Step 2: Data validation
    print("\n🔍 Step 2: Validating data structure...")
    is_valid, errors, warnings = data_handler.validate_data_structure(sample_df)
    
    if is_valid:
        print("✅ Data structure is valid!")
        if warnings:
            print("⚠️ Warnings:")
            for warning in warnings:
                print(f"   - {warning}")
    else:
        print("❌ Data structure issues:")
        for error in errors:
            print(f"   - {error}")
        return
    
    # Step 3: Initialize and train model
    print("\n🤖 Step 3: Training churn prediction model...")
    model = ChurnPredictionModel()
    
    start_time = time.time()
    model_performance = model.train_models(sample_df, 'churn')
    training_time = time.time() - start_time
    
    print(f"✅ Model training completed in {training_time:.2f} seconds")
    
    # Step 4: Show model performance
    print("\n📊 Step 4: Model Performance Results")
    print("-" * 40)
    
    for model_name, metrics in model_performance.items():
        print(f"\n{model_name.replace('_', ' ').title()}:")
        print(f"  Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        print(f"  F1 Score: {metrics['f1_score']:.3f} ({metrics['f1_score']*100:.1f}%)")
        print(f"  Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
        print(f"  Recall: {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
        print(f"  AUC: {metrics['auc']:.3f} ({metrics['auc']*100:.1f}%)")
    
    # Get best model
    best_model_name = max(model_performance.keys(), 
                         key=lambda x: model_performance[x]['f1_score'])
    best_metrics = model_performance[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name.replace('_', ' ').title()}")
    print(f"   F1 Score: {best_metrics['f1_score']:.3f} ({best_metrics['f1_score']*100:.1f}%)")
    
    # Step 5: Generate predictions
    print("\n🔮 Step 5: Generating churn predictions...")
    
    start_time = time.time()
    results_df = model.predict_churn(sample_df, 'churn')
    prediction_time = time.time() - start_time
    
    print(f"✅ Predictions generated in {prediction_time:.2f} seconds")
    
    # Step 6: Show prediction results
    print("\n📈 Step 6: Prediction Results Summary")
    print("-" * 40)
    
    total_customers = len(results_df)
    high_risk = len(results_df[results_df['churn_probability'] > 0.7])
    medium_risk = len(results_df[(results_df['churn_probability'] > 0.3) & (results_df['churn_probability'] <= 0.7)])
    low_risk = len(results_df[results_df['churn_probability'] <= 0.3])
    
    print(f"Total Customers: {total_customers:,}")
    print(f"High Risk (>70%): {high_risk:,} ({high_risk/total_customers*100:.1f}%)")
    print(f"Medium Risk (30-70%): {medium_risk:,} ({medium_risk/total_customers*100:.1f}%)")
    print(f"Low Risk (<30%): {low_risk:,} ({low_risk/total_customers*100:.1f}%)")
    
    print(f"\nAverage Churn Probability: {results_df['churn_probability'].mean():.3f} ({results_df['churn_probability'].mean()*100:.1f}%)")
    
    # Step 7: Show sample predictions
    print("\n📋 Step 7: Sample Predictions")
    print("-" * 40)
    
    # Show some high-risk customers
    high_risk_sample = results_df[results_df['churn_probability'] > 0.7].head(5)
    print("\nHigh Risk Customers (Sample):")
    for _, row in high_risk_sample.iterrows():
        print(f"  Customer {row['customer_id']}: {row['churn_probability']:.1%} churn probability")
    
    # Show some low-risk customers
    low_risk_sample = results_df[results_df['churn_probability'] < 0.3].head(5)
    print("\nLow Risk Customers (Sample):")
    for _, row in low_risk_sample.iterrows():
        print(f"  Customer {row['customer_id']}: {row['churn_probability']:.1%} churn probability")
    
    # Step 8: Feature importance
    print("\n🎯 Step 8: Feature Importance")
    print("-" * 40)
    
    if model.feature_importance is not None:
        print("Top 10 Most Important Features:")
        for i, (_, row) in enumerate(model.feature_importance.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.3f}")
    
    # Step 9: Model summary
    print("\n📊 Step 9: Complete Model Summary")
    print("-" * 40)
    
    model_summary = model.get_model_summary()
    print(f"Best Model: {model_summary['best_model']['name'].replace('_', ' ').title()}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Prediction Time: {prediction_time:.2f} seconds")
    print(f"Total Processing Time: {training_time + prediction_time:.2f} seconds")
    
    # Step 10: Save model (optional)
    print("\n💾 Step 10: Model Persistence")
    print("-" * 40)
    
    try:
        os.makedirs('models', exist_ok=True)
        model_path = f"models/churn_model_demo_{int(time.time())}.joblib"
        model.save_model(model_path)
        print(f"✅ Model saved to: {model_path}")
    except Exception as e:
        print(f"⚠️ Could not save model: {e}")
    
    print("\n🎉 Demo completed successfully!")
    print("=" * 50)
    print("\n💡 Next steps:")
    print("1. Run 'streamlit run app.py' to use the web interface")
    print("2. Upload your own customer data")
    print("3. Customize model parameters in config.py")
    print("4. Integrate with your business processes")
    
    return results_df, model, model_performance

def quick_test():
    """
    Quick test of core functionality
    """
    print("🧪 Running quick test...")
    
    try:
        # Test data creation
        data_handler = DataHandler()
        sample_df = data_handler.create_sample_data()
        print(f"✅ Sample data created: {len(sample_df)} records")
        
        # Test model initialization
        model = ChurnPredictionModel()
        print("✅ Model initialized")
        
        # Test training
        model_performance = model.train_models(sample_df, 'churn')
        print(f"✅ Model trained: {len(model_performance)} algorithms tested")
        
        # Test predictions
        results_df = model.predict_churn(sample_df, 'churn')
        print(f"✅ Predictions generated: {len(results_df)} results")
        
        print("🎉 Quick test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

if __name__ == "__main__":
    import os
    
    print("Churn Prediction Tool - Demo Script")
    print("Choose an option:")
    print("1. Run full demo")
    print("2. Run quick test")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        results, model, performance = run_demo()
    elif choice == "2":
        success = quick_test()
        if success:
            print("\n🚀 Ready to run the full application!")
        else:
            print("\n❌ Please check your installation and dependencies.")
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Running quick test instead...")
        quick_test()
