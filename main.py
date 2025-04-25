#!/usr/bin/env python
# Main script for car insurance claims prediction

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb
import time
import pickle
import logging

# Import updated custom modules
from data_processing import (
    load_and_preprocess_data, 
    feature_engineering, 
    prepare_features_for_xgboost
)
from visualization import (
    plot_distributions, 
    plot_feature_importance, 
    plot_model_performance,
    plot_lift_chart,
    plot_precision_recall_curve_func
)
from models import (
    train_xgboost_model, 
    train_xgboost_poisson_model, 
    train_xgboost_tweedie_model,
    evaluate_model
)

def main():
    # --- Setup --- 
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots/distributions', exist_ok=True)
    os.makedirs('plots/model', exist_ok=True)
    
    # Set up logging (same config as notebook)
    log_file = 'car_insurance_model_main.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting main script execution...")
    print(f"Logging to {log_file}")

    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Start timer
    start_time = time.time()
    
    # --- Data Loading and Preparation --- 
    data_path = "freMTPL2freq.csv"
    
    # Load and preprocess data using the updated function
    logger.info("Loading and preprocessing data...")
    df = load_and_preprocess_data(data_path)
    
    # Feature engineering using the updated function
    logger.info("Performing feature engineering...")
    df = feature_engineering(df)
    
    # Exploratory data analysis and visualization
    logger.info("Generating exploratory visualizations...")
    plot_distributions(df)
    
    # --- Data Splitting and Preparation for Modeling --- 
    logger.info("Splitting data into train and test sets...")
    # Define columns to drop - IDpol and the target ClaimNb are always dropped
    # ClaimFreq might not exist depending on preprocessing choices, handle gracefully
    columns_to_drop = ['ClaimNb', 'IDpol']
    if 'ClaimFreq' in df.columns:
        logger.info("'ClaimFreq' column found and will be dropped.")
        columns_to_drop.append('ClaimFreq')
        
    # Check if all columns to drop actually exist in the dataframe
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    logger.info(f"Dropping columns: {columns_to_drop}")

    features = df.drop(columns=columns_to_drop, axis=1)
    target = df['ClaimNb']
    exposure = df['Exposure']
    
    logger.info(f"Feature set shape: {features.shape}")
    logger.info(f"Target shape: {target.shape}")
    logger.info(f"Exposure shape: {exposure.shape}")
    
    X_train, X_test, y_train, y_test, exposure_train, exposure_test = train_test_split(
        features, target, exposure, test_size=0.2, random_state=42
    )
    logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # Prepare features for XGBoost (handle categorical)
    logger.info("Preparing features for XGBoost (converting categoricals)...")
    X_train_prep = prepare_features_for_xgboost(X_train)
    X_test_prep = prepare_features_for_xgboost(X_test)
    logger.info(f"Prepared training features shape: {X_train_prep.shape}")
    logger.info(f"Prepared test features shape: {X_test_prep.shape}")
    
    # --- Model Training --- 
    
    # 1. Standard XGBoost Model
    logger.info("Training Standard XGBoost Model...")
    standard_model, _ = train_xgboost_model(X_train_prep, y_train)
    logger.info("Saving Standard XGBoost model...")
    with open('models/standard_xgboost_model.pkl', 'wb') as f:
        pickle.dump(standard_model, f)
    
    # 2. Poisson XGBoost Model
    logger.info("Training Poisson XGBoost Model...")
    poisson_model, poisson_predictions, poisson_feature_importance = train_xgboost_poisson_model(
        X_train_prep, y_train, X_test_prep, exposure_train, exposure_test
    )
    logger.info("Saving Poisson XGBoost model...")
    with open('models/poisson_xgboost_model.pkl', 'wb') as f:
        pickle.dump(poisson_model, f)
        
    # 3. Tweedie XGBoost Model
    logger.info("Training Tweedie XGBoost Model...")
    tweedie_model, tweedie_predictions, tweedie_feature_importance = train_xgboost_tweedie_model(
        X_train_prep, y_train, X_test_prep, exposure_train, exposure_test, tweedie_variance_power=1.5
    )
    logger.info("Saving Tweedie XGBoost model...")
    with open('models/tweedie_xgboost_model.pkl', 'wb') as f:
        pickle.dump(tweedie_model, f)

    # --- Model Evaluation --- 
    logger.info("Evaluating models...")

    # Standard Model Evaluation
    logger.info("Evaluating Standard XGBoost model...")
    standard_predictions = standard_model.predict(X_test_prep)
    # Recalculate feature importance for standard model based on the trained model
    std_importance_vals = standard_model.feature_importances_
    standard_feature_importance = pd.DataFrame({
        'Feature': X_train_prep.columns,
        'Importance': std_importance_vals
    }).sort_values(by='Importance', ascending=False)
    standard_metrics, _, standard_eval_df = evaluate_model(
        y_test, standard_predictions, "Standard XGBoost", X_test=X_test_prep, feature_importance=standard_feature_importance
    )

    # Poisson Model Evaluation (predictions already generated during training)
    logger.info("Evaluating Poisson XGBoost model...")
    poisson_metrics, _, poisson_eval_df = evaluate_model(
        y_test, poisson_predictions, "Poisson XGBoost", X_test=X_test_prep, feature_importance=poisson_feature_importance
    )

    # Tweedie Model Evaluation (predictions already generated during training)
    logger.info("Evaluating Tweedie XGBoost model...")
    tweedie_metrics, _, tweedie_eval_df = evaluate_model(
        y_test, tweedie_predictions, "Tweedie XGBoost", X_test=X_test_prep, feature_importance=tweedie_feature_importance
    )
    
    # --- Visualize Model Results --- 
    logger.info("Generating model result visualizations...")

    # Feature Importance Plots
    plot_feature_importance(standard_feature_importance, model_name="Standard XGBoost")
    plot_feature_importance(poisson_feature_importance, model_name="Poisson XGBoost")
    plot_feature_importance(tweedie_feature_importance, model_name="Tweedie XGBoost")

    # Performance Plots (Actual vs Pred, Error Dist)
    plot_model_performance(y_test, standard_predictions, model_name="Standard XGBoost")
    plot_model_performance(y_test, poisson_predictions, model_name="Poisson XGBoost")
    plot_model_performance(y_test, tweedie_predictions, model_name="Tweedie XGBoost")
    
    # Lift Charts
    plot_lift_chart(y_test, standard_predictions, model_name="Standard XGBoost")
    plot_lift_chart(y_test, poisson_predictions, model_name="Poisson XGBoost")
    plot_lift_chart(y_test, tweedie_predictions, model_name="Tweedie XGBoost")

    # Precision-Recall Curves
    plot_precision_recall_curve_func(y_test, standard_predictions, model_name="Standard XGBoost")
    plot_precision_recall_curve_func(y_test, poisson_predictions, model_name="Poisson XGBoost")
    plot_precision_recall_curve_func(y_test, tweedie_predictions, model_name="Tweedie XGBoost")

    # --- Model Comparison --- 
    logger.info("Comparing model performance...")
    comparison_df = pd.DataFrame([
        standard_metrics,
        poisson_metrics,
        tweedie_metrics
    ])
    # Set index to model name for clarity
    comparison_df = comparison_df.set_index('Model') 
    # Select key metrics for display
    key_metrics = ['RMSE', 'MAE', 'R2', 'Poisson Deviance', 'Claims Detection F1']
    # Ensure key metrics exist before selection
    key_metrics = [m for m in key_metrics if m in comparison_df.columns]
    print("\n==== Model Performance Comparison ====")
    print(comparison_df[key_metrics].to_string())
    logger.info(f"Model Comparison:\n{comparison_df[key_metrics].to_string()}")
    
    # --- Error Analysis --- 
    logger.info("Performing error analysis...")
    # Create DataFrame for error analysis including all models
    error_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted_Std': standard_predictions,
        'Predicted_Poisson': poisson_predictions,
        'Predicted_Tweedie': tweedie_predictions,
        'Error_Std': standard_predictions - y_test,
        'Error_Poisson': poisson_predictions - y_test, 
        'Error_Tweedie': tweedie_predictions - y_test,
        'Abs_Error_Std': np.abs(standard_predictions - y_test),
        'Abs_Error_Poisson': np.abs(poisson_predictions - y_test),
        'Abs_Error_Tweedie': np.abs(tweedie_predictions - y_test)
    })
    
    # Add some key original test features for context (use non-prepared X_test)
    key_features_for_error = ['Exposure', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']
    # Check which keys exist in original X_test before adding
    key_features_to_add = [f for f in key_features_for_error if f in X_test.columns]
    if key_features_to_add:
         error_df = pd.concat([error_df, X_test[key_features_to_add].reset_index(drop=True)], axis=1)
    
    # Analyze top 10 largest errors for each model
    print("\n--- Top 10 Largest Errors Analysis ---")
    error_cols_to_show = ['Actual', 'Predicted_Std', 'Error_Std'] + key_features_to_add
    std_largest_errors = error_df.sort_values('Abs_Error_Std', ascending=False).head(10)
    print("\nStandard XGBoost Largest Errors:")
    print(std_largest_errors[error_cols_to_show].to_string())
    logger.info(f"Standard XGBoost Largest Errors:\n{std_largest_errors[error_cols_to_show].to_string()}")
    
    error_cols_to_show = ['Actual', 'Predicted_Poisson', 'Error_Poisson'] + key_features_to_add
    poisson_largest_errors = error_df.sort_values('Abs_Error_Poisson', ascending=False).head(10)
    print("\nPoisson XGBoost Largest Errors:")
    print(poisson_largest_errors[error_cols_to_show].to_string())
    logger.info(f"Poisson XGBoost Largest Errors:\n{poisson_largest_errors[error_cols_to_show].to_string()}")

    error_cols_to_show = ['Actual', 'Predicted_Tweedie', 'Error_Tweedie'] + key_features_to_add
    tweedie_largest_errors = error_df.sort_values('Abs_Error_Tweedie', ascending=False).head(10)
    print("\nTweedie XGBoost Largest Errors:")
    print(tweedie_largest_errors[error_cols_to_show].to_string())
    logger.info(f"Tweedie XGBoost Largest Errors:\n{tweedie_largest_errors[error_cols_to_show].to_string()}")
    
    # Save model comparison summary to file
    summary_filename = 'model_comparison_summary.txt'
    logger.info(f"Saving model comparison summary to {summary_filename}...")
    with open(summary_filename, 'w') as f:
        f.write("Car Insurance Claims Prediction Model Comparison\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Model Metrics Comparison (Key Metrics):\n")
        f.write("-" * 30 + "\n")
        f.write(comparison_df[key_metrics].to_string() + "\n\n")
        
        f.write("Standard XGBoost - Top Features:\n")
        f.write("-" * 30 + "\n")
        f.write(standard_feature_importance.head(10).to_string(index=False) + "\n\n")
        
        f.write("Poisson XGBoost - Top Features:\n")
        f.write("-" * 30 + "\n")
        f.write(poisson_feature_importance.head(10).to_string(index=False) + "\n\n")
        
        f.write("Tweedie XGBoost - Top Features:\n")
        f.write("-" * 30 + "\n")
        f.write(tweedie_feature_importance.head(10).to_string(index=False) + "\n")
        
    print(f"\nModel comparison summary saved to: {summary_filename}")

    # --- Completion --- 
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    print("\nCompleted successfully!")
    logger.info("Main script execution completed successfully.")
    print(f"Results saved to 'plots/', 'models/', '{summary_filename}', and logs saved to '{log_file}'")

if __name__ == "__main__":
    main() 