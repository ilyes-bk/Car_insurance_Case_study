#!/usr/bin/env python
# Visualization module for car insurance claims prediction

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging # Added import
from sklearn.metrics import precision_recall_curve, average_precision_score # Added imports

# Setup logger (can be configured further if needed)
logger = logging.getLogger(__name__)

# Create output directories if they don't exist
os.makedirs('plots/distributions', exist_ok=True)
os.makedirs('plots/model', exist_ok=True)

# Updated plot_distributions function from the notebook
def plot_distributions(df):
    """
    Create exploratory visualizations of the data
    
    Args:
        df (pandas.DataFrame): Preprocessed data (should include engineered features like BM_Segment)
    """
    print("\n=== Creating Exploratory Visualizations ===")
    
    # Set plotting style (can be customized)
    sns.set_style("whitegrid")
    sns.set_context("talk")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Plot distribution of target variable
    print("Plotting claim numbers distribution...")
    plt.figure(figsize=(12, 6))
    # Filter for ClaimNb < 5 for better visibility
    sns.countplot(x='ClaimNb', data=df[df['ClaimNb'] < 5])
    plt.title('Distribution of Claim Numbers (ClaimNb < 5)')
    plt.xlabel('Number of Claims')
    plt.ylabel('Count')
    plt.savefig('plots/distributions/claim_numbers.png')
    plt.close()
    
    # Plot distribution of continuous variables
    continuous_vars = ['Exposure', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']
    
    for var in continuous_vars:
        print(f"Plotting {var} distribution...")
        plt.figure(figsize=(12, 6))
        # Handle potential NaN/inf values before plotting
        sns.histplot(df[var].dropna(), bins=30, kde=True) 
        plt.title(f'Distribution of {var}')
        plt.xlabel(var)
        plt.ylabel('Count')
        plt.savefig(f'plots/distributions/{var.lower()}.png')
        plt.close()
    
    # Plot relationship between claim numbers and continuous variables
    for var in continuous_vars:
        print(f"Plotting ClaimNb vs {var}...")
        plt.figure(figsize=(12, 6))
        # Filter for ClaimNb < 5 for better visibility
        sns.boxplot(x='ClaimNb', y=var, data=df[df['ClaimNb'] < 5])
        plt.title(f'Relationship between Claim Numbers and {var}')
        plt.xlabel('Number of Claims')
        plt.ylabel(var)
        plt.savefig(f'plots/distributions/claimnb_vs_{var.lower()}.png')
        plt.close()
    
    # Plot distribution of categorical variables
    # Note: These were label encoded in preprocessing, adjust if using original labels
    categorical_vars = ['Area', 'VehBrand', 'VehGas', 'Region'] 
    
    for var in categorical_vars:
        print(f"Plotting {var} distribution...")
        plt.figure(figsize=(14, 6))
        # Use value_counts().index for ordering
        sns.countplot(y=var, data=df, order=df[var].value_counts().index) 
        plt.title(f'Distribution of {var}')
        plt.xlabel('Count')
        plt.ylabel(var) # Assumes var name is meaningful
        plt.tight_layout() # Added for better spacing
        plt.savefig(f'plots/distributions/{var.lower()}_distribution.png')
        plt.close()
        
        # --- Plotting Claims vs Categorical (Optional, depends if needed outside notebook) ---
        # print(f"Plotting Claims vs {var}...")
        # plt.figure(figsize=(14, 6))
        # category_claims = df.groupby(var)['ClaimNb'].mean().reset_index()
        # sns.barplot(x=var, y='ClaimNb', data=category_claims, order=category_claims.sort_values('ClaimNb', ascending=False)[var])
        # plt.title(f'Average Claim Numbers by {var}')
        # plt.xlabel(var)
        # plt.ylabel('Average Claim Number')
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.savefig(f'plots/distributions/{var.lower()}_vs_claims.png')
        # plt.close()

    # Plot risk-based encodings (Example, adapt if using different engineered features)
    risk_vars = [col for col in df.columns if col.endswith('_Risk') or col in ['CompositeRiskScore']] # Adjust as needed
    if risk_vars:
        print("Plotting risk-based encodings...")
        try:
            plt.figure(figsize=(14, 8))
            # Melt might fail if types are inconsistent, select numeric risk vars
            numeric_risk_vars = df[risk_vars].select_dtypes(include=np.number).columns
            if not numeric_risk_vars.empty:
                 risk_df = df[numeric_risk_vars].melt(var_name='Category', value_name='Risk_Level')
                 sns.boxplot(x='Category', y='Risk_Level', data=risk_df)
                 plt.title('Risk Levels by Category')
                 plt.xticks(rotation=45)
                 plt.tight_layout()
                 plt.savefig('plots/distributions/risk_encodings.png')
                 plt.close()
            else:
                print("No numeric risk variables found to plot.")
        except Exception as e:
            print(f"Could not plot risk encodings: {e}")
            plt.close() # Close plot if error occurred
    
    # Correlation Matrix
    print("Creating correlation matrix...")
    try:
        # Select only numeric columns for correlation
        numeric_df = df.select_dtypes(include=[np.number])
        # Limit to key variables to keep the matrix readable
        key_vars = ['ClaimNb', 'Exposure', 'VehPower', 'VehAge', 
                    'DrivAge', 'BonusMalus', 'Density', 'LogDensity']
        # Include engineered features if they are numeric and important
        engineered_numeric = ['LogExposure', 'VehAge_DrivAge_Ratio', 'DrivExperience', 
                              'PowerToExperience', 'BM_Exp', 'BM_Power', 'BM_Age', 
                              'CompositeRiskScore', 'RiskFactorScore', 'Area_Region', 
                              'Brand_Gas', 'VehAgePower']
        key_vars.extend([v for v in engineered_numeric if v in numeric_df.columns])                
        key_vars = list(set([var for var in key_vars if var in numeric_df.columns])) # Unique vars present
        
        if len(key_vars) > 1:
            plt.figure(figsize=(18, 14)) # Adjusted size for more vars
            corr_matrix = numeric_df[key_vars].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', 
                        linewidths=0.5, cbar_kws={'shrink': .8})
            plt.title('Correlation Matrix of Key Features')
            plt.tight_layout()
            plt.savefig('plots/distributions/correlation_matrix.png')
            plt.close()
        else:
            print("Not enough numeric key variables for correlation matrix.")
    except Exception as e:
        print(f"Could not create correlation matrix: {e}")
        plt.close() # Close plot if error occurred

    
    # Additional visualization: ClaimNb distribution by BonusMalus segments
    # Ensure 'BM_Segment' exists from feature engineering
    if 'BM_Segment' in df.columns:
        print("Plotting claims by BonusMalus segments...")
        plt.figure(figsize=(12, 6))
        # Ensure BM_Segment is treated correctly (might be category type)
        # Calculate mean claim number (not frequency or pct here) by segment
        # Check if BM_Segment is categorical and handle observed=True if needed
        if pd.api.types.is_categorical_dtype(df['BM_Segment']):
            bm_stats = df.groupby('BM_Segment', observed=True)['ClaimNb'].agg(['mean', 'count'])
        else:
             bm_stats = df.groupby('BM_Segment')['ClaimNb'].agg(['mean', 'count'])
        # bm_stats['mean_pct'] = bm_stats['mean'] * 100 # Plotting mean count directly
        
        # Plot
        ax = sns.barplot(x=bm_stats.index.astype(str), y='mean', data=bm_stats) # Convert index to str just in case
        plt.title('Mean Claim Count by BonusMalus Segment')
        plt.xlabel('BonusMalus Segment')
        plt.ylabel('Mean Claim Count')
        
        # Add count labels
        for i, p in enumerate(ax.patches):
             # Check index type before accessing bm_stats['count']
            current_label = ax.get_xticklabels()[i].get_text()
            if current_label in bm_stats.index:
                 count_val = bm_stats.loc[current_label, 'count']
                 ax.annotate(f"n={count_val:,}", 
                           (p.get_x() + p.get_width()/2., p.get_height()),
                           ha='center', va='bottom')
            else: # Fallback if index matching is tricky
                 ax.annotate(f"n={bm_stats['count'].iloc[i]:,}", 
                           (p.get_x() + p.get_width()/2., p.get_height()),
                           ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('plots/distributions/claims_by_bonusmalus.png')
        plt.close()
    else:
        print("Skipping 'Claims by BonusMalus Segment' plot: 'BM_Segment' column not found.")

    
    print("=== Exploratory Visualizations Complete ===\n") 

# Updated plot_feature_importance from notebook
def plot_feature_importance(feature_importance_df, model_name):
    """
    Plot feature importance from a trained model.
    Saves plot with model name in the filename.
    
    Args:
        feature_importance_df (pandas.DataFrame): DataFrame containing 'Feature' and 'Importance'
        model_name (str): Name of the model (e.g., 'Standard XGBoost') for title and filename
    """
    print(f"Plotting feature importance for {model_name}...")
    plt.figure(figsize=(12, 10))
    
    # Ensure correct column names are used ('Feature', 'Importance')
    if 'Feature' not in feature_importance_df.columns or 'Importance' not in feature_importance_df.columns:
        logger.error("Feature importance DataFrame must contain 'Feature' and 'Importance' columns.")
        return
        
    # Get top 20 features
    top_features = feature_importance_df.nlargest(20, 'Importance')
    
    # Create horizontal bar plot
    sns.barplot(x='Importance', y='Feature', data=top_features)
    
    plt.title(f'Top 20 Feature Importance - {model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Create unique filename
    filename = f'plots/model/feature_importance_{model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'
    plt.savefig(filename)
    plt.close()
    print(f"Feature importance plot saved to {filename}")

# Updated plot_model_performance from notebook
def plot_model_performance(y_true, y_pred, model_name):
    """
    Plot actual vs predicted values and error distribution.
    Saves plots with model name in the filename.
    
    Args:
        y_true (array-like): Actual values
        y_pred (array-like): Predicted values
        model_name (str): Name of the model for plot title and filenames.
    """
    print(f"Plotting model performance for {model_name}...")
    # Calculate error
    error = y_true - y_pred
    
    # Plot error distribution (limited range for visibility)
    plt.figure(figsize=(12, 6))
    # Filter errors for plot: handle potential NaNs from y_pred
    valid_errors = error[~np.isnan(error)]
    sns.histplot(valid_errors[(np.abs(valid_errors) < 2)], bins=50, kde=True) 
    plt.title(f'Error Distribution - {model_name} (|Error| < 2)')
    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Count')
    plt.grid(True)
    filename_err = f'plots/model/error_distribution_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename_err)
    plt.close()
    print(f"Error distribution plot saved to {filename_err}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    
    # Create a scatter plot with alpha for density visualization
    # Filter out NaNs before plotting
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    plt.scatter(y_true[mask], y_pred[mask], alpha=0.1)
    
    # Add perfect prediction line (handle potential NaNs in max calculation)
    if mask.sum() > 0: # Only plot line if there's data
        max_val = max(y_true[mask].max(), y_pred[mask].max())
        min_val = min(y_true[mask].min(), y_pred[mask].min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--') # Adjust line to data range
    else:
        max_val = 1 # Default if no valid data
        plt.plot([0, 1], [0, 1], 'r--')

    plt.title(f'Actual vs Predicted - {model_name}')
    plt.xlabel('Actual Claims') # More specific label
    plt.ylabel('Predicted Claims') # More specific label
    plt.grid(True)
    # Consider adding limits if predictions are very skewed, e.g., plt.ylim(bottom=0)
    filename_avp = f'plots/model/actual_vs_predicted_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename_avp)
    plt.close()
    print(f"Actual vs Predicted plot saved to {filename_avp}")

# Added plot_lift_chart from notebook
def plot_lift_chart(y_true, y_pred, model_name, n_bins=10):
    """
    Plots a lift chart to evaluate model performance in ranking risks.
    
    Args:
        y_true (array-like): Actual target values.
        y_pred (array-like): Predicted target values.
        model_name (str): Name of the model for the plot title.
        n_bins (int): Number of bins (deciles) to create based on predictions.
    """
    logger.info(f"Generating Lift Chart for {model_name}...")
    print(f"Generating Lift Chart for {model_name}...") # Also print

    
    # Ensure inputs are numpy arrays and finite
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.all(valid_mask):
        logger.warning(f"Removing { (~valid_mask).sum()} non-finite values for lift chart: {model_name}")
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
    if len(y_true) == 0:
        logger.error(f"No valid data to plot lift chart for {model_name}.")
        return

    # Create DataFrame and sort by prediction
    lift_df = pd.DataFrame({'actual': y_true, 'predicted': y_pred})
    lift_df = lift_df.sort_values('predicted')
    
    # Create bins based on predicted values
    # Using qcut for deciles, handling potential duplicate edges
    try:
        # Ensure bins have unique edges by using rank for tie-breaking
        lift_df['bin'] = pd.qcut(lift_df['predicted'].rank(method='first'), q=n_bins, labels=False, duplicates='drop')
    except ValueError: # Fallback if qcut still fails
        logger.warning(f"qcut failed for lift chart of {model_name}, using simple ranking (np.linspace).")
        # Use np.linspace for potentially more robust binning with duplicates
        bin_edges = np.linspace(0, len(lift_df), n_bins + 1)
        lift_df['rank'] = np.arange(len(lift_df))
        lift_df['bin'] = pd.cut(lift_df['rank'], bins=bin_edges, labels=False, include_lowest=True, right=False)

    # Calculate average actual and predicted values per bin
    grouped = lift_df.groupby('bin').agg(
        avg_actual=('actual', 'mean'),
        avg_predicted=('predicted', 'mean'),
        count=('actual', 'size')
    )
    
    # Handle potential NaN bins if qcut/cut produced them
    grouped = grouped.dropna(subset=['avg_actual', 'avg_predicted'])
    if grouped.empty:
         logger.error(f"No valid bins found after grouping for lift chart: {model_name}.")
         return

    # Calculate overall average actual rate
    overall_avg_actual = lift_df['actual'].mean()
    if overall_avg_actual == 0: # Avoid division by zero
        overall_avg_actual = 1e-10
        logger.warning(f"Overall average actual is zero for {model_name}. Lift calculation might be less meaningful.")
    
    # Calculate lift
    grouped['lift'] = grouped['avg_actual'] / overall_avg_actual
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    bin_indices = grouped.index.astype(int) # Ensure index is integer for plotting

    # Bar chart for average actual rate per bin
    bars = ax1.bar(bin_indices, grouped['avg_actual'], color='skyblue', label='Avg Actual Rate')
    ax1.set_xlabel(f'Prediction Bin ({n_bins}-Quantile, 0=Lowest Risk)') # Updated label
    ax1.set_ylabel('Average Actual Claim Rate', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_xticks(bin_indices)
    # ax1.set_xticklabels(bin_indices) # Use default labels if index is 0, 1, 2...
    # Add count labels on bars (optional, can be noisy)
    # ax1.bar_label(bars, labels=grouped['count'].map('{:,.0f}'.format), padding=3)

    # Line chart for lift on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(bin_indices, grouped['lift'], color='red', marker='o', linestyle='--', label='Lift')
    ax2.set_ylabel('Lift vs Overall Average', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.axhline(1.0, color='grey', linestyle=':', linewidth=1)
    ax2.grid(False) # Turn off grid for secondary axis

    ax1.grid(True, axis='y', linestyle=':', alpha=0.7) # Keep primary grid, make it subtle

    plt.title(f'Lift Chart - {model_name}')
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    fig.tight_layout()
    filename = f'plots/model/lift_chart_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename)
    plt.close()
    logger.info(f"Lift Chart saved to {filename}")
    print(f"Lift Chart saved to {filename}")

# Added plot_precision_recall_curve_func from notebook
def plot_precision_recall_curve_func(y_true, y_pred_prob, model_name):
    """
    Plots the Precision-Recall curve for claim detection.
    
    Args:
        y_true (array-like): Actual binary target (1 if claim, 0 otherwise).
        y_pred_prob (array-like): Predicted probabilities or scores for the positive class (claim).
        model_name (str): Name of the model for the plot title.
    """
    logger.info(f"Generating Precision-Recall Curve for {model_name}...")
    print(f"Generating Precision-Recall Curve for {model_name}...") # Also print
    
    # Ensure y_true is binary (0 or 1)
    y_true_binary = (np.asarray(y_true) > 0).astype(int)
    y_pred_prob = np.asarray(y_pred_prob)
    
    # Handle NaNs or Infs
    valid_mask = np.isfinite(y_true_binary) & np.isfinite(y_pred_prob)
    if not np.all(valid_mask):
        logger.warning(f"Removing {(~valid_mask).sum()} non-finite values for PR curve: {model_name}")
        y_true_binary = y_true_binary[valid_mask]
        y_pred_prob = y_pred_prob[valid_mask]
        
    if len(y_true_binary) == 0 or len(np.unique(y_true_binary)) < 2:
        logger.error(f"Not enough valid data or only one class present for PR curve: {model_name}. Cannot plot.")
        return
    
    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_pred_prob)
    average_precision = average_precision_score(y_true_binary, y_pred_prob)
    
    # Calculate F1 score for each threshold - handle division by zero
    # Ensure precision and recall are arrays
    precision = np.asarray(precision)
    recall = np.asarray(recall)
    f1_scores = np.divide(2 * recall * precision, recall + precision, 
                          out=np.zeros_like(precision, dtype=float), 
                          where=(recall + precision) != 0)
    
    # Find the threshold index that gives the best F1 score
    if len(f1_scores) > 0:
        best_threshold_idx = np.argmax(f1_scores)
        # Need to be careful with threshold length (it's len(precision)-1)
        # Find threshold closest to the best precision/recall point
        if best_threshold_idx < len(thresholds):
             best_threshold = thresholds[best_threshold_idx]
        else: # Handle edge case where best F1 is at the end (threshold not explicitly defined)
             best_threshold = thresholds[-1] if len(thresholds) > 0 else 0.5 # Use last or default
             best_threshold_idx = len(f1_scores) - 1 # Adjust index to last valid point

        best_f1 = f1_scores[best_threshold_idx]
        best_recall = recall[best_threshold_idx]
        best_precision = precision[best_threshold_idx]
    else:
        best_threshold_idx = -1 # Indicate failure
        best_f1 = np.nan
        best_threshold = np.nan
        print(f"Warning: Could not calculate F1 scores for {model_name}")

    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, marker='.', label=f'{model_name} (AP={average_precision:.2f})')
    
    # Plot the point for best F1 if found
    if best_threshold_idx != -1:
        plt.scatter(best_recall, best_precision, marker='o', color='red', s=100,
                    label=f'Best F1 ({best_f1:.2f}) at Thr~{best_threshold:.3f}')

    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.05) # Ensure y-axis goes up to 1
    plt.xlim(0, 1.0) # Ensure x-axis goes up to 1
    filename = f'plots/model/precision_recall_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename)
    plt.close()
    logger.info(f"Precision-Recall Curve saved to {filename}")
    print(f"Precision-Recall Curve saved to {filename}") 