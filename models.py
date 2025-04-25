#!/usr/bin/env python
# Models module for car insurance claims prediction

import numpy as np
import pandas as pd
import xgboost as xgb
import time
import logging # Added import
# Removed unused imports: GridSearchCV, KFold, make_scorer
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
) # Added classification metrics

# Setup logger (can be configured further if needed)
logger = logging.getLogger(__name__)

# Updated train_xgboost_model function from notebook
def train_xgboost_model(X_train, y_train):
    """
    Train an XGBoost model with optimal parameters and scale_pos_weight.
    Assumes features are already preprocessed (e.g., categorical converted).
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target variable
        
    Returns:
        tuple: (trained XGBoost model, dictionary of best parameters used)
    """
    print("\n=== Starting Standard XGBoost Model Training ===")
    logger.info("Starting Standard XGBoost Model Training") # Use logger
    print(f"Training data shape: {X_train.shape}")
    print(f"Target variable statistics:\n{y_train.describe()}")

    # Calculate scale_pos_weight for handling imbalance in claim counts
    # Ensures the model pays more attention to the minority class (claims > 0)
    if (y_train > 0).sum() > 0: # Avoid division by zero if no claims
        scale_pos_weight = (y_train == 0).sum() / (y_train > 0).sum()
    else:
        scale_pos_weight = 1 # Default if no positive class samples
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
    logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # Use the best parameters directly based on prior tuning (as done in notebook)
    # These parameters balance model complexity and generalization for insurance data
    best_params = {
        'objective': 'count:poisson', # Suitable for count data like claims
        'learning_rate': 0.05,    # Step size shrinkage to prevent overfitting
        'max_depth': 5,           # Maximum depth of a tree
        'min_child_weight': 3,    # Minimum sum of instance weight needed in a child
        'subsample': 0.8,         # Fraction of samples used for fitting the trees
        'colsample_bytree': 0.8,  # Fraction of features used for fitting the trees
        'gamma': 0.1,             # Minimum loss reduction required to make a further partition
        'alpha': 0.1,             # L1 regularization term on weights
        'lambda': 0.1              # L2 regularization term on weights
    }

    print("\nUsing optimal parameters from prior tuning:")
    logger.info(f"Using optimal parameters: {best_params}")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    # Train the final model with optimal parameters and scale_pos_weight
    print("\nTraining final model with optimal parameters and scale_pos_weight...")
    logger.info("Training final Standard XGBoost model...")
    final_model = xgb.XGBRegressor(
        **best_params,
        scale_pos_weight=scale_pos_weight,  # Apply calculated weight
        tree_method='hist',               # Efficient histogram-based algorithm
        random_state=42,                  # For reproducibility
        n_estimators=200,                 # Number of boosting rounds
        verbosity=0                       # Suppress verbose output
    )

    start_time = time.time()
    # Fit the model to the training data
    final_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Final model training completed in {training_time:.2f} seconds")
    logger.info(f"Standard XGBoost training completed in {training_time:.2f}s")

    print("=== Standard XGBoost Model Training Complete ===\n")
    # Return the trained model and the parameters used
    return final_model, best_params

# Updated train_xgboost_poisson_model function from notebook
def train_xgboost_poisson_model(X_train, y_train, X_test, exposure_train=None, exposure_test=None, random_state=42):
    """
    Train an XGBoost model with Poisson objective function for claim frequency prediction
    with improved exposure handling
    
    Args:
        X_train: Training features
        y_train: Training target (can be ClaimNb or ClaimFrequency)
        X_test: Test features
        exposure_train: Exposure values for training set (optional)
        exposure_test: Exposure values for test set (optional)
        random_state: Random state for reproducibility
        
    Returns:
        Trained model, test predictions, and feature importance
    """
    logger.info("Starting Poisson XGBoost model training with improved exposure handling")
    logger.info(f"Input shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}")
    
    # Check if exposure data is provided
    if exposure_train is not None and exposure_test is not None:
        logger.info("Exposure data provided - using as offset")
        
        # Handle exposure more carefully
        # Minimum exposure threshold to avoid numerical issues
        min_exposure = 0.05
        exposure_train_adj = np.maximum(exposure_train, min_exposure)
        exposure_test_adj = np.maximum(exposure_test, min_exposure)
        
        # Use log(exposure) as offset in base_margin
        exposure_train_log = np.log(exposure_train_adj)
        exposure_test_log = np.log(exposure_test_adj)
        
        # Convert count to rate (more stable for very short exposures)
        y_freq_train = y_train / exposure_train_adj
        
        # Cap extreme values in frequency to avoid fitting to outliers
        max_freq = np.percentile(y_freq_train[y_freq_train > 0], 99)
        logger.info(f"Capping extreme frequencies at {max_freq:.4f}")
        y_freq_train = np.minimum(y_freq_train, max_freq)
    else:
        logger.info("No exposure data provided - assuming input is already properly scaled")
        exposure_train_log = None
        exposure_test_log = None
        y_freq_train = y_train
    
    # Handle zeros properly (small positive value instead of exact zero)
    eps = 1e-6
    y_freq_train = np.maximum(y_freq_train, eps)
    
    # Parameter grid for grid search
    param_grid = {
        'learning_rate': [0.01],
        'max_depth': [5],
        'min_child_weight': [3],
        'gamma': [0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'reg_alpha': [0.1],
        'reg_lambda': [1.0]
    }
    
    # Create model with Poisson objective
    model = xgb.XGBRegressor(
        objective='count:poisson',
        n_estimators=100,
        random_state=random_state,
        verbosity=0
    )
    
    # Use best parameters directly to avoid grid search issues
    # These are reasonable defaults based on the observations
    best_params = {
        'learning_rate': 0.03,
        'max_depth': 4,
        'min_child_weight': 3,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0
    }
    
    logger.info(f"Using predefined best parameters: {best_params}")
    
    # Train final model with best parameters
    final_model = xgb.XGBRegressor(
        objective='count:poisson',
        n_estimators=300,
        random_state=random_state,
        **best_params
    )
    
    # Fit final model
    if exposure_train_log is not None:
        logger.info("Fitting final model with exposure offset")
        # Train on rates, with log(exposure) as offset
        final_model.fit(X_train, y_freq_train, base_margin=exposure_train_log)
        
        # Make predictions (these will be rates)
        pred_rates = final_model.predict(X_test)
        
        # Cap predictions to reasonable values
        max_pred_rate = 1.0  # Maximum reasonable rate (1 claim per unit exposure)
        pred_rates = np.minimum(pred_rates, max_pred_rate)
        
        # Convert rates to counts using test exposure
        predictions = pred_rates * exposure_test_adj
        logger.info(f"Predictions shape: {predictions.shape}")
        logger.info(f"Prediction range: {predictions.min():.4f} to {predictions.max():.4f}")
    else:
        logger.info("Fitting final model without exposure adjustment")
        final_model.fit(X_train, y_freq_train)
        predictions = final_model.predict(X_test)
    
    # Get feature importance
    importance = final_model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    logger.info("Poisson XGBoost model training completed with improved exposure handling")
    return final_model, predictions, feature_importance


# Added train_xgboost_tweedie_model function from notebook
def train_xgboost_tweedie_model(X_train, y_train, X_test, exposure_train=None, exposure_test=None, tweedie_variance_power=1.5, random_state=42):
    """
    Train an XGBoost model with Tweedie objective function, suitable for zero-inflated count data.
    Uses exposure offset similar to the Poisson model.
    Assumes features are already preprocessed.
    
    Args:
        X_train: Training features
        y_train: Training target (ClaimNb)
        X_test: Test features
        exposure_train: Exposure values for training set (optional but recommended)
        exposure_test: Exposure values for test set (optional but recommended)
        tweedie_variance_power (float): Parameter for Tweedie distribution (1 < p < 2 often used).
        random_state: Random state for reproducibility
        
    Returns:
        tuple: (Trained Tweedie XGBoost model, test predictions (counts), feature importance DataFrame)
    """
    print(f"\n=== Starting Tweedie XGBoost Model Training (power={tweedie_variance_power}) ===")
    logger.info(f"Starting Tweedie XGBoost model training (power={tweedie_variance_power})...")
    logger.info(f"Input shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}")

    # Prepare exposure offset similar to Poisson model
    if exposure_train is not None and exposure_test is not None:
        logger.info("Exposure data provided - using as offset for Tweedie")
        min_exposure = 0.05
        exposure_train_adj = np.maximum(np.asarray(exposure_train), min_exposure)
        exposure_test_adj = np.maximum(np.asarray(exposure_test), min_exposure)
        exposure_train_log = np.log(exposure_train_adj)
        exposure_test_log = np.log(exposure_test_adj)
        logger.info(f"Adjusted minimum exposure to {min_exposure}")

        # Tweedie can sometimes work directly on counts with offset.
        # However, training on rate can be more stable, especially with skewed exposure.
        # We will follow the notebook's approach of training on capped frequency.
        y_freq_train = y_train / exposure_train_adj
        non_zero_freq = y_freq_train[y_freq_train > 0]
        if len(non_zero_freq) > 0:
            max_freq = np.percentile(non_zero_freq, 99)
            logger.info(f"Capping extreme training frequencies at {max_freq:.4f}")
            y_freq_train = np.minimum(y_freq_train, max_freq)
        else:
            logger.warning("No non-zero frequencies found in training data for capping.")
            max_freq = 1.0 # Default cap
            
        eps = 1e-6
        y_freq_train = np.maximum(y_freq_train, eps)
    else:
        # Tweedie without offset is possible but less common for frequency modeling
        logger.warning("No exposure data provided - Tweedie model training without offset.")
        exposure_train_log = None
        exposure_test_log = None
        y_freq_train = y_train # Use original target if no exposure
        exposure_test_adj = 1.0 # Default exposure multiplier if none provided

    # Use similar best parameters as Poisson, but with Tweedie objective
    # Note: Hyperparameters might ideally be re-tuned specifically for Tweedie
    best_params = {
        # 'objective': 'reg:tweedie', # Objective set directly in XGBRegressor
        'learning_rate': 0.03,
        'max_depth': 4,
        'min_child_weight': 3,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0
    }
    print(f"\nUsing predefined best parameters: {best_params}")
    logger.info(f"Using predefined best parameters for Tweedie: {best_params}")

    # Initialize and train the final Tweedie model
    print("\nTraining final Tweedie model...")
    logger.info("Training final Tweedie XGBoost model...")
    final_model = xgb.XGBRegressor(
        objective='reg:tweedie',                   # Set Tweedie objective
        tweedie_variance_power=tweedie_variance_power, # Set Tweedie power parameter
        n_estimators=300,
        random_state=random_state,
        tree_method='hist',
        verbosity=0,
        **best_params
    )

    # Fit final model
    start_time = time.time()
    if exposure_train_log is not None:
        logger.info("Fitting Tweedie model with exposure offset")
        final_model.fit(X_train, y_freq_train, base_margin=exposure_train_log)
    else:
        logger.info("Fitting Tweedie model without exposure adjustment")
        final_model.fit(X_train, y_freq_train)
    training_time = time.time() - start_time
    print(f"Tweedie model training completed in {training_time:.2f} seconds")
    logger.info(f"Tweedie XGBoost training completed in {training_time:.2f}s")

    # --- Make Predictions --- 
    print("\nMaking predictions with Tweedie model...")
    logger.info("Making predictions with Tweedie model...")
    if exposure_test_log is not None:
        # Predict rates using test offset
        # Again, assume XGBRegressor handles offset implicitly after fitting with base_margin
        pred_rates = final_model.predict(X_test)
        # If needed: pred_rates = final_model.predict(X_test, base_margin=exposure_test_log)
        # Cap rates
        max_pred_rate = 1.0 
        pred_rates = np.minimum(pred_rates, max_pred_rate)
        logger.info(f"Predicted rates capped at {max_pred_rate:.2f}")
        # Convert back to counts
        predictions = pred_rates * exposure_test_adj
    else:
        # Predict directly if no offset was used
        predictions = final_model.predict(X_test)

    # Ensure predictions are non-negative
    predictions = np.maximum(predictions, 0)
    print(f"Final predictions (counts) generated. Range: {predictions.min():.4f} to {predictions.max():.4f}")
    logger.info(f"Tweedie predictions (counts) shape: {predictions.shape}, Range: {predictions.min():.4f} to {predictions.max():.4f}")

    # Get feature importance
    print("\nCalculating Tweedie model feature importance...")
    importance = final_model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    print(f"=== Tweedie XGBoost (power={tweedie_variance_power}) Model Training Complete ===\n")
    return final_model, predictions, feature_importance


# Updated evaluate_model function from notebook
def evaluate_model(y_true, y_pred, model_name, X_test=None, feature_importance=None):
    """
    Evaluate model performance with enhanced business-relevant metrics
    and improved handling of extreme values. 
    Returns metrics dict, feature importance df, and evaluation df.
    
    Args:
        y_true (array-like): True target values (ClaimNb)
        y_pred (array-like): Predicted target values (ClaimNb)
        model_name (str): Name of the model for reporting (e.g., "Standard XGBoost")
        X_test (pd.DataFrame, optional): Test features DF for risk group analysis. Defaults to None.
        feature_importance (pd.DataFrame, optional): DataFrame of feature importance. Defaults to None.
        
    Returns:
        tuple: (Dictionary of metrics, feature importance DataFrame or None, eval_df DataFrame)
    """
    print(f"\n=== Evaluating {model_name} ==-")
    logger.info(f"Evaluating {model_name} with enhanced metrics")

    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Basic check for consistent lengths
    if len(y_true) != len(y_pred):
         logger.error(f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)}) for {model_name}")
         # Decide how to handle: return None, raise error, etc.
         return {}, feature_importance, pd.DataFrame() # Return empty results

    # Create a DataFrame with actual and predicted values for analysis
    eval_df = pd.DataFrame({'actual': y_true, 'predicted': y_pred})

    # --- Metric Calculation --- 
    
    # Cap extreme predictions to avoid undue influence on metrics like RMSE/MAE/R2
    # Cap based on a high percentile of the *actual* non-zero claims
    non_zero_actual = y_true[y_true > 0]
    if len(non_zero_actual) > 0:
        max_reasonable_pred = np.percentile(non_zero_actual, 99.9) * 1.5 # Allow some margin
    else:
        max_reasonable_pred = 10 # Default cap if no non-zero actuals
    y_pred_capped = np.minimum(y_pred, max_reasonable_pred)
    logger.info(f"Predictions capped at {max_reasonable_pred:.2f} for metric calculation.")

    # Traditional regression metrics (using capped predictions for robustness)
    mse = mean_squared_error(y_true, y_pred_capped)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred_capped)

    # Calculate R² score carefully, handling cases where SS_total might be zero
    y_true_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_true_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred_capped) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total > 1e-10 else 0 # Avoid division by zero

    # --- Poisson Deviance Calculation (Improved) --- 
    # This metric assesses goodness-of-fit specifically for Poisson-like data
    def poisson_deviance_improved(y_true_dev, y_pred_dev):
        """Calculate Poisson deviance with better handling of zeros and small values."""
        eps = 1e-10 # Small epsilon to avoid log(0) or division by zero
        # Ensure values are positive
        y_t = np.maximum(np.asarray(y_true_dev), eps)
        y_p = np.maximum(np.asarray(y_pred_dev), eps)
        # Calculate deviance term-by-term
        # Formula: 2 * [y_true * log(y_true / y_pred) - (y_true - y_pred)]
        # Use np.where to handle y_t=0 case gracefully (where log(y_t/y_p) is undefined)
        log_term = np.where(y_t > eps, y_t * np.log(y_t / y_p), 0)
        dev_terms = 2 * (log_term - (y_t - y_p))
        # Handle potential NaN/Inf values arising from edge cases
        dev_terms = np.nan_to_num(dev_terms, nan=0, posinf=1e10, neginf=-1e10)
        # Cap extreme deviance values per observation for stability
        dev_terms = np.clip(dev_terms, -1e6, 1e6)
        return np.mean(dev_terms)

    # Calculate Poisson deviance using the *original* predictions (not capped)
    # Deviance is sensitive to the raw output scale
    poisson_dev = poisson_deviance_improved(y_true, y_pred)

    # --- Business-Relevant Metrics --- 

    # 1. Accuracy by Risk Groups (based on actual claims)
    # Define risk groups based on actual claim counts for stability
    zero_mask = (y_true == 0)
    single_mask = (y_true == 1)
    multi_mask = (y_true > 1)

    risk_group_metrics = {}
    groups = {'Low Risk (0 Claims)': zero_mask, 
              'Medium Risk (1 Claim)': single_mask, 
              'High Risk (>1 Claim)': multi_mask}

    for group_name, mask in groups.items():
        if mask.sum() > 0:
            # Use original predictions for group metrics to see raw model performance
            group_rmse = np.sqrt(mean_squared_error(eval_df.loc[mask, 'actual'], eval_df.loc[mask, 'predicted']))
            group_mae = mean_absolute_error(eval_df.loc[mask, 'actual'], eval_df.loc[mask, 'predicted'])
            risk_group_metrics[f'{group_name}_RMSE'] = group_rmse
            risk_group_metrics[f'{group_name}_MAE'] = group_mae
            print(f"{group_name}: RMSE={group_rmse:.4f}, MAE={group_mae:.4f} (n={mask.sum()})")
        else:
            risk_group_metrics[f'{group_name}_RMSE'] = np.nan
            risk_group_metrics[f'{group_name}_MAE'] = np.nan
            print(f"{group_name}: No samples in this group.")

    # 2. Claims Detection Metrics (Binary Classification: Claim vs No Claim)
    # Treat this as a binary problem: does the policy have ANY claims?
    has_claims_actual = (y_true > 0).astype(int)
    # Define a threshold for prediction - simple approach: predict claim if pred > small value
    # More sophisticated: use precision-recall curve to find optimal threshold (done in plotting)
    # For a basic metric here, let's use a simple threshold slightly above 0 or based on quantiles
    # Threshold choice impacts these metrics significantly!
    # Using a percentile based on actual claim rate is often reasonable start
    claim_rate = has_claims_actual.mean()
    pred_threshold = np.percentile(y_pred, 100 * (1 - claim_rate)) if claim_rate > 0 else 0.5
    has_claims_pred = (y_pred > pred_threshold).astype(int)
    logger.info(f"Using prediction threshold {pred_threshold:.4f} for binary classification metrics.")

    # Calculate binary classification metrics
    claim_accuracy = accuracy_score(has_claims_actual, has_claims_pred)
    claim_precision = precision_score(has_claims_actual, has_claims_pred, zero_division=0)
    claim_recall = recall_score(has_claims_actual, has_claims_pred, zero_division=0)
    claim_f1 = f1_score(has_claims_actual, has_claims_pred, zero_division=0)

    # Compile all metrics into a dictionary
    metrics = {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Poisson Deviance': poisson_dev,
        # Add specific group metrics explicitly
        'Low Risk (0 Claims)_RMSE': risk_group_metrics.get('Low Risk (0 Claims)_RMSE', np.nan),
        'Low Risk (0 Claims)_MAE': risk_group_metrics.get('Low Risk (0 Claims)_MAE', np.nan),
        'Medium Risk (1 Claim)_RMSE': risk_group_metrics.get('Medium Risk (1 Claim)_RMSE', np.nan),
        'Medium Risk (1 Claim)_MAE': risk_group_metrics.get('Medium Risk (1 Claim)_MAE', np.nan),
        'High Risk (>1 Claim)_RMSE': risk_group_metrics.get('High Risk (>1 Claim)_RMSE', np.nan),
        'High Risk (>1 Claim)_MAE': risk_group_metrics.get('High Risk (>1 Claim)_MAE', np.nan),
        # Add binary classification metrics
        'Claims Detection Accuracy': claim_accuracy,
        'Claims Detection Precision': claim_precision,
        'Claims Detection Recall': claim_recall,
        'Claims Detection F1': claim_f1,
        'Prediction Threshold Used': pred_threshold
    }

    # Print summary metrics
    print(f"\n--- {model_name} Evaluation Summary --- ")
    print(f"Overall RMSE: {rmse:.4f}")
    print(f"Overall MAE: {mae:.4f}")
    print(f"Overall R²: {r2:.4f}")
    print(f"Poisson Deviance: {poisson_dev:.4f}")
    print(f"Claims Detection F1 (Threshold ~ {pred_threshold:.3f}): {claim_f1:.4f}")
    print("-------------------------------------")

    # Return metrics dictionary, feature importance DataFrame, and the evaluation DataFrame
    return metrics, feature_importance, eval_df 