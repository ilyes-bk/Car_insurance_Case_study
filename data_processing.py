#!/usr/bin/env python
# Data processing module for car insurance claims prediction

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Updated load_and_preprocess_data function from the notebook
def load_and_preprocess_data(filepath):
    """
    Load and preprocess the data from CSV file with enhanced cleaning
    and transformation of skewed variables

    Args:
        filepath (str): Path to the CSV data file

    Returns:
        pandas.DataFrame: Preprocessed data
    """
    print("\n=== Starting Data Loading and Preprocessing ===")

    # Load the data
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Basic data cleaning
    print("\n=== Data Cleaning ===")
    print("Checking for missing values...")
    missing_values = df.isnull().sum()
    print(f"Missing values per column:\n{missing_values}")

    # Handle categorical variables
    categorical_cols = ['Area', 'VehBrand', 'VehGas', 'Region']
    print(f"\nEncoding categorical variables: {categorical_cols}")

    # Encode categorical variables
    for col in categorical_cols:
        print(f"Encoding {col}...")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        print(f"Unique values in {col}: {df[col].nunique()}")

    # Analyze target variable
    print("\nAnalyzing claim distribution...")
    print(f"ClaimNb value counts:\n{df['ClaimNb'].value_counts().sort_index()}")
    print(f"Zero claims: {(df['ClaimNb'] == 0).mean()*100:.2f}%")
    print(f"Single claims: {(df['ClaimNb'] == 1).mean()*100:.2f}%")
    print(f"Multiple claims: {(df['ClaimNb'] > 1).mean()*100:.2f}%")

    # Handle extreme values in Exposure (capping very small exposures)
    min_exposure = 0.05  # Minimum meaningful exposure period (about 2.5 weeks)
    print(f"\nCapping very small exposure values below {min_exposure}...")
    small_exposure_pct = (df['Exposure'] < min_exposure).mean() * 100
    print(f"Very small exposures: {small_exposure_pct:.2f}%")

    # Flag rather than modifying the original exposure
    df['VeryShortExposure'] = (df['Exposure'] < min_exposure).astype(int)

    # Log transform of Density to handle skewness
    print("\nApplying log transform to Density...")
    df['LogDensity'] = np.log1p(df['Density'])
    print(f"Density statistics before transform:\n{df['Density'].describe()}")
    print(f"Density statistics after transform:\n{df['LogDensity'].describe()}")

    # Create population density categories (more granular)
    density_bins = [0, 50, 200, 500, 1000, 5000, float('inf')]
    density_labels = [0, 1, 2, 3, 4, 5]
    df['DensityGroup'] = pd.cut(df['Density'], bins=density_bins, labels=density_labels)
    df['DensityGroup'] = df['DensityGroup'].astype(int)

    # Handle infinite values
    print("\nChecking for infinite values...")
    inf_count = np.isinf(df).sum().sum()
    print(f"Found {inf_count} infinite values")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN values
    print("\nFilling NaN values...")
    for col in df.columns:
        if df[col].isnull().any():
            # Impute with median instead of mean for potentially skewed distributions
            median_val = df[col].median()
            print(f"Filling NaN in {col} with median: {median_val:.2f}")
            df[col].fillna(median_val, inplace=True)

    print(f"\nFinal processed data shape: {df.shape}")
    print("=== Data Loading and Preprocessing Complete ===\n")
    return df

# Updated feature_engineering function from the notebook
def feature_engineering(df):
    """
    Perform enhanced feature engineering on the preprocessed data
    with focus on risk factors and exposure handling

    Args:
        df (pandas.DataFrame): Preprocessed data

    Returns:
        pandas.DataFrame: Data with engineered features
    """
    print("\n=== Starting Enhanced Feature Engineering ===")

    # Create a copy to avoid modifying the original
    df_fe = df.copy()
    print(f"Initial shape: {df_fe.shape}")

    # Create log transform of exposure to handle skewness
    df_fe['LogExposure'] = np.log1p(df_fe['Exposure'])

    # Age-related interactions (enhanced)
    print("\nCreating enhanced age-related features...")
    df_fe['VehAge_DrivAge_Ratio'] = df_fe['VehAge'] / np.maximum(df_fe['DrivAge'], 1)

    # More granular vehicle age groups
    df_fe['NewVehicle'] = (df_fe['VehAge'] <= 1).astype(int)
    df_fe['YoungVehicle'] = ((df_fe['VehAge'] > 1) & (df_fe['VehAge'] <= 3)).astype(int)
    df_fe['MatureVehicle'] = ((df_fe['VehAge'] > 3) & (df_fe['VehAge'] <= 10)).astype(int)
    df_fe['OldVehicle'] = (df_fe['VehAge'] > 10).astype(int)
    df_fe['VeryOldVehicle'] = (df_fe['VehAge'] > 15).astype(int)

    # Driver age categories (more detailed for young drivers)
    df_fe['TeenDriver'] = ((df_fe['DrivAge'] >= 18) & (df_fe['DrivAge'] < 21)).astype(int)
    df_fe['YoungDriver'] = ((df_fe['DrivAge'] >= 21) & (df_fe['DrivAge'] < 25)).astype(int)
    df_fe['AdultDriver'] = ((df_fe['DrivAge'] >= 25) & (df_fe['DrivAge'] < 65)).astype(int)
    df_fe['SeniorDriver'] = (df_fe['DrivAge'] >= 65).astype(int)

    # Experience-related features
    print("\nCreating enhanced experience-related features...")
    df_fe['DrivExperience'] = df_fe['DrivAge'] - 18
    df_fe['DrivExperience'] = df_fe['DrivExperience'].clip(lower=0)

    # Better experience categories with more granularity for novice drivers
    df_fe['VeryNoviceDriver'] = (df_fe['DrivExperience'] <= 1).astype(int)
    df_fe['NoviceDriver'] = ((df_fe['DrivExperience'] > 1) & (df_fe['DrivExperience'] <= 3)).astype(int)
    df_fe['ModerateExperienceDriver'] = ((df_fe['DrivExperience'] > 3) & (df_fe['DrivExperience'] <= 10)).astype(int)
    df_fe['ExperiencedDriver'] = (df_fe['DrivExperience'] > 10).astype(int)

    # Power-related features
    print("\nCreating enhanced power-related features...")
    # Better power to age ratio (considering experience, not just age)
    df_fe['PowerToExperience'] = df_fe['VehPower'] / np.maximum(df_fe['DrivExperience'], 1)

    # More granular power categories
    df_fe['LowPower'] = (df_fe['VehPower'] <= 5).astype(int)
    df_fe['MediumPower'] = ((df_fe['VehPower'] > 5) & (df_fe['VehPower'] <= 8)).astype(int)
    df_fe['HighPower'] = ((df_fe['VehPower'] > 8) & (df_fe['VehPower'] <= 10)).astype(int)
    df_fe['VeryHighPower'] = (df_fe['VehPower'] > 10).astype(int)

    # High-risk combinations
    df_fe['YoungDriverHighPower'] = ((df_fe['DrivAge'] < 25) & (df_fe['VehPower'] > 8)).astype(int)
    df_fe['NewDriverNewCar'] = ((df_fe['DrivExperience'] <= 3) & (df_fe['VehAge'] <= 3)).astype(int)
    df_fe['InexperiencePowerRatio'] = df_fe['VehPower'] / np.maximum(df_fe['DrivExperience'], 1)

    # BonusMalus interactions (improved)
    print("\nCreating enhanced BonusMalus features...")
    # Exponential scale to emphasize high BM values
    df_fe['BM_Exp'] = np.exp((df_fe['BonusMalus'] - 50) / 50)

    df_fe['BM_Power'] = df_fe['BonusMalus'] * df_fe['VehPower']
    df_fe['BM_Age'] = df_fe['BonusMalus'] / np.maximum(df_fe['DrivAge'], 1)

    # Better BonusMalus risk categories (adjusted thresholds)
    # Most policies have BM=50, so we need more meaningful thresholds
    df_fe['HighRiskBM'] = (df_fe['BonusMalus'] > 100).astype(int)
    df_fe['ModerateRiskBM'] = ((df_fe['BonusMalus'] > 50) & (df_fe['BonusMalus'] <= 100)).astype(int)
    df_fe['LowRiskBM'] = (df_fe['BonusMalus'] == 50).astype(int)  # Exactly 50 is the base rate
    
    # Create BonusMalus segments based on notebook (added here for completeness)
    df_fe['BM_Segment'] = pd.cut(df_fe['BonusMalus'], 
                           bins=[0, 50, 75, 100, float('inf')],
                           labels=['Base (50)', 'Moderate (51-75)', 'High (76-100)', 'Very High (>100)'])

    # Density features (improved)
    print("\nCreating enhanced density-related features...")
    # LogDensity already created in preprocessing
    # High risk intersections
    df_fe['HighDensityBM'] = ((df_fe['Density'] > df_fe['Density'].mean()) &
                             (df_fe['BonusMalus'] > 75)).astype(int)

    # Composite risk score (improved with weights based on importance)
    print("\nCreating enhanced composite risk score...")
    df_fe['CompositeRiskScore'] = (
        df_fe['VeryHighPower'] * 3 +
        df_fe['YoungDriverHighPower'] * 5 +
        df_fe['HighRiskBM'] * 4 +
        df_fe['TeenDriver'] * 3 +
        df_fe['VeryOldVehicle'] * 2 +
        # Use VeryShortExposure flag created in preprocessing
        df_fe['VeryShortExposure'] * 4 +
        (df_fe['Density'] > df_fe['Density'].quantile(0.9)).astype(int) * 2
    )
    print(f"Composite risk score statistics:\n{df_fe['CompositeRiskScore'].describe()}")

    # Create policy segments for business-relevant analysis
    df_fe['RiskSegment'] = 'Medium'
    high_risk_mask = df_fe['CompositeRiskScore'] >= 4
    low_risk_mask = df_fe['CompositeRiskScore'] == 0
    df_fe.loc[high_risk_mask, 'RiskSegment'] = 'High'
    df_fe.loc[low_risk_mask, 'RiskSegment'] = 'Low'
    
    # Convert RiskSegment and BM_Segment to category type for potential later use
    df_fe['RiskSegment'] = df_fe['RiskSegment'].astype('category')
    df_fe['BM_Segment'] = df_fe['BM_Segment'].astype('category')


    # Print distribution of risk segments
    print("\nRisk segment distribution:")
    print(df_fe['RiskSegment'].value_counts(normalize=True) * 100)

    # Add Interaction features from Multiple Claims Analysis section of notebook
    print("\nCreating interaction features based on identified risk factors...")
    df_fe['RiskFactorScore'] = (
        df_fe['VehPower'] * df_fe['BonusMalus'] / np.maximum(df_fe['DrivAge'], 1) # Use np.maximum instead of +1 for robustness
    )
    print(f"RiskFactorScore statistics:\n{df_fe['RiskFactorScore'].describe()}")
    df_fe['Area_Region'] = df_fe['Area'] * df_fe['Region']
    df_fe['Brand_Gas'] = df_fe['VehBrand'] * df_fe['VehGas']
    df_fe['VehAgePower'] = df_fe['VehAge'] * df_fe['VehPower']

    # Cleanup - remove unnecessary columns
    # Removed ClaimFreq drop as it wasn't created here anymore
    # No need to drop columns here, better handled before splitting in main script

    print(f"\nFinal feature engineered data shape: {df_fe.shape}")
    print("=== Enhanced Feature Engineering Complete ===\n")
    return df_fe

# Helper function (from notebook) for converting categorical features for XGBoost
# Place it here or in models.py depending on preference
def prepare_features_for_xgboost(X):
    """
    Convert categorical features to numeric for XGBoost compatibility

    Args:
        X (pandas.DataFrame): Input features

    Returns:
        pandas.DataFrame: Features with categorical variables converted to numeric
    """
    X_prep = X.copy()

    # Find categorical columns
    categorical_columns = X_prep.select_dtypes(include=['category', 'object']).columns
    print(f"Found {len(categorical_columns)} categorical columns: {list(categorical_columns)}")

    # Convert categorical columns to numeric
    for col in categorical_columns:
        if X_prep[col].dtype == 'category':
            # Use codes, ensure consistency if unseen values appear in test
            # Consider saving category mappings if deploying
            print(f"Converting category column '{col}' using .cat.codes")
            X_prep[col] = X_prep[col].cat.codes
        elif X_prep[col].dtype == 'object':
            # Use factorize for object columns
            print(f"Converting object column '{col}' using pd.factorize")
            X_prep[col] = pd.factorize(X_prep[col])[0]
        # If column is already numeric, do nothing

    print("Categorical columns converted to numeric.")
    return X_prep 