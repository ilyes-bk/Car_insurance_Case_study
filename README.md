# Car Insurance Claims Prediction

This project provides a solution for predicting the number of car insurance claims based on the provided `freMTPL2freq.csv` dataset. The solution uses XGBoost to build regression models that predict the claim count for insurance policies.

## Project Structure

```
├── README.md                   # Project documentation
├── main.py                     # Main script orchestrating the workflow
├── data_processing.py          # Data loading, cleaning and feature engineering
├── models.py                   # Model training and evaluation functions
├── visualization.py            # Data and model visualization functions
├── plots/                      # Generated visualizations (created at runtime)
│   ├── distributions/          # EDA visualizations
│   └── model/                  # Model performance visualizations
├── models/                     # Saved models (created at runtime)
└── top_errors_analysis.csv     # Analysis of largest prediction errors (created at runtime)
```

## Requirements

The solution requires the following Python packages:
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- scipy

Install dependencies with:

```
pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy
```

## Data

The dataset (`freMTPL2freq.csv`) contains information about car insurance policies, including:

- **IDpol**: Policy ID
- **ClaimNb**: Number of claims during the exposure period (target variable)
- **Exposure**: The exposure period
- **Area**: The area code
- **VehPower**: The power of the car
- **VehAge**: The vehicle age in years
- **DrivAge**: The driver age in years
- **BonusMalus**: Bonus/malus score between 50 and 350
- **VehBrand**: The car brand
- **VehGas**: The car gas type (Diesel or Regular)
- **Density**: Population density where the driver lives
- **Region**: The policy region

## Solution Approach

The solution follows these main steps:

1. **Data Preprocessing**:
   - Handling categorical variables using Label Encoding
   - Calculating claim frequency (ClaimNb/Exposure)
   - Log transformation of skewed features
   - Handling missing or infinite values

2. **Feature Engineering**:
   - Creating interaction features (e.g., vehicle age to driver age ratio)
   - Driver experience estimation
   - Power-to-age risk factor
   - Density-based features
   - Age group categorization
   - Multiple claims indicator
   - Regional risk factors

3. **Modeling**:
   - Standard XGBoost Regression with count:poisson objective
   - XGBoost Poisson Regression with exposure offset
   - Hyperparameter tuning using GridSearchCV

4. **Evaluation**:
   - MSE, RMSE, MAE, R² metrics
   - Poisson Deviance (appropriate for count data)
   - Feature importance analysis
   - Top 10 largest prediction errors analysis

## Running the Solution

To run the complete solution:

```
python main.py
```

This will:
1. Load and preprocess the data
2. Perform feature engineering
3. Generate exploratory visualizations
4. Train both standard and Poisson XGBoost models
5. Evaluate and compare model performance
6. Save models, visualizations, and error analysis

## Key Insights

The solution provides several key insights:

1. **Feature Importance**: Reveals which factors most strongly influence claim frequency
2. **Model Comparison**: Compares standard regression vs. Poisson regression with exposure offset
3. **Error Analysis**: Examines cases where the model makes the largest prediction errors
4. **Visualization**: Provides comprehensive visualizations of data distributions and relationships

## Business Applications

The model can be used to:

1. **Risk Assessment**: Better understand and quantify policy risk factors
2. **Premium Calculation**: Inform actuarial decisions for policy pricing
3. **Customer Segmentation**: Identify high vs. low-risk customer segments
4. **Underwriting Decisions**: Support efficient policy approval processes 