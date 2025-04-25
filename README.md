# Car Insurance Claims Prediction

This project provides a comprehensive solution for predicting car insurance claims frequency to support Ominimo's European expansion strategy. Using the freMTPL2freq dataset, we developed and compared three XGBoost-based models: a Standard regression model, a Poisson regression model with exposure offset, and a Tweedie regression model for zero-inflated data.

## Project Structure

```
├── README.md                        # Project documentation
├── Car_Insurance_Claims_Notebook.py # Jupyter notebook with full analysis
├── main.py                          # Main script orchestrating the workflow
├── data_processing.py               # Data loading, cleaning and feature engineering
├── models.py                        # Model training and evaluation functions
├── visualization.py                 # Data and model visualization functions
├── Technical_Report.md              # Markdown technical report
├── Technical_Report.tex             # LaTeX technical report
├── plots/                           # Generated visualizations
│   ├── distributions/               # EDA visualizations
│   └── model/                       # Model performance visualizations
├── models/                          # Saved models
│   ├── standard_xgboost_model.pkl   # Standard XGBoost model
│   ├── poisson_xgboost_model.pkl    # Poisson XGBoost model
│   └── tweedie_xgboost_model.pkl    # Tweedie XGBoost model
└── model_comparison_summary.txt     # Summary of model performance metrics
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

The dataset (`freMTPL2freq.csv`) contains information about car insurance policies in France, including:

- **IDpol**: Policy ID
- **ClaimNb**: Number of claims during the exposure period (target variable)
- **Exposure**: The policy coverage period in years
- **Area**: The area code
- **VehPower**: The power of the car
- **VehAge**: The vehicle age in years
- **DrivAge**: The driver age in years
- **BonusMalus**: Bonus/malus score between 50 and 350 (higher means higher risk)
- **VehBrand**: The car brand
- **VehGas**: The car gas type (Diesel or Regular)
- **Density**: Population density where the driver lives
- **Region**: The policy region

## Solution Approach

### 1. Data Preprocessing

- Handling categorical variables using Label Encoding
- Log transformation of skewed features (Density)
- Creating density categories based on population distribution
- Identifying and flagging very short exposure periods (< 0.05 years)
- Handling missing or infinite values using median imputation

### 2. Feature Engineering

- **Age-related features**: 
  - VehAge_DrivAge_Ratio, DrivExperience
  - TeenDriver, YoungDriver, AdultDriver, SeniorDriver categories
  - NewVehicle, YoungVehicle, MatureVehicle, OldVehicle, VeryOldVehicle categories

- **Risk indicators**:
  - YoungDriverHighPower, NewDriverNewCar, InexperiencePowerRatio
  - HighRiskBM, ModerateRiskBM, LowRiskBM categories

- **BonusMalus features**:
  - BM_Exp, BM_Power, BM_Age
  - BM_Segment categories based on risk thresholds

- **Composite risk features**:
  - CompositeRiskScore (weighted combination of key risk factors)
  - RiskSegment (Low/Medium/High categorization)
  - RiskFactorScore (Vehicle power × BonusMalus / Driver age)

### 3. Modeling Approach

We implemented three complementary modeling approaches:

1. **Standard XGBoost Regression**:
   - Objective: 'count:poisson'
   - Direct prediction of claim counts
   - Parameters: learning_rate=0.05, max_depth=5, min_child_weight=3

2. **Poisson XGBoost with Exposure Offset**:
   - Objective: 'count:poisson'
   - Log(exposure) used as offset via base_margin
   - Target: Claim frequency (claims per unit exposure)
   - Parameters: learning_rate=0.03, max_depth=4

3. **Tweedie XGBoost Model**:
   - Objective: 'reg:tweedie'
   - Tweedie variance power: 1.5
   - Better handling of zero-inflated data
   - Parameters: learning_rate=0.03, max_depth=4

### 4. Evaluation

- **Traditional metrics**: RMSE, MAE, R²
- **Distribution-specific**: Poisson Deviance
- **Risk segment performance**: Performance across low/medium/high-risk policies
- **Business metrics**: Lift charts, precision-recall curves
- **Feature importance analysis**: Identifying key predictive factors
- **Error analysis**: Patterns in prediction errors

## Latest Results

### Model Performance Metrics

| Metric | Standard XGBoost | Poisson XGBoost | Tweedie XGBoost |
|--------|-----------------|-----------------|-----------------|
| RMSE | 0.2339 | 0.2350 | 0.2346 |
| MAE | 0.0976 | 0.1020 | 0.0961 |
| R² | 0.0369 | 0.0274 | 0.0310 |
| Poisson Deviance | 0.2938 | 0.2997 | 0.2964 |
| Claims Detection F1 | 0.1941 | 0.1753 | 0.1808 |

### Top Feature Importance

The top features for the Standard XGBoost model were:

| Feature | Importance |
|---------|------------|
| BM_Segment | 0.1908 |
| NewVehicle | 0.0641 |
| BM_Exp | 0.0563 |
| BonusMalus | 0.0555 |
| Brand_Gas | 0.0539 |
| Exposure | 0.0506 |
| VehAge | 0.0468 |
| VeryShortExposure | 0.0406 |
| VeryHighPower | 0.0357 |
| LogExposure | 0.0329 |

## Running the Solution

To run the complete solution:

```
python main.py
```

This will:
1. Load and preprocess the data
2. Perform feature engineering
3. Generate exploratory visualizations
4. Train all three XGBoost models
5. Evaluate and compare model performance
6. Create visualizations and save results

To run the notebook version:

```
jupyter notebook Car_Insurance_Claims_Notebook.py
```

## Key Insights

### Technical Insights

1. The Standard XGBoost model achieves the best overall performance
2. The Tweedie model shows advantages for low-risk policies and has the best MAE
3. All models struggle with predicting policies having multiple claims in short exposure periods
4. BonusMalus-related features, exposure information, and vehicle characteristics are consistently the most important predictors

### Business Insights

1. **Risk Factors**:
   - Policies with BM scores >100 show 3-4x higher claim frequencies
   - Young drivers with high-powered vehicles present 2.3x elevated risk
   - High density areas show 1.5x higher claim frequencies

2. **Market-Specific Strategies**:
   - **Netherlands**: Leverage density-based pricing tiers
   - **Poland**: Adjust vehicle age factors for older vehicle fleet
   - **Sweden**: Implement seasonal risk factors and higher-value vehicle focus

## Business Applications

The models developed in this project can be used for:

1. **Pricing Strategy**: Data-driven premium adjustment based on risk factors
2. **Underwriting Guidelines**: Special rules for high-risk combinations
3. **Market Expansion**: Targeted entry strategies for the Netherlands, Poland, and Sweden
4. **Portfolio Management**: Better monitoring of risk concentration
5. **Product Development**: Specialized offerings based on regional and demographic patterns

## Advanced AI Applications

The project lays groundwork for future advanced applications:

1. **Computer Vision**: Vehicle image analysis for risk assessment
2. **NLP**: Policy document analysis and claims narrative assessment
3. **Synthetic Data**: Generate realistic scenarios for new markets
4. **Dynamic Risk Scoring**: Real-time risk assessment with temporal models

For more details, please refer to the Technical_Report.md or Technical_Report.tex files. 