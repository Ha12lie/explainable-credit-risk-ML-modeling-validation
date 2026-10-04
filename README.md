# Explainable Credit Risk Modeling with Machine Learning Method: A Temporal Validation Study of Fannie Mae Mortgage Delinquency

This repository implements an end-to-end credit-risk workflow using public Fannie Mae Single-Family Loan Performance data. The README documents the execution logic. Detailed interpretation, formulas, results, and limitations are in [`PROJECT_ANALYSIS.md`](PROJECT_ANALYSIS.md).

This project develops an end-to-end, explainable credit-risk modeling workflow using Fannie Mae Single-Family Loan Performance data. It transforms quarterly disclosure files into a validated loan-level panel, constructs a forward three-month 30+ day delinquency target, and joins lagged macroeconomic indicators to preserve point-in-time feature availability. The modeling pipeline compares class-weighted logistic regression with a random forest using chronological train, calibration, and test periods. Platt scaling is applied to improve probability calibration, while evaluation covers discrimination, rare-event ranking, calibration, event capture, feature attribution, feature-block performance, population stability, and macroeconomic sensitivity.


## Project Pipeline

### Step 1: Data Preparation

**Source of raw data:** https://datadynamics.fanniemae.com/data-dynamics/#/reportMenu;category=HP
**Raw data:** 
2025Q1-2026Q1 Simple Family Loan Performance Data
**Data Pre-processing:**
Parses each 112-field record according to the documented disclosure layout, extracts key loan, borrower, seller, servicer, geographic, collateral, delinquency, modification, and zero-balance variables, validates field counts and duplicate loan-month keys, merges compatible prepared panels, and standardizes loan identifiers, reporting dates, and delinquency values. Invalid records are removed, observations are sorted by loan and month, and one validated record is retained for each loan-month for modeling.

### Step 2: Construct the forward target

The baseline must be current. The code requires three consecutive future monthly records and sets:

```text
target = 1 if delinquency >= 1 in month t+1, t+2, or t+3
target = 0 otherwise
```

The first eligible baseline per loan is retained, preventing the same loan from appearing in multiple temporal partitions. This is a three-month forward 30+ day delinquency target, not a charge-off, loss, or lifetime default label.

### Step 3: Join point-in-time macro features

Extract month-end `^TNX`, `^VIX`, and `SPY` prices using `yfinance`. It computes the trailing 12-month SPY return and shifts all macro features one month forward before joining them to the observation month. This lag is intended to avoid using information that was not available before the observation date.

### Step 4: Define feature blocks

The model matrix contains four blocks:

- **Bureau:** origination credit score, DTI, LTV, original UPB, and original interest rate.
- **Dealer proxy:** seller name, servicer name, and origination channel.
- **Alternative:** MSA, occupancy status, property type, and number of units.
- **Macro:** lagged Treasury yield, VIX, and trailing 12-month SPY return.

Current delinquency, payment history, zero-balance status, modification status, current balances/rates, and other post-origination performance fields are excluded from predictors and used only for labels or audit logic.

### Step 5: Preprocess the model matrix

Numeric variables use training-set median imputation and standardization. Categorical variables use most-frequent imputation and one-hot encoding with rare-category grouping. Unknown categories at scoring time are tolerated. Preprocessing is fitted inside the modeling pipeline to prevent train/test leakage.

### Step 6: Data Split

Unique months are ordered chronologically and split into approximately 60% training, 20% calibration/model selection, and 20% final test data. The future test window is never used to select the model or fit the calibration mapping.

### Step 7: Predictive Model Training

Two predictive models are trained using the same feature matrix:

1. **Class-weighted logistic regression** serves as an interpretable baseline.
2. **Random forest** uses 300 trees, `min_samples_leaf=20`, and balanced bootstrap class weights.

The model with the higher ROC-AUC on the calibration period is selected for probability calibration and final testing.

### Step 8:  Platt Probability Calibration

The selected model first produces raw probabilities. Their log-odds are passed to a one-variable logistic regression fitted on the calibration partition. This is Platt scaling. It changes probability accuracy while preserving score ordering, so ROC-AUC, PR-AUC, KS, and top-k ranking should remain effectively unchanged.

### Step 9: Out-of-Time Discrimination, Calibration, and Capture Validation

The test output reports ROC-AUC, average precision/PR-AUC, Brier score, precision, recall, and F1. Additional validation outputs report KS separation and the share of all events captured in the top 1%, 5%, and 10% of scores. Because the event rate is low, ranking lift and calibration are emphasized over an arbitrary 0.5 threshold.

### Step 10: SHAP-Based Global Model Explainability

This step uses SHAP values to summarize how features contribute to model predictions. Global importance, based on mean absolute SHAP values, helps identify which variables the fitted model relies on most. SHAP describes predictive attribution within the model; it does not establish that a feature causes delinquency.

### Step 11: Run feature-block ablation

The selected model is retrained using all features and each block separately. This isolates how much ranking information is available from bureau, participant, alternative, and macro data groups.

### Step 12: Population Stability Index (PSI) Drift Monitoring

This step compares feature distributions in the training and test populations. A high PSI can indicate changes in portfolio composition, origination vintages, economic conditions, or data definitions. Monitoring these shifts helps identify when the scoring population differs from the population used to develop the model and signals the need for further performance and calibration review.

### Step 13: Macroeconomic Scenario Stress Testing

The stress module applies explicit shocks to Treasury yield, VIX, and trailing SPY return and measures how portfolio-level predicted risk changes. It tests the model’s sensitivity to changes in interest rates, market volatility, and equity-market returns. These are hypothetical sensitivity scenarios, not forecasts or reconstructions of historical crises.
pytest -q
```

The detailed result interpretation is intentionally separated from this execution guide. Read [`docs/PROJECT_ANALYSIS.md`](docs/PROJECT_ANALYSIS.md) for the numerical findings, financial-risk interpretation, formulas, validation caveats, and recommended next steps.
