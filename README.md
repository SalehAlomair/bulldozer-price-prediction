# Bulldozer Price Prediction with Machine Learning

A comprehensive end-to-end machine learning project that predicts the sale prices of bulldozers using historical auction data from the Kaggle Blue Book for Bulldozers competition.

## Project Overview

This project demonstrates a complete machine learning workflow, from data exploration and preprocessing to model training and evaluation. Using Random Forest Regressor with hyperparameter tuning, the model achieves competitive performance in predicting bulldozer auction prices.

## Dataset

The dataset comes from the [Kaggle Blue Book for Bulldozers competition](https://www.kaggle.com/competitions/bluebook-for-bulldozers/data) and includes:

- **Training Data**: Historical bulldozer sales through end of 2011
- **Validation Data**: Sales from January 1, 2012 - April 30, 2012
- **Test Data**: Sales from May 1, 2012 - November 2012
- **Features**: 50+ attributes including machine specifications, sale location, and timing

### Key Features
- `YearMade`: Year of manufacture
- `MachineHoursCurrentMeter`: Usage hours at time of sale
- `UsageBand`: Usage classification (Low/Medium/High)
- `fiModelDesc`: Machine model description
- `State`: US state where sale occurred
- `SalePrice`: Target variable (auction sale price in USD)

## Project Structure

```
bulldozer-price-prediction/
├── end-to-end-bulldozer-price-regression.ipynb  # Main analysis notebook
├── data/                                         # Dataset files
│   ├── TrainAndValid.csv
│   ├── Test.csv
│   ├── train_tmp.csv
│   └── Data Dictionary.xlsx
├── env/                                         # Python environment
├── README.md
└── requirements.txt
```

## Technologies Used

- **Python 3.13+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib** - Data visualization
- **Jupyter Notebook** - Interactive development environment

## Machine Learning Pipeline

### 1. Data Exploration & Analysis
- Comprehensive exploratory data analysis (EDA)
- Missing value analysis and visualization
- Feature distribution analysis
- Temporal data patterns investigation

### 2. Data Preprocessing
- **Date Feature Engineering**: Extracted year, month, day, day of week, and day of year
- **Missing Value Handling**: 
  - Numeric features: Filled with median values
  - Categorical features: Encoded missing values as separate category
- **Categorical Encoding**: Converted categorical variables to numerical codes
- **Feature Alignment**: Ensured test data matches training data structure

### 3. Model Development
- **Algorithm**: Random Forest Regressor
- **Hyperparameter Tuning**: RandomizedSearchCV with 100 iterations
- **Cross-Validation**: 5-fold cross-validation
- **Performance Metrics**: 
  - RMSLE (Root Mean Squared Log Error) - Primary metric
  - MAE (Mean Absolute Error)
  - R² Score

### 4. Model Optimization
- Experimented with different sample sizes (`max_samples`)
- Optimized hyperparameters:
  - `n_estimators`: Number of trees
  - `max_depth`: Maximum tree depth
  - `min_samples_split`: Minimum samples to split node
  - `min_samples_leaf`: Minimum samples in leaf
  - `max_features`: Features considered for best split

## Results!

The final optimized model achieved:
- **Training RMSLE**: ~0.14
- **Validation RMSLE**: ~0.25
- **Training R²**: ~0.98
- **Validation R²**: ~0.89

**Prediction Statistics on Test Set:**
- Mean predicted price: $29,269
- Price range: $6,395 - $86,091
- Standard deviation: $18,197

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.13+
pip or conda package manager
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SalehAlomair/bulldozer-price-prediction.git
cd bulldozer-price-prediction
```

2. **Create virtual environment**
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
   - Visit the [Kaggle competition page](https://www.kaggle.com/competitions/bluebook-for-bulldozers/data)
   - Download the dataset files to the `data/` directory

5. **Run the notebook**
```bash
jupyter notebook end-to-end-bulldozer-price-regression.ipynb
```

## Key Insights

1. **Feature Engineering**: Date-time features significantly improved model performance
2. **Missing Value Strategy**: Creating binary indicators for missing values preserved valuable information
3. **Hyperparameter Tuning**: RandomizedSearchCV effectively optimized model performance
4. **Data Consistency**: Ensuring feature alignment between training and test sets is crucial for deployment

## 🔧 Usage

### Making Predictions

```python
# Load and preprocess your data
df_new = preprocess_data(your_bulldozer_data)

# Align features with training data
df_new = df_new[X_train.columns]

# Make predictions
predictions = ideal_model.predict(df_new)
```

### Model Evaluation

```python
# Use the built-in evaluation function
scores = show_scores(your_model)
print(f"RMSLE: {scores['Valid RMSLE']:.4f}")
print(f"R² Score: {scores['Valid R^2']:.4f}")
```

## Requirements

See `requirements.txt` for full list of dependencies:

```
numpy>=2.3.1
pandas>=2.3.1
matplotlib>=3.10.0
scikit-learn>=1.7.1
jupyter>=1.0.0
```
