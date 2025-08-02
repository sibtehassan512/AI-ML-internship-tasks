
# AI/ML Internship Project Series

This repository contains multiple tasks designed to practice data exploration, regression, and classification using various datasets and models.

---

# Table of Contents
1. [Task 1: Iris Dataset Exploration and Visualization](#task-1-iris)
2. [Task 2: Short-Term Stock Price Prediction](#task-2-stock)
3. [Task 3: Heart Disease Risk Classification](#task-3-heart)
4. [Task 6: House Price Prediction](#task-6-house-price-prediction)

---

## Task 1: Iris Dataset Exploration and Visualization

### Objective
Understand how to load, inspect, and visualize the Iris dataset to reveal trends, distributions, and outliers.

### Dataset
- **Name**: Iris Dataset  
- **Source**: Seaborn library or UCI repository  
- **Size**: 150 entries, 5 columns (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`, `species`)

### Workflow
1. **Data Loading & Inspection**  
   - Loaded via `sns.load_dataset("iris")` or `pd.read_csv`.  
   - Checked shape, column names, `.head()`, `.info()`, and `.describe()`.
2. **Visualizations**  
   - Scatter plot of `sepal_length` vs `petal_length` (hue by `species`).  
   - Histograms for each numeric feature.  
   - Box plots to detect outliers.  
   - Pairplot to view all feature relationships.

### Insights
- Petal length and width are strong separators between species.  
- Some outliers observed in petal width (especially for virginica).

---

## Task 2: Short-Term Stock Price Prediction

### Objective
Predict the next day’s closing price of a chosen stock using historical Open, High, Low, and Volume data.

### Dataset
- **Source**: Yahoo Finance via `yfinance` Python library  
- **Example Stock**: AAPL (Apple Inc.)  
- **Time Range**: 2020-01-01 to 2023-12-31 (modifiable)

### ⚙️ Workflow
1. **Data Loading**  
   - Fetched with `yf.download("AAPL", start, end)`.  
   - Created `Next_Close` by shifting `Close` by -1.
2. **Feature Engineering**  
   - Used `Open`, `High`, `Low`, and `Volume` as predictors.  
   - Dropped rows with NaN in `Next_Close`.
3. **Model Training**  
   - Split data (80% train, 20% test) without shuffle.  
   - Trained **Linear Regression** as a baseline.  
   - Trained **Random Forest Regressor** for nonlinear modeling.
4. **Evaluation**  
   - Calculated R² and RMSE for both models.  
   - Plotted actual vs predicted closing prices.

### Results
| Model                | R² (approx.) | RMSE (approx.) |
|----------------------|--------------|----------------|
| Linear Regression    | 0.95         | 2.23           |
| Random Forest        | 0.083        | 10.40          |

### Insights
- Random Forest outperforms Linear Regression.  
- Time-series split (no shuffle) is essential.  
- Potential enhancements: technical indicators, lag features, LSTM models.

---

## Task 3: Heart Disease Risk Classification

### Objective
Build a binary classification model to predict the presence of heart disease based on patient health metrics.

### Dataset
- **Name**: Heart Disease UCI Dataset  
- **Source**: Kaggle (`heart.csv`) or UCI repository  
- **Size**: ~303 samples, 14 attributes (e.g., `age`, `sex`, `cp`, `chol`, etc.), binary target `target`

### Workflow
1. **Data Loading & Cleaning**  
   - Loaded via `pd.read_csv("heart.csv")`.  
   - Checked for missing values; dataset contains none.
2. **Exploratory Data Analysis**  
   - Plotted distribution of `target`.  
   - Correlation heatmap of features.  
   - Box plots for key variables (e.g., age by target).
3. **Model Training**  
   - Split data (80% train, 20% test).  
   - Trained **Logistic Regression**.
4. **Evaluation**  
   - Metrics: Accuracy, ROC-AUC, Confusion Matrix.  
   - Plotted ROC curve and confusion matrix heatmap.
5. **Feature Importance**  
   - Extracted coefficients (Logistic Regression)

### Results
| Metric            | Value (approx.) |
|-------------------|-----------------|
| Accuracy          | 0.79            |
| ROC-AUC           | 0.88            |

###  Insights
- Top predictors: chest pain type, thalassemia, age, cholesterol.  
- Logistic Regression gives interpretable weights

---

## Task 6: House Price Prediction

### Objective
The goal of this task is to build a regression model that can predict house prices based on various features like:
- Area (square footage)
- Number of Bedrooms
- Location (categorical)

### Dataset
- **Name**: House Price Prediction Dataset  
- **Source**: [Kaggle](https://www.kaggle.com/datasets/zafarali27/house-price-prediction-dataset)  
- **Size**: 2000+ entries  
- **Features Used**:  
  - `Area`: Total area of the house  
  - `Bedrooms`: Number of bedrooms  
  - `Location`: Area or locality of the house  
  - `Price`: Target variable (house price)

### Model Used
- **Linear Regression** (as a baseline model)

### Workflow
1. **Data Preprocessing**
   - Handled missing values (none in this dataset)
   - One-hot encoded categorical `Location`
   - Scaled numeric features (`Area`, `Bedrooms`)
2. **Model Training**
   - Used `LinearRegression` from Scikit-learn
   - Split dataset (80% training, 20% testing)
3. **Evaluation**
   - Calculated Mean Absolute Error (MAE)
   - Calculated Root Mean Squared Error (RMSE)
   - Visualized Actual vs Predicted prices

### Results

| Metric | Value (approx.) |
|--------|------------------|
| MAE    | 241,340.17 |
| RMSE   | 278,152.90 |

### Insights
- The Linear Regression model gives a good baseline performance.
- There’s room for improvement using more features (e.g., `Bathrooms`, `Garage`, `Condition`) or ensemble methods like **Gradient Boosting**.

---

## Repository Structure
```
├── Task1_Iris_Exploration.ipynb
├── Task2_Stock_Prediction.ipynb
├── Task3_Heart_Disease_Classification.ipynb
├── Task6_House_Price_Prediction.ipynb
└── readme.txt
```
