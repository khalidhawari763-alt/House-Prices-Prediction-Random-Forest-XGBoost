# 🏠 House Prices Prediction

A machine learning project for predicting residential house prices using the **Ames Housing dataset**. The project covers data preprocessing, feature engineering, exploratory data analysis, categorical encoding, missing-value handling, and machine learning model training.

---

## 📌 Project Overview

The goal of this project is to build a machine learning pipeline that can predict the **SalePrice** of houses based on their characteristics.

The dataset contains information about different aspects of a house, including:

* Overall quality
* Living area
* Basement area
* Garage information
* Number of rooms
* Bathrooms
* Year built
* Neighborhood
* Exterior materials
* And many other features

The project focuses on preparing the data correctly and creating useful features before training the prediction model.

---

## 📂 Project Structure

```text
House-Prices-Prediction/
│
├── images/
│   ├── distribution_saleprice.png
│   ├── top_features_correlation.png
│   └── correlation_heatmap.png
│
├── PreProcess.py
├── random_forest_model.py
├── XGboost_model.py
├── train.csv
├── test.csv
├── submission.csv
└── README.md
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost

---

## 🔄 Machine Learning Pipeline

The project follows this general workflow:

```text
Raw Dataset
     │
     ▼
Data Loading
     │
     ▼
Feature Engineering
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Train / Validation Split
     │
     ▼
Missing Value Handling
     │
     ▼
Categorical Encoding
     │
     ▼
Feature Processing
     │
     ▼
Model Training
     │
     ▼
Validation
     │
     ▼
Prediction
```

---

## 🧹 Data Preprocessing

The preprocessing pipeline handles both numerical and categorical features.

### Numerical Features

Missing numerical values are handled using **median imputation**.

```python
SimpleImputer(strategy="median")
```

### Categorical Features

Missing categorical values are filled using the most frequent value.

Categorical variables are then converted into numerical representations using **One-Hot Encoding**.

```python
SimpleImputer(strategy="most_frequent")
OneHotEncoder(handle_unknown="ignore")
```

The preprocessing is implemented using a Scikit-learn `ColumnTransformer` and `Pipeline`.

---

## 🧠 Feature Engineering

Several additional features were created to provide the models with more meaningful information.

### TotalSF

Combines:

* Basement area
* 1st floor area
* 2nd floor area

```text
TotalSF =
TotalBsmtSF + 1stFlrSF + 2ndFlrSF
```

### TotalBathrooms

Combines full and half bathrooms, including basement bathrooms.

```text
TotalBathrooms =
FullBath
+ 0.5 × HalfBath
+ BsmtFullBath
+ 0.5 × BsmtHalfBath
```

### TotalPorchSF

Combines different porch and deck areas:

```text
TotalPorchSF =
OpenPorchSF
+ 3SsnPorch
+ EnclosedPorch
+ ScreenPorch
+ WoodDeckSF
```

### TotalHouseArea

Combines above-ground living area and basement area.

```text
TotalHouseArea =
GrLivArea + TotalBsmtSF
```

### TotalRooms

Combines rooms and bathrooms:

```text
TotalRooms =
TotRmsAbvGrd + FullBath + HalfBath
```

---

## 📊 Exploratory Data Analysis

The project includes several visualizations to understand the dataset and relationships between features.

### Sale Price Distribution

This plot shows the distribution of the target variable, `SalePrice`.

![Sale Price Distribution](images/distribution_saleprice.png)

---

### Top Features Correlation

This visualization shows the numerical features with the strongest absolute correlation with `SalePrice`.

![Top Features Correlation](images/top_features_correlation.png)

---

### Correlation Heatmap

The heatmap provides a visual overview of the relationships between the most relevant numerical features and the target.

![Correlation Heatmap](images/correlation_heatmap.png)

---

## 🔢 Dataset Processing

The original dataset contains numerical and categorical variables.

The preprocessing pipeline separates them automatically:

```python
numerical_cols = X.select_dtypes(
    include=[np.number]
).columns.tolist()

categorical_cols = X.select_dtypes(
    include=["object"]
).columns.tolist()
```

The data is then transformed before being passed to the machine learning model.

Example processed dimensions:

```text
Original training data:
1168 samples × 84 features

Processed training data:
1168 samples × 290 features
```

One-hot encoding increases the number of features because categorical variables can contain multiple categories.

---

## 🤖 Machine Learning Models

The project includes machine learning models for house-price prediction.

### Random Forest

Random Forest is an ensemble learning algorithm based on multiple decision trees.

It can capture nonlinear relationships between house characteristics and prices.

The model is implemented in:

```text
random_forest_model.py
```

---

### XGBoost

XGBoost is a gradient boosting algorithm that builds decision trees sequentially to improve prediction performance.

The model is implemented in:

```text
XGboost_model.py
```

---

## 📈 Model Evaluation

The data is divided into training and validation sets using an 80/20 split:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The validation set is used to evaluate how well the trained model performs on previously unseen data.

---

## 🧪 Reproducibility

A fixed random state is used for the train/validation split:

```python
random_state=42
```

This allows the same split to be reproduced when running the project again.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/khalidhawari763-alt/House-Prices-Prediction-Random-Forest-XGBoost.git
```

### 2. Enter the project directory

```bash
cd House-Prices-Prediction-Random-Forest-XGBoost
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### 4. Run preprocessing

```bash
python PreProcess.py
```

This will:

* Load the datasets
* Perform feature engineering
* Create EDA visualizations
* Handle missing values
* Encode categorical features
* Prepare the processed data

### 5. Train a model

Run the appropriate model script:

```bash
python random_forest_model.py
```

or:

```bash
python XGboost_model.py
```

---

## 📁 Generated Files

The EDA section generates:

```text
images/
├── distribution_saleprice.png
├── top_features_correlation.png
└── correlation_heatmap.png
```

These visualizations are also displayed in this README.

---

## 🎯 Project Goals

This project was built to practice and demonstrate:

* Data preprocessing
* Exploratory Data Analysis
* Feature engineering
* Missing-value handling
* Categorical encoding
* Scikit-learn pipelines
* Regression
* Random Forest
* XGBoost
* Model validation
* Machine learning workflow design

---

## 🔮 Future Improvements

Possible future improvements include:

* Hyperparameter tuning
* Feature selection
* Outlier analysis
* Log transformation of skewed features
* Cross-validation
* Additional regression models
* Model comparison
* Improved feature engineering
* Error analysis
* Visualization of model predictions

---

## 👨‍💻 Author

**Khalid Hawari**

Robotics Science / AI & Robotics student interested in:

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Computer Vision
* Robotics
* Autonomous Systems

---

