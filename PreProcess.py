import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# =========================================================
# Load Data
# =========================================================

df = pd.read_csv("train.csv")
Test = pd.read_csv("test.csv")


# =========================================================
# Feature Engineering
# =========================================================

# Total square footage
df["TotalSF"] = (
    df["TotalBsmtSF"]
    + df["1stFlrSF"]
    + df["2ndFlrSF"]
)

Test["TotalSF"] = (
    Test["TotalBsmtSF"]
    + Test["1stFlrSF"]
    + Test["2ndFlrSF"]
)


# Total bathrooms
df["TotalBathrooms"] = (
    df["FullBath"]
    + 0.5 * df["HalfBath"]
    + df["BsmtFullBath"]
    + 0.5 * df["BsmtHalfBath"]
)

Test["TotalBathrooms"] = (
    Test["FullBath"]
    + 0.5 * Test["HalfBath"]
    + Test["BsmtFullBath"]
    + 0.5 * Test["BsmtHalfBath"]
)


# Total porch area
df["TotalPorchSF"] = (
    df["OpenPorchSF"]
    + df["3SsnPorch"]
    + df["EnclosedPorch"]
    + df["ScreenPorch"]
    + df["WoodDeckSF"]
)

Test["TotalPorchSF"] = (
    Test["OpenPorchSF"]
    + Test["3SsnPorch"]
    + Test["EnclosedPorch"]
    + Test["ScreenPorch"]
    + Test["WoodDeckSF"]
)


# Total house area
df["TotalHouseArea"] = (
    df["GrLivArea"]
    + df["TotalBsmtSF"]
)

Test["TotalHouseArea"] = (
    Test["GrLivArea"]
    + Test["TotalBsmtSF"]
)


# Total rooms
df["TotalRooms"] = (
    df["TotRmsAbvGrd"]
    + df["FullBath"]
    + df["HalfBath"]
)

Test["TotalRooms"] = (
    Test["TotRmsAbvGrd"]
    + Test["FullBath"]
    + Test["HalfBath"]
)


# =========================================================
# Features and Target
# =========================================================

X = df.drop(columns=["SalePrice", "Id"])
y = df["SalePrice"]

test = Test.drop(columns=["Id"])


# =========================================================
# Separate Numerical and Categorical Features
# =========================================================

numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()


print("Number of numerical features:", len(numerical_cols))
print("Number of categorical features:", len(categorical_cols))

print("\nNumerical columns:")
print(numerical_cols)

print("\nCategorical columns:")
print(categorical_cols)


# =========================================================
# Train / Validation Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# Numerical Pipeline
# =========================================================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])


# =========================================================
# Categorical Pipeline
# =========================================================

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# =========================================================
# Preprocessor
# =========================================================

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])


# =========================================================
# Fit Preprocessor ONLY on Training Data
# =========================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

test_processed = preprocessor.transform(test)


# =========================================================
# Information
# =========================================================

print("\nOriginal training shape:")
print(X_train.shape)

print("\nProcessed training shape:")
print(X_train_processed.shape)

print("\nProcessed validation shape:")
print(X_test_processed.shape)

print("\nProcessed Kaggle test shape:")
print(test_processed.shape)