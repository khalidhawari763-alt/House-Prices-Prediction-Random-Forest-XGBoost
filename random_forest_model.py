from sklearn.ensemble import RandomForestRegressor
from PreProcess import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
# 1. Train Random Forest
print("Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest trained!")

predictions = rf_model.predict(X_test)

print(predictions[:20])
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

errors = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

errors["Error"] = errors["Actual"] - errors["Predicted"]
errors["AbsoluteError"] = abs(errors["Error"])

print(errors.sort_values("AbsoluteError", ascending=False).head(10))

sns.scatterplot(x=y_test, y=predictions)

plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Actual vs Predicted")
plt.show()

importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importance.head(15))

importance.head(15).sort_values().plot(kind="barh")

plt.title("Top 15 Feature Importances")
plt.xlabel("Importance")
plt.show()

predictions = rf_model.predict(test)

# Create submission
submission = pd.DataFrame({
    "Id": Test["Id"],
    "SalePrice": predictions
})

# Save
submission.to_csv("submission2.csv", index=False)

print(submission.head())
print(submission.shape)