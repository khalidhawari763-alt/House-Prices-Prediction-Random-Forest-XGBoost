from xgboost import XGBRegressor
from PreProcess import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
xgb_model = XGBRegressor(n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
                         )
# xgb_model.fit(X_train,y_train)
#
# predictions = xgb_model.predict(X_test)
#
#
# mae = mean_absolute_error(y_test, predictions)
# rmse = np.sqrt(mean_squared_error(y_test, predictions))
# r2 = r2_score(y_test, predictions)
#
# print("MAE:", mae)
# print("RMSE:", rmse)
# print("R²:", r2)
# errors = pd.DataFrame({
#     "Actual": y_test,
#     "Predicted": predictions
# })
#
# errors["Error"] = errors["Actual"] - errors["Predicted"]
# errors["AbsoluteError"] = abs(errors["Error"])
#
# print(errors.sort_values("AbsoluteError", ascending=False).head(10))
#
# sns.scatterplot(x=y_test, y=predictions)
#
# plt.xlabel("Actual SalePrice")
# plt.ylabel("Predicted SalePrice")
# plt.title("Actual vs Predicted")
# plt.show()
#
# importance = pd.Series(
#     xgb_model.feature_importances_,
#     index=X.columns
# ).sort_values(ascending=False)
#
# print(importance.head(15))
#
# importance.head(15).sort_values().plot(kind="barh")
#
# plt.title("Top 15 Feature Importances")
# plt.xlabel("Importance")
# plt.show()
#
# pred=xgb_model.predict(test)
#
# print(pred[:10])

# Train the best model
xgb_model.fit(X_train_processed, y_train)

pred = xgb_model.predict(test_processed)

submission = pd.DataFrame({
    "Id": Test["Id"],
    "SalePrice": pred
})

submission.to_csv("submission.csv", index=False)

print("Submission file created successfully!")
print(submission.head())