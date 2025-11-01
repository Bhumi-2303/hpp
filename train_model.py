# train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# ====================================================
# SAFE SAVE FUNCTION
# ====================================================
def safe_save(obj, path):
    """Safely save a Python object using joblib with overwrite protection."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # If existing file is corrupted or locked, delete it
        if os.path.exists(path):
            os.remove(path)

        joblib.dump(obj, path, compress=3)
        print(f"✅ Successfully saved: {path}")
    except Exception as e:
        print(f"❌ Error saving {path}: {e}")

# ====================================================
# LOAD AND CLEAN DATA
# ====================================================
data_path = "housing.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Could not find {data_path}. Make sure it exists in your project folder.")

print("📥 Loading dataset...")
data = pd.read_csv(data_path)

# Drop missing values
data.dropna(inplace=True)
print(f"✅ Data Loaded: {data.shape[0]} rows, {data.shape[1]} columns")

# ====================================================
# SPLIT DATA
# ====================================================
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_data = X_train.join(y_train)

# ====================================================
# FEATURE TRANSFORMATION
# ====================================================
log_cols = ["total_rooms", "total_bedrooms", "population", "households"]
for col in log_cols:
    train_data[col] = np.log(train_data[col] + 1)

# One-hot encode categorical column
ocean_dummies = pd.get_dummies(train_data["ocean_proximity"], dtype=int)
train_data = train_data.join(ocean_dummies)
train_data.drop(["ocean_proximity"], axis=1, inplace=True)

# Feature Engineering
train_data["bedroom_ratio"] = train_data["total_bedrooms"] / train_data["total_rooms"]
train_data["household_rooms"] = train_data["total_rooms"] / train_data["households"]

# ====================================================
# SCALING
# ====================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_data.drop("median_house_value", axis=1))
y_train = train_data["median_house_value"]

# ====================================================
# LINEAR REGRESSION BASELINE
# ====================================================
print("\n📊 Training Linear Regression (baseline)...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
print("✅ Linear Regression Trained")

# ====================================================
# TEST DATA PREPARATION
# ====================================================
test_data = X_test.join(y_test)

for col in log_cols:
    test_data[col] = np.log(test_data[col] + 1)

ocean_dummies_test = pd.get_dummies(test_data["ocean_proximity"], dtype=int)
test_data = test_data.join(ocean_dummies_test)
test_data.drop(["ocean_proximity"], axis=1, inplace=True)

test_data["bedroom_ratio"] = test_data["total_bedrooms"] / test_data["total_rooms"]
test_data["household_rooms"] = test_data["total_rooms"] / test_data["households"]

# Align columns
test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

X_test_scaled = scaler.transform(test_data.drop("median_house_value", axis=1))
y_test = test_data["median_house_value"]

# ====================================================
# BASELINE EVALUATION
# ====================================================
lr_pred = lr_model.predict(X_test_scaled)
lr_r2 = r2_score(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

print(f"📈 Linear Regression R²: {lr_r2:.4f}, RMSE: {lr_rmse:,.2f}")

# ====================================================
# RANDOM FOREST MODEL
# ====================================================
print("\n🌲 Training Random Forest...")
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(train_data.drop("median_house_value", axis=1), y_train)

rf_pred = rf_model.predict(test_data.drop("median_house_value", axis=1))
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print(f"📊 Random Forest R² (before tuning): {rf_r2:.4f}, RMSE: {rf_rmse:,.2f}")

# ====================================================
# GRID SEARCH (TUNING)
# ====================================================
print("\n🔍 Running Grid Search for Random Forest tuning (may take time)...")

param_grid = {
    "n_estimators": [30, 50, 100],
    "max_features": [8, 12, 20],
    "min_samples_split": [2, 4, 6],
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(train_data.drop("median_house_value", axis=1), y_train)

best_model = grid_search.best_estimator_
print(f"\n✅ Best Parameters Found: {grid_search.best_params_}")

# ====================================================
# FINAL EVALUATION
# ====================================================
best_pred = best_model.predict(test_data.drop("median_house_value", axis=1))
best_r2 = r2_score(y_test, best_pred)
best_rmse = np.sqrt(mean_squared_error(y_test, best_pred))

print(f"🎯 Final Random Forest R²: {best_r2:.4f}, RMSE: {best_rmse:,.2f}")

# ====================================================
# SAVE MODELS SAFELY
# ====================================================
safe_save(best_model, "model/model.pkl")
safe_save(scaler, "model/scaler.pkl")

print("\n🎉 Training complete! All files saved successfully.")
