import joblib
import pickle
import os

# Paths
model_path = os.path.join("model", "model.pkl")
scaler_path = os.path.join("model", "scaler.pkl")

# Load existing joblib-saved files
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Re-save them using pickle
with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

print("✅ Conversion complete! Both files are now pickle-compatible.")
