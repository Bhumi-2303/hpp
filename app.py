# app.py

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="California Housing Price Predictor 🏡",
    page_icon="🏠",
    layout="centered",
)

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_artifacts():
    """Safely load the trained model and scaler using pickle."""
    try:
        with open(os.path.join("model", "model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join("model", "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model or Scaler not found! Please ensure 'model/model.pkl' and 'model/scaler.pkl' exist.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Failed to load model files: {e}")
        st.stop()

# Load model + scaler
model, scaler = load_artifacts()

# ---------------------- HEADER SECTION ----------------------
st.title("🏘️ California Housing Price Prediction App")
st.markdown(
    """
    This app predicts the **Median House Value** (🏠 price) for a region in California  
    based on real-world housing and location factors.  
    Enter the details below and get an **instant price estimate** 💵.
    """
)

st.divider()
st.subheader("📋 About the Inputs")

with st.expander("ℹ️ Click to understand each input feature"):
    st.markdown("""
    - **Longitude 📍** → East–West position on the globe.  
      In California, it typically ranges from **-125 (West)** to **-114 (East)**.
    - **Latitude 🌍** → North–South position.  
      In California, it ranges roughly from **32 (South)** to **42 (North)**.
    - **Total Rooms 🛋️** → Number of rooms in all houses of the area/block combined.
    - **Total Bedrooms 🛏️** → Number of bedrooms in all houses of the area.
    - **Population 👨‍👩‍👧‍👦** → Total number of people living in that neighborhood or block.
    - **Households 🏠** → Number of occupied housing units (families or individuals).
    - **Median Income 💰** → The average (median) income of residents in tens of thousands of USD.  
      e.g., `4.5` means **$45,000** average annual income.
    - **Ocean Proximity 🌊** → How close the location is to the Pacific Ocean.
    """)
    st.info("💡 *Note:* The app automatically performs log-scaling for some features (like rooms and population) to make predictions more accurate.")

st.divider()
st.subheader("🏗️ Enter House Details")

# ---------------------- INPUT SECTION ----------------------
col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input(
        "Longitude 📍",
        min_value=-125.0,
        max_value=-114.0,
        step=0.01,
        value=-120.0,
        help="East–West coordinate (California: -125 to -114). Smaller = more westward."
    )

    latitude = st.number_input(
        "Latitude 🌍",
        min_value=32.0,
        max_value=42.0,
        step=0.01,
        value=37.0,
        help="North–South coordinate (California: 32 to 42). Larger = more northward."
    )

    total_rooms = st.number_input(
        "Total Rooms 🛋️",
        min_value=1.0,
        value=2000.0,
        step=100.0,
        help="Total number of rooms across all houses in the area."
    )

    total_bedrooms = st.number_input(
        "Total Bedrooms 🛏️",
        min_value=1.0,
        value=400.0,
        step=50.0,
        help="Total number of bedrooms in all houses of the block."
    )

with col2:
    population = st.number_input(
        "Population 👨‍👩‍👧‍👦",
        min_value=1.0,
        value=1000.0,
        step=50.0,
        help="Total number of residents living in this area."
    )

    households = st.number_input(
        "Households 🏠",
        min_value=1.0,
        value=350.0,
        step=10.0,
        help="Number of families or occupied housing units."
    )

    median_income = st.number_input(
        "Median Income 💰 (in $10,000)",
        min_value=0.1,
        value=4.0,
        step=0.1,
        help="Median income in tens of thousands of USD (e.g. 4 = $40,000)."
    )

# ---------------------- CATEGORICAL INPUT ----------------------
st.markdown("### 🌊 Ocean Proximity")
ocean_proximity = st.selectbox(
    "Select how close the area is to the ocean:",
    options=[
        "INLAND",
        "NEAR BAY",
        "NEAR OCEAN",
        "<1H OCEAN",
        "ISLAND",
    ],
    help="Proximity of the location to the Pacific Ocean."
)

st.divider()

# ---------------------- PREDICTION ----------------------
if st.button("🔮 Predict House Value"):
    try:
        user_data = {
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": 30,  # default value for simplicity
            "total_rooms": np.log(total_rooms + 1),
            "total_bedrooms": np.log(total_bedrooms + 1),
            "population": np.log(population + 1),
            "households": np.log(households + 1),
            "median_income": median_income,
            "bedroom_ratio": total_bedrooms / total_rooms,
            "household_rooms": total_rooms / households,
        }

        # Add dummy vars for proximity
        proximity_categories = [
            "NEAR BAY",
            "INLAND",
            "NEAR OCEAN",
            "<1H OCEAN",
            "ISLAND",
        ]
        for cat in proximity_categories:
            user_data[cat] = 1 if cat == ocean_proximity else 0

        input_df = pd.DataFrame([user_data])
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        st.success("✅ Prediction Complete!")
        st.metric(
            label="🏡 Estimated Median House Value",
            value=f"${prediction:,.2f}"
        )

        st.balloons()

        st.write("### 📊 Prediction Summary")
        st.bar_chart(pd.DataFrame({"Estimated Value": [prediction]}, index=["Price ($)"]))

        st.caption("💡 Model internally applies scaling and feature transformation for better accuracy.")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# ---------------------- FOOTER ----------------------
st.divider()
st.caption(
    "Developed by Anuj & Bhavini with ❤️ | Powered by Random Forest Regressor 🌲"
)
