import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk
from pathlib import Path

st.set_page_config(
    page_title="UrbanHeat AI | Bengaluru",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- PROJECT CONFIGURATION --------------------
MODEL_PATH = Path("UrbanHeat_AI_Final_Model (15).pkl")
IMPORTANCE_PATH = Path("UrbanHeat_Feature_Importance.csv")
MAP_PATH = Path("UrbanHeat_Map_Data.csv")

FEATURES = [
    "NDVI", "NDBI", "EVI", "MNDWI", "NO2_Emissions", "Pop_Density",
    "Relative_Humidity", "SAVI", "Wind_Speed", "Latitude", "Longitude",
    "Building_Density_Ratio", "Dist_Water", "Dist_Park", "Dist_Highway",
    "Street_Density", "Height_Variability", "SVF_Proxy", "Albedo",
    "Soil_Moisture", "Surface_Pressure_hPa", "Relative_Wealth_Index",
    "Dist_to_RWI_Node"
]

# -------------------- STYLING --------------------
st.markdown("""
<style>
    .stApp {background: linear-gradient(135deg, #07111f 0%, #101d30 100%);}
    h1, h2, h3 {color: #eaf2ff;}
    p, label, li {color: #d5deeb;}
    div[data-testid="stMetric"] {
        background: rgba(28, 48, 72, 0.75);
        border: 1px solid #29415e;
        padding: 15px;
        border-radius: 12px;
    }
    div[data-testid="stMetricLabel"] {color: #b8c9dd;}
    div[data-testid="stMetricValue"] {color: #ffffff;}
    .small-note {color:#aabbd0; font-size: 0.88rem;}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD FILES --------------------
@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_PATH)
    if isinstance(saved, dict):
        model = saved.get("model", saved.get("geo_model"))
        model_features = saved.get("features", FEATURES)
    else:
        model = saved
        model_features = FEATURES
    if model is None:
        raise ValueError("The model file does not contain a model under 'model' or 'geo_model'.")
    return model, list(model_features)

@st.cache_data
def load_importance():
    if not IMPORTANCE_PATH.exists():
        return pd.DataFrame(columns=["Feature", "Importance"])
    data = pd.read_csv(IMPORTANCE_PATH)
    if {"Feature", "Importance"}.issubset(data.columns):
        return data.sort_values("Importance", ascending=False)
    return pd.DataFrame(columns=["Feature", "Importance"])

@st.cache_data
def load_map_data():
    if not MAP_PATH.exists():
        return pd.DataFrame()
    data = pd.read_csv(MAP_PATH)
    required = {"Latitude", "Longitude", "LST"}
    if not required.issubset(data.columns):
        return pd.DataFrame()
    return data

if not MODEL_PATH.exists():
    st.error("Model file not found. Upload UrbanHeat_AI_Final_Model.pkl beside app.py.")
    st.stop()

try:
    model, model_features = load_model()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

importance = load_importance()
map_data = load_map_data()

# -------------------- SIDEBAR --------------------
st.sidebar.title("🌡️ UrbanHeat AI")
st.sidebar.caption("Bengaluru Urban Heat Intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "LST Prediction", "Bengaluru Heat Map",
     "Model Performance", "Feature Analysis", "About"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Supervised machine learning • Random Forest")

# -------------------- DASHBOARD --------------------
if page == "Dashboard":
    st.title("🌡️ UrbanHeat AI")
    st.subheader("AI-Powered Urban Heat Prediction & Smart City Planning")
    st.markdown("**Study area:** Bengaluru, Karnataka")
    st.write(
        "Explore land-surface-temperature patterns in the sampled Bengaluru "
        "locations and estimate LST using a trained Random Forest regression model."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Geographic validation R²", "0.517")
    c2.metric("Geographic validation MAE", "1.068 °C")
    c3.metric("Geographic validation RMSE", "1.465 °C")

    st.info(
        "These are the geographic-validation results from the project experiment. "
        "They are not live weather measurements or future-temperature forecasts."
    )

    if not map_data.empty:
        a, b, c = st.columns(3)
        a.metric("Sampled records", f"{len(map_data):,}")
        b.metric("Mean observed LST", f"{map_data['LST'].mean():.2f} °C")
        c.metric("Highest observed LST", f"{map_data['LST'].max():.2f} °C")

        st.subheader("Observed LST distribution")
        st.line_chart(map_data["LST"].reset_index(drop=True), height=250)
    else:
        st.warning("Map data file is missing. Upload UrbanHeat_Map_Data.csv to enable map analytics.")

    st.subheader("Project workflow")
    st.markdown(
        "1. Prepare environmental, built-up and location-related input features.\n"
        "2. Train a supervised Random Forest regression model.\n"
        "3. Evaluate performance using geographically grouped test data.\n"
        "4. Explore sampled locations and estimate LST through this dashboard."
    )

# -------------------- PREDICTION --------------------
elif page == "LST Prediction":
    st.title("🧠 Land Surface Temperature Prediction")
    st.write("Enter environmental and location features to estimate LST for a sampled-location scenario.")
    st.caption(
        "Use values in the same units and scale as the training dataset. "
        "The preset values below are illustrative starting points; replace them with known study data when available."
    )

    defaults = {
        "NDVI": 0.25, "NDBI": 0.15, "EVI": 0.20, "MNDWI": 0.05,
        "NO2_Emissions": 20.0, "Pop_Density": 5000.0,
        "Relative_Humidity": 55.0, "SAVI": 0.20, "Wind_Speed": 2.0,
        "Latitude": 12.95, "Longitude": 77.60,
        "Building_Density_Ratio": 0.45, "Dist_Water": 2.0,
        "Dist_Park": 1.0, "Dist_Highway": 1.5, "Street_Density": 5.0,
        "Height_Variability": 8.0, "SVF_Proxy": 0.5, "Albedo": 0.2,
        "Soil_Moisture": 0.2, "Surface_Pressure_hPa": 950.0,
        "Relative_Wealth_Index": 0.0, "Dist_to_RWI_Node": 1.0
    }

    descriptions = {
        "NDVI": "Vegetation index", "NDBI": "Built-up index",
        "EVI": "Enhanced vegetation index", "MNDWI": "Water index",
        "NO2_Emissions": "NO₂ emissions feature",
        "Pop_Density": "Population density feature",
        "Relative_Humidity": "Relative humidity (%)",
        "SAVI": "Soil-adjusted vegetation index",
        "Wind_Speed": "Wind speed feature",
        "Latitude": "Latitude", "Longitude": "Longitude",
        "Building_Density_Ratio": "Building density ratio",
        "Dist_Water": "Distance to water feature",
        "Dist_Park": "Distance to park feature",
        "Dist_Highway": "Distance to highway feature",
        "Street_Density": "Street density feature",
        "Height_Variability": "Building height variability",
        "SVF_Proxy": "Sky-view-factor proxy",
        "Albedo": "Surface reflectance feature",
        "Soil_Moisture": "Soil moisture feature",
        "Surface_Pressure_hPa": "Surface pressure (hPa)",
        "Relative_Wealth_Index": "Relative wealth index",
        "Dist_to_RWI_Node": "Distance to wealth-index node"
    }

    with st.form("prediction_form"):
        st.markdown("#### Enter model features")
        values = {}
        columns = st.columns(3)
        for i, feature in enumerate(model_features):
            with columns[i % 3]:
                values[feature] = st.number_input(
                    f"{feature}",
                    value=float(defaults.get(feature, 0.0)),
                    help=descriptions.get(feature, feature),
                    format="%.6f"
                )
        submitted = st.form_submit_button("Predict LST", type="primary", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([[values[f] for f in model_features]], columns=model_features)
        try:
            prediction = float(model.predict(input_df)[0])
            st.success(f"Estimated Land Surface Temperature: {prediction:.2f} °C")
            st.caption(
                "This is a model estimate for the supplied feature values, not a live "
                "measurement, official heat alert, or future forecast."
            )
            report = input_df.copy()
            report["Predicted_LST_C"] = prediction
            st.download_button(
                "Download this prediction (CSV)",
                report.to_csv(index=False).encode("utf-8"),
                file_name="UrbanHeat_AI_Prediction_Report.csv",
                mime="text/csv"
            )
        except Exception as exc:
            st.error(f"Prediction could not be generated: {exc}")

# -------------------- MAP --------------------
elif page == "Bengaluru Heat Map":
    st.title("🗺️ Bengaluru Sampled-Location Heat Map")
    if map_data.empty:
        st.error("UrbanHeat_Map_Data.csv is missing or does not contain Latitude, Longitude and LST.")
    else:
        st.write("Explore the sampled locations included in the project dataset.")
        metric = st.selectbox("Color locations by", ["Actual LST", "Predicted LST"])
        color_col = "Predicted_LST" if metric == "Predicted LST" and "Predicted_LST" in map_data.columns else "LST"
        plot_data = map_data.dropna(subset=["Latitude", "Longitude", color_col]).copy()
        low = plot_data[color_col].quantile(0.25)
        high = plot_data[color_col].quantile(0.75)

        def color_for_temp(t):
            if t <= low:
                return [55, 145, 255, 190]
            if t <= high:
                return [255, 190, 55, 210]
            return [245, 65, 65, 225]

        plot_data["color"] = plot_data[color_col].apply(color_for_temp)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=plot_data,
            get_position="[Longitude, Latitude]",
            get_fill_color="color",
            get_radius=110,
            pickable=True,
            auto_highlight=True
        )
        view = pdk.ViewState(
            latitude=float(plot_data["Latitude"].mean()),
            longitude=float(plot_data["Longitude"].mean()),
            zoom=11,
            pitch=35
        )
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip={"html": f"<b>{metric}:</b> {{{color_col}}} °C<br/><b>Latitude:</b> {{Latitude}}<br/><b>Longitude:</b> {{Longitude}}"}
        ))
        st.caption(
            "Blue, yellow and red indicate lower, middle and higher relative values "
            "within this dataset. They are not official warning categories. "
            "The points represent sampled locations, not every location in Bengaluru."
        )
        st.subheader("Sampled location records")
        st.dataframe(plot_data.drop(columns=["color"]), use_container_width=True)
        st.download_button(
            "Download map data (CSV)",
            map_data.to_csv(index=False).encode("utf-8"),
            file_name="UrbanHeat_AI_Map_Data.csv",
            mime="text/csv"
        )

# -------------------- MODEL PERFORMANCE --------------------
elif page == "Model Performance":
    st.title("📊 Model Performance")
    st.write("Geographic validation evaluates the model on held-out spatial groups.")
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", "0.517")
    c2.metric("MAE", "1.068 °C")
    c3.metric("RMSE", "1.465 °C")

    st.markdown("""
    **Metric interpretation**
    - **R²:** proportion of target variation explained by the model on the geographic test split.
    - **MAE:** average absolute prediction error, in the target's temperature units.
    - **RMSE:** square root of mean squared error; larger errors receive more weight.
    """)
    st.warning(
        "The geographic validation split groups nearby coordinates rounded to two decimal places. "
        "It is a spatial-group holdout, not a ward-based validation."
    )

# -------------------- FEATURE ANALYSIS --------------------
elif page == "Feature Analysis":
    st.title("🔍 Feature Importance & Explainability")
    if importance.empty:
        st.warning("UrbanHeat_Feature_Importance.csv is missing or has unexpected columns.")
    else:
        top_n = st.slider("Number of features to display", 5, min(23, len(importance)), min(10, len(importance)))
        top = importance.head(top_n).sort_values("Importance")
        st.bar_chart(top.set_index("Feature")["Importance"], height=420)
        st.dataframe(importance, use_container_width=True)
        st.caption(
            "Feature importance indicates how much the trained Random Forest relied on each feature. "
            "It does not prove that a feature causes temperature changes."
        )
        st.download_button(
            "Download feature importance (CSV)",
            importance.to_csv(index=False).encode("utf-8"),
            file_name="UrbanHeat_AI_Feature_Importance.csv",
            mime="text/csv"
        )

# -------------------- ABOUT --------------------
elif page == "About":
    st.title("ℹ️ About UrbanHeat AI")
    st.markdown("""
    **UrbanHeat AI** is a supervised machine-learning project that estimates
    Land Surface Temperature (LST) from environmental, built-up and
    location-related features.

    **Study area:** sampled locations in Bengaluru, Karnataka.

    **Algorithm:** Random Forest Regression.

    **Project objectives**
    - Explore relationships between urban features and LST.
    - Estimate LST for user-supplied feature combinations.
    - Visualize sampled-location temperature patterns.
    - Support learning and discussion around urban heat and planning.

    **Important limitations**
    - The current dataset covers sampled central Bengaluru coordinates, not the entire city.
    - The dashboard is not connected to live satellite feeds or live weather services.
    - Predictions are not future-month forecasts.
    - The map displays sampled points, not ward boundaries or official ward averages.
    - Geographic validation results describe this particular dataset and split.
    """)
    st.markdown("---")
    st.caption("UrbanHeat AI • Academic supervised-learning project")
