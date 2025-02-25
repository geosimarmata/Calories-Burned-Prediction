# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import streamlit as st
import joblib
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

#############################################
#      Data Loading & Model Training
#############################################
@st.cache_data
def load_data():
    # Update these paths if needed
    exercise = pd.read_csv(r'E:\# AI Project\Calories Burned Prediction\exercise.csv')
    calories = pd.read_csv(r'E:\# AI Project\Calories Burned Prediction\calories.csv')
    df = pd.merge(exercise, calories, on='User_ID')
    # Replace gender values with numeric
    df.replace({'male': 0, 'female': 1}, inplace=True)
    # Remove columns that are less informative (if they exist)
    to_remove = ['Weight', 'Duration']
    df.drop(to_remove, axis=1, inplace=True, errors='ignore')
    return df

@st.cache_resource
def train_models(df):
    # Prepare data (assume features: Age, Height, Gender, Body_Temp, Heart_Rate)
    X = df.drop(['User_ID', 'Calories'], axis=1)
    y = df['Calories']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=22)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train Random Forest
    rf_model = RandomForestRegressor(random_state=22)
    rf_model.fit(X_train_scaled, y_train)
    
    # Train XGBoost
    xgb_model = XGBRegressor(random_state=22)
    xgb_model.fit(X_train_scaled, y_train)
    
    # Evaluate models
    rf_pred = rf_model.predict(X_val_scaled)
    xgb_pred = xgb_model.predict(X_val_scaled)
    rf_mae = mean_absolute_error(y_val, rf_pred)
    xgb_mae = mean_absolute_error(y_val, xgb_pred)
    rf_r2 = r2_score(y_val, rf_pred)
    xgb_r2 = r2_score(y_val, xgb_pred)
    
    performance = {
        "rf_mae": rf_mae,
        "xgb_mae": xgb_mae,
        "rf_r2": rf_r2,
        "xgb_r2": xgb_r2,
    }
    
    return X, scaler, rf_model, xgb_model, performance

# Load data and train models
df = load_data()
X, scaler, rf_model, xgb_model, performance = train_models(df)

# (Optional) Save models/scaler to disk if needed
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

#############################################
#           Sidebar Navigation
#############################################
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About", "Prediction", "EDA", "Model Performance"])

#############################################
#                About Page
#############################################
if page == "About":
    st.title("Calories Burn Prediction Dashboard")
    st.markdown("""
    ### Project Overview
    This dashboard predicts the number of calories burned based on user input features.
    
    **Features include:**
    - **Age (years)**
    - **Height (cm)**
    - **Gender (male/female)**
    - **Body Temperature (°C)**
    - **Heart Rate (bpm)**
    
    The models used are **Random Forest** and **XGBoost**. You can compare their performance and explore the data interactively in this dashboard.
    
    **How to Use:**
    - Go to the **Prediction** page to enter your details and get a prediction.
    - Explore the data on the **EDA** page.
    - View model metrics and feature importance on the **Model Performance** page.
    """)

#############################################
#              Prediction Page
#############################################
elif page == "Prediction":
    st.title("Make a Prediction")
    
    # Define features and display units
    features_list = ["Age", "Height", "Gender", "Body_Temp", "Heart_Rate"]
    feature_info = {
        "Age": "years",
        "Height": "cm",
        "Gender": "(male/female)",
        "Body_Temp": "°C",
        "Heart_Rate": "bpm"
    }
    
    st.sidebar.header("Select Model & Enter Input Data")
    model_choice = st.sidebar.selectbox("Choose a model:", ["Random Forest", "XGBoost"])
    
    # Collect user inputs
    user_inputs = {}
    for feature in features_list:
        label = f"Enter {feature}"
        if feature in feature_info and feature != "Gender":
            label += f" ({feature_info[feature]})"
        
        if feature.lower() == "gender":
            gender = st.sidebar.selectbox(label, ["male", "female"])
            user_inputs[feature] = 0 if gender == "male" else 1
        else:
            # Default value set to the mean (if available) or 0
            default_val = str(round(X[feature].mean(), 2)) if feature in X.columns else "0"
            user_input = st.sidebar.text_input(label, value=default_val)
            try:
                user_inputs[feature] = float(user_input)
            except ValueError:
                user_inputs[feature] = 0.0
    
    # Create DataFrame for prediction (ensure same column order as training)
    input_df = pd.DataFrame([user_inputs], columns=X.columns)
    
    # Prepare a display version with units
    display_user_inputs = {}
    for feature in X.columns:
        value = user_inputs[feature]
        if feature.lower() == "gender":
            display_user_inputs[feature] = "male" if value == 0 else "female"
        elif feature in feature_info and feature_info[feature]:
            display_user_inputs[f"{feature} ({feature_info[feature]})"] = f"{value} {feature_info[feature]}"
        else:
            display_user_inputs[feature] = value
    st.subheader("User Input Data")
    st.table(pd.DataFrame([display_user_inputs]))
    
    # Scale input data and predict when button is clicked
    input_scaled = scaler.transform(input_df)
    if st.sidebar.button("Predict"):
        model = rf_model if model_choice == "Random Forest" else xgb_model
        prediction = model.predict(input_scaled)
        st.subheader(f"Predicted Calories Burned: {prediction[0]:.2f} kcal")
        
        # (Optional) Display a note on the chosen model's performance
        if model_choice == "Random Forest":
            st.caption(f"Random Forest - MAE: {performance['rf_mae']:.2f}, R²: {performance['rf_r2']:.2f}")
        else:
            st.caption(f"XGBoost - MAE: {performance['xgb_mae']:.2f}, R²: {performance['xgb_r2']:.2f}")

#############################################
#                EDA Page
#############################################
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    
    st.subheader("Data Overview")
    st.write(df.head())
    
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.subheader("Interactive Scatter Plot")
    # Let user select x and y variables from the numerical columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    x_axis = st.selectbox("Select X-axis", numeric_cols, index=0)
    y_axis = st.selectbox("Select Y-axis", numeric_cols, index= numeric_cols.index("Calories") if "Calories" in numeric_cols else 1)
    
    fig_scatter = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
    st.plotly_chart(fig_scatter, use_container_width=True)

#############################################
#         Model Performance Page
#############################################
elif page == "Model Performance":
    st.title("Model Performance & Feature Importance")
    
    st.subheader("Evaluation Metrics")
    st.markdown(f"""
    **Random Forest:**  
    - MAE: {performance['rf_mae']:.2f}  
    - R²: {performance['rf_r2']:.2f}  
    
    **XGBoost:**  
    - MAE: {performance['xgb_mae']:.2f}  
    - R²: {performance['xgb_r2']:.2f}  
    """)
    
    st.subheader("Feature Importance (Random Forest)")
    # Ensure the model has feature_importances_
    if hasattr(rf_model, "feature_importances_"):
        imp = rf_model.feature_importances_
        feat_imp = pd.DataFrame({"Feature": X.columns, "Importance": imp}).sort_values(by="Importance", ascending=False)
        fig_imp = px.bar(feat_imp, x="Feature", y="Importance", title="Feature Importance")
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.write("Random Forest does not provide feature importance.")

