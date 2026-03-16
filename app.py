import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Title
st.title("Life Expectancy Prediction App")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Life Expectancy Data.csv')
    return df

df = load_data()

# Display data overview
st.header("Dataset Overview")
st.write(df.head())
st.write(f"Shape: {df.shape}")

# Preprocessing function
@st.cache_data
def preprocess_data(df):
    # Handle missing values
    df_clean = df.fillna(df.median(numeric_only=True))

    # Identify numerical columns
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns

    # Check skewness and apply log transformation
    skewness = df_clean[numerical_cols].apply(lambda x: skew(x.dropna()))
    skewed_features = skewness[abs(skewness) > 0.5].index
    df_transformed = df_clean.copy()
    for col in skewed_features:
        if col != 'Life expectancy ' and (df_clean[col] > 0).all():
            df_transformed[col] = np.log1p(df_clean[col])

    # Handle outliers by capping
    for col in numerical_cols:
        if col != 'Life expectancy ':
            Q1 = df_transformed[col].quantile(0.25)
            Q3 = df_transformed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_transformed[col] = np.clip(df_transformed[col], lower_bound, upper_bound)

    # Feature scaling
    features = [col for col in numerical_cols if col != 'Life expectancy ']
    scaler = StandardScaler()
    df_scaled = df_transformed.copy()
    df_scaled[features] = scaler.fit_transform(df_transformed[features])

    return df_scaled, features, scaler, skewed_features

df_processed, features, scaler, skewed_features = preprocess_data(df)

# Split data
X = df_processed[features]
y = df_processed['Life expectancy ']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate models
@st.cache_resource
def load_model():
    import pickle
    with open('best_life_expectancy_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

best_model = load_model()

# Prediction section
st.header("Make a Prediction")
st.write("Enter the feature values below. Note: Some features may be log-transformed if skewed.")

inputs = {}
for feature in features:
    default_value = float(df[feature].median())  # Use median as default
    inputs[feature] = st.number_input(f"{feature}", value=default_value)

if st.button("Predict Life Expectancy"):
    # Preprocess input
    input_df = pd.DataFrame([inputs])

    # Apply log transformation to skewed features
    for col in skewed_features:
        if col in input_df.columns and input_df[col].iloc[0] > 0:
            input_df[col] = np.log1p(input_df[col])

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = best_model.predict(input_scaled)[0]
    st.success(f"Predicted Life Expectancy: {prediction:.2f} years")

# Visualization
st.header("Data Visualization")
option = st.selectbox("Choose a plot", ["Histogram of Life Expectancy", "Correlation Heatmap", "Scatter Plot"])

if option == "Histogram of Life Expectancy":
    fig, ax = plt.subplots()
    sns.histplot(df_processed['Life expectancy '], kde=True, ax=ax)
    ax.set_title("Distribution of Life Expectancy")
    st.pyplot(fig)
elif option == "Correlation Heatmap":
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df_processed.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
elif option == "Scatter Plot":
    x_axis = st.selectbox("X-axis", features)
    fig, ax = plt.subplots()
    sns.scatterplot(x=df_processed[x_axis], y=df_processed['Life expectancy '], ax=ax)
    ax.set_title(f"Life Expectancy vs {x_axis}")
    st.pyplot(fig)