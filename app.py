import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Life Expectancy Prediction - End-to-End ML",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-result {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ca02c;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏥 Life Expectancy Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">End-to-End Machine Learning Project</p>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a section:", [
    "📊 Data Overview",
    "🔍 Exploratory Data Analysis",
    "⚙️ Model Training & Evaluation",
    "🎯 Make Predictions",
    "📈 Model Performance",
    "ℹ️ About"
])

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Life Expectancy Data.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Error: 'Life Expectancy Data.csv' file not found. Please ensure the dataset file is in the same directory as the app.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

df = load_data()

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
    outlier_bounds = {}
    for col in numerical_cols:
        if col != 'Life expectancy ':
            Q1 = df_transformed[col].quantile(0.25)
            Q3 = df_transformed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_transformed[col] = np.clip(df_transformed[col], lower_bound, upper_bound)
            outlier_bounds[col] = (lower_bound, upper_bound)

    # Feature scaling
    features = [col for col in numerical_cols if col != 'Life expectancy ']
    scaler = StandardScaler()
    df_scaled = df_transformed.copy()
    df_scaled[features] = scaler.fit_transform(df_transformed[features])

    return df_scaled, features, scaler, skewed_features, outlier_bounds, df_clean

df_processed, features, scaler, skewed_features, outlier_bounds, df_clean = preprocess_data(df)

# Split data
X = df_processed[features]
y = df_processed['Life expectancy ']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load or train models
@st.cache_resource
def load_or_train_models():
    try:
        with open('best_life_expectancy_model.pkl', 'rb') as f:
            best_model = pickle.load(f)
        return best_model, True
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=0.1),
            'Lasso Regression': Lasso(alpha=0.1),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42),
            'SVM': SVR(kernel='rbf'),
            'XGBoost': XGBRegressor(random_state=42)
        }

        # Train and evaluate models
        results = {}
        trained_models = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
            results[name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'CV_R2': cv_r2}
            trained_models[name] = model

        # Best model
        best_model_name = max(results, key=lambda x: results[x]['R2'])
        best_model = trained_models[best_model_name]

        # Save best model
        with open('best_life_expectancy_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)

        return best_model, False

best_model, model_loaded = load_or_train_models()

# Page 1: Data Overview
if page == "📊 Data Overview":
    st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{df.shape[0]:,}")
    with col2:
        st.metric("Features", f"{df.shape[1]}")
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum()}")

    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Data Summary")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Data Types & Missing Values")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Data Types:**")
        dtype_df = pd.DataFrame(df.dtypes.astype(str), columns=['Data Type'])
        st.dataframe(dtype_df, use_container_width=True)

    with col2:
        st.write("**Missing Values:**")
        missing_df = pd.DataFrame(df.isnull().sum(), columns=['Missing Count'])
        missing_df['Missing %'] = (missing_df['Missing Count'] / len(df)) * 100
        st.dataframe(missing_df, use_container_width=True)

# Page 2: Exploratory Data Analysis
elif page == "🔍 Exploratory Data Analysis":
    st.markdown('<h2 class="section-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔗 Correlations", "📊 Scatter Plots", "🔍 Outliers"])

    with tab1:
        st.subheader("Feature Distributions")
        selected_feature = st.selectbox("Select feature to visualize:", features + ['Life expectancy '])
        fig, ax = plt.subplots(figsize=(10, 6))
        if selected_feature == 'Life expectancy ':
            sns.histplot(df_clean[selected_feature], kde=True, ax=ax, color='skyblue')
        else:
            sns.histplot(df_clean[selected_feature], kde=True, ax=ax, color='lightcoral')
        ax.set_title(f'Distribution of {selected_feature}')
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        # Skewness information
        if selected_feature != 'Life expectancy ':
            skewness_val = skew(df_clean[selected_feature].dropna())
            st.info(f"Skewness: {skewness_val:.3f} ({'Right-skewed' if skewness_val > 0.5 else 'Left-skewed' if skewness_val < -0.5 else 'Approximately symmetric'})")

    with tab2:
        st.subheader("Correlation Analysis")
        # Only use numerical columns for correlation
        numerical_df = df_processed.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = numerical_df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=.5, ax=ax)
        ax.set_title('Correlation Heatmap (Upper Triangle)')
        st.pyplot(fig)

        st.subheader("Top Correlations with Life Expectancy")
        if 'Life expectancy ' in corr_matrix.columns:
            life_exp_corr = corr_matrix['Life expectancy '].sort_values(ascending=False)
            st.dataframe(life_exp_corr[1:11], use_container_width=True)  # Top 10 correlations
        else:
            st.warning("Life expectancy column not found in correlation matrix.")

    with tab3:
        st.subheader("Scatter Plot Analysis")
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("X-axis feature:", features, key='scatter_x')
        with col2:
            y_feature = st.selectbox("Y-axis feature:", ['Life expectancy '] + features, key='scatter_y')

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=df_processed[x_feature], y=df_processed[y_feature], ax=ax, alpha=0.6)
        ax.set_title(f'{y_feature} vs {x_feature}')
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)

        # Add regression line if target is life expectancy
        if y_feature == 'Life expectancy ':
            sns.regplot(x=df_processed[x_feature], y=df_processed[y_feature], ax=ax,
                       scatter=False, color='red', line_kws={'linewidth': 2})

        st.pyplot(fig)

    with tab4:
        st.subheader("Outlier Analysis")
        selected_outlier_feature = st.selectbox("Select feature for outlier analysis:", features)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Box plot
        sns.boxplot(y=df_clean[selected_outlier_feature], ax=ax1, color='lightblue')
        ax1.set_title(f'Box Plot of {selected_outlier_feature}')
        ax1.set_ylabel(selected_outlier_feature)

        # Before and after outlier treatment
        original_data = df_clean[selected_outlier_feature]
        processed_data = df_processed[selected_outlier_feature]

        ax2.hist(original_data, alpha=0.5, label='Original', bins=30, color='red')
        ax2.hist(processed_data, alpha=0.5, label='After Outlier Treatment', bins=30, color='blue')
        ax2.set_title(f'Histogram: {selected_outlier_feature}')
        ax2.set_xlabel(selected_outlier_feature)
        ax2.set_ylabel('Frequency')
        ax2.legend()

        st.pyplot(fig)

        # Outlier statistics
        Q1 = original_data.quantile(0.25)
        Q3 = original_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((original_data < lower_bound) | (original_data > upper_bound)).sum()

        st.write(f"**Outlier Statistics for {selected_outlier_feature}:**")
        st.write(f"- Q1 (25th percentile): {Q1:.2f}")
        st.write(f"- Q3 (75th percentile): {Q3:.2f}")
        st.write(f"- IQR: {IQR:.2f}")
        st.write(f"- Lower bound: {lower_bound:.2f}")
        st.write(f"- Upper bound: {upper_bound:.2f}")
        st.write(f"- Number of outliers: {outliers}")

# Page 3: Model Training & Evaluation
elif page == "⚙️ Model Training & Evaluation":
    st.markdown('<h2 class="section-header">Model Training & Evaluation</h2>', unsafe_allow_html=True)

    if st.button("🔄 Retrain All Models", type="primary"):
        with st.spinner("Training models... This may take a few minutes."):
            # Define models with hyperparameters
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=0.1),
                'Lasso Regression': Lasso(alpha=0.1),
                'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
                'SVM': SVR(kernel='rbf', C=1.0),
                'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
            }

            # Train and evaluate models
            results = {}
            trained_models = {}

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, (name, model) in enumerate(models.items()):
                status_text.text(f"Training {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2').mean()

                results[name] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R²': r2,
                    'CV R²': cv_r2
                }
                trained_models[name] = model

                progress_bar.progress((i + 1) / len(models))

            status_text.text("Training completed!")

            # Save results to session state
            st.session_state['model_results'] = results
            st.session_state['trained_models'] = trained_models

            # Select best model
            best_model_name = max(results, key=lambda x: results[x]['R²'])
            best_model = trained_models[best_model_name]

            # Save best model
            with open('best_life_expectancy_model.pkl', 'wb') as f:
                pickle.dump(best_model, f)

            st.success(f"✅ Best model ({best_model_name}) saved successfully!")
            st.session_state['best_model_name'] = best_model_name

    # Display results if available
    if 'model_results' in st.session_state:
        st.subheader("📊 Model Performance Comparison")

        results_df = pd.DataFrame(st.session_state['model_results']).T
        results_df = results_df.round(4)

        # Highlight best model
        best_model_name = st.session_state.get('best_model_name', max(st.session_state['model_results'], key=lambda x: st.session_state['model_results'][x]['R²']))

        def highlight_best(s):
            is_best = s.name == best_model_name
            return ['background-color: #e8f5e8' if is_best else '' for _ in s]

        # Apply styling and display
        styled_df = results_df.style.apply(highlight_best, axis=0)
        st.dataframe(styled_df, use_container_width=True)
        # Model comparison visualization
        st.subheader("📈 Model Comparison Visualization")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # R² comparison
        models_names = list(results_df.index)
        r2_scores = results_df['R²']
        bars1 = ax1.barh(models_names, r2_scores, color='skyblue')
        ax1.set_title('R² Scores Comparison')
        ax1.set_xlabel('R² Score')
        ax1.barh([best_model_name], [results_df.loc[best_model_name, 'R²']], color='green')

        # RMSE comparison
        rmse_scores = results_df['RMSE']
        bars2 = ax2.barh(models_names, rmse_scores, color='lightcoral')
        ax2.set_title('RMSE Comparison (Lower is Better)')
        ax2.set_xlabel('RMSE')
        ax2.barh([best_model_name], [results_df.loc[best_model_name, 'RMSE']], color='red')

        # MAE comparison
        mae_scores = results_df['MAE']
        bars3 = ax3.barh(models_names, mae_scores, color='lightgreen')
        ax3.set_title('MAE Comparison (Lower is Better)')
        ax3.set_xlabel('MAE')
        ax3.barh([best_model_name], [results_df.loc[best_model_name, 'MAE']], color='darkgreen')

        # CV R² comparison
        cv_scores = results_df['CV R²']
        bars4 = ax4.barh(models_names, cv_scores, color='gold')
        ax4.set_title('Cross-Validation R² Scores')
        ax4.set_xlabel('CV R² Score')
        ax4.barh([best_model_name], [results_df.loc[best_model_name, 'CV R²']], color='orange')

        plt.tight_layout()
        st.pyplot(fig)

# Page 4: Make Predictions
elif page == "🎯 Make Predictions":
    st.markdown('<h2 class="section-header">Make Life Expectancy Predictions</h2>', unsafe_allow_html=True)

    if not model_loaded:
        st.warning("⚠️ No trained model found. Please go to 'Model Training & Evaluation' to train models first.")
    else:
        st.info("💡 Enter the feature values below to predict life expectancy. Default values are set to median values from the dataset.")

        # Get min/max values for validation
        feature_ranges = {}
        for feature in features:
            feature_ranges[feature] = (df[feature].min(), df[feature].max())

        # Create input form
        with st.form("prediction_form"):
            st.subheader("📝 Input Features")

            # Organize inputs in columns for better layout
            col1, col2 = st.columns(2)

            inputs = {}
            feature_descriptions = {
                'Year': 'Year of observation',
                'Adult Mortality': 'Adult mortality rate (per 1000 population)',
                'infant deaths': 'Number of infant deaths per 1000 population',
                'Alcohol': 'Alcohol consumption (litres of pure alcohol per capita)',
                'percentage expenditure': 'Expenditure on health as % of GDP',
                'Hepatitis B': 'Hepatitis B immunization coverage (%)',
                'Measles ': 'Number of reported measles cases per 1000 population',
                ' BMI ': 'Average BMI of population',
                'under-five deaths ': 'Number of under-five deaths per 1000 population',
                'Polio': 'Polio immunization coverage (%)',
                'Total expenditure': 'Government expenditure on health as % of total expenditure',
                'Diphtheria ': 'Diphtheria immunization coverage (%)',
                ' HIV/AIDS': 'HIV/AIDS deaths per 1000 live births',
                'GDP': 'GDP per capita (USD)',
                'Population': 'Population of the country',
                ' thinness  1-19 years': 'Prevalence of thinness among children aged 1-19 years (%)',
                ' thinness 5-9 years': 'Prevalence of thinness among children aged 5-9 years (%)',
                'Income composition of resources': 'Income composition of resources (0-1 scale)',
                'Schooling': 'Average years of schooling'
            }

            for i, feature in enumerate(features):
                current_col = col1 if i % 2 == 0 else col2
                with current_col:
                    min_val, max_val = feature_ranges[feature]
                    default_value = float(df[feature].median())
                    description = feature_descriptions.get(feature, feature)

                    inputs[feature] = st.number_input(
                        f"{feature}",
                        value=default_value,
                        min_value=float(min_val),
                        max_value=float(max_val),
                        help=f"{description}\nRange: {min_val:.2f} - {max_val:.2f}",
                        key=f"input_{feature}"
                    )

            # Submit button
            submitted = st.form_submit_button("🔮 Predict Life Expectancy", type="primary")

            if submitted:
                with st.spinner("Processing prediction..."):
                    # Preprocess input the same way as training data
                    input_df = pd.DataFrame([inputs])

                    # Apply log transformation to skewed features
                    for col in skewed_features:
                        if col in input_df.columns and input_df[col].iloc[0] > 0:
                            input_df[col] = np.log1p(input_df[col])

                    # Apply outlier capping using the same bounds as training data
                    for col in features:
                        if col in input_df.columns and col in outlier_bounds:
                            lower_bound, upper_bound = outlier_bounds[col]
                            input_df[col] = np.clip(input_df[col], lower_bound, upper_bound)

                    # Ensure correct column order and create a copy
                    input_processed = input_df[features].copy()

                    # Scale using the same scaler
                    input_scaled = scaler.transform(input_processed)

                    # Create DataFrame with correct feature names for prediction
                    input_scaled_df = pd.DataFrame(input_scaled, columns=features)

                    # Predict
                    prediction = best_model.predict(input_scaled_df)[0]

                    # Display result
                    st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                    st.markdown(f"🎯 **Predicted Life Expectancy: {prediction:.2f} years**")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Additional insights
                    st.subheader("📊 Prediction Insights")

                    # Compare with dataset statistics
                    mean_life_exp = df['Life expectancy '].mean()
                    median_life_exp = df['Life expectancy '].median()
                    min_life_exp = df['Life expectancy '].min()
                    max_life_exp = df['Life expectancy '].max()

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Dataset Mean", f"{mean_life_exp:.1f} years")
                    with col2:
                        st.metric("Dataset Median", f"{median_life_exp:.1f} years")
                    with col3:
                        st.metric("Dataset Min", f"{min_life_exp:.1f} years")
                    with col4:
                        st.metric("Dataset Max", f"{max_life_exp:.1f} years")

                    # Prediction interpretation
                    if prediction > mean_life_exp + 10:
                        st.success("🌟 This prediction is above average life expectancy!")
                    elif prediction < mean_life_exp - 10:
                        st.warning("⚠️ This prediction is below average life expectancy.")
                    else:
                        st.info("📈 This prediction is around average life expectancy.")

# Page 5: Model Performance
elif page == "📈 Model Performance":
    st.markdown('<h2 class="section-header">Model Performance Analysis</h2>', unsafe_allow_html=True)

    # Try to load the saved model
    try:
        with open('best_life_expectancy_model.pkl', 'rb') as f:
            best_model = pickle.load(f)
        model_available = True
        best_model_name = "Best Saved Model"
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        model_available = False
        best_model = None
        best_model_name = "Unknown"

    if not model_available:
        st.warning("⚠️ No trained model found. Please train models first in the 'Model Training & Evaluation' section.")
    else:
        # Make predictions with the loaded model
        y_pred = best_model.predict(X_test)

        # Performance metrics
        st.subheader("📊 Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        with col1:
            st.metric("Mean Squared Error", f"{mse:.4f}")
        with col2:
            st.metric("Root Mean Squared Error", f"{rmse:.4f}")
        with col3:
            st.metric("Mean Absolute Error", f"{mae:.4f}")
        with col4:
            st.metric("R² Score", f"{r2:.4f}")

        # Actual vs Predicted plot
        st.subheader("📈 Actual vs Predicted Values")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolors='black')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
        ax.set_xlabel('Actual Life Expectancy')
        ax.set_ylabel('Predicted Life Expectancy')
        ax.set_title('Actual vs Predicted Life Expectancy')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Residual plot
        st.subheader("📉 Residual Analysis")
        residuals = y_test - y_pred

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.6, color='green', edgecolors='black')
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Life Expectancy')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.grid(True, alpha=0.3)

        # Residual distribution
        ax2.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residual Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residual Distribution')
        ax2.grid(True, alpha=0.3)

        st.pyplot(fig)

        # Feature importance (if available)
        if hasattr(best_model, 'feature_importances_'):
            st.subheader("🔍 Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)

            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
            ax.set_xlabel('Importance')
            ax.set_title(f'Top 10 Feature Importances ({best_model_name})')
            ax.grid(True, alpha=0.3)

            # Color the bars
            for bar in bars:
                bar.set_color('skyblue')

            st.pyplot(fig)

            st.dataframe(feature_importance.head(10), use_container_width=True)

# Page 6: About
elif page == "ℹ️ About":
    st.markdown('<h2 class="section-header">About This Project</h2>', unsafe_allow_html=True)

    st.write("""
    ## 🏥 Life Expectancy Prediction - End-to-End Machine Learning Project

    This comprehensive Streamlit application demonstrates a complete machine learning workflow for predicting life expectancy based on various health, economic, and social factors.

    ### 🎯 Project Objectives
    - **Data Exploration**: Comprehensive analysis of the Life Expectancy dataset
    - **Preprocessing**: Handle missing values, outliers, feature scaling, and transformations
    - **Model Training**: Compare multiple machine learning algorithms
    - **Model Evaluation**: Rigorous performance assessment and validation
    - **Prediction Interface**: User-friendly prediction tool
    - **Visualization**: Interactive charts and performance metrics

    ### 📊 Dataset Information
    - **Source**: World Health Organization (WHO) Life Expectancy Data
    - **Features**: 19 health, economic, and social indicators
    - **Target**: Life expectancy in years
    - **Records**: 2,938 observations from 193 countries (2000-2015)

    ### 🤖 Machine Learning Models
    - Linear Regression
    - Ridge & Lasso Regression
    - Decision Tree
    - Random Forest
    - Support Vector Machine (SVM)
    - XGBoost

    ### 🛠️ Technologies Used
    - **Streamlit**: Interactive web application
    - **Scikit-learn**: Machine learning algorithms
    - **Pandas & NumPy**: Data manipulation
    - **Matplotlib & Seaborn**: Data visualization
    - **XGBoost**: Gradient boosting
    - **SciPy**: Statistical functions

    ### 📈 Key Features
    - **Multi-page Navigation**: Organized sections for different ML phases
    - **Interactive Visualizations**: Dynamic charts and plots
    - **Model Comparison**: Side-by-side performance metrics
    - **Real-time Predictions**: Instant life expectancy predictions
    - **Feature Importance**: Understanding model decisions
    - **Cross-validation**: Robust model evaluation

    ### 🔬 Methodology
    1. **Data Preprocessing**: Missing value imputation, outlier treatment, feature scaling
    2. **Feature Engineering**: Log transformations for skewed features
    3. **Model Selection**: Hyperparameter tuning and cross-validation
    4. **Performance Evaluation**: Multiple metrics (R², RMSE, MAE)
    5. **Model Deployment**: Pickle serialization for production use

    ### 📝 Usage Instructions
    1. **Data Overview**: Explore the dataset structure and summary statistics
    2. **EDA**: Analyze distributions, correlations, and relationships
    3. **Model Training**: Train and compare different ML algorithms
    4. **Predictions**: Input feature values to get life expectancy predictions
    5. **Performance**: Analyze model performance and feature importance

    ---
    **Built with ❤️ for educational and demonstration purposes**
    """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>© 2024 Life Expectancy Prediction Project | End-to-End Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)