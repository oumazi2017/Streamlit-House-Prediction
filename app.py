import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("🏠 House Price Prediction App")
st.markdown("""
This application predicts house prices based on various features using a Random Forest model.
Upload your data or use the provided dataset to get predictions.
""")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Home", "Model Training", "Predictions", "Data Exploration"])

# Load data
@st.cache_data
def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

# Preprocess data
def preprocess_data(df, is_train=True):
    df_copy = df.copy()
    
    # Store target if training data
    if is_train and 'SalePrice' in df_copy.columns:
        target = df_copy['SalePrice']
        df_copy = df_copy.drop('SalePrice', axis=1)
    else:
        target = None
    
    # Drop Id column
    if 'Id' in df_copy.columns:
        df_copy = df_copy.drop('Id', axis=1)
    
    # Select numeric and categorical columns
    numeric_features = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df_copy.select_dtypes(include=['object']).columns.tolist()
    
    # Handle missing values - Fixed to avoid FutureWarning
    for col in numeric_features:
        df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    
    for col in categorical_features:
        df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0] if not df_copy[col].mode().empty else 'None')
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df_copy[col] = le.fit_transform(df_copy[col].astype(str))
        label_encoders[col] = le
    
    return df_copy, target, label_encoders

# Train model
@st.cache_resource
def train_model(X, y):
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Validation metrics
    y_pred_val = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    r2 = r2_score(y_val, y_pred_val)
    
    return model, mae, rmse, r2

# Home Page
if page == "Home":
    st.header("Welcome to House Price Prediction App")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 About the Dataset")
        st.write("""
        This app uses the famous Ames Housing dataset to predict house prices.
        The dataset includes 79 features describing various aspects of residential homes.
        
        **Key Features:**
        - Overall Quality and Condition
        - Square Footage (Living Area, Lot Size)
        - Number of Rooms, Bathrooms, Bedrooms
        - Year Built and Remodeled
        - Garage, Basement, and Porch details
        - And many more!
        """)
    
    with col2:
        st.subheader("🤖 Model Information")
        st.write("""
        **Algorithm:** Random Forest Regressor
        
        **Why Random Forest?**
        - Handles non-linear relationships
        - Robust to outliers
        - Provides feature importance
        - Good performance out of the box
        
        The model is trained on 80% of the data and validated on the remaining 20%.
        """)
    
    st.markdown("---")
    st.info("👈 Use the sidebar to navigate through different sections of the app!")

# Model Training Page
elif page == "Model Training":
    st.header("🎯 Model Training & Performance")
    
    try:
        train_df, test_df = load_data()
        
        st.subheader("Dataset Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Training Samples", len(train_df))
            st.metric("Test Samples", len(test_df))
        
        with col2:
            st.metric("Total Features", len(train_df.columns) - 1)
            st.metric("Target Variable", "SalePrice")
        
        # Preprocess data
        with st.spinner("Preprocessing data..."):
            X, y, encoders = preprocess_data(train_df, is_train=True)
        
        st.success("✅ Data preprocessed successfully!")
        
        # Train model
        if st.button("Train Model", type="primary"):
            with st.spinner("Training model... This may take a minute."):
                model, mae, rmse, r2 = train_model(X, y)
                
                # Save model
                with open('house_price_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
                
                st.success("✅ Model trained successfully!")
                
                st.subheader("📈 Model Performance Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Mean Absolute Error", f"${mae:,.2f}")
                
                with col2:
                    st.metric("Root Mean Squared Error", f"${rmse:,.2f}")
                
                with col3:
                    st.metric("R² Score", f"{r2:.4f}")
                
                # Feature importance
                st.subheader("🔍 Top 10 Most Important Features")
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(10)
                
                st.bar_chart(feature_importance.set_index('feature'))
        
        # Load existing model if available
        if os.path.exists('house_price_model.pkl'):
            st.info("ℹ️ A trained model already exists. Click 'Train Model' to retrain.")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Predictions Page
elif page == "Predictions":
    st.header("🔮 Make Predictions")
    
    # Check if model exists
    if not os.path.exists('house_price_model.pkl'):
        st.warning("⚠️ No trained model found. Please train the model first in the 'Model Training' section.")
    else:
        try:
            # Load model
            with open('house_price_model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            train_df, test_df = load_data()
            
            st.subheader("Choose Prediction Method")
            pred_method = st.radio("Prediction Method", ["Predict on Test Set", "Custom Input"], label_visibility="visible")
            
            if pred_method == "Predict on Test Set":
                st.write("Generate predictions for the entire test dataset.")
                
                if st.button("Generate Predictions", type="primary"):
                    with st.spinner("Generating predictions..."):
                        X_test, _, _ = preprocess_data(test_df, is_train=False)
                        predictions = model.predict(X_test)
                        
                        # Create submission dataframe
                        submission = pd.DataFrame({
                            'Id': test_df['Id'],
                            'SalePrice': predictions
                        })
                        
                        st.success("✅ Predictions generated!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Total Predictions", len(predictions))
                            st.metric("Average Predicted Price", f"${predictions.mean():,.2f}")
                        
                        with col2:
                            st.metric("Min Predicted Price", f"${predictions.min():,.2f}")
                            st.metric("Max Predicted Price", f"${predictions.max():,.2f}")
                        
                        st.subheader("Preview Predictions")
                        st.dataframe(submission.head(20), use_container_width=True)
                        
                        # Download button
                        csv = submission.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions (CSV)",
                            data=csv,
                            file_name="house_price_predictions.csv",
                            mime="text/csv"
                        )
            
            else:  # Custom Input
                st.write("Enter custom house features to get a price prediction.")
                
                # Create input form with most important features
                st.subheader("Enter House Details")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 7)
                    gr_liv_area = st.number_input("Ground Living Area (sq ft)", 500, 5000, 1500)
                    garage_cars = st.number_input("Garage Capacity (cars)", 0, 4, 2)
                    total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 3000, 1000)
                
                with col2:
                    year_built = st.number_input("Year Built", 1900, 2025, 2000)
                    full_bath = st.number_input("Full Bathrooms", 0, 4, 2)
                    fireplaces = st.number_input("Fireplaces", 0, 3, 1)
                    lot_area = st.number_input("Lot Area (sq ft)", 1000, 50000, 8000)
                
                with col3:
                    bedroom_abv_gr = st.number_input("Bedrooms Above Grade", 0, 8, 3)
                    kitchen_abv_gr = st.number_input("Kitchens Above Grade", 0, 3, 1)
                    tot_rms_abv_grd = st.number_input("Total Rooms Above Grade", 2, 15, 7)
                    garage_area = st.number_input("Garage Area (sq ft)", 0, 1500, 500)
                
                if st.button("Predict Price", type="primary"):
                    # Create a sample with all features (using median for other features)
                    X_train, _, _ = preprocess_data(train_df, is_train=True)
                    sample = pd.DataFrame([X_train.median()], columns=X_train.columns)
                    
                    # Update with user inputs (mapping to actual column names)
                    feature_map = {
                        'OverallQual': overall_qual,
                        'GrLivArea': gr_liv_area,
                        'GarageCars': garage_cars,
                        'TotalBsmtSF': total_bsmt_sf,
                        'YearBuilt': year_built,
                        'FullBath': full_bath,
                        'Fireplaces': fireplaces,
                        'LotArea': lot_area,
                        'BedroomAbvGr': bedroom_abv_gr,
                        'KitchenAbvGr': kitchen_abv_gr,
                        'TotRmsAbvGrd': tot_rms_abv_grd,
                        'GarageArea': garage_area
                    }
                    
                    for col, val in feature_map.items():
                        if col in sample.columns:
                            sample[col] = val
                    
                    prediction = model.predict(sample)[0]
                    
                    st.success("✅ Prediction Complete!")
                    st.markdown(f"### Predicted House Price: **${prediction:,.2f}**")
                    
                    # Price range
                    st.info(f"💡 Confidence Range: ${prediction*0.9:,.2f} - ${prediction*1.1:,.2f}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Data Exploration Page
elif page == "Data Exploration":
    st.header("📊 Data Exploration")
    
    try:
        train_df, test_df = load_data()
        
        dataset_choice = st.selectbox("Choose Dataset", ["Training Data", "Test Data"])
        df = train_df if dataset_choice == "Training Data" else test_df
        
        st.subheader(f"{dataset_choice} Overview")
        
        tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📈 Statistics", "📊 Missing Values"])
        
        with tab1:
            st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head(20), use_container_width=True)
        
        with tab2:
            st.write("**Numerical Features Statistics**")
            st.dataframe(df.describe(), use_container_width=True)
            
            if 'SalePrice' in df.columns:
                st.subheader("🏷️ Target Variable: SalePrice")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Mean Price", f"${df['SalePrice'].mean():,.2f}")
                
                with col2:
                    st.metric("Median Price", f"${df['SalePrice'].median():,.2f}")
                
                with col3:
                    st.metric("Std Dev", f"${df['SalePrice'].std():,.2f}")
        
        with tab3:
            missing = df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            
            if len(missing) > 0:
                st.write(f"**Features with Missing Values: {len(missing)}**")
                missing_df = pd.DataFrame({
                    'Feature': missing.index,
                    'Missing Count': missing.values,
                    'Percentage': (missing.values / len(df) * 100).round(2)
                })
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing values in this dataset!")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit | House Price Prediction App</p>
</div>
""", unsafe_allow_html=True)
