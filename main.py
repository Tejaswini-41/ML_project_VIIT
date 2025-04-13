# testing

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Traffic Accident Prediction",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("🚦 Traffic Accident Prediction System")
st.markdown("""
This application predicts the likelihood of a traffic accident based on various road, vehicle, and driver conditions.
Enter the relevant information below and click 'Predict' to get the accident risk assessment.
""")

# Define feature categories (from feature engineering notebook)
nominal_features = ['Weather', 'Road_Type', 'Road_Condition', 'Vehicle_Type', 'Road_Light_Condition']
ordinal_features = ['Time_of_Day']
numerical_features = ['Traffic_Density', 'Speed_Limit', 'Number_of_Vehicles', 'Driver_Alcohol', 'Driver_Age']

# Define ordinal feature orders
time_categories = ['Morning', 'Afternoon', 'Evening', 'Night']

# Load the model
@st.cache_resource
def load_model():
    try:
        # Load the saved model (adjust filename as needed)
        model_filename = "accident_pred_logistic_regression.pkl"  # Replace with your actual model filename
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Create and fit preprocessor (this would normally be saved with the model)
@st.cache_resource
def get_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('ord', OrdinalEncoder(categories=[time_categories]), ordinal_features),
            ('nom', OneHotEncoder(drop='first'), nominal_features)
        ])
    
    # Create a sample DataFrame to fit the preprocessor
    sample_data = pd.DataFrame({
        'Weather': ['Clear', 'Rainy', 'Foggy', 'Snowy', 'Stormy'],
        'Road_Type': ['City Road', 'Highway', 'Rural Road', 'Mountain Road', 'City Road'],
        'Road_Condition': ['Dry', 'Wet', 'Icy', 'Under Construction', 'Dry'],
        'Vehicle_Type': ['Car', 'Truck', 'Motorcycle', 'Bus', 'Car'],
        'Road_Light_Condition': ['Daylight', 'Artificial Light', 'No Light', 'Daylight', 'Artificial Light'],
        'Time_of_Day': ['Morning', 'Afternoon', 'Evening', 'Night', 'Morning'],
        'Traffic_Density': [50, 70, 30, 20, 60],
        'Speed_Limit': [50, 80, 60, 40, 35],
        'Number_of_Vehicles': [10, 20, 5, 15, 8],
        'Driver_Alcohol': [0, 1, 0, 0, 1],
        'Driver_Age': [25, 45, 35, 60, 22]
    })
    
    # Fit the preprocessor
    preprocessor.fit(sample_data)
    return preprocessor

# Function to get feature names after preprocessing
def get_feature_names(preprocessor):
    feature_names = []
    
    # Numerical feature names
    feature_names.extend([f'{col}_scaled' for col in numerical_features])
    
    # Ordinal feature name
    feature_names.append('Time_of_Day_encoded')
    
    # Get one-hot encoded feature names
    ohe = preprocessor.named_transformers_['nom']
    for feature, categories in zip(nominal_features, ohe.categories_):
        # Skip the first category because we used drop='first'
        for category in categories[1:]:
            feature_names.append(f'{feature}_{category}')
    
    return feature_names

# Function to make prediction
def predict_accident(features_df, model, preprocessor):
    # First apply the standard preprocessing
    X_transformed = preprocessor.transform(features_df)
    X_df = pd.DataFrame(X_transformed, columns=get_feature_names(preprocessor))
    
    # Now add the additional engineered features
    
    # 1. Create interaction features
    X_df['Alcohol_Night'] = X_df['Driver_Alcohol_scaled'] * (X_df['Time_of_Day_encoded'] >= 3).astype(int)
    X_df['Speed_Wet_Road'] = X_df['Speed_Limit_scaled'] * ((features_df['Road_Condition'] == 'Wet').values | 
                                              (features_df['Road_Condition'] == 'Icy').values).astype(int)
    X_df['Motorcycle_Speed'] = X_df['Speed_Limit_scaled'] * (features_df['Vehicle_Type'] == 'Motorcycle').values.astype(int)
    X_df['Low_Visibility_Speed'] = X_df['Speed_Limit_scaled'] * (features_df['Road_Light_Condition'] == 'No Light').values.astype(int)
    
    # 2. Create Alcohol Risk Factor
    X_df['Alcohol_Risk_Factor_scaled'] = X_df['Driver_Alcohol_scaled'] * 3.0
    
    # 3. Create age groups
    age_scaled = (features_df['Driver_Age'].values[0] - 16) / (90 - 16)  # Normalize to 0-1 scale
    
    # Add age group dummies
    X_df['Age_Adult'] = 1 if 0.25 <= age_scaled < 0.5 else 0
    X_df['Age_Middle'] = 1 if 0.5 <= age_scaled < 0.75 else 0
    X_df['Age_Senior'] = 1 if age_scaled >= 0.75 else 0
    
    # Make prediction using the complete feature set
    prediction = model.predict(X_df.values)[0]
    probability = model.predict_proba(X_df.values)[0][1]
    
    return prediction, probability

# Function to get safety recommendations
def get_recommendations(features):
    recommendations = []
    
    # Weather-based recommendations
    if features['Weather'] in ['Rainy', 'Stormy', 'Foggy', 'Snowy']:
        recommendations.append(f"⚠️ Drive carefully in {features['Weather']} conditions. Reduce speed and increase following distance.")
    
    # Speed-based recommendations
    if features['Speed_Limit'] > 70:
        recommendations.append("⚠️ High speed area. Maintain proper speed limits and stay alert.")
    
    # Road condition recommendations
    if features['Road_Condition'] in ['Wet', 'Icy', 'Under Construction']:
        recommendations.append(f"⚠️ Hazardous road condition: {features['Road_Condition']}. Adjust driving accordingly.")
    
    # Time of day recommendations
    if features['Time_of_Day'] in ['Night', 'Evening']:
        recommendations.append("⚠️ Limited visibility during evening/night hours. Use headlights and stay vigilant.")
    
    # Traffic density recommendations
    if features['Traffic_Density'] > 70:
        recommendations.append("⚠️ High traffic density. Maintain safe distance between vehicles.")
    
    # Alcohol recommendations
    if features['Driver_Alcohol'] == 1:
        recommendations.append("🚫 NEVER drive under the influence of alcohol. It significantly increases accident risk.")
    
    # Vehicle type recommendations
    if features['Vehicle_Type'] == 'Motorcycle':
        recommendations.append("⚠️ Motorcyclists should wear proper safety gear and remain highly visible to other drivers.")
    elif features['Vehicle_Type'] == 'Truck':
        recommendations.append("⚠️ Trucks require longer stopping distances. Maintain extra space around large vehicles.")
    
    # Add general recommendations
    recommendations.append("✅ Always wear seatbelts and follow traffic rules.")
    recommendations.append("✅ Stay focused and avoid distractions while driving.")
    
    return recommendations

# Create the main form
st.header("Input Traffic Conditions")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Road Conditions")
        weather = st.selectbox("Weather", options=['Clear', 'Rainy', 'Foggy', 'Snowy', 'Stormy'])
        road_type = st.selectbox("Road Type", options=['City Road', 'Highway', 'Rural Road', 'Mountain Road'])
        road_condition = st.selectbox("Road Condition", options=['Dry', 'Wet', 'Icy', 'Under Construction'])
        road_light = st.selectbox("Road Light Condition", options=['Daylight', 'Artificial Light', 'No Light'])
        time_of_day = st.selectbox("Time of Day", options=['Morning', 'Afternoon', 'Evening', 'Night'])
        speed_limit = st.slider("Speed Limit (km/h)", 20, 120, 60)
        
    with col2:
        st.subheader("Vehicle & Driver Information")
        vehicle_type = st.selectbox("Vehicle Type", options=['Car', 'Truck', 'Motorcycle', 'Bus'])
        traffic_density = st.slider("Traffic Density (%)", 0, 100, 50)
        num_vehicles = st.slider("Number of Vehicles", 1, 50, 10)
        driver_age = st.slider("Driver Age", 16, 90, 35)
        alcohol = st.radio("Driver Alcohol Consumption", options=["No", "Yes"])
        
        # Convert alcohol to binary
        driver_alcohol = 1 if alcohol == "Yes" else 0
    
    predict_button = st.form_submit_button("Predict Accident Risk")

# Try to load the model and preprocessor
try:
    model = load_model()
    preprocessor = get_preprocessor()
    feature_names = get_feature_names(preprocessor)
    
    # Run prediction if form is submitted
    if predict_button:
        # Create input features dataframe
        input_features = pd.DataFrame({
            'Weather': [weather],
            'Road_Type': [road_type],
            'Road_Condition': [road_condition],
            'Vehicle_Type': [vehicle_type],
            'Road_Light_Condition': [road_light],
            'Time_of_Day': [time_of_day],
            'Traffic_Density': [traffic_density],
            'Speed_Limit': [speed_limit],
            'Number_of_Vehicles': [num_vehicles],
            'Driver_Age': [driver_age],
            'Driver_Alcohol': [driver_alcohol]
        })
        
        # Make prediction
        prediction, probability = predict_accident(input_features, model, preprocessor)
        
        # Display results
        st.header("Prediction Results")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Adjust threshold from 0.5 to 0.25 for more reasonable risk assessment
            if probability > 0.25:  # Lower threshold to be more conservative
                st.error("⚠️ High Risk of Accident")
                risk_level = "High"
            else:
                st.success("✅ Low Risk of Accident")
                risk_level = "Low"
            
            st.metric("Accident Probability", f"{probability:.2%}")
        
        with col2:
            # Risk gauge visualization
            fig, ax = plt.subplots(figsize=(4, 0.8))
            ax.barh(0, probability, color='red', height=0.2)
            ax.barh(0, 1, color='lightgrey', height=0.2)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
            plt.tight_layout()
            st.pyplot(fig)
        
        # Display recommendations
        st.header("Safety Recommendations")
        recommendations = get_recommendations({
            'Weather': weather,
            'Road_Type': road_type,
            'Road_Condition': road_condition,
            'Vehicle_Type': vehicle_type,
            'Road_Light_Condition': road_light,
            'Time_of_Day': time_of_day,
            'Traffic_Density': traffic_density,
            'Speed_Limit': speed_limit,
            'Number_of_Vehicles': num_vehicles,
            'Driver_Age': driver_age,
            'Driver_Alcohol': driver_alcohol
        })
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        # If the model has feature importances, display them
        if hasattr(model, 'feature_importances_'):
            st.header("Risk Factor Analysis")
            
            # Get feature importances
            importances = model.feature_importances_
            
            # Create and display feature importance plot
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
            ax.set_title('Top 10 Features Contributing to Accident Risk')
            plt.tight_layout()
            st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.warning("Note: This application requires the trained model file to make predictions.")
    st.info("In a production environment, make sure the model file is properly saved and accessible.")

# Sidebar information
st.sidebar.header("About")
st.sidebar.info("""
This Traffic Accident Prediction System analyzes various factors to predict accident likelihood.

The model was trained on traffic data considering factors like:
- Weather conditions
- Road characteristics
- Driver attributes
- Vehicle information
- Traffic conditions
""")

st.sidebar.header("Safe Driving Tips")
st.sidebar.markdown("""
- 🛑 Follow speed limits
- 🚫 Never drive impaired 
- 🌧️ Adjust to weather conditions
- 👀 Avoid distractions
- 🚗 Maintain your vehicle
- 🛣️ Be cautious on unfamiliar roads
- 💺 Always wear seatbelts
""")