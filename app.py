import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder


# Set page title and configuration
st.set_page_config(page_title="Traffic Accident Prediction", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    with open('accident_pred_random_forest.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Header
st.title("🚗 Traffic Accident Prediction System")
st.markdown("### Enter the traffic conditions to predict accident probability")

# Create columns for better layout
col1, col2 = st.columns(2)

# Feature inputs
with col1:
    st.subheader("Road & Weather Conditions")
    
    weather = st.selectbox(
        "Weather",
        options=["Clear", "Rainy", "Foggy", "Snowy", "Stormy"],
        index=0
    )
    
    road_type = st.selectbox(
        "Road Type",
        options=["City Road", "Highway", "Rural Road", "Mountain Road"],
        index=0
    )
    
    road_condition = st.selectbox(
        "Road Condition",
        options=["Dry", "Wet", "Icy", "Under Construction"],
        index=0
    )
    
    road_light = st.selectbox(
        "Road Light Condition",
        options=["Daylight", "Artificial Light", "No Light"],
        index=0
    )
    
    time_of_day = st.selectbox(
        "Time of Day",
        options=["Morning", "Afternoon", "Evening", "Night"],
        index=0
    )

with col2:
    st.subheader("Vehicle & Driver Information")
    
    traffic_density = st.slider(
        "Traffic Density (scale 0-3)",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1
    )
    
    speed_limit = st.number_input(
        "Speed Limit (km/h)",
        min_value=20,
        max_value=200,
        value=60
    )
    
    num_vehicles = st.number_input(
        "Number of Vehicles",
        min_value=1,
        max_value=20,
        value=3
    )
    
    vehicle_type = st.selectbox(
        "Vehicle Type",
        options=["Car", "Truck", "Bus", "Motorcycle"],
        index=0
    )
    
    driver_age = st.slider(
        "Driver Age",
        min_value=16,
        max_value=90,
        value=35
    )
    
    driver_alcohol = st.radio(
        "Driver Alcohol Consumption",
        options=["No", "Yes"],
        index=0,
        horizontal=True
    )
    driver_alcohol_val = 1 if driver_alcohol == "Yes" else 0

# Predict button
predict_button = st.button("Predict Accident Probability", type="primary", use_container_width=True)

# Function to preprocess inputs
def preprocess_input(input_data):
    # Create a DataFrame from input
    df = pd.DataFrame(input_data, index=[0])
    
    # Define feature types
    nominal_features = ['Weather', 'Road_Type', 'Road_Condition', 'Vehicle_Type', 'Road_Light_Condition']
    ordinal_features = ['Time_of_Day']
    numerical_features = ['Traffic_Density', 'Speed_Limit', 'Number_of_Vehicles', 
                         'Driver_Alcohol', 'Driver_Age']
    
    # Scale numerical features
    scaler = StandardScaler()
    # Use means and stds from training (approximate values)
    means = {'Traffic_Density': 1.0, 'Speed_Limit': 80.0, 'Number_of_Vehicles': 4.0, 
             'Driver_Alcohol': 0.1, 'Driver_Age': 40.0}
    stds = {'Traffic_Density': 0.8, 'Speed_Limit': 30.0, 'Number_of_Vehicles': 2.0, 
            'Driver_Alcohol': 0.3, 'Driver_Age': 15.0}
    
    for col in numerical_features:
        df[col + '_scaled'] = (df[col] - means[col]) / stds[col]
    
    # Encode Time_of_Day
    time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    df['Time_of_Day_encoded'] = df['Time_of_Day'].map(time_mapping)
    
    # One-hot encode categorical features
    # Weather (default is Clear)
    df['Weather_Foggy'] = 1 if df['Weather'].iloc[0] == 'Foggy' else 0
    df['Weather_Rainy'] = 1 if df['Weather'].iloc[0] == 'Rainy' else 0
    df['Weather_Snowy'] = 1 if df['Weather'].iloc[0] == 'Snowy' else 0
    df['Weather_Stormy'] = 1 if df['Weather'].iloc[0] == 'Stormy' else 0
    
    # Road_Type (default is City Road)
    df['Road_Type_Highway'] = 1 if df['Road_Type'].iloc[0] == 'Highway' else 0
    df['Road_Type_Mountain Road'] = 1 if df['Road_Type'].iloc[0] == 'Mountain Road' else 0
    df['Road_Type_Rural Road'] = 1 if df['Road_Type'].iloc[0] == 'Rural Road' else 0
    
    # Road_Condition (default is Dry)
    df['Road_Condition_Icy'] = 1 if df['Road_Condition'].iloc[0] == 'Icy' else 0
    df['Road_Condition_Under Construction'] = 1 if df['Road_Condition'].iloc[0] == 'Under Construction' else 0
    df['Road_Condition_Wet'] = 1 if df['Road_Condition'].iloc[0] == 'Wet' else 0
    
    # Vehicle_Type (default is Bus)
    df['Vehicle_Type_Car'] = 1 if df['Vehicle_Type'].iloc[0] == 'Car' else 0
    df['Vehicle_Type_Motorcycle'] = 1 if df['Vehicle_Type'].iloc[0] == 'Motorcycle' else 0
    df['Vehicle_Type_Truck'] = 1 if df['Vehicle_Type'].iloc[0] == 'Truck' else 0
    
    # Road_Light_Condition (default is Artificial Light)
    df['Road_Light_Condition_Daylight'] = 1 if df['Road_Light_Condition'].iloc[0] == 'Daylight' else 0
    df['Road_Light_Condition_No Light'] = 1 if df['Road_Light_Condition'].iloc[0] == 'No Light' else 0
    
    # Select only the transformed features
    feature_names = [
        'Traffic_Density_scaled', 'Speed_Limit_scaled', 'Number_of_Vehicles_scaled',
        'Driver_Alcohol_scaled', 'Driver_Age_scaled', 'Time_of_Day_encoded',
        'Weather_Foggy', 'Weather_Rainy', 'Weather_Snowy', 'Weather_Stormy',
        'Road_Type_Highway', 'Road_Type_Mountain Road', 'Road_Type_Rural Road',
        'Road_Condition_Icy', 'Road_Condition_Under Construction', 'Road_Condition_Wet',
        'Vehicle_Type_Car', 'Vehicle_Type_Motorcycle', 'Vehicle_Type_Truck',
        'Road_Light_Condition_Daylight', 'Road_Light_Condition_No Light'
    ]
    
    return df[feature_names]

if predict_button:
    # Gather all inputs
    input_data = {
        'Weather': weather,
        'Road_Type': road_type,
        'Time_of_Day': time_of_day,
        'Traffic_Density': traffic_density,
        'Speed_Limit': speed_limit,
        'Number_of_Vehicles': num_vehicles,
        'Driver_Alcohol': driver_alcohol_val,
        'Road_Condition': road_condition,
        'Vehicle_Type': vehicle_type,
        'Driver_Age': driver_age,
        'Road_Light_Condition': road_light
    }
    
    # Preprocess the inputs
    processed_input = preprocess_input(input_data)
    
    # Make prediction
    with st.spinner('Calculating accident probability...'):
        prediction_proba = model.predict_proba(processed_input)
        accident_probability = prediction_proba[0][1] * 100  # Probability of class 1 (accident)
    
    # Display results
    st.markdown("---")
    st.header("Prediction Result")
    
    col1, col2 = st.columns(2)
    
    # Initialize risk factors list
    risk_factors = []
    
    if weather in ["Rainy", "Snowy", "Foggy", "Stormy"]:
        risk_factors.append(f"Bad weather condition ({weather})")
    
    if road_condition in ["Icy", "Wet", "Under Construction"]:
        risk_factors.append(f"Poor road condition ({road_condition})")
        
    if time_of_day in ["Night", "Evening"]:
        risk_factors.append("Reduced visibility due to time of day")
        
    if speed_limit > 100:
        risk_factors.append(f"High speed limit ({speed_limit} km/h)")
        
    if driver_alcohol_val == 1:
        risk_factors.append("Driver alcohol consumption")
        
    if driver_age < 25 or driver_age > 65:
        risk_factors.append(f"Driver age ({driver_age} years)")
        
    if road_type == "Mountain Road":
        risk_factors.append("Challenging road type (Mountain Road)")
    
    # Now display risk level with awareness of risk factors
    with col1:
        if accident_probability > 60:
            st.error(f"⚠️ High Accident Risk: {accident_probability:.1f}%")
        elif accident_probability > 30:
            st.warning(f"⚠️ Moderate Accident Risk: {accident_probability:.1f}%")
        else:
            # Now this condition will work correctly
            if len(risk_factors) >= 3:
                st.warning(f"⚠️ Low Model Risk but Multiple Risk Factors Present: {accident_probability:.1f}%")
            else:
                st.success(f"✅ Low Accident Risk: {accident_probability:.1f}%")
    
    with col2:
        # Create a gauge chart
        import plotly.graph_objects as go
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=accident_probability,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Accident Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': accident_probability
                }
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors analysis
    st.subheader("Contributing Risk Factors")
    
    # Example logic for risk factors (based on some domain knowledge)
    risk_factors = []
    
    if weather in ["Rainy", "Snowy", "Foggy", "Stormy"]:
        risk_factors.append(f"Bad weather condition ({weather})")
    
    if road_condition in ["Icy", "Wet", "Under Construction"]:
        risk_factors.append(f"Poor road condition ({road_condition})")
        
    if time_of_day in ["Night", "Evening"]:
        risk_factors.append("Reduced visibility due to time of day")
        
    if speed_limit > 100:
        risk_factors.append(f"High speed limit ({speed_limit} km/h)")
        
    if driver_alcohol_val == 1:
        risk_factors.append("Driver alcohol consumption")
        
    if driver_age < 25 or driver_age > 65:
        risk_factors.append(f"Driver age ({driver_age} years)")
        
    if road_type == "Mountain Road":
        risk_factors.append("Challenging road type (Mountain Road)")
    
    if not risk_factors:
        st.write("No major risk factors identified.")
    else:
        for factor in risk_factors:
            st.write(f"• {factor}")
    
    # Safety recommendations
    st.subheader("Safety Recommendations")
    
    recommendations = []
    if "Bad weather" in " ".join(risk_factors):
        recommendations.append("Reduce speed and increase following distance in adverse weather")
    
    if "road condition" in " ".join(risk_factors).lower():
        recommendations.append("Drive cautiously on compromised road surfaces")
    
    if "visibility" in " ".join(risk_factors).lower():
        recommendations.append("Use proper lighting and maintain extra vigilance at night")
    
    if "speed limit" in " ".join(risk_factors).lower():
        recommendations.append("Stay within posted speed limits, especially in high-speed zones")
    
    if "alcohol" in " ".join(risk_factors).lower():
        recommendations.append("Never drive under the influence of alcohol")
    
    if "age" in " ".join(risk_factors).lower():
        if driver_age < 25:
            recommendations.append("Young drivers should gain experience in low-risk conditions")
        else:
            recommendations.append("Senior drivers should consider daytime driving when possible")
    
    if not recommendations:
        recommendations = ["Maintain safe driving practices", 
                          "Always use seat belts", 
                          "Stay alert and avoid distractions"]
    
    for rec in recommendations:
        st.write(f"• {rec}")