import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Custom CSS function that accepts theme parameters
def apply_theme(primary_color, secondary_color, text_color, bg_color, sidebar_color):
    st.markdown(f"""
    <style>
    .main .block-container {{
        background-color: {bg_color};
        padding: 2rem;
        border-radius: 10px;
    }}
    
    .sidebar .sidebar-content {{
        background-color: {sidebar_color};
    }}
    
    h1, h2, h3 {{
        color: {primary_color};
    }}
    
    .stButton>button {{
        background-color: {primary_color};
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }}
    
    .stButton>button:hover {{
        background-color: {secondary_color};
    }}
    
    .metric-card {{
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 5px solid {primary_color};
    }}
    
    .recommendation {{
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        border-radius: 5px;
    }}
    
    .high-risk {{
        background-color: #ff6b6b;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }}
    
    .low-risk {{
        background-color: #51cf66;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }}
    
    .sidebar .sidebar-content {{
        background-color: {sidebar_color};
    }}
    
    .sidebar-header {{
        color: {primary_color};
        font-weight: bold;
    }}
    
    /* Customize sidebar text and elements */
    .sidebar [data-testid="stMarkdownContainer"] {{
        color: {text_color};
    }}
    
    /* Style sliders in sidebar */
    .sidebar [data-testid="stSlider"] > div > div {{
        background-color: {primary_color};
    }}
    </style>
    """, unsafe_allow_html=True)

# Set page configuration
st.set_page_config(
    page_title="Traffic Accident Prediction System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define theme presets
themes = {
    "Blue (Default)": {
        "primary": "#4a74df",
        "secondary": "#3a64cf",
        "text": "#1f3672",
        "bg": "#f5f7ff",
        "sidebar": "#e0e8ff"
    },
    "Green": {
        "primary": "#2ecc71",
        "secondary": "#27ae60",
        "text": "#1e8449",
        "bg": "#eafaf1",
        "sidebar": "#d5f5e3"
    },
    "Red": {
        "primary": "#e74c3c",
        "secondary": "#c0392b",
        "text": "#922b21",
        "bg": "#fdedec",
        "sidebar": "#f5b7b1"
    },
    "Purple": {
        "primary": "#9b59b6",
        "secondary": "#8e44ad",
        "text": "#6c3483",
        "bg": "#f4ecf7",
        "sidebar": "#e8daef"
    }
}
# Load the trained model
@st.cache_resource
def load_model():
    model = pickle.load(open('accident_pred_logistic_regression.pkl', 'rb'))
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# Define constants and categories
WEATHER_OPTIONS = ['Clear', 'Rainy', 'Foggy', 'Snowy', 'Stormy']
ROAD_TYPE_OPTIONS = ['City Road', 'Highway', 'Rural Road', 'Mountain Road']
ROAD_CONDITION_OPTIONS = ['Dry', 'Wet', 'Icy', 'Under Construction']
LIGHT_CONDITION_OPTIONS = ['Daylight', 'Artificial Light', 'No Light']
TIME_OF_DAY_OPTIONS = ['Morning', 'Afternoon', 'Evening', 'Night']
VEHICLE_TYPE_OPTIONS = ['Car', 'Truck', 'Motorcycle', 'Bus']

# ----- INTERACTIVE SIDEBAR SECTION -----
st.sidebar.title("🚗 Road Safety Hub")
st.sidebar.markdown('<div class="sidebar-header"></div>', unsafe_allow_html=True)

# Theme selection
theme_options = list(themes.keys())
selected_theme = st.sidebar.selectbox("Select Theme", theme_options)

# Apply the selected theme
current_theme = themes[selected_theme]
apply_theme(
    current_theme["primary"],
    current_theme["secondary"],
    current_theme["text"],
    current_theme["bg"],
    current_theme["sidebar"]
)

# About section
with st.sidebar.expander("ℹ️ About This System", expanded=True):
    st.markdown("""
    This Traffic Accident Prediction System analyzes various factors to predict accident likelihood.

    The model was trained on traffic data considering factors like:
    - Weather conditions
    - Road characteristics
    - Driver attributes
    - Vehicle information
    - Traffic conditions
    """)

# Safety Tips section
with st.sidebar.expander("🛡️ Safe Driving Tips", expanded=True):
    st.markdown("""
    - 🛑 Follow speed limits
    - 🚫 Never drive impaired 
    - 🌧️ Adjust to weather conditions
    - 👀 Avoid distractions
    - 🚗 Maintain your vehicle
    - 🛣️ Be cautious on unfamiliar roads
    - 💺 Always wear seatbelts
    """)

# Interactive safety quiz
with st.sidebar.expander("🔍 Test Your Safety Knowledge", expanded=False):
    st.write("Quick Safety Quiz!")
    q1 = st.radio("What's the #1 cause of traffic accidents?", 
                 ["Speeding", "Distracted driving", "Weather", "Vehicle failure"])
    if q1:
        if q1 == "Distracted driving":
            st.success("✅ Correct! Distracted driving is the leading cause of accidents.")
        else:
            st.error("❌ Actually, distracted driving is the leading cause.")
    
    safe_score = st.slider("Rate your driving safety (1-10):", 1, 10, 7)
    if safe_score < 5:
        st.warning("Consider improving your driving habits!")
    elif safe_score >= 8:
        st.success("Great safety consciousness!")

# Emergency resources
with st.sidebar.expander("🚨 Emergency Resources", expanded=False):
    st.markdown("""
    - **Emergency**: 911
    - **Roadside Assistance**: 1-800-AAA-HELP
    - **Weather Updates**: [National Weather Service](https://www.weather.gov/)
    - **Road Conditions**: [DOT Road Info](https://www.fhwa.dot.gov/)
    """)

# Feedback section
with st.sidebar.expander("📝 Provide Feedback", expanded=False):
    st.text_area("Share your thoughts or report issues:", "")
    if st.button("Submit Feedback", key="sidebar_feedback"):
        st.sidebar.success("Thank you for your feedback!")

st.sidebar.markdown("---")
st.sidebar.info("© 2025 Traffic Safety Initiative")

# ----- MAIN APP CONTENT -----
# Title and description
st.title("🚦 Traffic Accident Prediction System")
st.markdown("""
This application predicts the likelihood of a traffic accident based on various road, vehicle, and driver conditions. 
Enter the relevant information below and click 'Predict' to get the accident risk assessment.
""")

# Create columns for the form
st.header("Input Traffic Conditions")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Road Conditions")
    weather = st.selectbox("Weather", WEATHER_OPTIONS)
    road_type = st.selectbox("Road Type", ROAD_TYPE_OPTIONS)
    road_condition = st.selectbox("Road Condition", ROAD_CONDITION_OPTIONS)
    road_light = st.selectbox("Road Light Condition", LIGHT_CONDITION_OPTIONS)
    time_of_day = st.selectbox("Time of Day", TIME_OF_DAY_OPTIONS)
    speed_limit = st.slider("Speed Limit (km/h)", 20, 120, 50)

with col2:
    st.subheader("Vehicle & Driver Information")
    vehicle_type = st.selectbox("Vehicle Type", VEHICLE_TYPE_OPTIONS)
    traffic_density = st.slider("Traffic Density (%)", 0, 100, 30)
    num_vehicles = st.slider("Number of Vehicles", 1, 50, 5)
    driver_age = st.slider("Driver Age", 16, 90, 35)
    driver_alcohol = st.radio("Driver Alcohol Consumption", ["No", "Yes"])

# Convert driver_alcohol to binary
driver_alcohol_binary = 1 if driver_alcohol == "Yes" else 0

# Preprocess function to match the training pipeline
def preprocess_input(features):
    # Create a DataFrame with the input features
    input_df = pd.DataFrame([features])
    
    # Apply the same preprocessing steps as in training
    time_categories = ['Morning', 'Afternoon', 'Evening', 'Night']
    time_encoded = time_categories.index(features['Time_of_Day'])
    
    # Handle one-hot encoding manually
    weather_encoded = [0] * (len(WEATHER_OPTIONS) - 1)
    if features['Weather'] != 'Clear':  # Assuming 'Clear' was the reference category
        idx = WEATHER_OPTIONS.index(features['Weather']) - 1
        if idx >= 0:
            weather_encoded[idx] = 1
            
    road_type_encoded = [0] * (len(ROAD_TYPE_OPTIONS) - 1)
    if features['Road_Type'] != 'City Road':  # Assuming 'City Road' was the reference
        idx = ROAD_TYPE_OPTIONS.index(features['Road_Type']) - 1
        if idx >= 0:
            road_type_encoded[idx] = 1
            
    road_cond_encoded = [0] * (len(ROAD_CONDITION_OPTIONS) - 1)
    if features['Road_Condition'] != 'Dry':  # Assuming 'Dry' was the reference
        idx = ROAD_CONDITION_OPTIONS.index(features['Road_Condition']) - 1
        if idx >= 0:
            road_cond_encoded[idx] = 1
            
    light_encoded = [0] * (len(LIGHT_CONDITION_OPTIONS) - 1)
    if features['Road_Light_Condition'] != 'Daylight':  # Assuming 'Daylight' was reference
        idx = LIGHT_CONDITION_OPTIONS.index(features['Road_Light_Condition']) - 1
        if idx >= 0:
            light_encoded[idx] = 1
            
    vehicle_encoded = [0] * (len(VEHICLE_TYPE_OPTIONS) - 1)
    if features['Vehicle_Type'] != 'Car':  # Assuming 'Car' was the reference
        idx = VEHICLE_TYPE_OPTIONS.index(features['Vehicle_Type']) - 1
        if idx >= 0:
            vehicle_encoded[idx] = 1
    
    # Standardize numerical features (approximate based on possible dataset stats)
    scaler = StandardScaler()
    numerical = np.array([features['Traffic_Density'], features['Speed_Limit'], 
                         features['Number_of_Vehicles'], features['Driver_Alcohol'], 
                         features['Driver_Age']])
    numerical_scaled = (numerical - np.array([50, 60, 10, 0.15, 40])) / np.array([25, 20, 8, 0.36, 15])
    
    # Create interaction features
    alcohol_night = numerical_scaled[3] * (1 if time_encoded >= 3 else 0)
    speed_wet_road = numerical_scaled[1] * (1 if features['Road_Condition'] in ['Wet', 'Icy'] else 0)
    motorcycle_speed = numerical_scaled[1] * (1 if features['Vehicle_Type'] == 'Motorcycle' else 0)
    low_visibility = numerical_scaled[1] * (1 if features['Road_Light_Condition'] == 'No Light' else 0)
    
    # Create alcohol risk factor (3x amplification as in the feature engineering)
    alcohol_risk = features['Driver_Alcohol'] * 3.0
    numerical_scaled = np.append(numerical_scaled, alcohol_risk)
    
    # Define age groups based on scaled age value
    age_scaled = (features['Driver_Age'] - 16) / (90 - 16)  # Scale to 0-1 range
    
    # Add age group dummies (assuming we drop first category, 'Young')
    if 0.25 <= age_scaled < 0.5:
        age_dummies = [1, 0, 0]  # Adult
    elif 0.5 <= age_scaled < 0.75:
        age_dummies = [0, 1, 0]  # Middle
    elif age_scaled >= 0.75:
        age_dummies = [0, 0, 1]  # Senior
    else:
        age_dummies = [0, 0, 0]  # Young (reference)
        
    # Combine all features into a single array
    processed_features = np.concatenate([
        numerical_scaled,
        [time_encoded],
        weather_encoded,
        road_type_encoded,
        road_cond_encoded,
        light_encoded,
        vehicle_encoded,
        [alcohol_night, speed_wet_road, motorcycle_speed, low_visibility],
        age_dummies
    ])
    
    return processed_features

# Generate safety recommendations based on input features
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

# Create a prediction button with the themed color
predict_button = st.button("Predict Accident Risk", use_container_width=True)

# Process when the predict button is clicked
if predict_button:
    # Create features dictionary
    features = {
        'Weather': weather,
        'Road_Type': road_type,
        'Road_Condition': road_condition,
        'Road_Light_Condition': road_light,
        'Time_of_Day': time_of_day,
        'Speed_Limit': speed_limit,
        'Traffic_Density': traffic_density,
        'Number_of_Vehicles': num_vehicles,
        'Vehicle_Type': vehicle_type,
        'Driver_Age': driver_age,
        'Driver_Alcohol': driver_alcohol_binary
    }
    
    # Add a progress bar for processing
    progress_text = "Analyzing risk factors... Please wait."
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        progress_bar.progress(percent_complete + 1)
        if percent_complete < 50:
            # Simulate slower processing
            import time
            time.sleep(0.01)
    
    # Preprocess the input
    processed_input = preprocess_input(features)
    
    # Make prediction
    probability = model.predict_proba([processed_input])[0][1]
    
    # Create columns for output in a styled container
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.header("Prediction Results")
    col1, col2 = st.columns(2)
    
    # Show the prediction with styled output
    with col1:
        if probability > 0.25:  # Lowered threshold for better risk detection
            st.markdown("<div class='high-risk'>⚠️ High Risk of Accident</div>", unsafe_allow_html=True)
            risk_level = "High"
        else:
            st.markdown("<div class='low-risk'>✅ Low Risk of Accident</div>", unsafe_allow_html=True)
            risk_level = "Low"
    
    with col2:
        st.metric("Accident Probability", f"{probability*100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Generate recommendations
    recommendations = get_recommendations(features)
    
    # Display the recommendations in a styled container
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.header("Safety Recommendations")
    for recommendation in recommendations:
        if "🚫" in recommendation or "⚠️" in recommendation:
            st.markdown(f"<div class='recommendation' style='background-color: #fff3cd; padding: 8px; border-left: 4px solid {current_theme['primary']};'>{recommendation}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='recommendation' style='background-color: #e7f5ff; padding: 8px; border-left: 4px solid {current_theme['primary']};'>{recommendation}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Risk factor analysis visualization in a styled container
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.header("Risk Factor Analysis")
    
    # Create a visual representation of risk factors
    risk_factors = {
        'Driver_Alcohol': driver_alcohol_binary * 0.3,  # Amplify alcohol importance
        'Weather_Risk': 0.1 if weather in ['Foggy', 'Snowy', 'Stormy'] else 0.0,
        'Road_Condition_Risk': 0.15 if road_condition in ['Wet', 'Icy', 'Under Construction'] else 0.0,
        'Speed_Risk': max(0, (speed_limit - 60) / 200),  # Normalize to 0-0.3 range
        'Light_Risk': 0.15 if road_light == 'No Light' else (0.05 if road_light == 'Artificial Light' else 0.0),
        'Time_Risk': 0.1 if time_of_day in ['Night'] else (0.05 if time_of_day == 'Evening' else 0.0),
        'Traffic_Density_Risk': traffic_density / 500,  # Normalize to 0-0.2 range
        'Vehicle_Risk': 0.15 if vehicle_type == 'Motorcycle' else (0.1 if vehicle_type == 'Truck' else 0.05),
        'Age_Risk': 0.1 if driver_age < 25 or driver_age > 65 else 0.05
    }
    
    # Sort by risk contribution
    sorted_risks = dict(sorted(risk_factors.items(), key=lambda item: item[1], reverse=True))
    
    # Create bar chart with theme colors
    fig, ax = plt.subplots(figsize=(10, 6))
    risks = list(sorted_risks.keys())
    values = list(sorted_risks.values())
    
    # Custom colors based on theme
    colors = [current_theme['primary'] if v > 0.1 else current_theme['secondary'] for v in values]
    
    ax.barh(risks, values, color=colors)
    ax.set_xlabel('Risk Contribution')
    ax.set_title('Major Risk Factors in Current Conditions')
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    # Explanation of risk factors
    st.subheader("Understanding Key Risk Factors")
    st.write("""
    The chart above shows the relative contribution of each factor to the overall accident risk.
    Factors with higher values contribute more significantly to the risk assessment.
    Pay special attention to factors shown in the darker color, as these represent the highest risk elements
    in the current conditions.
    """)
    st.markdown("</div>", unsafe_allow_html=True)