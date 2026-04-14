import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. UI STYLING ---
st.set_page_config(page_title="Content Monetization Modeler", layout="wide", page_icon="💰")

st.markdown("""
    <style>
    body {background-color:#0e1117; color:#fff;}
    .sidebar .sidebar-content {background-color:#1c1c24;}
    h1, h2, h3, h4 { color: #F9FAFB !important; }
    .stNumberInput label, .stSelectbox label { color: #E0E0E0 !important; }
    </style>
""", unsafe_allow_html=True)

st.title("▶️ Content Monetization Modeler")

# --- 2. LOAD FILES ---
MODEL_PATH = 'linear_regression_model.pkl'
SCALER_PATH = 'scaler.pkl'
COLS_PATH = 'model_columns.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Model files not found! Make sure your .pkl files are in the exact same folder as this app.py file.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
model_columns = joblib.load(COLS_PATH)

# --- 3. USER INPUTS ---
st.subheader("📊 Enter Video Feature Values")

col1, col2 = st.columns(2)

with col1:
    views = st.number_input("Total Views", min_value=0, value=10000)
    likes = st.number_input("Total Likes", min_value=0, value=500)
    comments = st.number_input("Total Comments", min_value=0, value=100)
    watch_time = st.number_input("Watch Time (Minutes)", min_value=0.0, value=5000.0)

with col2:
    video_length = st.number_input("Video Length (Minutes)", min_value=0.0, value=10.0)
    subscribers = st.number_input("Channel Subscribers", min_value=0, value=50000)
    
st.subheader("🌍 Contextual Information")
col3, col4 = st.columns(2)

with col3:
    category = st.selectbox("Category", ["Entertainment", "Gaming", "Lifestyle", "Music", "Tech"])
    device = st.selectbox("Primary Device", ["Mobile", "TV", "Tablet"])
    
with col4:
    country = st.selectbox("Top Country", ["US", "IN", "UK", "CA", "DE"])

# --- 4. PREDICTION & CHART LOGIC ---
if st.button("Predict"):
    # Calculate your engineered feature
    engagement_rate = (likes + comments) / views if views > 0 else 0
    
    # Create the dictionary of inputs
    input_data = {
        'views': views, 'likes': likes, 'comments': comments,
        'watch_time_minutes': watch_time, 'video_length_minutes': video_length,
        'subscribers': subscribers, 'engagement_rate': engagement_rate,
        # Default date values so the model doesn't break
        'year': 2024, 'month': 1, 'day': 1, 'day_of_week': 0, 'hour': 12, 'minute': 0,
        f'category_{category}': 1, f'device_{device}': 1, f'country_{country}': 1
    }

    input_df = pd.DataFrame([input_data])

    # Ensure all columns from training are present (fill missing with 0)
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # Reorder columns to exactly match the training data
    input_df = input_df[model_columns]

    try:
        # Scale the data using your saved scaler
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        prediction = max(0, prediction)  # Prevent negative revenue
        
        st.success(f"💰 **Predicted Ad Revenue:** ${prediction:,.2f}")
        st.balloons()

        if prediction == 0:
            st.info("This video may not generate ad revenue based on the current input values.")

        # --- BEST OF BOTH WORLDS CHART ---
        st.markdown("---")
        st.subheader("📊 How strongly did each feature influence THIS prediction?")
        
        # 1. Calculate dynamic impact based on YOUR numbers
        weights = model.coef_
        specific_impact = weights * input_scaled[0]
        
        # 2. Put it in a dataframe
        importance_df = pd.DataFrame({'Feature': model_columns, 'Impact': specific_impact})
        
        # 3. FILTER: Only keep the main dashboard features
        dashboard_features = [
            'views', 'likes', 'comments', 'watch_time_minutes', 
            'video_length_minutes', 'engagement_rate'
        ]
        importance_df = importance_df[importance_df['Feature'].isin(dashboard_features)]
        
        # 4. THE MAGIC TRICK: Turn everything into a positive number (Influence Strength)
        importance_df['Influence Strength'] = importance_df['Impact'].abs()
        
        # 5. Sort by biggest influence
        importance_df = importance_df.sort_values(by='Influence Strength', ascending=False)
        
        # 6. Plot the dynamic, positive-only bar chart!
        st.bar_chart(data=importance_df.set_index('Feature')['Influence Strength'])
        st.caption("Taller bars mean the AI cared heavily about that specific input when calculating your revenue. Try changing a number to see it shift!")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")