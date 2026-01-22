# app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Geothermal Well Predictor",
    page_icon="♨️",
    layout="wide"
)

# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource
def load_model():
    """Load pre-trained model"""
    try:
        model = joblib.load('final_model.sav')
        return model
    except FileNotFoundError:
        st.error("Error: 'final_model.sav' not found. Please ensure the model file is in the same directory.")
        return None

model = load_model()

# ============================================================================
# HEADER
# ============================================================================

st.title("♨️ Geothermal Well Conversion Predictor")
st.markdown("Enter the well parameters below to predict conversion feasibility.")
st.markdown("---")


if model:
    # Create 3 columns for cleaner layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🌡️ Thermal & Depth")
        
        temp = st.slider(
            "Bottomhole Temperature (°C)", 
            min_value=85, max_value=220, value=150, step=1,
            help="Temperature at the bottom of the well."
        )
        
        depth = st.slider(
            "Well Depth (m)", 
            min_value=800, max_value=5000, value=2500, step=50
        )
        
        perm = st.number_input(
            "Permeability (mD)", 
            min_value=0.1, max_value=100.0, value=25.0, step=0.1
        )

    with col2:
        st.subheader("💧 Fluid Dynamics")
        
        flow_rate = st.slider(
            "Flow Rate (m³/day)", 
            min_value=0, max_value=6500, value=2000, step=50
        )
        
        flow_stability = st.slider(
            "Flow Rate Stability (%)", 
            min_value=0, max_value=100, value=35, step=1,
            help="Percentage of time flow rate is stable."
        )
        
        tds = st.number_input(
            "TDS (mg/L)", 
            min_value=0, max_value=10000, value=5000, step=100
        )
        
        h2s = st.number_input(
            "H2S (ppm)", 
            min_value=0, max_value=1000, value=100, step=10
        )

    with col3:
        st.subheader("⚙️ Operational")
        
        power = st.number_input(
            "Expected Power Output (kW)", 
            min_value=0, max_value=12000, value=1500, step=100
        )
        
        well_age = st.slider(
            "Well Age (Years)", 
            min_value=0, max_value=70, value=15, step=1
        )
        
        distance = st.slider(
            "Distance to Demand (km)", 
            min_value=0, max_value=150, value=50, step=1
        )
    
    st.markdown("---")
    
    # Center the predict button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        predict_btn = st.button("🔮 Predict Feasibility", use_container_width=True, type="primary")

    if predict_btn:
        # CRITICAL FIX: Creating a DataFrame with explicit column names
        # The keys below MUST match the columns your model was trained on exactly.
        input_data = pd.DataFrame({
            'Well_Depth_m': [depth],
            'Bottomhole_Temp_C': [temp],
            'Flow_Rate_m3_day': [flow_rate],
            'Flow_Rate_Stability_pct': [flow_stability],
            'Power_Output_kW': [power],
            'Well_Age_Years': [well_age],
            'Distance_to_Demand_km': [distance],
            'Permeability_mD': [perm],
            'TDS_mg_L': [tds],
            'H2S_ppm': [h2s]
        })

        try:
            # 1. Get Prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            success_prob = probability[1] * 100
            is_success = prediction == 1
            
            # 2. Display Results
            st.divider()
            
            # Top level metric
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                if is_success:
                    st.success("## Project viable")
                else:
                    st.error("## High Risk,  not viable!")

            with res_col2:
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = success_prob,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Feasibility Score"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "green" if is_success else "red"},
                        'steps': [
                            {'range': [0, 50], 'color': "mistyrose"},
                            {'range': [50, 100], 'color': "honeydew"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Ensure the feature names in 'input_data' exactly match the column names used during model training.")

else:
    st.warning("Model not loaded. Please check the file path.")
