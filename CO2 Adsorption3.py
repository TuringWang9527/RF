import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('xgb.pkl')

# Define feature names
feature_names = ['Temperature', 'CO2 Partial Pressure', 'Carbonization Time',
       'CO2 Concentration', 'Particle Diameter', 'CaO', 'MgO', 'SiO2', 'Al2O3',
       'Fe2O3', 'MnO', 'L/S']

# Streamlit user interface
st.title("CO2 Adsorption")

# Temperature: numerical input
Temperature = st.number_input("Temperature:", min_value=10, max_value=180, value=50)

# CO2 Partial Pressure: numerical input
CO2_Partial_Pressure = st.number_input("CO2 Partial Pressure:", min_value=0, max_value=39, value=10)

# Carbonization Time: numerical input
Carbonization_Time = st.number_input("Carbonization Time:", min_value=0, max_value=240, value=100)

# CO2 Concentration: numerical input
CO2_Concentration = st.number_input("CO2 Concentration:", min_value=0, max_value=100, value=50)

# Particle Diameter: numerical input
Particle_Diameter = st.number_input("Particle Diameter:", min_value=6, max_value=532, value=100)

# CaO: numerical input
CaO = st.number_input("CaO:", min_value=20, max_value=58, value=30)

# MgO: numerical input
MgO = st.number_input("MgO:", min_value=1.65, max_value=10.9, value=5.0)

# SiO2: numerical input
SiO2 = st.number_input("SiO2:", min_value=6.39, max_value=32.0, value=15.0)

# SiO2: numerical input
Al2O3 = st.number_input("Al2O3:", min_value=0.38, max_value=13.0, value=5.0)

# SiO2: numerical input
Fe2O3 = st.number_input("Fe2O3:", min_value=0.35, max_value=46.93, value=20.0)

# MnO: numerical input
MnO = st.number_input("MnO:", min_value=0.0, max_value=10.62, value=5.0)

# Fe2O3: numerical input
Fe2O3 = st.number_input("Fe2O3:", min_value=0.35, max_value=46.93, value=20.0)

# L/S: numerical input
L_S = st.number_input("L/S:", min_value=0, max_value=30, value=15)
# Process inputs and make predictions
feature_values = [Temperature, CO2_Partial_Pressure, Carbonization_Time,
       CO2_Concentration, Particle_Diameter, CaO, MgO, SiO2, Al2O3,
       Fe2O3, MnO, L_S]
features = np.array([feature_values])
input_df = pd.DataFrame(features,columns = ['Temperature', 'CO2 Partial Pressure', 'Carbonization Time',
       'CO2 Concentration', 'Particle Diameter', 'CaO', 'MgO', 'SiO2', 'Al2O3',
       'Fe2O3', 'MnO', 'L/S']


if st.button("Predict"):
	# Process inputs and make predictions
	feature_values = [Temperature, CO2_Partial_Pressure, Carbonization_Time,
       CO2_Concentration, Particle_Diameter, CaO, MgO, SiO2, Al2O3,
       Fe2O3, MnO, L_S]
	features = np.array([feature_values])
	input_df = pd.DataFrame(features,columns = ['Temperature', 'CO2 Partial Pressure', 'Carbonization Time',
       'CO2 Concentration', 'Particle Diameter', 'CaO', 'MgO', 'SiO2', 'Al2O3',
       'Fe2O3', 'MnO', 'L/S']
    # Predict class and probabilities
    predicted_number = model.predict(input_df)

    # Display prediction results
    st.write(f"**CO2吸收量:** {predicted_number}")

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=500)

    st.image("shap_force_plot.png")