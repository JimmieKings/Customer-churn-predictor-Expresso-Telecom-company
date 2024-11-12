import sklearn
import streamlit as st
import pandas as pd
import pickle

# Load the trained Random Forest model
with open('churn_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit app title
st.title("Expresso Customer Churn Prediction")
st.write("Enter the required features to predict customer churn.")

# Define mappings for categorical features
region_mapping = {'Region1': 0, 'Region2': 1, 'Region3': 2, 'Unknown': 3}
top_pack_mapping = {'Pack1': 0, 'Pack2': 1, 'Pack3': 2, 'No Pack': 3}

# Input fields for relevant features
region = st.selectbox('Region', list(region_mapping.keys()))
tenure = st.number_input('Tenure (Months)', min_value=0, max_value=100)
montant = st.number_input('Montant', min_value=0.0)
frequency_rech = st.number_input('Frequency of Recharge')
revenue = st.number_input('Revenue', min_value=0.0)
arpu_segment = st.number_input('ARPU Segment', min_value=0.0)
frequency = st.number_input('Frequency', min_value=0)
data_volume = st.number_input('Data Volume (MB)', min_value=0.0)
on_net = st.number_input('On Net Calls (Minutes)', min_value=0.0)
orange = st.number_input('Orange Network Usage (Minutes)', min_value=0.0)
tigo = st.number_input('Tigo Network Usage (Minutes)', min_value=0.0)
zone1 = st.number_input('Zone 1 Usage (Minutes)', min_value=0.0)
zone2 = st.number_input('Zone 2 Usage (Minutes)', min_value=0.0)
mrg = st.number_input('MRG (Monthly Recharge Group)', min_value=0)
regularity = st.number_input('Regularity', min_value=0)
top_pack = st.selectbox('Top Pack', list(top_pack_mapping.keys()))
freq_top_pack = st.number_input('Frequency of Top Pack', min_value=0)

# Map categorical inputs to numerical values
region_encoded = region_mapping[region]
top_pack_encoded = top_pack_mapping[top_pack]

# Prepare input data for the model
input_data = pd.DataFrame({
    'REGION': [region_encoded],
    'TENURE': [tenure],
    'MONTANT': [montant],
    'FREQUENCE_RECH': [frequency_rech],
    'REVENUE': [revenue],
    'ARPU_SEGMENT': [arpu_segment],
    'FREQUENCE': [frequency],
    'DATA_VOLUME': [data_volume],
    'ON_NET': [on_net],
    'ORANGE': [orange],
    'TIGO': [tigo],
    'ZONE1': [zone1],
    'ZONE2': [zone2],
    'MRG': [mrg],
    'REGULARITY': [regularity],
    'TOP_PACK': [top_pack_encoded],
    'FREQ_TOP_PACK': [freq_top_pack]
})

# Predict and display the result
if st.button("Predict Churn"):
    prediction = model.predict(input_data)
    churn_prob = model.predict_proba(input_data)[0][1]
    st.write(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
    st.write(f"Churn Probability: {churn_prob:.2f}")
