import streamlit as st
import pandas as pd
import joblib

# Carregar o modelo treinado
model = joblib.load("gb_model.joblib")

# Título do app
st.title("Previsão de Obesidade com Machine Learning")

st.markdown("Insira os dados do paciente abaixo para prever o nível de obesidade.")

# Inputs numéricos
age = st.number_input("Idade", min_value=1, max_value=120, value=25)
height = st.number_input("Altura (em metros)", min_value=1.0, max_value=2.5, value=1.70)
weight = st.number_input("Peso (em kg)", min_value=30.0, max_value=200.0, value=70.0)
fcvc = st.slider("Frequência de consumo de vegetais (0 a 3)", 0.0, 3.0, 2.0)
ncp = st.slider("Número de refeições principais ao dia", 1.0, 4.0, 3.0)
ch2o = st.slider("Consumo de água por dia (litros)", 0.0, 3.0, 2.0)
faf = st.slider("Atividade física (horas por semana)", 0.0, 20.0, 3.0)
tue = st.slider("Tempo diante de tela (horas por dia)", 0.0, 24.0, 8.0)

# Inputs categóricos
gender = st.selectbox("Gênero", ["Male", "Female"])
family_history = st.selectbox("Histórico familiar de obesidade", ["yes", "no"])
favc = st.selectbox("Consome alimentos calóricos com frequência?", ["yes", "no"])
caec = st.selectbox("Come entre as refeições?", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Fuma?", ["yes", "no"])
scc = st.selectbox("Monitora consumo de calorias?", ["yes", "no"])
calc = st.selectbox("Consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Meio de transporte", ["Public_Transportation", "Walking", "Automobile", "Bike", "Motorbike"])

# Botão de previsão
if st.button("Prever"):
    input_data = pd.DataFrame([{
        "Age": age,
        "Height": height,
        "Weight": weight,
        "FCVC": fcvc,
        "NCP": ncp,
        "CH2O": ch2o,
        "FAF": faf,
        "TUE": tue,
        "Gender": gender,
        "family_history": family_history,
        "FAVC": favc,
        "CAEC": caec,
        "SMOKE": smoke,
        "SCC": scc,
        "CALC": calc,
        "MTRANS": mtrans
    }])

    prediction = model.predict(input_data)[0]
    st.success(f"💡 Previsão: **{prediction}**")

