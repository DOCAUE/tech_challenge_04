import streamlit as st
import joblib
import pandas as pd

# Carrega o modelo
modelo = joblib.load("gb_model.joblib")

st.set_page_config(page_title="App Preditivo", layout="centered")
st.title("Previsão com Modelo Treinado")

# Entradas (exemplo — ajuste conforme seu modelo)
feature_1 = st.number_input("Idade", min_value=0, max_value=120, value=30)
feature_2 = st.number_input("Salário", min_value=0.0, value=5000.0)

# Botão de previsão
if st.button("Realizar Previsão"):
    dados = pd.DataFrame([[feature_1, feature_2]], columns=["idade", "salario"])
    pred = modelo.predict(dados)
    st.success(f"Resultado: {pred[0]}")
