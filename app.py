import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# 1. Carregar dados e modelo
@st.cache_data
def load_data():
    return pd.read_csv("data/Obesity.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/gb_model.joblib")

df = load_data()
model = load_model()

st.title("Dashboard Analítico de Obesidade")
st.markdown("Insights sobre perfil de obesidade e previsão individual")

# 2. Sidebar de filtros
st.sidebar.header("Filtros")
genders = st.sidebar.multiselect("Gênero", df["Gender"].unique(), default=df["Gender"].unique())
ages = st.sidebar.slider("Idade", int(df["Age"].min()), int(df["Age"].max()),
                         (int(df["Age"].min()), int(df["Age"].max())))
df_filt = df[df["Gender"].isin(genders) & df["Age"].between(*ages)]

# 3. Métricas gerais
st.subheader("Visão Geral")
col1, col2, col3 = st.columns(3)
col1.metric("Total de registros", len(df_filt))
col2.metric("Média de Peso (kg)", f"{df_filt['Weight'].mean():.1f}")
col3.metric("Média de Altura (m)", f"{df_filt['Height'].mean():.2f}")

# 4. Distribuição de níveis de obesidade
st.subheader("Distribuição de Obesity_level")
dist = df_filt["Obesity_level"].value_counts(normalize=True).mul(100)
st.bar_chart(dist)

# 5. Correlação entre variáveis numéricas
st.subheader("Matriz de Correlação")
corr = df_filt[["Age","Height","Weight","CH2O","FAF"]].corr()
fig, ax = plt.subplots()
ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45)
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
st.pyplot(fig)

# 6. Previsão individual
st.sidebar.header("Nova previsão")
with st.sidebar.form("predict_form"):
    inputs = {
        "Gender": st.selectbox("Gender", df["Gender"].unique()),
        "Age": st.number_input("Age", int(df["Age"].min()), int(df["Age"].max()), int(df["Age"].mean())),
        "Height": st.number_input("Height", float(df["Height"].min()), float(df["Height"].max()), float(df["Height"].mean())),
        "Weight": st.number_input("Weight", float(df["Weight"].min()), float(df["Weight"].max()), float(df["Weight"].mean())),
        # … repita para as demais features categóricas/numéricas …
    }
    submitted = st.form_submit_button("Prever Obesidade")
    if submitted:
        X_new = pd.DataFrame([inputs])
        pred = model.predict(X_new)[0]
        st.sidebar.success(f"Nível previsto: **{pred}**")
