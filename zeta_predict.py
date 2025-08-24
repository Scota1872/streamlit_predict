import streamlit as st
import altair as alt
import pandas as pd
import os
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from catboost import CatBoostRegressor

#---------------------------CONFIGURAÇÕES STREAMLIT---------------------------------#
st.set_page_config(page_title='Simulador de Preço de Veículos', layout='wide')

#---------------------------CARREGAR DADOS-------------------------------------------#
df_raw = pd.read_csv('Dataset_price.csv', sep=',', encoding='LATIN')


X = df_raw[['Make','Model','Year','Engine Size','Mileage','Fuel Type','Transmission']]
y = df_raw['Price']


cat_columns = [0,1,5,6]
process = ColumnTransformer(
    transformers=[('OneHot', OneHotEncoder(handle_unknown='ignore'), cat_columns)],
    remainder='passthrough'
)

X_proc = process.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.3, random_state=42)


ctb = CatBoostRegressor(learning_rate=0.06, random_state=5, max_depth=5, n_estimators=150, verbose=0)
ctb.fit(X_train, y_train)


kfold = KFold(shuffle=True, random_state=5, n_splits=5)
cv_result = cross_val_score(ctb, X_proc, y, cv=kfold)
r2_medio = cv_result.mean() * 100

#--------------------SIDEBAR-------------------------------#
def barra_simulador():
    with st.sidebar:
        st.title("Filtros Simulador")
        selecoes = {}
        selecoes['Marca'] = st.selectbox("Marca", df_raw['Make'].unique(), index=None, placeholder="Ex.: BMW")
        selecoes['Modelo'] = st.selectbox("Modelo", df_raw['Model'].unique(), index=None, placeholder="Ex.: Serie 3")
        selecoes['Combustível'] = st.selectbox("Combustível", df_raw['Fuel Type'].unique(), index=None, placeholder="Ex.: Petrol")
        selecoes['Transmissão'] = st.selectbox("Transmissão", df_raw['Transmission'].unique(), index=None, placeholder="Ex.: Manual")
        selecoes['Ano'] = st.number_input("Ano do veículo:")
        selecoes['Motor'] = st.number_input("Tamanho do motor (L):")
        selecoes['Quilometragem'] = st.number_input("Quilometragem:")

        col1, col2 = st.columns(2)
        with col1:
            btn_simular = st.button("Simular", type='primary')
        with col2:
            btn_limpar = st.button("Limpar", type='primary')

    return selecoes, btn_simular, btn_limpar

#------------------------------------TELA PRINCIPAL--------------------------------------------------#
def tela_principal(selecoes, btn_simular):
    col1, col2 = st.columns(2)
    fotos = ("C:/Users/Junior/projeto_git/fotos")

    with col1:
        st.subheader('')
        st.subheader("Descrição do veículo")
        st.subheader('')  
        for label, valor in selecoes.items():
            if valor:
                st.write(f"**{label}:** {valor}")

    with col2:
        marca = selecoes.get("Marca")
        st.subheader('') 
        st.subheader("Imagem do Carro")
        st.subheader('') 
        if marca:
            caminho_foto = os.path.join(fotos, f"{marca}.jpg")
            if os.path.exists(caminho_foto):
                st.image(caminho_foto, caption=marca)
            else:
                st.warning(f"Foto da Marca {marca} não existe!")

   
    if btn_simular:
        novo_carro = pd.DataFrame([{
            'Make': selecoes['Marca'],
            'Model': selecoes['Modelo'],
            'Year': selecoes['Ano'],
            'Engine Size': selecoes['Motor'],
            'Mileage': selecoes['Quilometragem'],
            'Fuel Type': selecoes['Combustível'],
            'Transmission': selecoes['Transmissão']
        }])

        new_car_proc = process.transform(novo_carro)
        price_predict = ctb.predict(new_car_proc)[0]

        st.success(f"Preço previsto para este carro: **U$ {price_predict:,.2f}**")

#-------------------------------------GRÁFICOS-----------------------------------------------------------#
import matplotlib.pyplot as plt
import seaborn as sbn

def graficos():
    st.subheader('')
    st.subheader("Correlação Entre as Colunas")
    st.subheader('')
    col_grafico1, col_grafico2 = st.columns(2)


    colunas_numericas = df_raw.select_dtypes(include=['number']).columns
    with col_grafico1:
        chart = (
            alt.Chart(df_raw)
            .mark_circle()
            .encode(
                y='Price',
                x='Mileage',
                color='Make',
            )
        )
        st.altair_chart(chart, use_container_width=True)


    with col_grafico2:
        correlacoes = df_raw[colunas_numericas].corr(method='spearman')

        fig, ax = plt.subplots(figsize=(6, 4))
        sbn.heatmap(correlacoes, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

   
    st.write(f"**Coeficiente de determinação médio: {r2_medio:.2f}%**")

#--------------------------------MAIN APP--------------------------------------#
selecoes, btn_simular, btn_limpar = barra_simulador()
tela_principal(selecoes, btn_simular)
graficos()
