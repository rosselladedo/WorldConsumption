import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from prophet import Prophet
from utils.parametri import seleziona_parametri  

def analisi_avanzate(df, df_fuel, favorites_file):
    """
    Funzione per l'analisi avanzata dei dati energetici.
    Usa `parametri.py` per la gestione dei filtri e previsioni.
    """
    filtered_dataset, selected_countries, selected_fuel, selected_years = seleziona_parametri(df, df_fuel, favorites_file)


    """Funzione per analisi avanzate e previsioni"""
    
    if filtered_dataset.empty:
        st.warning("⚠️ Nessun dato disponibile con i parametri selezionati.")
        return
    
    st.write("### 📊 Previsione della Produzione Energetica")
    
    selected_country = st.selectbox("Seleziona il Paese per la previsione", filtered_dataset['country'].unique())

    # Filtrare il dataset
    prediction_df = filtered_dataset[filtered_dataset['country'] == selected_country]
    # 📌 **Selezione del Paese per la Previsione**
    if not filtered_dataset.empty:
        forecast_data = filtered_dataset[
            (filtered_dataset['country'] == selected_country) & 
            (filtered_dataset['fuel'] == selected_fuel)
        ]

        # 📈 **Previsione con Regressione Lineare**
        st.write("### Scenario Futuro - Previsione della Produzione Energetica")
        X = forecast_data['year'].values.reshape(-1, 1)
        y = forecast_data['production'].values
        model = LinearRegression()
        model.fit(X, y)
        future_years = np.array(range(selected_years[1] + 1, selected_years[1] + 6)).reshape(-1, 1)
        predictions = model.predict(future_years)

        fig_pred = px.line(
            x=future_years.flatten(), y=predictions,
            labels={'x': 'Anno', 'y': 'Produzione Energetica'},
            title=f"Previsione della Produzione Energetica Futura per {selected_country}"
        )
        st.plotly_chart(fig_pred)

        # 📌 **Previsione con Prophet**
        st.header("Previsione della Produzione Energetica con Prophet")
        if not forecast_data.empty:
            forecast_data['date'] = pd.to_datetime(forecast_data['year'].astype(str) + '-01-01')
            forecast_data = forecast_data[['date', 'production']].rename(columns={'date': 'ds', 'production': 'y'})
            model = Prophet()
            model.fit(forecast_data)
            future = model.make_future_dataframe(periods=1825)
            forecast = model.predict(future)

            st.write("### Previsione della Produzione Energetica con Prophet")
            fig = model.plot(forecast)
            st.pyplot(fig)

            csv = forecast.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Scarica CSV", csv, "previsione_prophet.csv", "text/csv")

        # 📌 **Clustering con K-Means**
        st.header("Clustering dei Paesi con K-Means")
        clustering_data = filtered_dataset[filtered_dataset['fuel'] == selected_fuel]
        clustering_data_grouped = clustering_data.groupby('country').agg({'production': 'sum'}).reset_index()

        k = st.slider("Seleziona il numero di cluster (K)", min_value=2, max_value=10, value=3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        clustering_data_grouped['cluster'] = kmeans.fit_predict(clustering_data_grouped[['production']])

        st.write(f"### Risultati del Clustering ({k} cluster)")
        st.dataframe(clustering_data_grouped[['country', 'production', 'cluster']])

        csv = clustering_data_grouped.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Scarica CSV", csv, "clustering.csv", "text/csv")

        fig_cluster = px.scatter(
            clustering_data_grouped, x='country', y='production', color='cluster',
            title="Clustering dei Paesi per Produzione Energetica"
        )
        st.plotly_chart(fig_cluster)
    else:
        st.warning("⚠️ Nessun dato disponibile per la previsione o il clustering.")
