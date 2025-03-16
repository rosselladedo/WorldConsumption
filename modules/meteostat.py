import streamlit as st
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from meteostat import Stations, Daily
from datetime import datetime
from utils.functions import leggi_preferiti


def meteostat(cities_df, df_prod_region, df_rinnovabili):
    # Input per la data (st.date_input restituisce un oggetto date)
  start_date_input = st.date_input("Data inizio", datetime(2018, 1, 1).date())
  end_date_input = st.date_input("Data fine", datetime(2018, 12, 31).date())

  # Converti le date in datetime.datetime usando datetime.combine
  start_date = datetime.combine(start_date_input, datetime.min.time())
  end_date = datetime.combine(end_date_input, datetime.min.time())

  # Selezione della città dal menu a tendina
  city_names = cities_df['city'].tolist()
  selected_city = st.selectbox("Seleziona una città", city_names)

  # estraggo la regione in base alla città
  city_data = cities_df[cities_df["city"] == selected_city].iloc[0]
  selected_region = city_data["region"]

  # Trova le coordinate della città selezionata
  lat, lon = city_data['latitude'], city_data['longitude']

  st.write(f"🌍 **Città selezionata:** {selected_city} | **Regione:** {selected_region}")

  # Filtraggio dati per la regione trovata
  region_data = df_prod_region[df_prod_region["Regione"] == selected_region]
  
  #df_rinnovabili_aggregato["Date"] = pd.to_datetime(df_rinnovabili_aggregato["Date"])
  start_year = start_date.year
  st.write(start_year)
  region_aggregated = region_data.groupby("Fonte")["Produzione (GWh)"].sum().reset_index()
  st.write(f"### Produzione di Energia Rinnovabile in {selected_region} ({start_year})")

    # Visualizzazione della produzione di energia per fonte
  fig = px.bar(
    region_aggregated,
    x="Fonte",
    y="Produzione (GWh)",
    title=f"Produzione Energetica per Fonte - {selected_region} ({start_year})",
    color="Fonte",
    labels={"Produzione (GWh)": "GWh"}
  )
  st.plotly_chart(fig)

  # Indicatori chiave sulla produzione energetica
  st.metric("Totale Produzione Rinnovabili", f"{region_data['Produzione (GWh)'].values[0]:,.2f} GWh")

  # Input per latitudine e longitudine
  #lat = st.number_input("Inserisci latitudine", value=41.9028, format="%.4f")
  #lon = st.number_input("Inserisci longitudine", value=12.4964, format="%.4f")

  st.write("Ricerca della stazione più vicina...")
  stations = Stations().nearby(lat, lon)
  station_df = stations.fetch(1)
  if not station_df.empty:
    st.write("Stazione trovata:")
    st.dataframe(station_df)
    station_id = station_df.index[0]
  else:
    st.error("Nessuna stazione trovata!")
    station_id = None

  if station_id:
    data = Daily(station_id, start_date, end_date).fetch()
    if not data.empty:
        st.write("### Dati Meteo")
        st.dataframe(data)
        
        # Verifica se le colonne 'tavg', 'tmin' e 'tmax' sono presenti nei dati
        if 'tavg' in data.columns and 'tmin' in data.columns and 'tmax' in data.columns:
            st.write("#### Andamento della Temperatura Media, Minima e Massima")
            
            # Plot delle 3 colonne (tavg, tmin, tmax) con colori distinti
            fig, ax = plt.subplots(figsize=(10, 6))
            data[['tavg', 'tmin', 'tmax']].plot(ax=ax, color=['blue', 'green', 'red'])
            
            
            # Etichette e titolo
            ax.set_title('Andamento delle Temperature Media, Minima e Massima', fontsize=16)
            ax.set_xlabel('Data', fontsize=12)
            ax.set_ylabel('Temperatura (°C)', fontsize=12)
            
            # Mostra il grafico in Streamlit
            st.pyplot(fig)
        else:
            st.write("Le colonne 'tavg', 'tmin' e 'tmax' non sono disponibili nei dati.")
    else:
        st.error("Nessun dato meteo disponibile per il periodo selezionato.")



#correlazione tra temperatura media e produzione id energia rinnovabile
  st.write("###  Correlazione tra Temperatura Media e Produzione di Energia Rinnovabile")

# Creiamo un DataFrame con la media giornaliera della temperatura e la produzione energetica totale
  production_vs_temp = pd.DataFrame({
                "Data": data.index,
                "Temperatura Media (°C)": data["tavg"],
                "Produzione Totale Rinnovabile (GWh)": region_data["Produzione (GWh)"].values[0]
            })

            # Creazione scatter plot interattivo
  fig_corr = px.scatter(
                production_vs_temp, x="Temperatura Media (°C)", y="Produzione Totale Rinnovabile (GWh)",
                title="Relazione tra Temperatura Media e Produzione Rinnovabile",
                trendline="ols", color_discrete_sequence=["blue"]
            )
  st.plotly_chart(fig_corr)