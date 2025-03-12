import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import json
import requests
from datetime import datetime
from meteostat import Stations, Daily
from streamlit_lottie import st_lottie


# Import funzionalità avanzate
from prophet import Prophet       
from sklearn.cluster import KMeans

# --- Caricamento dei dataset
df_fuel = pd.read_csv('EnergyDecription.csv')
df = pd.read_csv('WorldConsumption_Prepdataset.csv')
df_prod_region= pd.read_csv("Produzione_elettrica_da_fonti_rinnovabili_regioni.csv")
#coordinates_df= pd.read_csv("cities_coordinates.csv")
cities_df= pd.read_csv("citta_italiane.csv")


# --- File JSON preferiti
favorites_file = 'preferiti.json' #preferiti



#funzione per importare json lottie
def load_lottie_file(file_path: str):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)  # Legge il file JSON
        print("File JSON caricato correttamente!")  # Debug
        return data
    except Exception as e:
        print(f"Errore durante il caricamento del file JSON: {e}")  # Mostra eventuali errori
        return None

# funzione per caricare JSON da GitHub
def load_lottie_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Errore nel caricamento del JSON.")
        return None

# Funzione per caricare l'immagine da URL
def load_image_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        st.error("Errore nel caricamento dell'immagine.")
        return None

#caricamento animazioni pagine
lottie_earth = load_lottie_url('https://raw.githubusercontent.com/rosselladedo/WorldConsumption/refs/heads/main/world_Page.json') #animazione mondo pagina iniziale
lottie_sun = load_lottie_url('https://raw.githubusercontent.com/rosselladedo/WorldConsumption/refs/heads/main/sun.json') #animazione sole pagine
lottie_spin = load_lottie_url('https://raw.githubusercontent.com/rosselladedo/WorldConsumption/refs/heads/main/world_spin.json') #animazione mondo pagine
lottie_clouds = load_lottie_url('https://raw.githubusercontent.com/rosselladedo/WorldConsumption/refs/heads/main/clouds.json') #animazione mondo pagine
lottie_wind = load_lottie_url('https://raw.githubusercontent.com/rosselladedo/WorldConsumption/refs/heads/main/wind.json') #animazione mondo pagine
lottie_click = load_lottie_url('https://raw.githubusercontent.com/rosselladedo/WorldConsumption/refs/heads/main/click.json') #animazione mondo pagine
logo=load_image_from_url('https://raw.githubusercontent.com/rosselladedo/WorldConsumption/main/logo_dashboard.jpg')

# --- Funzioni per la gestione dei preferiti
def leggi_preferiti(favorites_file):
    try:
        with open(favorites_file, 'r') as json_file:
            preferiti = json.load(json_file)
        return preferiti["preferiti"]
    except FileNotFoundError:
        return []

def scrivi_preferiti(favorites_file, preferiti):
    try:
        with open(favorites_file, 'w', encoding='utf-8') as json_file:
            json.dump({"preferiti": preferiti}, json_file, indent=4, ensure_ascii=False)
        print(f"Preferiti salvati con successo in {favorites_file}")
    except Exception as e:
        print(f"Errore nel salvataggio dei preferiti: {e}")

def elimina_preferito(favorites_file, preferito_da_eliminare):
    preferiti_list = leggi_preferiti(favorites_file)
    preferiti_list = [p for p in preferiti_list if p['nome preferito'] != preferito_da_eliminare]
    scrivi_preferiti(favorites_file, preferiti_list)

# --- Estrai la lista unica dei paesi dal dataset
paesi_disponibili = df['country'].unique().tolist()

# --- Carica i preferiti
preferiti = leggi_preferiti(favorites_file)

# --- Menu di navigazione multipage nella sidebar
with st.sidebar:
    st_lottie(lottie_click, speed=1, width=250, height=250, key="sidebar_animation")
st.sidebar.title(" Scegli la pagina")
page = st.sidebar.selectbox(
    " ",
    ["🌎Home", "✅Preferiti", "📊Analisi", "⚒️Funzionalità Avanzate", "🌫️Meteostat"]
)

# -------------------- PAGINA HOME --------------------
if page == "🌎Home":
    
    #st.snow()
    # Mostra l'animazione
    st_lottie(lottie_earth, speed=1, width=700, height=300, key="earth_animation")

    
    st.markdown("""
      <h3 style='text-align: center;'>Benvenuto nella World Energy dashboard </h3>
      <p style='text-align: center;'>Usa il menu laterale per navigare tra le diverse sezioni. Scopri l'andamento dei consumi energetici e le tendenze future</p>
      """, unsafe_allow_html=True) 
    
  

    st.image(logo, use_container_width=True)


     

# -------------------- PAGINA PREFERITI --------------------
elif page == "✅Preferiti":
    col1, col2 = st.columns([1, 1])
    with col2:
      st_lottie(lottie_sun, speed=1, width=300, height=300, key="sun_animation")
    with col1:
      st.title(" ")
      st.title("Gestione Preferiti")
    
    st.write("### I tuoi preferiti salvati:")
    if len(preferiti) > 0:
        for p in preferiti:
            st.write(f"**{p['nome preferito']}** - Paesi: {', '.join(p['paese'])}, Anni: {p['anno da']} - {p['anno a']}, Energia: {p['energia']}")
    else:
        st.write("Non ci sono preferiti salvati.")
    
    # Azioni per aggiungere o eliminare preferiti (il blocco per selezionare un preferito esistente è rimosso da qui)
    azione_preferito = st.selectbox(
        "Scegli un'azione",
        ["Aggiungi un nuovo preferito", "Elimina un preferito"]
    )
    
    if azione_preferito == "Aggiungi un nuovo preferito":
        nome_preferito = st.text_input("Nome Preferito")
        paesi_input = st.multiselect("Seleziona il/i Paesi", paesi_disponibili)
        anno_da_input = st.number_input("Anno da", min_value=1900, max_value=2022)
        anno_a_input = st.number_input("Anno a", min_value=1900, max_value=2022)
        energia_input = st.selectbox("Tipo di Energia", ["biofuel", "coal", "gas", "hydro", "nuclear", "oil", "other_renewable", "solar", "wind"])
        
        if st.button("Salva Preferito"):
            if not paesi_input:
                st.warning("Devi selezionare almeno un paese.")
            else:
                nuovo_preferito = {
                    "nome preferito": nome_preferito,
                    "paese": paesi_input,
                    "anno da": anno_da_input,
                    "anno a": anno_a_input,
                    "energia": energia_input
                }
                preferiti.append(nuovo_preferito)
                scrivi_preferiti(favorites_file, preferiti)
                st.success(f"Preferito aggiunto: {nome_preferito}, Paesi: {', '.join(paesi_input)}, Anni: {anno_da_input}-{anno_a_input}, Energia: {energia_input}")
                st.rerun()
                
    elif azione_preferito == "Elimina un preferito":
        if len(preferiti) > 0:
            preferiti_names = [p['nome preferito'] for p in preferiti]
            selected_preferito = st.selectbox("Scegli un preferito da cancellare", preferiti_names)
            if st.button("Elimina Preferito"):
                elimina_preferito(favorites_file, selected_preferito)
                st.success(f"Preferito '{selected_preferito}' eliminato con successo!")
                st.rerun()
        else:
            st.warning("Non ci sono preferiti salvati.")

    col1, col2 = st.columns([6, 1])
    with col2:
      st.image(logo, use_container_width=True)
    with col1:
      st.title(" ")

# -------------------- PAGINA ANALISI --------------------
if page == "📊Analisi":
    col1, col2 = st.columns([1, 1])
    with col2:
      st_lottie(lottie_clouds, speed=1, width=300, height=300, key="clouds_animation")
    with col1:
      st.title(" ")
      st.title("Analisi dei Dati")

    # Se ci sono preferiti salvati, offri due modalità tramite un radio button
    if len(preferiti) > 0:
        mode = st.radio(
            "Modalità di selezione dei parametri:",
            ("Preferito esistente", "Inserisci i dati manualmente")
        )
        if mode == "Preferito esistente":
            preferito_selezionato = st.selectbox(
                "Scegli un preferito", 
                [p["nome preferito"] for p in preferiti]
            )
            if preferito_selezionato:
                selected_preference = next(p for p in preferiti if p["nome preferito"] == preferito_selezionato)
                # Estrai i parametri dal preferito
                selected_countries = selected_preference['paese']  # deve essere una lista
                anno_da_selezionato = selected_preference['anno da']
                anno_a_selezionato = selected_preference['anno a']
                selected_fuel = selected_preference['energia']
                selected_years = (anno_da_selezionato, anno_a_selezionato)
                st.write(f"Preferito selezionato: **{selected_preference['nome preferito']}**")
                st.write(f"Paesi: {', '.join(selected_countries)}")
                st.write(f"Anni: {anno_da_selezionato} - {anno_a_selezionato}")
                st.write(f"Energia: {selected_fuel}")
        else:  # Modalità manuale
            selected_countries = st.multiselect("Seleziona uno o più Paesi", df['country'].unique())
            selected_fuel = st.selectbox("Seleziona il Tipo di Energia", df['fuel'].unique())
            min_year = int(df['year'].min())
            max_year = int(df['year'].max())
            selected_years = st.slider("Seleziona intervallo di anni", min_year, max_year, (min_year, max_year))
    else:
        # Se non ci sono preferiti, usa direttamente la modalità manuale
        st.write("Nessun preferito salvato. Inserisci i dati manualmente.")
        selected_countries = st.multiselect("Seleziona uno o più Paesi", df['country'].unique())
        selected_fuel = st.selectbox("Seleziona il Tipo di Energia", df['fuel'].unique())
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
        selected_years = st.slider("Seleziona intervallo di anni", min_year, max_year, (min_year, max_year))
    
    try:
      fuel_description = df_fuel[df_fuel['fuel'] == selected_fuel]['description'].values[0]
      st.write(f"### Descrizione di {selected_fuel.capitalize()}")
      st.write(fuel_description)
    except IndexError:
        st.warning("Descrizione non trovata per il fuel selezionato.")

    
    # --- Crea il dataset filtrato in base ai parametri definiti
    filtered_dataset = df[
        (df['country'].isin(selected_countries)) &
        (df['fuel'] == selected_fuel) &
        (df['year'].between(selected_years[0], selected_years[1]))
    ]
    st.write("### Dati Filtrati")
    st.dataframe(filtered_dataset)
    
    if not filtered_dataset.empty:
        # 1. Analisi della Crescita
        filtered_df_growth = filtered_dataset.groupby('country')['production'].agg(['min', 'max']).reset_index()
        filtered_df_growth['growth_rate'] = (filtered_df_growth['max'] - filtered_df_growth['min']) / filtered_df_growth['min'] * 100
        st.write("### Tasso di Crescita della Produzione Energetica")
        st.dataframe(filtered_df_growth[['country', 'growth_rate']])
        
        # 2. Redditività dell'Investimento (normalizzati)
        if 'normalized_production_per_gdp' in df.columns:
            gdp_comparison = filtered_dataset.groupby('country').agg({
                'normalized_production': 'sum', 
                'normalized_gdp': 'sum', 
                'normalized_production_per_gdp': 'sum'
            }).reset_index()
            st.write("### Redditività dell'Investimento (Produzione / PIL Normalizzati)")
            st.dataframe(gdp_comparison[['country', 'normalized_production_per_gdp']])
        else:
            st.write("### Redditività dell'Investimento")
            st.warning("Non è possibile calcolare la redditività perché i dati normalizzati non sono disponibili.")
        
        # 3. Matrice di Correlazione
        columns_to_check = ['production', 'gdp', 'population', 'per_capita']
        existing_columns = [col for col in columns_to_check if col in filtered_dataset.columns]
        if len(existing_columns) == len(columns_to_check):
            correlation_df = filtered_dataset[existing_columns]
            corr_matrix = correlation_df.corr()
            st.write("### Matrice di Correlazione")
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.write("### Matrice di Correlazione")
            st.warning("Non tutte le colonne richieste sono disponibili.")
        
        # 4. Ranking Paesi
        ranking_df = filtered_df_growth.sort_values(by='growth_rate', ascending=False)
        st.write("### Ranking dei Paesi per Crescita")
        st.dataframe(ranking_df[['country', 'growth_rate']])
        

        #Grafico a barre: Produzione Energetica per Paese
        st.write("### Produzione Energetica per Paese")
        filtered_prod = df[df['country'].isin(selected_countries) & (df['fuel'] == selected_fuel) & (df['year'].between(selected_years[0], selected_years[1]))]
        fig_prod = px.bar(
          filtered_prod,
          x='country', y='production', color='country',
          title="Produzione Energetica per Paese",
          labels={'production': 'Produzione Energetica', 'country': 'Paese'},
          color_discrete_sequence=px.colors.qualitative.Set2
    )
        
        st.plotly_chart(fig_prod)
        #st.write("Paesi selezionati:", selected_countries)

        #Grafico a dispersione: Correlazione tra Produzione Energetica e Popolazione
        st.write("### Correlazione tra Produzione Energetica e Popolazione")
        scatter_df = df[
            (df['year'].between(selected_years[0], selected_years[1])) &
            (df['country'].isin(selected_countries))
        ]
        scatter_plot = px.scatter(
            scatter_df, 
            x='population', 
            y='production', 
            color='country', 
            title="Correlazione tra Produzione Energetica e Popolazione",
            labels={'production': 'Produzione Energetica', 'population': 'Popolazione'}
        )
        st.plotly_chart(scatter_plot)

        #Box plot: Distribuzione della Produzione Energetica
        distribution_prod = df[df['country'].isin(selected_countries) & (df['fuel'] == selected_fuel) & (df['year'].between(selected_years[0], selected_years[1]))]
        distribution_prod = distribution_prod.groupby('country')['production'].sum().reset_index()


        if not distribution_prod.empty:
          fig_dist = px.pie(
              distribution_prod, values='production', names='country',
              title="Distribuzione della Produzione Energetica per Paese",
              color_discrete_sequence=px.colors.qualitative.Set3
          )
          st.plotly_chart(fig_dist)
        else:
          st.write("Nessun dato disponibile per la distribuzione della produzione energetica.")
                  

    # --- Grafico a dispersione (scatter plot) Produzione vs PIL con dimensione = popolazione
    # Assicurati che la colonna 'gdp' e 'population' siano disponibili nel dataset filtrato
    if 'gdp' in filtered_prod.columns and 'population' in filtered_prod.columns:
        fig_scatter = px.scatter(
            filtered_prod,
            x='gdp',
            y='production',
            size='population',
            color='country',
            title="Produzione vs PIL (dimensione = Popolazione)",
            hover_data=['country']
        )
        st.plotly_chart(fig_scatter)
    else:
        st.write("Colonne 'gdp' o 'population' non disponibili per il grafico scatter.")

    # --- Box Plot della distribuzione della produzione per Tipo di Energia
    fig_box = px.box(
        filtered_prod,
        x='fuel',
        y='production',
        title="Distribuzione della Produzione per Tipo di Energia"
    )
    st.plotly_chart(fig_box)


    # --- Grafico Radar per confrontare variabili chiave per ogni Paese
    # Raggruppa per paese e calcola i totali/medie delle variabili interessate
    radar_data = filtered_prod.groupby('country').agg({
        'production': 'sum',
        'gdp': 'sum',
        'population': 'mean',
        'per_capita': 'mean'
    }).reset_index()

    # Seleziona le categorie da mostrare (potresti normalizzarle se i range sono molto diversi)
    categories = ['production', 'gdp', 'population', 'per_capita']

    # Indicatori chiave
    st.write("### Indicatori Chiave")
    st.metric("Produzione Totale", f"{filtered_prod['production'].sum():,.2f} MWh")
    st.metric("Produzione Media", f"{filtered_prod['production'].mean():,.2f} MWh")
    st.metric("Massima Produzione", f"{filtered_prod['production'].max():,.2f} MWh")

    # Mappa della produzione energetica globale
    if 'iso_code' in df.columns:
        map_df = df[(df['fuel'] == selected_fuel) & df['year'].between(selected_years[0], selected_years[1])]
        fig_map = px.choropleth(map_df, locations='iso_code', color='production',
                                hover_name='country', title=f"Produzione di {selected_fuel} nel Mondo",
                                color_continuous_scale='viridis', projection='natural earth')
        st.plotly_chart(fig_map)

    import plotly.graph_objects as go
    fig_radar = go.Figure()
    for _, row in radar_data.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row[cat] for cat in categories],
            theta=categories,
            fill='toself',
            name=row['country']
        ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                # Se i valori sono molto diversi, potresti voler normalizzare o usare un range adatto
                # range=[0, max(radar_data[categories].max())] 
            )
        ),
        showlegend=True,
        title="Confronto Variabili per Paese"
    )
    st.plotly_chart(fig_radar)



    # Tabella con i dati filtrati
    st.write("### Dati Filtrati")
    st.dataframe(filtered_prod)

    # Download dei dati filtrati
    csv = filtered_prod.to_csv(index=False).encode('utf-8')
    st.download_button("Scarica CSV", csv, "dati_filtrati.csv", "text/csv")




    col1, col2 = st.columns([6, 1])
    with col2:
      st.image(logo, use_container_width=True)
    with col1:
      st.title(" ")        
# -------------------- PAGINA FUNZIONALITÀ AVANZATE --------------------
elif page == "⚒️Funzionalità Avanzate": 
    col1, col2 = st.columns([2, 1])
    with col2:
      st_lottie(lottie_spin, speed=1, width=200, height=200, key="spin_animation")
    with col1:
     st.title("Funzionalità Avanzate")
    

    # Selezione dei parametri tramite preferito
    # Se ci sono preferiti salvati, offri due modalità tramite un radio button
    if len(preferiti) > 0:
        mode = st.radio(
            "Modalità di selezione dei parametri:",
            ("Preferito esistente", "Inserisci i dati manualmente")
        )
        if mode == "Preferito esistente":
            preferito_selezionato = st.selectbox(
                "Scegli un preferito", 
                [p["nome preferito"] for p in preferiti]
            )
            if preferito_selezionato:
                selected_preference = next(p for p in preferiti if p["nome preferito"] == preferito_selezionato)
                # Estrai i parametri dal preferito
                selected_countries = selected_preference['paese']  # deve essere una lista
                anno_da_selezionato = selected_preference['anno da']
                anno_a_selezionato = selected_preference['anno a']
                selected_fuel = selected_preference['energia']
                selected_years = (anno_da_selezionato, anno_a_selezionato)
                st.write(f"Preferito selezionato: **{selected_preference['nome preferito']}**")
                st.write(f"Paesi: {', '.join(selected_countries)}")
                st.write(f"Anni: {anno_da_selezionato} - {anno_a_selezionato}")
                st.write(f"Energia: {selected_fuel}")
        else:  # Modalità manuale
            selected_countries = st.multiselect("Seleziona uno o più Paesi", df['country'].unique())
            selected_fuel = st.selectbox("Seleziona il Tipo di Energia", df['fuel'].unique())
            min_year = int(df['year'].min())
            max_year = int(df['year'].max())
            selected_years = st.slider("Seleziona intervallo di anni", min_year, max_year, (min_year, max_year))
    else:
        # Se non ci sono preferiti, usa direttamente la modalità manuale
        st.write("Nessun preferito salvato. Inserisci i dati manualmente.")
        selected_countries = st.multiselect("Seleziona uno o più Paesi", df['country'].unique())
        selected_fuel = st.selectbox("Seleziona il Tipo di Energia", df['fuel'].unique())
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
        selected_years = st.slider("Seleziona intervallo di anni", min_year, max_year, (min_year, max_year))
    
    try:
      fuel_description = df_fuel[df_fuel['fuel'] == selected_fuel]['description'].values[0]
      st.write(f"### Descrizione di {selected_fuel.capitalize()}")
      st.write(fuel_description)
    except IndexError:
        st.warning("Descrizione non trovata per il fuel selezionato.")

    
    # --- Crea il dataset filtrato in base ai parametri definiti
    filtered_dataset = df[
        (df['country'].isin(selected_countries)) &
        (df['fuel'] == selected_fuel) &
        (df['year'].between(selected_years[0], selected_years[1]))
    ]
    st.write("### Dati Filtrati ")
    st.dataframe(filtered_dataset)

    # Filtra il dataset per il paese e il carburante selezionato
    selected_country = st.selectbox('Seleziona il Paese per la previsione', selected_countries)
    #st.write(selected_country)
    forecast_data = filtered_dataset[
        (filtered_dataset['country']==selected_country) & 
        (filtered_dataset['fuel'] == selected_fuel)
    ]
    
    
    # Funzionalità 2: Previsione della Produzione Energetica


    st.write("### Scenario Futuro - Previsione della Produzione Energetica")
    country_for_prediction = st.selectbox('Seleziona il Paese per la previsione', selected_countries, key=selected_country)
    st.write(country_for_prediction)

    # Filtrare i dati per il paese selezionato
    prediction_df = df[(df['country'] == country_for_prediction) & 
                        (df['fuel'] == selected_fuel) & 
                        (df['year'].between(selected_years[0], selected_years[1]))]

    # Previsione con regressione lineare
    X = prediction_df['year'].values.reshape(-1, 1)
    y = prediction_df['production'].values
    model = LinearRegression()
    model.fit(X, y)

    # Previsione per gli anni futuri
    future_years = np.array(range(selected_years[1] + 1, selected_years[1] + 6)).reshape(-1, 1)
    predictions = model.predict(future_years)

    # Visualizzazione del grafico delle previsioni
    fig_pred = px.line(x=future_years.flatten(), y=predictions, labels={'x': 'Anno', 'y': 'Produzione Energetica'},
                        title=f"Previsione della Produzione Energetica Futura per {country_for_prediction}")
    st.plotly_chart(fig_pred)



    # Funzionalità 2: Previsione con Prophet
    st.header("Previsione della Produzione Energetica con Prophet")
    

    if not forecast_data.empty:
        st.write(f"### Dati per la previsione di {selected_country} con {selected_fuel}")
        st.dataframe(forecast_data[['year', 'production']])
        forecast_data['date'] = pd.to_datetime(df['year'].astype(str) + '-01-01')
        st.write(forecast_data)



        # Preprocessamento dei dati per Prophet
        forecast_data = forecast_data[['date', 'production']]
        st.write(forecast_data)
        forecast_data = forecast_data.rename(columns={'date': 'ds', 'production': 'y'})  # Prophet richiede colonne 'ds' e 'y'
        
        model = Prophet()
        model.fit(forecast_data)

    
        # Previsione per i prossimi 5 anni
        future = model.make_future_dataframe(periods=1825)
        forecast = model.predict(future)

        # Imposta la previsione per partire dalla data minima nel dataset
        min_date = forecast_data['ds'].min()  # Ottieni la data minima dal dataset
        future['ds'] = future['ds'].apply(lambda x: max(x, min_date))  # Assicurati che parta dalla data minima
        
        # Grafico della previsione
        st.write("### Previsione della Produzione Energetica")
        fig = model.plot(forecast)
        fig2 = model.plot_components(forecast)
        st.pyplot(fig)

    else:
        st.warning("Nessun dato disponibile per la previsione di questo paese e carburante.")

    #scarica i dati della previsione
    csv = forecast.to_csv(index=False).encode('utf-8')
    st.download_button("Scarica CSV", csv, "dati_filtrati.csv", "text/csv", key='download_button_1')

    # Funzionalità 2: Clustering con K-Means
    st.header("Clustering dei Paesi con K-Means")

    # Filtra il dataset per il carburante selezionato
    clustering_data = filtered_dataset[filtered_dataset['fuel'] == selected_fuel]

    # Raggruppa i dati per paese e calcola la produzione totale
    clustering_data_grouped = clustering_data.groupby('country').agg({'production': 'sum'}).reset_index()

    # Applica K-Means per il clustering
    k = st.slider("Seleziona il numero di cluster (K)", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clustering_data_grouped['cluster'] = kmeans.fit_predict(clustering_data_grouped[['production']])

    # Mostra i risultati del clustering
    st.write(f"### Risultati del Clustering ({k} cluster)")
    st.dataframe(clustering_data_grouped[['country', 'production', 'cluster']])

    # Download del risultato del clustering
    csv = clustering_data_grouped.to_csv(index=False).encode('utf-8')
    st.download_button("Scarica CSV", csv, "dati_filtrati.csv", "text/csv", key='download_button_2')

    # Grafico dei cluster
    fig_cluster = px.scatter(clustering_data_grouped, x='country', y='production', color='cluster', title="Clustering dei Paesi per Produzione Energetica")
    st.plotly_chart(fig_cluster)





    col1, col2 = st.columns([6, 1])
    with col2:
      st.image(logo, use_container_width=True)
    with col1:
      st.title(" ")



# -------------------- PAGINA METEOSTAT --------------------
elif page == "🌫️Meteostat":
  col1, col2 = st.columns([2, 1])
  with col2:
      st_lottie(lottie_wind, speed=1, width=200, height=200, key="wind_animation")
  with col1:
      st.title("Funzionalità MeteoStat")

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
  region_data = df_prod_region[df_prod_region["Regioni"] == selected_region]
  
    # Visualizzazione della produzione di energia per fonte
  st.write(f"### Produzione di Energia Rinnovabile in {selected_region}")
  fig = px.bar(
        region_data.melt(id_vars=["Regioni"], var_name="Fonte", value_name="Produzione"),
        x="Fonte", y="Produzione", title=f"Produzione Energetica per Fonte - {selected_region}",
        color="Fonte", labels={"Produzione": "GWh"}
    )
  st.plotly_chart(fig)

  # Indicatori chiave sulla produzione energetica
  st.metric("Totale Rinnovabili", f"{region_data['Totale rinnov.'].values[0]:,.2f} GWh")
  st.metric("Totale Produzione", f"{region_data['Totale'].values[0]:,.2f} GWh")

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
  st.write("### 🔗 Correlazione tra Temperatura Media e Produzione di Energia Rinnovabile")

# Creiamo un DataFrame con la media giornaliera della temperatura e la produzione energetica totale
  production_vs_temp = pd.DataFrame({
                "Data": data.index,
                "Temperatura Media (°C)": data["tavg"],
                "Produzione Totale Rinnovabile (GWh)": region_data["Totale rinnov."].values[0]
            })

            # Creazione scatter plot interattivo
  fig_corr = px.scatter(
                production_vs_temp, x="Temperatura Media (°C)", y="Produzione Totale Rinnovabile (GWh)",
                title="Relazione tra Temperatura Media e Produzione Rinnovabile",
                trendline="ols", color_discrete_sequence=["blue"]
            )
  st.plotly_chart(fig_corr)

col1, col2 = st.columns([6, 1])
with col2:
      st.image(logo, use_container_width=True)
with col1:
      st.title(" ")
