import streamlit as st
import pandas as pd
import plotly.express as px
import base64
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
from modules.home import home
from modules.preferiti import mostra_preferiti 
from utils import functions, parametri
from modules import analisi
from modules.analisi_avanzate import analisi_avanzate
from modules.meteostat import meteostat

# Import funzionalità avanzate
from prophet import Prophet       
from sklearn.cluster import KMeans

# --- Caricamento dei dataset
df_fuel = pd.read_csv('data/EnergyDecription.csv')
df = pd.read_csv('data/WorldConsumption_Prepdataset.csv')
df_prod_region= pd.read_csv("data/Produzione_rinnovabile_regioni212223.csv")
cities_df= pd.read_csv("data/citta_italiane.csv")
df_rinnovabili_aggregato= pd.read_csv("data/dataset_giornaliero_aggregato.csv")


# --- File JSON preferiti
favorites_file = 'utils/preferiti.json' #preferiti



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
lottie_earth = load_lottie_url('https://raw.githubusercontent.com/rosselladedo/WorldConsumption/refs/heads/main/animations/world_Page.json') #animazione mondo pagina iniziale
lottie_sun = load_lottie_url('https://raw.githubusercontent.com/rosselladedo/WorldConsumption/refs/heads/main/animations/sun.json') #animazione sole pagine
lottie_spin = load_lottie_url('https://raw.githubusercontent.com/rosselladedo/WorldConsumption/refs/heads/main/animations/world_spin.json') #animazione mondo pagine
lottie_clouds = load_lottie_url('https://raw.githubusercontent.com/rosselladedo/WorldConsumption/refs/heads/main/animations/clouds.json') #animazione mondo pagine
lottie_wind = load_lottie_url('https://raw.githubusercontent.com/rosselladedo/WorldConsumption/refs/heads/main/animations/wind.json') #animazione mondo pagine
lottie_click = load_lottie_url('https://raw.githubusercontent.com/rosselladedo/WorldConsumption/refs/heads/main/animations/click.json') #animazione mondo pagine
logo=load_image_from_url('https://raw.githubusercontent.com/rosselladedo/WorldConsumption/main/utils/logo_dashboard.jpg')



# --- Estrai la lista unica dei paesi dal dataset
paesi_disponibili = df['country'].unique().tolist()

# --- Carica i preferiti
preferiti = functions.leggi_preferiti(favorites_file)

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

    home()
    

# -------------------- PAGINA PREFERITI --------------------
elif page == "✅Preferiti":
    col1, col2 = st.columns([1, 3])
    with col1:
      st_lottie(lottie_sun, speed=1, width=100, height=100, key="sun_animation")
    with col2:
      st.title("Gestione Preferiti")
    mostra_preferiti(paesi_disponibili)


# -------------------- PAGINA ANALISI --------------------
if page == "📊Analisi":
    col1, col2 = st.columns([1, 3])
    with col1:
      st_lottie(lottie_clouds, speed=1, width=100, height=100, key="clouds_animation")
    with col2:
      st.title("Analisi dei Dati")

    filtered_dataset, selected_countries, selected_fuel, selected_years= parametri.seleziona_parametri(df, df_fuel, favorites_file,key_prefix="analisi")
    analisi.analizza_dati(filtered_dataset, selected_fuel, df_fuel)
    
# -------------------- PAGINA FUNZIONALITÀ AVANZATE --------------------
elif page == "⚒️Funzionalità Avanzate": 
    col1, col2 = st.columns([1, 4])
    with col1:
      st_lottie(lottie_spin, speed=1, width=100, height=100, key="spin_animation")
    with col2:
     st.title("Funzionalità Avanzate")
    
    analisi_avanzate(df, df_fuel, favorites_file)
  


# -------------------- PAGINA METEOSTAT --------------------
elif page == "🌫️Meteostat":
  col1, col2 = st.columns([1, 3])
  with col1:
      st_lottie(lottie_wind, speed=1, width=100, height=100, key="wind_animation")
  with col2:
      st.title("Funzionalità MeteoStat")

  meteostat(cities_df, df_prod_region, df_rinnovabili_aggregato)

#logo dashboard

logo_url='https://raw.githubusercontent.com/rosselladedo/WorldConsumption/main/utils/logo_dashboard.jpg'
st.markdown(""" 
    <style>
        .footer-line {
            width: 50%;  /* Larghezza ridotta per rispettare i margini */
            height: 1px;  /* Spessore sottile */
            background-color: grey;
            margin: 20px auto;  /* Distanza dal contenuto e centratura */
            opacity: 0.7; /* Leggera trasparenza per un effetto più elegante */
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="footer-line"></div>  <!-- Riga sottile grigia -->
    <div style="display: flex; justify-content: center;">
        <img src="{logo_url}" style="width: 150px;">
    </div>
    """,
    unsafe_allow_html=True
)


