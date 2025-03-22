import streamlit as st
import pandas as pd
from utils.functions import leggi_preferiti

def seleziona_parametri(df, df_fuel, favorites_file,key_prefix="default"):
    """Seleziona i parametri e filtra il dataset in base a preferiti o input manuale"""
    
    preferiti = leggi_preferiti(favorites_file)  # Carica i preferiti salvati

    # 🟢 Se ci sono preferiti, offri due modalità di selezione: preferito o manuale
    if len(preferiti) > 0:
        mode = st.radio(
            "Modalità di selezione dei parametri:",
            ("Preferito esistente", "Inserisci i dati manualmente"),
            key=f"{key_prefix}_selezione_mode"
        )
        
        if mode == "Preferito esistente":
            preferito_selezionato = st.selectbox(
                "Scegli un preferito", 
                [p["nome preferito"] for p in preferiti],
                key=f"{key_prefix}_preferito_select"
            )
            if preferito_selezionato:
                selected_preference = next(p for p in preferiti if p["nome preferito"] == preferito_selezionato)
                selected_countries = selected_preference['paese']  
                selected_fuel = selected_preference['energia']
                selected_years = (selected_preference['anno da'], selected_preference['anno a'])

                # Mostra il riepilogo della selezione
                st.write(f"**Preferito selezionato:** {selected_preference['nome preferito']}")
                st.write(f"🌍 Paesi: {', '.join(selected_countries)}")
                st.write(f"📅 Anni: {selected_years[0]} - {selected_years[1]}")
                st.write(f"⚡ Energia: {selected_fuel}")
    
        # 🔵 Modalità di selezione manuale
    if len(preferiti) == 0 or mode == "Inserisci i dati manualmente":
        st.write("🔧 **Inserisci i dati manualmente**")

        # Selezione dei paesi disponibili nel dataset
        selected_countries = st.multiselect(
            "Seleziona uno o più Paesi", 
            options=sorted(df['country'].unique()),
            key=f"{key_prefix}_countries_multiselect"
        )

        # Selezione del tipo di energia
        selected_fuel = st.selectbox(
            "Seleziona il Tipo di Energia", 
            options=sorted(df['fuel'].unique()),
            key=f"{key_prefix}_fuel_select"
        )

        # Selezione intervallo temporale
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
        selected_years = st.slider(
            "Seleziona intervallo di anni",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            key=f"{key_prefix}_years_slider"
        )
            
    # 🟠 **Filtraggio del dataset in base ai parametri selezionati**
    filtered_dataset = df[
        (df['country'].isin(selected_countries)) &
        (df['fuel'] == selected_fuel) &
        (df['year'].between(selected_years[0], selected_years[1]))
    ]
    
    return filtered_dataset, selected_countries, selected_fuel, selected_years
