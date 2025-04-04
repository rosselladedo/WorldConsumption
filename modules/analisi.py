
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

def analizza_dati(filtered_dataset, selected_fuel, df_fuel, selected_years):
    """
    Funzione per l'analisi dei dati energetici.
    Mostra statistiche, visualizzazioni e tendenze sulla produzione energetica.
    """

    if 'description' in df_fuel.columns:
        descrizioni = df_fuel[df_fuel['fuel'] == selected_fuel]['description']
        if not descrizioni.empty:
         st.write(f"### Descrizione di {selected_fuel.capitalize()}")
         st.write(descrizioni.values[0])
        else:
         st.warning("Descrizione non trovata per il fuel selezionato.")
    else:
     st.info("Il dataset delle descrizioni non contiene la colonna 'description'.")



    if not filtered_dataset.empty:
        # 1️⃣ **Analisi della Crescita**
               # 📈 Tasso di Crescita tra gli anni selezionati
        start_year, end_year = selected_years

        st.write(f"### 📈 Tasso di Crescita ({start_year}–{end_year}) per Paese")

        # Filtra i dati per gli anni selezionati
        df_years = filtered_dataset[filtered_dataset['year'].isin([start_year, end_year])]

        # Tiene solo i paesi che hanno entrambi gli anni
        year_counts = df_years.groupby('country')['year'].nunique()
        valid_countries = year_counts[year_counts == 2].index.tolist()
        df_valid = df_years[df_years['country'].isin(valid_countries)]

        # Raggruppa per anno e paese
        prod_by_country_year = df_valid.groupby(['country', 'year'])['production'].sum().reset_index()

        # Calcola produzione per inizio e fine anno
        first_year = prod_by_country_year[prod_by_country_year['year'] == start_year].set_index('country')
        last_year = prod_by_country_year[prod_by_country_year['year'] == end_year].set_index('country')

        # Calcolo della crescita
        growth_df = first_year[['production']].rename(columns={'production': 'start_prod'}).join(
            last_year[['production']].rename(columns={'production': 'end_prod'})
        )

        import numpy as np
        growth_df['growth_%'] = ((growth_df['end_prod'] - growth_df['start_prod']) / growth_df['start_prod']) * 100
        growth_df = growth_df.replace([np.inf, -np.inf], np.nan).dropna()
        growth_df = growth_df.sort_values(by='growth_%', ascending=False)

        st.dataframe(growth_df.reset_index()[['country', 'growth_%']].head(10))



        # 2️⃣ **Redditività dell'Investimento (normalizzati)**
        if 'normalized_production_per_gdp' in filtered_dataset.columns:
            gdp_comparison = filtered_dataset.groupby('country').agg({
                'normalized_production': 'sum', 
                'normalized_gdp': 'sum', 
                'normalized_production_per_gdp': 'sum'
            }).reset_index()
            st.write("### 📊 Redditività dell'Investimento (Produzione / PIL Normalizzati)")
            st.dataframe(gdp_comparison[['country', 'normalized_production_per_gdp']])
        else:
            st.write("### 💰 Redditività dell'Investimento")
            st.warning("Dati normalizzati non disponibili.")

        # 3️⃣ **Matrice di Correlazione**
        columns_to_check = ['production', 'gdp', 'population', 'per_capita']
        existing_columns = [col for col in columns_to_check if col in filtered_dataset.columns]
        if len(existing_columns) == len(columns_to_check):
            correlation_df = filtered_dataset[existing_columns]
            corr_matrix = correlation_df.corr()
            st.write("### 🔗 Matrice di Correlazione")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Non tutte le colonne richieste sono disponibili.")


        

        # 4️⃣ **Ranking Paesi per Crescita**
        filtered_df_growth = filtered_dataset.groupby('country')['production'].agg(['min', 'max']).reset_index()
        filtered_df_growth = filtered_df_growth[filtered_df_growth['min'] > 0]  # evita divisione per zero
        filtered_df_growth['growth_rate'] = (filtered_df_growth['max'] - filtered_df_growth['min']) / filtered_df_growth['min'] * 100
        
        ranking_df = filtered_df_growth.sort_values(by='growth_rate', ascending=False)
        st.write("### 🏆 Ranking dei Paesi per Crescita")
        st.dataframe(ranking_df[['country', 'growth_rate']])

        # 5️⃣ **Grafico a barre: Produzione Energetica per Paese**
        st.write("### ⚡ Produzione Energetica per Paese")
        fig_prod = px.bar(
            filtered_dataset,
            x='country', y='production', color='country',
            title="Produzione Energetica per Paese",
            labels={'production': 'Produzione Energetica', 'country': 'Paese'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_prod)

        # 6️⃣ **Grafico a dispersione: Correlazione tra Produzione Energetica e Popolazione**
        st.write("### 📊 Correlazione tra Produzione Energetica e Popolazione")
        scatter_plot = px.scatter(
            filtered_dataset, 
            x='population', 
            y='production', 
            color='country', 
            title="Produzione vs Popolazione",
            labels={'production': 'Produzione Energetica', 'population': 'Popolazione'}
        )
        st.plotly_chart(scatter_plot)

        # 7️⃣ **Grafico a torta: Distribuzione della Produzione Energetica**
        st.write("### 🔄 Distribuzione della Produzione Energetica")
        distribution_prod = filtered_dataset.groupby('country')['production'].sum().reset_index()
        if not distribution_prod.empty:
            fig_dist = px.pie(
                distribution_prod, values='production', names='country',
                title="Distribuzione della Produzione Energetica",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_dist)
        else:
            st.warning("Nessun dato disponibile.")

        # 8️⃣ **Grafico scatter Produzione vs PIL con dimensione = Popolazione**
        if 'gdp' in filtered_dataset.columns and 'population' in filtered_dataset.columns:
            fig_scatter = px.scatter(
                filtered_dataset,
                x='gdp',
                y='production',
                size='population',
                color='country',
                title="Produzione vs PIL (dimensione = Popolazione)",
                hover_data=['country']
            )
            st.plotly_chart(fig_scatter)
        else:
            st.warning("Dati su PIL o Popolazione mancanti.")

        # 9️⃣ **Box Plot: Distribuzione della Produzione per Tipo di Energia**
        fig_box = px.box(
            filtered_dataset,
            x='fuel',
            y='production',
            title="Distribuzione della Produzione per Tipo di Energia"
        )
        st.plotly_chart(fig_box)

        # 🔟 **Grafico Radar per confrontare i Paesi**
        radar_data = filtered_dataset.groupby('country').agg({
            'production': 'sum',
            'gdp': 'sum',
            'population': 'mean',
            'per_capita': 'mean'
        }).reset_index()

        categories = ['production', 'gdp', 'population', 'per_capita']
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
                    visible=True
                )
            ),
            showlegend=True,
            title="Confronto Variabili per Paese"
        )
        st.plotly_chart(fig_radar)

        # 1️⃣1️⃣ Mappa mondiale della produzione energetica
        st.write("### 🗺️ Mappa Mondiale della Produzione Energetica")

        # Raggruppamento per paese
        map_data = filtered_dataset.groupby(['iso_code', 'country'])['production'].sum().reset_index()

        # Generazione della mappa
        fig_map = px.choropleth(
            map_data,
            locations="iso_code",
            color="production",
            hover_name="country",
            color_continuous_scale=px.colors.sequential.YlOrRd,
            title="Produzione Energetica Totale per Paese"
        )

        st.plotly_chart(fig_map)


        # 🔽 **Tabella Dati Filtrati + Download CSV**
        st.write("### 📋 Dati Filtrati")
        st.dataframe(filtered_dataset)

        csv = filtered_dataset.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Scarica CSV", csv, "dati_filtrati.csv", "text/csv")

