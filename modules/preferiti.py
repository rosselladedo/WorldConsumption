import streamlit as st
from utils import functions  # Assumendo che le funzioni siano in utils/functions.py

def mostra_preferiti(paesi_disponibili):
    """
    Modulo Streamlit per la gestione dei preferiti
    """
    favorites_file = "utils/preferiti.json"  

    # Leggi i preferiti dal file JSON
    preferiti = functions.leggi_preferiti(favorites_file)

    st.write("### I tuoi preferiti salvati:")
    if len(preferiti) > 0:
        for p in preferiti:
            st.write(f"**{p['nome preferito']}** - Paesi: {', '.join(p['paese'])}, Anni: {p['anno da']} - {p['anno a']}, Energia: {p['energia']}")
    else:
        st.write("Non ci sono preferiti salvati.")

    # Azioni per aggiungere o eliminare preferiti
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
                functions.scrivi_preferiti(favorites_file, preferiti)
                st.success(f"Preferito aggiunto: {nome_preferito}, Paesi: {', '.join(paesi_input)}, Anni: {anno_da_input}-{anno_a_input}, Energia: {energia_input}")
                st.rerun()

    elif azione_preferito == "Elimina un preferito":
        if len(preferiti) > 0:
            preferiti_names = [p['nome preferito'] for p in preferiti]
            selected_preferito = st.selectbox("Scegli un preferito da cancellare", preferiti_names)
            if st.button("Elimina Preferito"):
                functions.elimina_preferito(favorites_file, selected_preferito)
                st.success(f"Preferito '{selected_preferito}' eliminato con successo!")
                st.rerun()
        else:
            st.warning("Non ci sono preferiti salvati.")
