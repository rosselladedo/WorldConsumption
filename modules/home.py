import streamlit as st
def home():    
    st.markdown(""" 
        <style>
            .justified-text {
                text-align: justify;
                max-width: 620px;  /* Limita la larghezza per una migliore leggibilità */
                margin: auto;  /* Centra il testo */
                padding: 10px; /* Riduce gli spazi interni */
            }

        </style>

        <div class="justified-text">
            <h3 style='text-align: center;'>BENVENUTO NELLA WORLD ENERGY DASHBOARD</h3>
            <br>
            <p>
            Questa dashboard interattiva è stata sviluppata per analizzare e visualizzare i dati sulla produzione e il consumo di energia a livello globale e regionale. Grazie agli strumenti di analisi avanzati, offre una panoramica dettagliata sull'evoluzione delle diverse fonti energetiche, permettendo di comprendere le tendenze e i cambiamenti nel settore.
            Attraverso questa piattaforma, è possibile esplorare i consumi energetici e confrontare l'andamento della produzione per diverse fonti, come idroelettrico, eolico, solare, nucleare e molte altre. L’integrazione con dati meteorologici consente inoltre di valutare l’influenza delle condizioni climatiche sulla produzione di energia rinnovabile.
            </p>
        </div>
        """, unsafe_allow_html=True)
