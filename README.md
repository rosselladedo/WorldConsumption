🌍 World Energy Dashboard
Una dashboard interattiva sviluppata in Python con Streamlit per l’analisi e la visualizzazione della produzione energetica mondiale, con particolare attenzione all'integrazione di dati storici, fonti rinnovabili, variabili meteorologiche e funzionalità avanzate di previsione.

📌 Obiettivi del progetto
Questo progetto nasce come parte di una tesi magistrale in Ingegneria Informatica e ha l’obiettivo di:
Analizzare la produzione energetica globale e per singolo paese;
Fornire strumenti visivi per il confronto temporale e geografico;
Permettere previsioni sull’andamento della produzione energetica;
Integrare dati esterni (Meteostat, descrizioni energetiche, fonti regionali);
Offrire un'interfaccia intuitiva accessibile anche a utenti non tecnici.

------📊 Link all' app 
https://worldconsumption.streamlit.app/

------⚙️ Tecnologie utilizzate
Python 3.11+ -> linguaggio utilizzato
Streamlit -> per la creazione della web app
Plotly, Seaborn, Matplotlib -> per la visualizzazione dei dati
Pandas & Numpy -> per l’elaborazione dei dataset
Prophet -> per le previsioni
scikit-learn -> clustering con K-Means
Meteostat API ->per i dati meteorologici

-----📁 Struttura del progetto
bash
Copia
Modifica
📂 WorldConsumption/
│
├── WorldConsumption.py          # File principale per l'avvio della dashboard
│
├── 📂 data/                     # Dataset utilizzati
│   ├── WorldConsumption_Prepdataset.csv
│   ├── EnergyDecription.csv
│   ├── citta_italiane.csv
│   ├── dataset_giornaliero_aggregato.csv
│   └── Produzione_elettrica_da_fonti_rinnovabili_regioni.csv
│
├── 📂 utils/                    # Funzioni di utilità
│   ├── functions.py            # Funzioni per la gestione preferiti e file
│   └── parametri.py            # Gestione dei filtri e parametri dinamici
│
├── 📂 modules/                 # Moduli della dashboard
│   ├── home.py
│   ├── analisi.py
│   ├── analisi_avanzate.py
│   ├── meteostat.py
│   └── preferiti.py
│
├── 📂 animations/              # Animazioni Lottie per la UI
│   └── *.json
│
├── requirements.txt            # Dipendenze Python
└── README.md                   # Questo file

-------🚀 Funzionalità principali
Analisi personalizzabile: filtro per paese, anno, fonte energetica.
Salvataggio preferenze: ogni utente può salvare combinazioni ricorrenti di analisi.
Visualizzazioni dinamiche: grafici interattivi 
Previsioni future: Regressione lineare e Prophet 
Clustering: segmentazione automatica dei paesi su base produttiva
Analisi meteo: dati meteo storici integrati grazie a Meteostat
Approfondimenti regionali italiani: analisi delle rinnovabili per regione.

-------📦 Installazione

git clone https://github.com/rosselladedo/WorldConsumption.git
cd WorldConsumption

python -m venv venv
source venv/bin/activate  # o venv\Scripts\activate su Windows

pip install -r requirements.txt

streamlit run WorldConsumption.py

-------📖 Guida all'utilizzo

## 🧭 Come utilizzare l'app

Una volta avviata l'app, l'interfaccia principale ti guiderà in diverse sezioni attraverso il sidebar laterale dal quale è possibile raggiungere le varie pagine.
Le pagine Analisi e Aanalisi Avanzate ti permettono di inserire i dati manualmente o richiamare i preferiti precedentemente salvati.

1. **Home Page**  
   Introduzione generale al progetto e panoramica del contesto energetico.
   
3. **Preferiti**  
   - Salva combinazioni ricorrenti di analisi (paesi, anni, fonte energetica).
   - Richiama facilmente i tuoi preferiti in ogni sezione dell'app.
   - Modifica o elimina preferiti già salvati.

4. **Analisi**  
   - Seleziona uno o più **paesi**, un intervallo di **anni** e una **fonte energetica** o scegli tra i preferiti esistenti
   - Visualizza grafici interattivi: andamento della produzione, correlazioni, ranking e confronto tra paesi.
   - Scarica i dati filtrati in formato CSV.

5. **Analisi Avanzate**
   - Seleziona uno o più **paesi**, un intervallo di **anni** e una **fonte energetica** o scegli tra i preferiti esistenti
   - Seleziona un paese tra quelli selezionati per eseguire **previsioni** della produzione energetica (con Regressione Lineare e Prophet).
   - Visualizza **clustering** dei paesi in base alla produzione totale (K-Means).
   - Esplora dinamiche future e relazioni tra gruppi di paesi.
   - Scarica i dati filtrati in formato CSV.

7. **Analisi Meteo**  
   - Seleziona una **città italiana** e un intervallo di tempo.
   - Visualizza i dati meteo (temperatura media, minima, massima) tramite l'API Meteostat.
   - Confronta la **produzione energetica rinnovabile regionale** della città scelta con le condizioni meteo.

---

➡️ Ogni sezione è interattiva e aggiorna i grafici in tempo reale in base ai parametri selezionati.







