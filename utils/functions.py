import json
import os
import subprocess

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
            aggiorna_github(favorites_file)
        print(f"Preferiti salvati con successo in {favorites_file}")
    except Exception as e:
        print(f"Errore nel salvataggio dei preferiti: {e}")

def elimina_preferito(favorites_file, preferito_da_eliminare):
    preferiti_list = leggi_preferiti(favorites_file)
    preferiti_list = [p for p in preferiti_list if p['nome preferito'] != preferito_da_eliminare]
    scrivi_preferiti(favorites_file, preferiti_list)

def aggiorna_github(file_path):
    """Esegue il commit e il push delle modifiche su GitHub"""
    try:
        repo_path = os.getcwd()  # Prende la cartella corrente del repository
        os.chdir(repo_path)  # Si sposta nella directory del repository

        subprocess.run(["git", "add", file_path], check=True)
        subprocess.run(["git", "commit", "-m", "Aggiornato file preferiti.json"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)

        print("✅ Modifiche committate e pushate su GitHub con successo!")
    except Exception as e:
        print(f"❌ Errore durante il push su GitHub: {e}")