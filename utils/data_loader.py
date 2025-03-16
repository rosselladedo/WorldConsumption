import pandas as pd

def load_datasets():
    """
    Carica i dataset e li restituisce come DataFrame.
    """
    df_fuel = pd.read_csv("data/EnergyDecription.csv")
    df = pd.read_csv("data/WorldConsumption_Prepdataset.csv")
    df_prod_region = pd.read_csv("data/Produzione_rinnovabile_regioni212223.csv")
    cities_df = pd.read_csv("data/citta_italiane.csv")
    df_rinnovabili_aggregato = pd.read_csv("data/dataset_giornaliero_aggregato.csv")

    return df_fuel, df, df_prod_region, cities_df, df_rinnovabili_aggregato
