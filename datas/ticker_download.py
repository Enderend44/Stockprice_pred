import os
import yfinance as yf
import random
import shutil

# Liste des tickers (indices/actions moins connues + DJI)
tickers = ["^DJI", "ARKK", "SPCE", "BYDDF", "NVAX", "PLTR", 
           "PSNY", "RIVN", "WKHS", "HITI", "SPWR", "PLUG", 
           "NNDM", "^VIX", "^FTSE", "^SSEC", "^N225", "^AXJO",
           "ETH-USD", "DOGE-USD", "SOL-USD", "SHOP", "SQ", 
           "TTD", "ETSY", "TEAM"]

# Plage de dates
start_date = "1900-01-01"
end_date = "2025-02-16"

# Création des dossiers de destination
os.makedirs("training_data", exist_ok=True)
os.makedirs("test_data", exist_ok=True)

# Télécharger les données pour chaque ticker et sauvegarder les fichiers CSV
csv_files = []
for ticker in tickers:
    print(f"Téléchargement des données pour {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    filename = f"{ticker}_Historical_Data.csv"
    data.to_csv(filename)
    csv_files.append(filename)
    print(f"Données sauvegardées dans {filename}")

# Répartition des fichiers entre training et test
random.shuffle(csv_files)  # Mélanger les fichiers de manière aléatoire
split_index = int(len(csv_files) * 2 / 3)  # 2/3 des fichiers pour training, le reste pour test

training_files = csv_files[:split_index]
test_files = csv_files[split_index:]

for file in training_files:
    shutil.move(file, "training_data/" + file)

for file in test_files:
    shutil.move(file, "test_data/" + file)

print(f"Fichiers répartis : {len(training_files)} dans 'training_data', {len(test_files)} dans 'test_data'.")
