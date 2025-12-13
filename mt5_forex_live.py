import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time

# 1. Connexion à MT5
if not mt5.initialize():
    print("Erreur MT5")
    quit()

# 2. Paires à analyser
symbols = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "EURJPY",
    "GBPJPY",
    "EURGBP"
]

# 3. Nom du fichier CSV
file_name = "forex_live_data.csv"

# 4. Colonnes du CSV
columns = ["time"] + symbols

# 5. Création du fichier
df = pd.DataFrame(columns=columns)
df.to_csv(file_name, index=False)

print("Enregistrement démarré...")

# 6. Boucle temps réel
while True:
    row = {}
    row["time"] = datetime.now()

    for symbol in symbols:
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            row[symbol] = tick.bid
        else:
            row[symbol] = None

    df = pd.DataFrame([row])
    df.to_csv(file_name, mode="a", header=False, index=False)

    time.sleep(1)  # 1 seconde
