import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time
import os

# -----------------------------
# 1. Connexion à MT5
# -----------------------------
if not mt5.initialize():
    print("Erreur MT5")
    quit()

# -----------------------------
# 2. Paires à analyser
# -----------------------------
symbols = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "EURJPY",
    "GBPJPY",
    "EURGBP"
]

# -----------------------------
# 3. Fichier CSV
# -----------------------------
file_name = "forex_live_data.csv"
columns = ["time"] + symbols

if not os.path.exists(file_name):
    pd.DataFrame(columns=columns).to_csv(file_name, index=False)

print("Enregistrement démarré...")

# -----------------------------
# 4. Mémoire des dernières valeurs
# -----------------------------
last_values = {symbol: None for symbol in symbols}

# -----------------------------
# 5. Boucle temps réel
# -----------------------------
while True:
    current_values = {}
    changed = False

    for symbol in symbols:
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            price = tick.bid
        else:
            price = None

        current_values[symbol] = price

        # Détection de changement
        if last_values[symbol] != price:
            changed = True

    # -----------------------------
    # 6. Écriture SI AU MOINS UN CHANGEMENT
    # -----------------------------
    if changed:
        row = {"time": datetime.now()}
        row.update(current_values)

        pd.DataFrame([row]).to_csv(
            file_name,
            mode="a",
            header=False,
            index=False
        )

        # Mise à jour mémoire
        last_values = current_values.copy()

        print("Snapshot enregistré")

    time.sleep(1)
