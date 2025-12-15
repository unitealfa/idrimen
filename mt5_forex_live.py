import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time
import os
import requests
import json
import threading

# -----------------------------
# Configuration
# -----------------------------
API_URL = "http://localhost:3005/api/tick"
# API_URL = "https://your-vercel-app.vercel.app/api/tick" # Uncomment for production

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
# 3. Fichier CSV (Backup)
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

def send_tick(symbol, price, timestamp):
    try:
        payload = {
            "symbol": symbol,
            "price": price,
            "timestamp": timestamp.isoformat()
        }
        requests.post(API_URL, json=payload, timeout=0.5)
    except Exception as e:
        # Ignore connection errors to avoid blocking
        pass

# -----------------------------
# 5. Boucle temps réel
# -----------------------------
while True:
    current_values = {}
    changed = False
    
    current_time = datetime.now()

    for symbol in symbols:
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            price = tick.bid
        else:
            price = None

        current_values[symbol] = price

        # Détection de changement
        # Note: In high freq, price changes often. We send update if price changed.
        if price is not None and last_values[symbol] != price:
            changed = True
            # Send to API in a thread to not block the loop
            threading.Thread(target=send_tick, args=(symbol, price, current_time)).start()

    # -----------------------------
    # 6. Écriture CSV SI AU MOINS UN CHANGEMENT (Old Logic kept for backup)
    # -----------------------------
    if changed:
        row = {"time": current_time}
        row.update(current_values)

        pd.DataFrame([row]).to_csv(
            file_name,
            mode="a",
            header=False,
            index=False
        )

        # Mise à jour mémoire
        last_values = current_values.copy()

        print(f"Snapshot enregistré & Sent to API: {current_time.strftime('%H:%M:%S')}")

    time.sleep(1)
