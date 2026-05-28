"""
Script untuk generate data saham sample.
Jalankan ini jika belum punya file saham.csv dari dosen.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

# Generate 3 tahun data harian (cukup untuk MA200)
dates = pd.date_range(start='2021-01-01', end='2024-12-31', freq='B')  # Business days
n = len(dates)

# Simulasi harga saham dengan random walk + trend
price = 1000.0  # Harga awal
prices = [price]

for i in range(1, n):
    # Tambahkan sedikit trend naik + noise
    change = np.random.normal(0.0005, 0.02) * price
    price = max(price + change, 100)  # Harga minimal 100
    prices.append(price)

prices = np.array(prices)

# Buat OHLCV data
df = pd.DataFrame({
    'Date': dates[:n],
    'Open': prices * (1 + np.random.uniform(-0.01, 0.01, n)),
    'High': prices * (1 + np.random.uniform(0.005, 0.025, n)),
    'Low': prices * (1 - np.random.uniform(0.005, 0.025, n)),
    'Close': prices,
    'Volume': np.random.randint(100000, 5000000, n)
})

# Round harga ke 2 desimal
for col in ['Open', 'High', 'Low', 'Close']:
    df[col] = df[col].round(2)

df.to_csv('data/saham.csv', index=False)
print(f"✅ Data sample berhasil dibuat: {len(df)} baris")
print(f"   Periode: {df['Date'].iloc[0]} s/d {df['Date'].iloc[-1]}")
print(f"   Harga awal: {df['Close'].iloc[0]}, Harga akhir: {df['Close'].iloc[-1]}")
