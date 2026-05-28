"""
Golden Cross Detection — Program Utama
Tugas Kelompok Sistem Cerdas

Alur eksekusi:
1. Load Data CSV
2. Preprocessing
3. Feature Engineering (MA50, MA200)
4. Deteksi Golden Cross
5. Buat Model Prediksi
6. Visualisasi
"""
import sys
import os

# Tambahkan src ke path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from load_data import load_data
from preprocess import preprocess
from features import add_moving_averages
from detect import detect_golden_cross
from model import build_model
from visualize import plot_golden_cross


def main():
    print("\n" + "═" * 60)
    print("  📈 GOLDEN CROSS DETECTION — SISTEM CERDAS")
    print("  Tugas Kelompok: Pattern Detection")
    print("═" * 60 + "\n")
    
    # Langkah 1: Load Data
    df = load_data('data/saham.csv')
    
    # Langkah 2: Preprocessing
    df = preprocess(df)
    
    # Langkah 3: Feature Engineering
    df = add_moving_averages(df)
    
    # Langkah 4: Deteksi Golden Cross
    df = detect_golden_cross(df)
    
    # Langkah 5: Model Prediksi
    model, accuracy, report, scaler = build_model(df, model_type='random_forest')
    
    # Langkah 6: Visualisasi
    chart_path = plot_golden_cross(df, save_path='golden_cross_chart.png')
    
    # Ringkasan
    print("═" * 60)
    print("  ✅ SELESAI — RINGKASAN HASIL")
    print("═" * 60)
    print(f"\n  📊 Total data: {len(df)} hari perdagangan")
    print(f"  🎯 Golden Cross terdeteksi: {df['Signal'].sum()} kali")
    print(f"  🤖 Akurasi model: {accuracy*100:.2f}%")
    print(f"  📉 Grafik tersimpan: {chart_path}")
    print(f"\n{'═' * 60}\n")


if __name__ == "__main__":
    main()
