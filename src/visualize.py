"""
Langkah 6: Visualisasi
Membuat grafik harga saham dengan MA dan Golden Cross markers.
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def plot_golden_cross(df, save_path='golden_cross_chart.png'):
    """
    Plot grafik harga saham dengan MA50, MA200, dan penanda Golden Cross.
    
    Parameters:
        df (pd.DataFrame): DataFrame lengkap dengan Close, MA50, MA200, Signal
        save_path (str): Path untuk menyimpan gambar
    """
    print("=" * 50)
    print("📉 LANGKAH 6: Visualisasi")
    print("=" * 50)
    
    # Set style
    sns.set_style("darkgrid")
    plt.rcParams['figure.facecolor'] = '#1a1a2e'
    plt.rcParams['axes.facecolor'] = '#16213e'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot harga Close
    ax.plot(df.index, df['Close'], label='Harga Close', 
            alpha=0.6, color='#e0e0e0', linewidth=1)
    
    # Plot MA50
    ax.plot(df.index, df['MA50'], label='MA50 (pendek)', 
            color='#00d2ff', linewidth=1.5)
    
    # Plot MA200
    ax.plot(df.index, df['MA200'], label='MA200 (panjang)', 
            color='#ff6b6b', linewidth=1.5)
    
    # Tandai Golden Cross
    gc = df[df['Signal'] == 1]
    if len(gc) > 0:
        ax.scatter(gc.index, gc['Close'], 
                   marker='^', color='#00ff88', s=200, 
                   label=f'Golden Cross ({len(gc)}x)', 
                   zorder=5, edgecolors='white', linewidths=0.5)
        
        # Tambahkan anotasi untuk setiap Golden Cross
        for date, row in gc.iterrows():
            ax.annotate(f'GC\n{date.strftime("%Y-%m-%d")}',
                       xy=(date, row['Close']),
                       xytext=(0, 25),
                       textcoords='offset points',
                       ha='center',
                       fontsize=7,
                       color='#00ff88',
                       fontweight='bold')
    
    # Styling
    ax.set_title('📈 Golden Cross Detection — Analisis Saham', 
                 fontsize=16, fontweight='bold', pad=20, color='white')
    ax.set_xlabel('Tanggal', fontsize=12)
    ax.set_ylabel('Harga (IDR)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10, 
              facecolor='#16213e', edgecolor='#333', labelcolor='white')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # Grid
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    
    print(f"\n✅ Grafik disimpan: {save_path}")
    print(f"   Resolusi: 150 DPI")
    print()
    
    return save_path


if __name__ == "__main__":
    from load_data import load_data
    from preprocess import preprocess
    from features import add_moving_averages
    from detect import detect_golden_cross
    
    df = load_data()
    df = preprocess(df)
    df = add_moving_averages(df)
    df = detect_golden_cross(df)
    plot_golden_cross(df)
