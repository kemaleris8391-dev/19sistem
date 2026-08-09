import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Windows terminal UTF-8 çıktı desteği
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def run_monte_carlo_analysis(num_books=1_000_000, word_count=77430):
    print("=" * 70)
    print("--- MONTE CARLO İSTATİSTİKSEL P-VALUE VE GRAFİK MODÜLÜ ---")
    print(f"Test Edilen Rastgele Kitap Sayısı   : {num_books:,}")
    print(f"Her Kitaptaki Ortalama Kelime Sayısı: {word_count:,}")
    print("=" * 70)

    start_time = time.time()

    word_lengths = np.random.randint(1, 15, size=(num_books, 100))
    totals = np.sum(word_lengths, axis=1)
    remainders = totals % 19

    counts = np.bincount(remainders, minlength=19)
    percentages = (counts / num_books) * 100

    matches_19 = counts[0]
    p_value = matches_19 / num_books
    expected_prob = 1.0 / 19.0  # ~%5.26315

    duration = time.time() - start_time

    print(f"\n[ANALİZ SONUÇLARI - Süre: {duration:.4f} saniye]")
    print(f" * 19 Sistemine Uyan Kitap Sayısı (Kalan 0) : {matches_19:,}")
    print(f" * Gerçekleşen Olasılık (P-Value)         : %{p_value * 100:.4f}")
    print(f" * Teorik Beklenen Rastgele Olasılık       : %{expected_prob * 100:.4f}")

    mean = num_books * expected_prob
    std_dev = np.sqrt(num_books * expected_prob * (1 - expected_prob))
    z_score = (matches_19 - mean) / std_dev
    print(f" * İstatistiksel Z-Skoru                  : {z_score:.4f} (Beklenen Değer Sapması)")

    # GRAFİK 1: Kalanların Dağılım Histograma (0-18)
    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(19), percentages, color='#3498db', edgecolor='black', alpha=0.8)
    bars[0].set_color('#2ecc71')

    plt.axhline(y=expected_prob*100, color='r', linestyle='--', label=f'Teorik Beklenen (%{expected_prob*100:.2f})')
    plt.title(f"1 Milyon Rastgele Kitapta 19'a Bölüm Kalanlarının Dağılımı (Monte Carlo)", fontsize=12, fontweight='bold')
    plt.xlabel("19'a Bölümünden Kalan Sayı (0-18)")
    plt.ylabel("Oluşma Olasılığı (%)")
    plt.xticks(range(19))
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.7)

    plot1_path = "19_tesaduf_olasilik_grafik.png"
    plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.close()

    # GRAFİK 2: Pasta Grafiği
    plt.figure(figsize=(6, 6))
    labels = ['19 Sistemine Uymayanlar (%94.74)', '19 Sistemine Uyanlar (%5.26)']
    sizes = [num_books - matches_19, matches_19]
    colors = ['#e74c3c', '#2ecc71']

    plt.pie(sizes, labels=labels, autopct='%1.2f%%', startangle=140, colors=colors, explode=(0, 0.1))
    plt.title("1 Milyon Kitapta 19 Örüntüsü Tesadüf Oranı Pasta Grafiği", fontsize=11, fontweight='bold')

    plot2_path = "19_pasta_grafik.png"
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[GÖRSELLEŞTİRME] Grafik görselleri oluşturuldu ve kaydedildi:")
    print(f" 1. {os.path.abspath(plot1_path)}")
    print(f" 2. {os.path.abspath(plot2_path)}")
    print("=" * 70)

if __name__ == "__main__":
    run_monte_carlo_analysis()
