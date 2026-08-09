import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows UTF-8 Terminal Uyumluluğu
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def run_kozmik_analiz():
    print("=" * 95)
    print(" 🪐 KOZMİK ZAMAN & EVREN ANALOJİSİ (8 MİLYAR İNSAN & 13.8 MİLYAR YIL) 🪐")
    print("=" * 95)

    # Temel Parametreler
    nufus = 8_000_000_000          # 8 Milyar İnsan
    hiz_saniye = 1                # 1 Kitap / Saniye / İnsan
    toplam_hiz = nufus * hiz_saniye # 8.000.000.000 Kitap / Saniye (Tüm İnsanlık)

    evren_yasi_yil = 13_800_000_000 # 13.8 Milyar Yıl
    saniye_per_yil = 365.25 * 24 * 3600 # 31.557.600 saniye/yıl
    evren_toplam_saniye = evren_yasi_yil * saniye_per_yil # ~4.355 x 10^17 saniye

    # Evrenin yaşı boyunca tüm insanlığın üretebileceği toplam kitap sayısı
    kozmik_toplam_kitap = toplam_hiz * evren_toplam_saniye # ~3.484 x 10^27 kitap

    print(f"\n📊 TEMEL PARAMETRELER:")
    print(f"  • Dünya Nüfusu                   : {nufus:,} (8 Milyar)")
    print(f"  • Bireysel Yazma Hızı            : {hiz_saniye} kitap/saniye/kişi")
    print(f"  • Tüm İnsanlığın Anlık Üretimi   : {toplam_hiz:,} kitap/saniye")
    print(f"  • Evrenin Yaşı                   : {evren_yasi_yil:,} Yıl (13.8 Milyar Yıl)")
    print(f"  • Evrenin Yaşı (Saniye)          : {evren_toplam_saniye:.4e} Saniye")
    print(f"  • Evren Boyunca Üretilen Kitap  : {kozmik_toplam_kitap:.4e} Kitap (~3.48 × 10²⁷ Kitap)")

    print("\n" + "=" * 95)
    print(" 🧮 19^K OLASILIK UZAYI VE İNSANLIĞIN HARCAMASI GEREKEN ZAMAN ÇİZELGESİ:")
    print("=" * 95)
    print(f"  {'Kriter (K)':<12} | {'Olasılık Uzayı (19^K)':<30} | {'Gereken Süre (Saniye)':<24} | Gereken Süre (Yıl / Zaman)")
    print("  " + "─" * 91)

    k_values = [1, 5, 10, 15, 19, 20, 21, 22]
    yillar_dict = {}

    for k in k_values:
        olasilik_uzayi = 19 ** k
        gereken_saniye = olasilik_uzayi / toplam_hiz
        gereken_yil = gereken_saniye / saniye_per_yil
        yillar_dict[k] = gereken_yil

        if gereken_yil < 1 / 365.25:
            zaman_str = f"{gereken_saniye:.2f} Saniye"
        elif gereken_yil < 1:
            zaman_str = f"{gereken_saniye / 86400:.2f} Gün"
        elif gereken_yil < 1_000_000:
            zaman_str = f"{gereken_yil:,.1f} Yıl"
        elif gereken_yil < 1_000_000_000:
            zaman_str = f"{gereken_yil / 1_000_000:.2f} Milyon Yıl"
        else:
            oran = gereken_yil / evren_yasi_yil
            zaman_str = f"{gereken_yil / 1_000_000_000:.2f} Milyar Yıl ({oran:.1f}x Evren Yaşı)"

        print(f"  19^{k:<9} | {olasilik_uzayi:<30,.0f} | {gereken_saniye:<24.2e} | {zaman_str}")

    print("\n" + "=" * 95)
    print(" 💡 KRİTİK KOZMİK SONUÇ:")
    print("=" * 95)
    print(f"  1. 19^19 = {19**19:,} (~1.98 × 10²⁴ kitap).")
    print(f"     8 Milyar insanın saniyede 1 kitap yazarak bu sayıya ulaşması tam 7.83 MİLYON YIL sürer!")
    print(f"  2. 19^21 = {19**21:,} (~7.14 × 10²⁶ kitap).")
    print(f"     Bu sayıya ulaşmak için 8 Milyar insan tam 2.83 MİLYAR YIL durmaksızın yazmalıdır.")
    print(f"  3. 19^22 = {19**22:,} (~1.36 × 10²⁸ kitap).")
    print(f"     Tüm insanlığın evrenin başından beri (13.8B yıl) yazdığı kitapların (~3.48 × 10²⁷) neredeyse 4 KATIDIR!")
    print(f"     19 sistemindeki 22 kuralın tesadüfen aynı anda oluşması kozmik ölçekte %100 İMKÂNSIZDIR.")
    print("=" * 95 + "\n")

    # --- MATPLOTLIB GRAFİK ÜRETİMİ ---
    os.makedirs("stüdyo_grafikler", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#1e1e1e')
    plt.tight_layout(pad=3.5)

    # SOL GRAFİK: Üretim Oranları (Log Ölçek)
    ax1.set_facecolor('#2b2b2b')
    etiketler = [
        '8M İnsan x 1 Gün',
        '8M İnsan x 1 Yıl',
        '8M İnsan x 1.000 Yıl',
        '19^19 Kitap Hedefi',
        '8M İnsan x Evren Yaşı',
        '19^22 Kitap Hedefi'
    ]

    degerler = [
        8e9 * 86400,                   # 1 Gün
        8e9 * saniye_per_yil,          # 1 Yıl
        8e9 * saniye_per_yil * 1000,   # 1000 Yıl
        float(19**19),                 # 19^19
        kozmik_toplam_kitap,            # Evren Yaşı
        float(19**22)                  # 19^22
    ]

    renkler = ['#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6']
    bars = ax1.barh(etiketler, degerler, color=renkler, edgecolor='white', height=0.55)
    ax1.set_xscale('log')
    ax1.set_xlabel("Kitap Sayısı (Logaritmik Ölçek)", color='white', fontsize=11, fontweight='bold')
    ax1.set_title("Kozmik Kitap Uretim Kapasitesi vs 19 Kriter Uzaylari", color='white', fontsize=12, fontweight='bold')
    ax1.tick_params(colors='white', labelsize=10)
    ax1.grid(True, which="both", ls=":", color="gray", alpha=0.4)

    for bar, val in zip(bars, degerler):
        ax1.text(val * 1.5, bar.get_y() + bar.get_height()/2, f"{val:.1e}", ha='left', va='center', color='white', fontsize=9, fontweight='bold')

    # SAĞ GRAFİK: 19^K için Gereken Yıl Sayısı (Zaman Çizelgesi)
    ax2.set_facecolor('#2b2b2b')
    ks = list(range(1, 23))
    yillar = [(19.0**int(k)) / (toplam_hiz * saniye_per_yil) for k in ks]

    ax2.plot(ks, yillar, marker='o', color='#e74c3c', linewidth=2.5, markersize=6, zorder=5, label="Gereken Süre (Yıl)")
    ax2.axhline(y=evren_yasi_yil, color='#f1c40f', linestyle='--', linewidth=2, label="Evrenin Yaşı (13.8B Yıl)")

    ax2.set_yscale('log')
    ax2.set_xlabel("Eşzamanlı Kriter Sayısı (K)", color='white', fontsize=11, fontweight='bold')
    ax2.set_ylabel("Gereken Süre (Yıl - Logaritmik)", color='white', fontsize=11, fontweight='bold')
    ax2.set_title("8 Milyar Insanin 19^K Kitap Uretmesi Icin Gereken Yil", color='white', fontsize=12, fontweight='bold')
    ax2.tick_params(colors='white', labelsize=10)
    ax2.grid(True, which="both", ls=":", color="gray", alpha=0.4)
    ax2.legend(facecolor='#34495E', edgecolor='white', labelcolor='white')

    # Özel noktaları işaretle (K=19 ve K=22)
    ax2.annotate(f"K=19: 7.84M Yıl\n(19^19)", (19, yillar[18]), xytext=(14, yillar[18] * 10),
                 arrowprops=dict(facecolor='#2ecc71', shrink=0.05, width=1.5, headwidth=6),
                 color='#2ecc71', fontweight='bold', fontsize=9)

    ax2.annotate(f"K=22: 53.7B Yıl\n(Evrenin ~4 Katı)", (22, yillar[21]), xytext=(16, yillar[21] / 5),
                 arrowprops=dict(facecolor='#e74c3c', shrink=0.05, width=1.5, headwidth=6),
                 color='#e74c3c', fontweight='bold', fontsize=9)

    grafik_yolu = "stüdyo_grafikler/kozmik_evren_19_analojisi.png"
    fig.savefig(grafik_yolu, dpi=150, bbox_inches='tight')
    print(f"📸 Grafikler kaydedildi: {grafik_yolu}")

if __name__ == "__main__":
    run_kozmik_analiz()
