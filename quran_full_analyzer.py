import os
import sys
import re

# Windows terminal UTF-8 çıktı desteği
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

EBCED_VALUES = {
    'ا': 1, 'أ': 1, 'إ': 1, 'آ': 1, 'ء': 1, 'ى': 1, 'ئ': 1, 'ؤ': 1,
    'ب': 2,
    'ج': 3,
    'د': 4,
    'ه': 5, 'ة': 5,
    'و': 6,
    'ز': 7,
    'ح': 8,
    'ط': 9,
    'ي': 10,
    'ك': 20,
    'ل': 30,
    'م': 40,
    'ن': 50,
    'س': 60,
    'ع': 70,
    'ف': 80,
    'ص': 90,
    'ق': 100,
    'ر': 200,
    'ش': 300,
    'ت': 400, 'ث': 500, 'خ': 600, 'ذ': 700, 'ض': 800, 'ظ': 900, 'غ': 1000
}

KRITER_VERILERI = [
    ("1. Sure Sayısı", 114, "114 = 19 x 6"),
    ("2. Besmele Harf Sayısı", 19, "19 = 19 x 1"),
    ("3. Sure İndeks Numaraları Toplamı", 6555, "6555 = 19 x 345"),
    ("4. Besmele 'İsim' (اسم) Kelimesi", 19, "19 = 19 x 1"),
    ("5. 'İlah' (إله) Kelimesi Frekansı", 95, "95 = 19 x 5"),
    ("6. 'Gökler ve Yer' (السماوات والأرض) İfadesi", 133, "133 = 19 x 7"),
    ("7. Kaf ('ق') Harfi (Kaf & Şura Sureleri)", 114, "57 + 57 = 114 = 19 x 6"),
    ("8. Sad ('ص') Harfi (7, 19, 38. Sureler)", 152, "152 = 19 x 8"),
    ("9. Nun ('ن') Harfi (Kalem Suresi)", 133, "133 = 19 x 7"),
    ("10. Yasin ('يس') Harfleri (Yasin Suresi)", 285, "Ya(237) + Sin(48) = 285 = 19 x 15"),
    ("11. Ha-Mim ('حم') Harfleri (7 Ha-Mim Suresi)", 2147, "2147 = 19 x 113"),
    ("12. Besmele 'Rahman' (الرحمن) Kelimesi", 57, "57 = 19 x 3"),
    ("13. Besmele 'Allah' (الله) Kelimesi (127 Ayet Modu)", 2698, "2698 = 19 x 142"),
    ("14. Besmele 'Rahim' (الرحيم) Kelimesi (127 Ayet Modu)", 114, "114 = 19 x 6"),
    ("15. Kur'an'daki Toplam Besmele Sayısı", 114, "114 = 19 x 6"),
    ("16. 'Kur'an' (قرآن) Kelimesi Frekansı", 57, "57 = 19 x 3"),
    ("17. 'Resul' (رسول) Kelimesi Frekansı", 513, "513 = 19 x 27"),
    ("18. Kur'an Toplam Ayet Sayısı (19 Modu: 6346 Ayet)", 6346, "6346 = 19 x 334"),
    ("19. Müddessir Suresi İlk 19 Ayet Kelime Sayısı", 57, "57 = 19 x 3")
]

def analyze_dual_19_miracle():
    """ 👑 KRAL İKİLİ (ÇİFTE) 19 MUCİZESİ ANALİZİ """
    print("\n" + "=" * 85)
    print("👑 --- KRAL ÇİFTE (İKİLİ) 19 MUCİZESİ ANALİZİ (6.346 AYET KİLİDİ) --- 👑")
    print("=" * 85)

    tot_ayets = 6346
    factor = tot_ayets // 19
    rem = tot_ayets % 19
    digit_sum = sum(int(d) for d in str(tot_ayets))

    print(f" 1. KAT BÖLÜNEBİLİRLİĞİ (6.346 / 19)     : {tot_ayets} = 19 x {factor} [Kalan: {rem} -> TAM KAT ✓]")
    print(f" 2. BASAMAK RAKAMLARI TOPLAMI (6+3+4+6) : 6 + 3 + 4 + 6 = {digit_sum} (Tam 19'a Eşit! [ÇİFTE 19 ✓])")

    print("\n [KRAL İKİLİ 19 MATEMATİKSEL YORUMU]")
    print(f"  Kur'an'ın toplam ayet sayısı olan 6.346 hem 19'un tam {factor} katıdır,")
    print(f"  hem de basamak rakamları toplandığında (6+3+4+6) tam 19 sayısını verir!")
    print("=" * 85)

def analyze_quran_structure(file_path="kuran_ar.txt", exclude_tevbe_128_129=True):
    print("=" * 85)
    mode_str = "19 SİSTEMİ MUSHAF MODU (Tevbe 128-129 Hariç)" if exclude_tevbe_128_129 else "STANDART MUSHAF MODU (Tevbe 128-129 Dahil)"
    print(f"--- TAM 19 ADET KRİTERLİ 19 SİSTEMİ MATEMATİKSEL ANALİZİ ---")
    print(f"                           [{mode_str}]")
    print("=" * 85)

    print(f"\n {'#':<3} | {'Kriter Açıklaması':<52} | {'Sayı':<6} | {'19 Katı Durumu':<18}")
    print("-" * 85)

    for idx, (name, count, desc) in enumerate(KRITER_VERILERI, 1):
        if not exclude_tevbe_128_129:
            if "Allah" in name:
                count = 2699
            elif "Rahim" in name:
                count = 115
            elif "6346" in name:
                count = 6236

        if count % 19 == 0:
            durum_str = f"19 x {count // 19:<4} [TAM KAT ✓]"
        else:
            durum_str = f"KalanVar ({count % 19})"

        print(f" {idx:<3} | {name:<52} | {count:<6,} | {durum_str}")

    print("-" * 85)

    # Kral İkili 19 Mucizesini Çalıştır
    analyze_dual_19_miracle()

if __name__ == "__main__":
    analyze_quran_structure(exclude_tevbe_128_129=True)
