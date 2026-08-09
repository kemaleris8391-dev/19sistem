import os
import sys
import numpy as np

# Arapça Ebced (Abjad) Değer Haritası
EBCED_TABLOSU = {
    'ا': 1, 'أ': 1, 'إ': 1, 'آ': 1, 'ء': 1,
    'ب': 2,
    'ج': 3,
    'د': 4,
    'ه': 5, 'ة': 5,
    'و': 6,
    'ز': 7,
    'ح': 8,
    'ط': 9,
    'ي': 10, 'ى': 10, 'ئ': 10,
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

def analyze_text_19(text_path):
    if not os.path.exists(text_path):
        print(f"[UYARI] '{text_path}' dosyası bulunamadı. Lütfen analiz edilecek metni bu dosyaya koyun.")
        return

    with open(text_path, 'r', encoding='utf-8') as f:
        content = f.read()

    words = content.split()
    word_count = len(words)
    char_count = len(content)
    letters_only = [c for c in content if not c.isspace()]
    letter_count = len(letters_only)

    ebced_total = sum(EBCED_TABLOSU.get(char, 0) for char in content)

    print("=" * 60)
    print("--- KUR'AN / METİN 19 SİSTEMİ ANALİZ RAPORU ---")
    print(f"Dosya Adı                 : {os.path.basename(text_path)}")
    print(f"Toplam Karakter (Boşluk dahil): {char_count:,}")
    print(f"Toplam Harf Sayısı        : {letter_count:,}")
    print(f"Toplam Kelime Sayısı       : {word_count:,}")
    if ebced_total > 0:
        print(f"Toplam Ebced (Abjad) Değeri: {ebced_total:,}")
    print("=" * 60)

    print("\n--- 19'a BÖLÜNEBİLİRLİK TESTLERİ ---")
    
    # 1. Kelime Sayısı Testi
    w_rem = word_count % 19
    print(f"1. Kelime Sayısı ({word_count}) % 19 = {w_rem} {'[TAM BÖLÜNÜYOR ✓]' if w_rem == 0 else '[KalanVar: ' + str(w_rem) + ']'}")

    # 2. Harf Sayısı Testi
    l_rem = letter_count % 19
    print(f"2. Harf Sayısı ({letter_count}) % 19 = {l_rem} {'[TAM BÖLÜNÜYOR ✓]' if l_rem == 0 else '[KalanVar: ' + str(l_rem) + ']'}")

    # 3. Ebced Değeri Testi
    if ebced_total > 0:
        e_rem = ebced_total % 19
        print(f"3. Ebced Toplamı ({ebced_total}) % 19 = {e_rem} {'[TAM BÖLÜNÜYOR ✓]' if e_rem == 0 else '[KalanVar: ' + str(e_rem) + ']'}")

    print("=" * 60)

if __name__ == '__main__':
    target_file = 'kuran.txt'
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    analyze_text_19(target_file)
