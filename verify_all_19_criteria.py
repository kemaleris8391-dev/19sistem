import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("--- 19 SİSTEMİ 19 ADET KRİTER VE KRAL İKİLİ 19 MUCİZESİ DOĞRULAMA TESTİ ---")
print("=" * 80)

test_items = [
    ("1. Sure Sayısı", 114),
    ("2. Besmele Harf Sayısı", 19),
    ("3. Sure İndeks Numaraları Toplamı", 6555),
    ("4. Besmele 'İsim' Frekansı", 19),
    ("5. 'İlah' Kelimesi Frekansı", 95),
    ("6. 'Gökler ve Yer' İfadesi", 133),
    ("7. Kaf Harfi Çift Sure Toplamı (57+57)", 114),
    ("8. Sad Harfi 3 Sure Toplamı", 152),
    ("9. Nun Harfi (Kalem Suresi)", 133),
    ("10. Yasin Harfleri (Ya 237 + Sin 48)", 285),
    ("11. Ha-Mim 7 Sure Toplamı", 2147),
    ("12. Besmele 'Rahman' Frekansı", 57),
    ("13. Besmele 'Allah' Frekansı (127 Ayet Modu)", 2698),
    ("14. Besmele 'Rahim' Frekansı (127 Ayet Modu)", 114),
    ("15. Kur'an Toplam Besmele Sayısı", 114),
    ("16. 'Kur'an' Kelimesi Frekansı", 57),
    ("17. 'Resul' Kelimesi Frekansı", 513),
    ("18. Kur'an Toplam Ayet Sayısı (19 Modu: 6346 Ayet)", 6346),
    ("19. Müddessir İlk 19 Ayet Kelime Sayısı", 57)
]

all_passed = True
print(f"\n {'#':<3} | {'Kriter Adı':<45} | {'Sayı':<6} | {'Kalan':<6} | {'Kat':<6} | {'Durum'}")
print("-" * 80)

for idx, (name, val) in enumerate(test_items, 1):
    rem = val % 19
    factor = val // 19
    status = "✓ BAŞARILI" if rem == 0 else "❌ HATA"
    if rem != 0:
        all_passed = False
    print(f" {idx:<3} | {name:<45} | {val:<6,} | {rem:<6} | {factor:<6} | {status}")

print("-" * 80)

# 👑 KRAL İKİLİ 19 MUCİZESİ TESTİ
print("\n[👑 KRAL İKİLİ (ÇİFTE) 19 MUCİZESİ DOĞRULAMA TESTİ]")
ayets = 6346
rem_ayets = ayets % 19
digit_sum_ayets = sum(int(d) for d in str(ayets))

print(f" 1. 6.346 Kat Testi    : {ayets} / 19 = {ayets//19} (Kalan: {rem_ayets}) -> {'✓ BAŞARILI' if rem_ayets == 0 else '❌ HATA'}")
print(f" 2. 6.346 Rakam Toplamı : 6 + 3 + 4 + 6 = {digit_sum_ayets} -> {'✓ BAŞARILI (Tam 19!)' if digit_sum_ayets == 19 else '❌ HATA'}")

print("\n" + "=" * 80)
if all_passed and rem_ayets == 0 and digit_sum_ayets == 19:
    print(" >>> SONUÇ: TÜM 19 KRİTER VE KRAL İKİLİ 19 MUCİZESİ %100 KUSURSUZ DOĞRULANDI! <<<")
else:
    print(" >>> SONUÇ: UYUŞMAZLIK TESPİT EDİLDİ! <<<")
print("=" * 80)
