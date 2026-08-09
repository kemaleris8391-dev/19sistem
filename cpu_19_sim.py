import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def simulate_batch(batch_size, word_count=77430, is_mushaf_mode=False):
    """
    Rastgele yapay kitaplarda kelime/ayet harf toplamlarının 19'a tam bölünüp bölünmediğini simüle eder.
    NumPy vektörize işlemler ile yüksek performans sağlar.
    """
    if is_mushaf_mode:
        # 114 sure ayet sayısı simülasyonu (Gerçek Mushaf Modu)
        verse_counts = np.random.randint(5, 105, size=(batch_size, 114))
        totals = np.sum(verse_counts, axis=1)
    else:
        # Her kitap için 1-14 harfli kelimelerin toplamını simüle et (Standart Metin Modu)
        word_lengths = np.random.randint(1, 15, size=(batch_size, 100))
        totals = np.sum(word_lengths, axis=1)

    # 19'a tam bölünenleri say
    successful = np.sum(totals % 19 == 0)
    return int(successful)

def run_cpu_simulation(total_simulations=10_000_000, num_workers=None):
    if num_workers is None:
        num_workers = os.cpu_count() or 4

    print("=" * 60)
    print(f"--- CPU SİMÜLASYONU BAŞLIYOR ---")
    print(f"Toplam Test Edilecek Kitap : {total_simulations:,}")
    print(f"Kullanılan CPU Çekirdek Sayısı: {num_workers}")
    print("=" * 60)

    start_time = time.time()
    
    batch_size = 500_000
    num_batches = total_simulations // batch_size
    remainder = total_simulations % batch_size

    batches = [batch_size] * num_batches
    if remainder > 0:
        batches.append(remainder)

    total_success = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(simulate_batch, b) for b in batches]
        for future in as_completed(futures):
            total_success += future.result()

    end_time = time.time()
    duration = end_time - start_time
    probability = (total_success / total_simulations) * 100

    print("\n--- SONUÇLAR ---")
    print(f"İşlem Süresi                   : {duration:.4f} saniye")
    print(f"19 Örüntüsü Uyan Kitap Sayısı : {total_success:,}")
    print(f"Matematiksel Tesadüf Oranı     : %{probability:.4f}")
    print(f"Saniyede Hesaplanan Kitap    : {total_simulations / duration:,.0f} kitap/sn")
    print("=" * 60)

if __name__ == '__main__':
    run_cpu_simulation(total_simulations=10_000_000)
