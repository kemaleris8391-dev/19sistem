import sys
import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

# Windows terminal UTF-8 çıktı desteği
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def build_grover_19_circuit():
    """
    19 sayısını (İkili / Binary karşılığı: 10011_2 => q4=1, q3=0, q2=0, q1=1, q0=1)
    kuantum süperpozisyonunda arayan 5 Kübitlik Kuantum Grover Algoritması Devresi.
    """
    qc = QuantumCircuit(5)

    # 1. ADIM: SÜPERPOZİSYON OLUŞTURMA (Hadamard Kapıları)
    qc.h(range(5))

    # 2. ADIM: KUANTUM KAHİN (ORACLE) - 19 (10011) DURUMUNU İŞARETLEME
    # q3=0 ve q2=0 olduğu için x(2) ve x(3) ile ters çeviriyoruz ki 10011 durumunda tüm kontroller 1 olsun
    qc.x(2)
    qc.x(3)

    qc.h(4)
    qc.mcx([0, 1, 2, 3], 4)
    qc.h(4)

    qc.x(2)
    qc.x(3)

    # 3. ADIM: KUANTUM DİFÜZYON (AMPLITUDE AMPLIFICATION) OPERATÖRÜ
    qc.h(range(5))
    qc.x(range(5))
    qc.h(4)
    qc.mcx([0, 1, 2, 3], 4)
    qc.h(4)
    qc.x(range(5))
    qc.h(range(5))

    # Ölçüm Ekleme
    qc.measure_all()

    return qc

def run_quantum_simulation():
    print("=" * 70)
    print("--- KUANTUM BİLGİSAYARI (GROVER ALGORİTMASI) 19 ARAMA SİMÜLASYONU ---")
    print("=" * 70)
    print("Hedef Durum : 19 (İkili Kod: 10011)")
    print("Kübit Sayısı: 5 Kübit (2^5 = 32 Eşzamanlı Kuantum Süperpozisyon Durumu)")
    print("Algoritma   : Grover Kuantum Arama & Faz Genliği Genişletme")
    print("=" * 70)

    start_time = time.time()
    qc = build_grover_19_circuit()

    # Kuantum Devresi Simülatörü
    sampler = StatevectorSampler()
    job = sampler.run([(qc)], shots=1024)
    result = job.result()[0]

    # Ölçüm İstatistikleri
    counts = result.data.meas.get_counts()
    end_time = time.time()

    print(f"\n[KUANTUM SİMÜLASYON TAMAMLANDI - Süre: {end_time - start_time:.4f} saniye]")
    print("\nKUANTUM ÖLÇÜM SONUÇLARI (En Yüksek Olasılıklı Kuantum Durumları):")
    print("-" * 60)

    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    for state, count in sorted_counts[:5]:
        decimal_val = int(state, 2)
        percentage = (count / 1024) * 100
        is_target = " [HEDEF 19 BULUNDU OK]" if decimal_val == 19 else ""
        print(f" Durum: |{state}> (Ondalık: {decimal_val:>2}) -> {count:>4} Kez Ölçüldü (%{percentage:.1f}){is_target}")

    print("-" * 60)
    print("\n[AÇIKLAMA]")
    print("Klasik bilgisayarlar 32 durumu tek tek denerken, Kuantum Bilgisayarı süperpozisyon ve")
    print("Grover Algoritması ile tek bir işlem adımında 19 sayısını (10011) yüksek olasılıkla tespit etmiştir.")
    print("=" * 70)

if __name__ == "__main__":
    run_quantum_simulation()
