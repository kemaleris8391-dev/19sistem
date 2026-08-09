import sys
import time
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# OpenCL AMD GPU C-Kernel
KERNEL_CODE = """
uint rand_lcg(uint *seed) {
    *seed = (*seed * 1664525u + 1013904223u);
    return *seed;
}

__kernel void simulate_19_books(
    __global const uint *seeds,
    __global uint *results_19,
    const uint num_criteria
) {
    int gid = get_global_id(0);
    uint seed = seeds[gid];

    uint all_match = 1;

    for (int k = 0; k < num_criteria; k++) {
        uint letter_sum = 0;
        for (int i = 0; i < 50; i++) {
            uint val = (rand_lcg(&seed) % 14) + 1;
            letter_sum += val;
        }
        if ((letter_sum % 19) != 0) {
            all_match = 0;
        }
    }

    results_19[gid] = all_match;
}
"""

def run_amd_gpu_simulation(total_simulations=10_000_000, num_criteria=1):
    print("=" * 70)
    print("--- AMD RADEON GPU (PYTORCH/DIRECTML HIZLANDIRMA) SİMÜLASYONU ---")
    print(f"Toplam Test Edilecek Kitap : {total_simulations:,}")
    print(f"Test Edilen Kriter Sayısı   : {num_criteria}")
    print("=" * 70)

    try:
        import pyopencl as cl

        platforms = cl.get_platforms()
        gpu_device = None
        for p in platforms:
            for d in p.get_devices(device_type=cl.device_type.GPU):
                gpu_device = d
                break
            if gpu_device:
                break

        if not gpu_device:
            print("[UYARI] AMD GPU bulunamadı, CPU moduna geçiliyor.")
            return None, None

        print(f"[INFO] AMD GPU Donanımı Bağlandı: {gpu_device.name} ({gpu_device.global_mem_size // (1024*1024)} MB VRAM)")

        ctx = cl.Context([gpu_device])
        queue = cl.CommandQueue(ctx)
        program = cl.Program(ctx, KERNEL_CODE).build()

        start_time = time.time()

        # Batch parçalama (Parça başı max 5M kitap)
        batch_size = min(5_000_000, total_simulations)
        num_batches = int(np.ceil(total_simulations / batch_size))

        total_success = 0
        mf = cl.mem_flags

        knl = program.simulate_19_books

        for b in range(num_batches):
            curr_size = min(batch_size, total_simulations - b * batch_size)
            seeds_host = np.random.randint(1, 2**31 - 1, size=curr_size, dtype=np.uint32)
            results_host = np.zeros(curr_size, dtype=np.uint32)

            seeds_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=seeds_host)
            results_buf = cl.Buffer(ctx, mf.WRITE_ONLY, results_host.nbytes)

            knl(queue, (curr_size,), None, seeds_buf, results_buf, np.uint32(num_criteria))
            cl.enqueue_copy(queue, results_host, results_buf)
            queue.finish()

            total_success += int(np.sum(results_host == 1))

        duration = time.time() - start_time
        prob = (total_success / total_simulations) * 100

        print("\n--- GPU SİMÜLASYON SONUÇLARI ---")
        print(f" İşlem Süresi                   : {duration:.4f} saniye")
        print(f" GPU Hızı                       : {total_simulations / duration:,.0f} kitap/sn")
        print(f" 19 Örüntüsü Uyan Kitap Sayısı : {total_success:,}")
        print(f" Gerçekleşen Olasılık           : %{prob:.4f}")
        print("=" * 70)

        return duration, total_success
    except Exception as e:
        print(f"[HATA] GPU Simülasyon hatası: {e}")
        return None, None

if __name__ == "__main__":
    run_amd_gpu_simulation(10_000_000)
