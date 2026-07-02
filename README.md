# HLK-LD2450 mmWave Radar — Comparison of PF & UKF Filtering

Sistem akuisisi, kalibrasi, tracking, filtering (Unscented Kalman Filter vs Particle Filter), dan visualisasi real-time untuk sensor mmWave radar **HLK-LD2450**, menggunakan **ESP32** sebagai jembatan (bridge) antara sensor dan laptop.

Proyek ini dibuat sebagai **Tugas Akhir**:

- **Nama**: Adjira Eka Dewanda (122400039)
- **Program Studi**: Teknik Telekomunikasi
- **Institusi**: Institut Teknologi Sumatera (ITERA)
- **Tahun**: 2026
- **Video Kalibrasi & Pengujian**: https://drive.google.com/drive/folders/1iD_9C1xGV8mYI9Wc8RvfNzljP1lo5GUw?usp=sharing

> Dokumentasi ini ditulis selengkap mungkin agar bisa langsung dilanjutkan/dikembangkan oleh adik tingkat tanpa perlu bertanya banyak ke penulis asli. Jika ada bagian yang kurang jelas, silakan buka *issue* atau baca langsung source code — setiap file disebutkan fungsinya di bagian [Struktur Proyek](#-struktur-proyek--penjelasan-tiap-file).

---

## 📑 Daftar Isi

1. [Ringkasan Sistem](#-ringkasan-sistem)
2. [Arsitektur & Alur Data](#-arsitektur--alur-data)
3. [Kebutuhan Hardware](#-kebutuhan-hardware)
4. [Skema Wiring / Pemasangan Kabel](#-skema-wiring--pemasangan-kabel)
5. [Format Data Mentah dari Sensor LD2450](#-format-data-mentah-dari-sensor-ld2450)
6. [Format Data ESP32 → Laptop](#-format-data-esp32--laptop)
7. [Instalasi Software](#-instalasi-software)
8. [Flashing Firmware ke ESP32](#-flashing-firmware-ke-esp32)
9. [Menjalankan Sistem (Quick Start)](#-menjalankan-sistem-quick-start)
10. [Kalibrasi Sensor](#-kalibrasi-sensor)
11. [Kontrol & Hotkey Saat Runtime](#-kontrol--hotkey-saat-runtime)
12. [Struktur Proyek & Penjelasan Tiap File](#-struktur-proyek--penjelasan-tiap-file)
13. [Detail Algoritma](#-detail-algoritma)
14. [Output & Format Log Excel](#-output--format-log-excel)
15. [Parameter Tuning](#-parameter-tuning)
16. [Troubleshooting](#-troubleshooting)
17. [Keterbatasan & Ide Pengembangan Lanjutan](#-keterbatasan--ide-pengembangan-lanjutan)
18. [FAQ Singkat](#-faq-singkat)

---

## 🧭 Ringkasan Sistem

Sistem ini membaca posisi hingga **3 target manusia** secara bersamaan dari sensor mmWave radar HLK-LD2450, lalu:

1. Mengonversi koordinat (x, y) menjadi jarak (distance) dan kecepatan (velocity).
2. **(Opsional)** Melakukan *data association* — melacak target yang sama antar frame agar ID (t1/t2/t3) tidak tertukar saat target bergerak/berpapasan.
3. **(Opsional)** Mengoreksi bias sensor menggunakan **model kalibrasi** (logaritmik untuk jarak, linear untuk kecepatan) yang telah dilatih sebelumnya.
4. Mem-filter data mentah yang noisy menggunakan dua metode estimasi state secara paralel:
   - **UKF** — Unscented Kalman Filter
   - **PF** — Particle Filter
5. Menghitung metrik perbandingan performa (**RMSE, MAE, MBE**) antara UKF vs PF terhadap data mentah.
6. Menampilkan semuanya secara **live** dalam dashboard `matplotlib` (grafik jarak, grafik kecepatan, panel metrik per target).
7. Menyimpan seluruh riwayat data + metrik ke file **Excel (.xlsx)** saat program ditutup.

Tujuan akhir dari Tugas Akhir ini adalah **membandingkan performa UKF vs PF** dalam mengestimasi jarak & kecepatan target dari data radar yang noisy.

---

## 🏗 Arsitektur & Alur Data

```
┌─────────────────┐   UART (256000 bps)   ┌───────────────┐   USB Serial (115200 bps)   ┌─────────────────────┐
│   HLK-LD2450     │ ───────────────────►  │     ESP32     │ ───────────────────────────► │   Laptop / PC        │
│  (mmWave radar)  │   protokol biner      │ (firmware .ino)│   teks: "x1,y1,x2,y2,x3,y3"  │  (Python)            │
└─────────────────┘   pabrik LD2450       └───────────────┘   1 baris per frame           └─────────────────────┘
```

Di sisi laptop (`main.py` sebagai entry point), alur pemrosesan per frame adalah:

```
parsing.py (read_radar_data)
   │  baca 1 baris serial → parse_radar_frame() → 3 deteksi mentah (x, y, distance)
   ▼
[opsional] data_association.py (DataAssociator.update)
   │  gating + cost matching (jarak, heading, momentum-x) → assign ke track t1/t2/t3
   ▼
[opsional] calibration_system.py (CalibrationManager.apply_calibration)
   │  distance_raw → model logaritmik ; velocity_raw → model linear
   ▼
main.py (data_loop thread)
   │  untuk setiap target t1/t2/t3:
   │     → ukf.py (UKF.predict + UKF.update)
   │     → pf.py  (ParticleFilter.predict + ParticleFilter.update)
   │     → metrics.py (hitung RMSE/MAE/MBE, raw vs UKF, raw vs PF)
   ▼
viz.py (RadarVisualizer)
   │  animasi matplotlib live (grafik + panel metrik)
   │  setiap frame juga dicatat ke radar_logger.py (Logger)
   ▼
[Tekan Q] → radar_logger.py menyimpan seluruh history ke logs/radar_log_<timestamp>.xlsx
```

**Poin penting:** `USE_DATA_ASSOCIATION` dan `USE_CALIBRATION` adalah flag global di `parsing.py` yang **default `False`** saat program mulai, dan bisa di-toggle live saat runtime dengan tombol `D` dan `C` (lihat [Kontrol & Hotkey](#-kontrol--hotkey-saat-runtime)).

---

## 🔩 Kebutuhan Hardware

| Komponen | Keterangan |
|---|---|
| Sensor **HLK-LD2450** | Radar mmWave 24GHz, mampu mendeteksi hingga 3 target (posisi x,y, kecepatan bawaan sensor — meski di proyek ini kecepatan dihitung ulang di host) |
| **ESP32 DevKit** (WROOM/WROVER, atau varian dengan USB native seperti S2/S3) | Sebagai bridge UART ↔ USB Serial. Firmware menggunakan `HardwareSerial(2)`, jadi board perlu punya UART2 yang bisa di-remap ke pin custom (hampir semua ESP32 klasik bisa) |
| Kabel jumper female-female (4 pin minimal: 5V/3V3, GND, TX, RX) | Menghubungkan LD2450 ↔ ESP32 |
| Kabel USB (Micro-USB atau USB-C sesuai board ESP32) | Menghubungkan ESP32 ↔ Laptop |
| Laptop/PC dengan port USB | Menjalankan seluruh sistem Python |
| (Opsional, untuk kalibrasi) Meteran/pita ukur & stopwatch | Untuk menentukan ground truth jarak & kecepatan saat kalibrasi |

> **Catatan daya**: Cek datasheet modul LD2450 yang kamu punya — sebagian breakout board LD2450 butuh 5V, sebagian lain toleran 3.3V–5V. Pastikan level tegangan **TX/RX** sensor **kompatibel dengan 3.3V logic ESP32** (LD2450 pada umumnya sudah 3.3V TTL pada pin UART-nya, tapi selalu cek dokumentasi board yang dipakai untuk menghindari kerusakan pin ESP32).

---

## 🔌 Skema Wiring / Pemasangan Kabel

Firmware (`hlk_ld2450_esp32.ino`) mengonfigurasi **UART2** ESP32 dengan pin berikut:

```cpp
#define RADAR_RX_PIN   16   // ESP32 GPIO16 (RX2) ← TX sensor LD2450
#define RADAR_TX_PIN   17   // ESP32 GPIO17 (TX2) → RX sensor LD2450
#define RADAR_BAUD     256000  // baud rate default pabrik LD2450
#define PC_BAUD        115200  // baud rate ke laptop via USB
```

### Tabel koneksi LD2450 → ESP32

| Pin LD2450 | Pin ESP32 | Keterangan |
|---|---|---|
| `VCC` | `5V` (atau `3V3`, cek datasheet board LD2450 kamu) | Power sensor |
| `GND` | `GND` | Harus common ground dengan ESP32 |
| `TX`  | `GPIO16` (RX2) | Data dari sensor masuk ke ESP32 |
| `RX`  | `GPIO17` (TX2) | Perintah dari ESP32 ke sensor (di proyek ini tidak dipakai untuk konfigurasi, tapi tetap disambung agar UART berfungsi normal) |

### Koneksi ESP32 → Laptop

Cukup pakai **kabel USB biasa** dari ESP32 ke laptop. Koneksi ini otomatis menjadi:
- Jalur **flashing firmware** (upload sketch dari Arduino IDE), dan
- Jalur **USB Serial** untuk mengirim data hasil parsing (`Serial.begin(115200)`) ke Python di laptop.

Tidak perlu USB-to-TTL converter terpisah karena hampir semua board ESP32 DevKit sudah punya chip USB-to-Serial onboard (CP2102/CH340/atau native USB tergantung varian board).

---

## 📡 Format Data Mentah dari Sensor LD2450

Sensor LD2450 mengirim data secara terus-menerus melalui UART dengan **protokol biner bawaan pabrik**. Firmware ESP32 mem-parsing protokol ini dengan state machine (`parseByte()` di `hlk_ld2450_esp32.ino`):

```
Header  : 0xAA 0xFF 0x03 0x00        (4 byte)
Data    : 24 byte  → 3 target × 8 byte/target
            per target: [x_lo, x_hi, y_lo, y_hi, speed_lo, speed_hi, res_lo, res_hi]
            (proyek ini hanya membaca x & y, speed/res dari sensor diabaikan
             karena velocity dihitung ulang di sisi host/laptop)
Tail    : 0x55 0xCC                   (2 byte)
─────────────────────────────────────────────
Total frame = 4 + 24 + 2 = 30 byte     (FRAME_SIZE)
```

**Decoding koordinat** (fungsi `decodeCoord()`):

```cpp
int16_t decodeCoord(uint8_t lo, uint8_t hi) {
  uint16_t raw = lo | (hi << 8);
  int16_t  val = (int16_t)(raw & 0x7FFF);   // 15 bit magnitude
  return (raw & 0x8000) ? val : -val;       // bit 15 = sign flag (1 = positif)
}
```
Nilai koordinat dalam satuan **milimeter (mm)**, relatif terhadap posisi sensor (x = kiri/kanan, y = jarak depan sensor).

> Jika suatu saat ingin memakai data kecepatan **bawaan sensor** (bukan hasil hitung ulang di host), byte `speed_lo`/`speed_hi` pada offset `o+4` dan `o+5` per target bisa ditambahkan ke `processFrame()` — saat ini byte tersebut tidak dibaca sama sekali oleh firmware.

---

## 📤 Format Data ESP32 → Laptop

Setelah satu frame valid berhasil di-parsing, ESP32 mengirim **1 baris teks CSV** ke laptop melalui `Serial.println()` (USB Serial, 115200 baud):

```
x1,y1,x2,y2,x3,y3\n
```

Contoh isi baris nyata:
```
520,1830,0,0,0,0
```
Artinya: Target 1 terdeteksi di x=520mm, y=1830mm. Target 2 dan 3 tidak terdeteksi (nilai 0,0 = slot kosong).

Baris ini kemudian dibaca dan diparsing di laptop oleh `parse_radar_frame()` pada **dua file berbeda** dengan format ekspektasi berbeda — **penting untuk dipahami saat development**:

| File | Fungsi | Ekspektasi jumlah nilai | Dipakai untuk |
|---|---|---|---|
| `parsing.py` | `parse_radar_frame(frame)` | **6 nilai** `x1,y1,x2,y2,x3,y3` | Live tracking utama (dipakai `main.py`) — cocok dengan output firmware ESP32 saat ini |
| `calibration_system.py` | `parse_radar_frame(frame)` | **9 nilai** `x1,y1,d1,x2,y2,d2,x3,y3,d3` | Skrip kalibrasi (`calibration_quick_test.py`, `run_distance_calibration_interactive`, dll) |

> ⚠️ **PERHATIAN UNTUK PENGEMBANGAN**: kedua fungsi `parse_radar_frame` ini **tidak identik** dan **tidak kompatibel** satu sama lain karena mengharapkan jumlah kolom serial yang berbeda (6 vs 9). Firmware `.ino` saat ini hanya mengirim **6 nilai** (tanpa kolom distance terpisah). Ini artinya skrip kalibrasi (`calibration_system.py`, `calibration_quick_test.py`) **mengasumsikan firmware versi lama/berbeda** yang mengirim jarak (`distance`) langsung dari ESP32, sementara firmware `.ino` yang ada di repo ini **tidak** mengirim kolom distance. **Sebelum menjalankan kalibrasi, cek dulu apakah firmware ESP32 di device kamu mengirim 6 atau 9 kolom**, dan sesuaikan salah satu sisi (firmware atau skrip kalibrasi) agar konsisten. Ini adalah salah satu titik yang perlu diperbaiki/diselaraskan oleh pengembang selanjutnya (lihat [Keterbatasan & Ide Pengembangan](#-keterbatasan--ide-pengembangan-lanjutan)).

Di `parsing.py`, setelah 6 nilai berhasil diparsing, jarak dihitung ulang secara manual di laptop:
```python
dist = math.sqrt(x**2 + y**2)   # dalam mm
```

---

## 💻 Instalasi Software

### 1. Prasyarat

- **Python** 3.9 – 3.12 (disarankan 3.10/3.11; hindari versi paling baru yang kadang belum didukung penuh oleh `scipy`/`matplotlib`)
- **Arduino IDE** 2.x (untuk flashing firmware ke ESP32)
- **Driver USB-to-Serial** untuk board ESP32 kamu (biasanya otomatis terpasang di Windows 10/11 modern, tapi kalau port tidak muncul, install driver **CP210x** (Silicon Labs) atau **CH340** sesuai chip USB di board kamu)

### 2. Clone repository

```bash
git clone <URL_REPO_INI>
cd <nama-folder-repo>
```

### 3. Buat virtual environment (opsional tapi sangat disarankan)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 4. Install dependency Python

Buat file `requirements.txt` (atau gunakan yang sudah ada di repo bila tersedia) dengan isi berikut, lalu install:

```
pyserial
numpy
scipy
matplotlib
pandas
openpyxl
```

```bash
pip install -r requirements.txt
```

**Rincian dependency & kegunaannya:**

| Package | Dipakai di | Fungsi |
|---|---|---|
| `pyserial` | `parsing.py`, `calibration_system.py`, `calibration_quick_test.py` | Komunikasi serial dengan ESP32 |
| `numpy` | hampir semua file (`ukf.py`, `pf.py`, `metrics.py`, dll) | Operasi matriks & numerik |
| `scipy` | `calibration_system.py` (`scipy.optimize.curve_fit`) | Fitting model kalibrasi logaritmik |
| `matplotlib` | `viz.py`, `main.py` | Visualisasi real-time (dengan backend `QtAgg`) |
| `pandas` | `radar_logger.py` | Menyusun & menyimpan data ke Excel |
| `openpyxl` | `radar_logger.py` (dipakai `pandas.to_excel`) | Engine penulisan file `.xlsx` |
| `tkinter` | `mode_selector.py` | GUI pemilihan mode (FULL/DISTANCE/VELOCITY) saat startup — **sudah termasuk built-in Python di Windows**, tapi di Linux mungkin perlu install terpisah: `sudo apt install python3-tk` |

> Karena `viz.py` menggunakan backend `matplotlib.use("QtAgg")`, pastikan salah satu binding Qt terpasang, contoh: `pip install PyQt6` (atau `PyQt5`/`PySide6`). Jika tidak ada binding Qt yang terpasang, matplotlib akan gagal membuka window dan program akan error saat start.

---

## ⚡ Flashing Firmware ke ESP32

1. Buka **Arduino IDE**, install **board support ESP32** jika belum:
   `File → Preferences → Additional Board Manager URLs`, tambahkan:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
   Lalu `Tools → Board → Boards Manager` → cari **"esp32"** → install (package by Espressif Systems).
2. Buka file `hlk_ld2450_esp32.ino` di Arduino IDE.
3. Pilih board yang sesuai di `Tools → Board → ESP32 Arduino → (pilih model board kamu, mis. "ESP32 Dev Module")`.
4. Pilih port COM/tty yang sesuai di `Tools → Port`.
5. Klik **Upload** (ikon panah kanan).
6. Setelah upload sukses, buka **Serial Monitor** (baud rate **115200**) untuk memastikan data mengalir — kamu akan melihat baris berformat `x1,y1,x2,y2,x3,y3` muncul terus-menerus saat ada target di depan sensor.
7. **Tutup Serial Monitor** sebelum menjalankan skrip Python — port serial hanya bisa dipakai oleh satu aplikasi dalam satu waktu (Serial Monitor Arduino akan mengunci port dan membuat Python gagal connect).

---

## 🚀 Menjalankan Sistem (Quick Start)

Ringkas, langkah demi langkah dari nol sampai dashboard muncul:

1. **Rangkai hardware** sesuai [skema wiring](#-skema-wiring--pemasangan-kabel): LD2450 → ESP32.
2. **Colokkan ESP32 ke laptop** via kabel USB.
3. (Jika firmware belum pernah di-flash / berubah) **Flash firmware** sesuai [langkah di atas](#-flashing-firmware-ke-esp32).
4. Pastikan **Serial Monitor Arduino IDE ditutup** (agar port tidak terkunci).
5. Aktifkan virtual environment (jika dipakai):
   ```bash
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Linux/macOS
   ```
6. Jalankan program utama:
   ```bash
   python main.py
   ```
7. Sebuah **jendela GUI kecil (Tkinter)** akan muncul meminta kamu memilih mode:
   - **FULL** — tampilkan & simpan data jarak DAN kecepatan
   - **DISTANCE** — hanya jarak (kecepatan di-nol-kan)
   - **VELOCITY** — hanya kecepatan (jarak di-nol-kan)
8. Setelah memilih mode, dashboard `matplotlib` akan terbuka menampilkan grafik live untuk target 1, 2, dan 3.
9. Untuk berhenti dan menyimpan data ke Excel, tekan tombol **`Q`** pada jendela dashboard (lihat [Kontrol & Hotkey](#-kontrol--hotkey-saat-runtime)).

> **Auto-deteksi port ESP32**: `parsing.py` (fungsi `find_esp32_port()`) mencari port secara otomatis berdasarkan USB VID chip (`0x10C4` = Silicon Labs CP210x, `0x1A86` = CH340, `0x303A` = Espressif native USB). Kamu **tidak perlu** mengatur nama COM port secara manual untuk `main.py`. Jika ESP32 tidak terdeteksi ("ESP32 Tidak Terdeteksi"), cek driver USB dan pastikan hanya 1 board ESP32 yang tercolok.

> Catatan: `calibration_system.py` dan `calibration_quick_test.py` **tidak** memakai auto-deteksi ini — keduanya memakai `SERIAL_PORT = 'COM5'` yang di-hardcode di bagian atas file. **Ubah nilai ini sesuai port ESP32 di device kamu** (cek di Device Manager pada Windows, atau `ls /dev/tty*` di Linux/macOS) sebelum menjalankan skrip kalibrasi.

---

## 🎯 Kalibrasi Sensor

Kalibrasi bersifat **opsional** — tanpa kalibrasi, sistem tetap berjalan menggunakan data mentah radar (model identitas, tanpa koreksi). Tapi kalibrasi disarankan untuk mengurangi bias sistematik sensor.

### Konsep

- **Kalibrasi jarak** memakai **model logaritmik**: `y = a * ln(b * x + 1) + c`, dilatih dengan `scipy.optimize.curve_fit`. Minimal butuh **3 sampel** titik jarak berbeda; kalau kurang dari 3, otomatis fallback ke model linear.
- **Kalibrasi kecepatan** memakai **model linear**: `y = a * x + b`, dilatih dengan `numpy.linalg.lstsq`. Minimal butuh **2 sampel**.
- Model hasil training disimpan (via `pickle`) ke file **`calibration_models.pkl`**, dan otomatis di-load setiap kali `parsing.py` diimpor (dipakai `main.py`).

### Langkah menjalankan kalibrasi jarak & kecepatan

> ⚠️ Sebelum lanjut, baca ulang [peringatan format 6 vs 9 kolom](#-format-data-esp32--laptop) di atas — `calibration_system.py` mengekspektasikan 9 kolom data serial (termasuk kolom distance eksplisit per target), sedangkan firmware `.ino` di repo ini hanya mengirim 6 kolom. **Selaraskan dulu** salah satu sisi sebelum kalibrasi dijalankan, kalau tidak, `parse_radar_frame()` di `calibration_system.py` akan selalu mengembalikan `None` dan tidak ada sampel yang bisa dikumpulkan.

1. Edit `SERIAL_PORT` di bagian atas `calibration_system.py` sesuai COM port ESP32 kamu.
2. Jalankan:
   ```bash
   python calibration_system.py
   ```
3. Ikuti instruksi interaktif di terminal:
   - **Kalibrasi jarak**: kamu akan diminta menempatkan target pada **6 titik jarak berbeda (1m, 2m, 3m, 4m, 5m, 6m)**. Untuk setiap titik: letakkan target, tekan Enter, sensor akan mengumpulkan sampel selama **5 detik**, lalu kamu konfirmasi apakah jarak sebenarnya sesuai atau input manual. Kamu bisa `[c]ontinue`, `[s]kip` titik tertentu, atau `[q]uit` di tengah proses.
   - **Kalibrasi kecepatan**: kamu tentukan panjang lintasan (misal 3 meter), lalu target berjalan melewati lintasan tersebut sambil sensor merekam **selama 10 detik** per percobaan, dan kamu ukur waktu tempuh sebenarnya dengan stopwatch untuk menghitung ground-truth kecepatan (`jarak/waktu`). Ulangi untuk beberapa kecepatan berbeda (jalan pelan, jalan cepat, dst) agar model linear lebih representatif.
4. Setelah kedua tahap selesai, data mentah kalibrasi otomatis disimpan ke `calibration_raw_data.json`, model dilatih dan disimpan ke `calibration_models.pkl`.

### Verifikasi hasil kalibrasi

Jalankan skrip cepat berikut untuk melihat perbandingan data mentah vs data terkalibrasi secara live (target harus diletakkan di jarak yang diketahui, misal 2.0m):

```bash
python calibration_quick_test.py
```
Edit dulu `SERIAL_PORT` di file tersebut sesuai port kamu. Skrip akan menampilkan `Raw` vs `Calibrated` distance selama 10 detik, lalu ringkasan statistik (mean ± std, rata-rata koreksi).

### Mengaktifkan kalibrasi saat live tracking

Kalibrasi **tidak otomatis aktif** saat `main.py` dijalankan (default `USE_CALIBRATION = False`). Aktifkan dengan menekan tombol **`C`** di jendela dashboard saat program berjalan (lihat bagian berikutnya).

---

## ⌨️ Kontrol & Hotkey Saat Runtime

Semua hotkey aktif saat jendela dashboard (`matplotlib`) sedang fokus:

| Tombol | Fungsi |
|---|---|
| `D` | Toggle **Data Association** ON/OFF (lihat status di pojok kiri bawah dashboard: `da:on`/`da:off`) |
| `C` | Toggle **Calibration** ON/OFF (`cal:on`/`cal:off`) — butuh `calibration_models.pkl` sudah ada, kalau tidak ada file, sistem otomatis fallback ke model identitas (tanpa koreksi) |
| `I` | Info (saat ini hanya mengambil status flag secara internal — **belum menampilkan apa pun ke layar**, lihat catatan di [Keterbatasan](#-keterbatasan--ide-pengembangan-lanjutan)) |
| `Q` | **Quit & Save** — menutup dashboard dan menyimpan seluruh history data + metrik ke `logs/radar_log_<timestamp>.xlsx` |
| `M` | Mode sudah ditentukan di awal program (dipilih lewat GUI Tkinter sebelum dashboard terbuka); menekan `M` hanya menampilkan pesan bahwa mode harus diganti dengan restart program |

Status mode aktif (`FULL`/`DISTANCE`/`VELOCITY`), status Data Association, dan Calibration selalu ditampilkan kecil di pojok kiri-bawah dashboard (teks abu-abu transparan), contoh: `mode:full | da:on cal:off`.

---

## 📂 Struktur Proyek & Penjelasan Tiap File

```
.
├── hlk_ld2450_esp32.ino       # Firmware ESP32 (C++/Arduino) — jembatan sensor ↔ laptop
├── main.py                     # Entry point utama sistem (jalankan file ini)
├── mode_selector.py            # GUI Tkinter pemilihan mode saat startup
├── parsing.py                  # Baca serial, parsing frame, orkestrasi data association + kalibrasi
├── data_association.py         # Algoritma tracking multi-target (gating, cost matching, ghost filter)
├── calibration_system.py       # Sistem kalibrasi (kolektor data, model linear/logaritmik, save/load)
├── calibration_quick_test.py   # Skrip cepat untuk verifikasi hasil kalibrasi secara live
├── calibration_models.pkl      # Model kalibrasi hasil training (binary, hasil `pickle.dump`)
├── ukf.py                      # Implementasi Unscented Kalman Filter (constant-velocity model)
├── pf.py                       # Implementasi Particle Filter (1000 partikel, systematic resampling)
├── metrics.py                  # Perhitungan RMSE, MAE, MBE per target per parameter
├── viz.py                      # Dashboard visualisasi live (matplotlib) + kontrol hotkey
├── radar_logger.py             # Pencatatan histori data & metrik, ekspor ke Excel
└── README.md                   # Dokumentasi ini
```

### Penjelasan detail per file

#### `hlk_ld2450_esp32.ino`
Firmware ESP32. Membaca UART2 dari sensor LD2450 (256000 baud), mem-parsing protokol biner pabrik via state machine (`WAIT_H1 → WAIT_H2 → WAIT_H3 → WAIT_H4 → READ_DATA`), memvalidasi tail byte (`0x55 0xCC`), mendekode koordinat x/y untuk 3 target, lalu mengirim hasilnya sebagai teks CSV ke laptop via USB Serial (115200 baud). **Tidak melakukan filtering/kalkulasi apa pun** — murni bridge protokol.

#### `main.py`
Entry point. Alurnya:
1. Panggil `select_mode_gui()` untuk memilih mode tampilan.
2. Inisialisasi `Metrics()`, `RadarVisualizer()`, generator `read_radar_data()`, serta dictionary `UKF` dan `ParticleFilter` untuk masing-masing target (`t1`, `t2`, `t3`).
3. Menjalankan `data_loop()` di **thread terpisah** (`threading.Thread(..., daemon=True)`) agar pembacaan data tidak memblokir event-loop GUI matplotlib.
4. Di setiap iterasi loop: hitung `dt`, jalankan `predict()` + `update()` UKF dan PF untuk tiap target yang datanya valid (bukan 0,0), catat metrik, lalu kirim ke `viz.append_data()`.
5. Reset filter (UKF, PF) dan metrik otomatis saat target hilang (raw distance & velocity = 0) agar filter tidak "menyeret" estimasi lama dari target yang sudah tidak ada.
6. `plt.show()` menjalankan main loop GUI matplotlib (blocking) sampai window ditutup.

#### `mode_selector.py`
GUI sederhana berbasis `tkinter` dengan 3 tombol (FULL/DISTANCE/VELOCITY). Window otomatis center di layar. Jika ditutup tanpa memilih (klik tombol X), default ke mode `FULL`.

#### `parsing.py`
"Otak" orkestrasi data di sisi laptop:
- `find_esp32_port()` — auto-deteksi port ESP32 berdasarkan USB VID.
- `parse_radar_frame(frame)` — parsing baris CSV 6-kolom dari ESP32 menjadi 3 dict deteksi (`posx`, `posy`, `distance` dalam mm).
- `read_radar_data()` — **generator** yang terus membaca serial, memilih jalur data association (kalau `USE_DATA_ASSOCIATION=True`) atau perhitungan velocity sederhana per-index (kalau `False`), lalu menerapkan kalibrasi jika `USE_CALIBRATION=True`, dan `yield` output final `{'t1': {...}, 't2': {...}, 't3': {...}}` dalam satuan meter & m/s.
- `toggle_data_association()`, `toggle_calibration()`, `get_flags()` — dipanggil dari hotkey `D`/`C`/`I` di `main.py`.

#### `data_association.py`
Algoritma tracking multi-target yang menjaga **konsistensi ID** (t1/t2/t3) antar frame walau target bergerak, berpapasan, atau sementara hilang dari jangkauan. Komponen kunci:
- **Gating** (`GATE_DIST = 1.5m`) — kandidat asosiasi harus berada dalam radius ini dari posisi prediksi track.
- **Cost gabungan** — kombinasi bobot dari jarak Euclidean (`ALPHA=0.5`), heading error / kesamaan arah gerak (`BETA=0.3`), dan momentum-x / potensi salah asosiasi saat target bergerak berlawanan arah horizontal (`GAMMA=0.2`).
- **Deduplication / ghost filter** (`GHOST_DIST=0.5m`) — LD2450 kadang mendeteksi 1 orang sebagai 2 titik berdekatan; deteksi kedua yang terlalu dekat dengan yang sudah dipilih akan dibuang.
- **Merge zone handling** (`MERGE_DIST=0.3m`) — saat 2 track saling berdekatan (berpotensi tertukar), posisi keduanya sementara di-*coasting* menggunakan posisi prediksi, bukan deteksi mentah, untuk mengurangi ID-switch.
- **Lost/Free state machine** — track yang tidak match selama `LOST_TIMEOUT=10` frame otomatis di-reset ke status `FREE` (slot kosong siap dipakai target baru).

#### `calibration_system.py`
- `CalibrationCollector` — mengumpulkan pasangan data (measured, ground_truth) untuk jarak & kecepatan, dan menyimpannya sebagai JSON mentah (`calibration_raw_data.json`).
- `LinearCalibration` — fit `y = a*x + b` dengan least squares.
- `LogarithmicCalibration` — fit `y = a*ln(b*x+1) + c` dengan `scipy.optimize.curve_fit`, fallback otomatis ke linear jika sampel < 3 atau fitting gagal konvergen.
- `CalibrationManager` — orkestrasi keseluruhan: `calibrate()` (training), `apply_calibration()` (dipakai saat runtime), `save()`/`load()` (persist ke `.pkl`).
- `run_distance_calibration_interactive()` / `run_velocity_calibration_interactive()` — wizard interaktif via terminal untuk mengumpulkan sampel kalibrasi (lihat [Kalibrasi Sensor](#-kalibrasi-sensor)).
- `main_calibration()` — fungsi utama yang dijalankan saat file ini dieksekusi langsung (`python calibration_system.py`).
- ⚠️ Berisi definisi `parse_radar_frame()` versi **9 kolom** (lihat peringatan format di atas) — berbeda dari versi di `parsing.py`.

#### `calibration_quick_test.py`
Skrip verifikasi cepat: load `calibration_models.pkl`, baca serial selama N detik (default 10), tampilkan perbandingan raw vs calibrated distance secara live plus ringkasan statistik di akhir.

#### `ukf.py`
Implementasi **Unscented Kalman Filter** custom (tanpa library eksternal seperti `filterpy`), dengan:
- State vector 2 dimensi: `[distance, velocity]`.
- Model proses **constant velocity**: `f(x, dt) = [x0 + x1*dt, x1]`.
- Model pengukuran identitas: `h(x) = x` (mengukur langsung distance & velocity).
- Parameter UKF standar: `alpha=1e-3`, `beta=2.0`, `kappa=0`.
- Noise proses `Q` (default `q_dist=0.1, q_vel=0.1`) dan noise pengukuran `R` (default `r_dist=0.8, r_vel=0.8`) — **bisa di-tuning** sesuai karakteristik noise sensor kamu.
- Ada fallback ke SVD jika Cholesky decomposition gagal (matriks kovarian tidak positive-definite) — melindungi dari crash numerik.

#### `pf.py`
Implementasi **Particle Filter** custom dengan:
- `N=1000` partikel default.
- Noise proses (`process_std_dist/vel=0.2`) dan noise pengukuran (`meas_std_dist/vel=0.005`) — **catatan: nilai default sangat kecil untuk meas_std, artinya likelihood sangat "tajam"/sensitif terhadap deviasi pengukuran; ini bisa jadi titik tuning penting**.
- Predict: random walk + constant velocity motion model, ditambah noise Gaussian.
- Update: hitung likelihood Gaussian per partikel terhadap pengukuran baru, normalisasi bobot.
- **Systematic resampling** dipicu otomatis saat **Effective Sample Size (ESS)** turun di bawah `ess_threshold * N` (default 50%), untuk mencegah *particle degeneracy*.
- Estimasi akhir = rata-rata partikel berbobot (`weighted mean`).

#### `metrics.py`
Menyimpan histori penuh (list) dari `actual` (data mentah/raw sebagai referensi), `ukf`, dan `pf` per target per parameter (`distance`, `velocity`). Menghitung:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MBE** (Mean Bias Error — menunjukkan arah bias, over/under estimation)

> Catatan metodologis: metrik ini membandingkan hasil filter terhadap **data mentah radar**, bukan terhadap ground truth fisik sebenarnya (kecuali data mentahnya sudah dikalibrasi). Untuk evaluasi Tugas Akhir yang lebih rigorous, pertimbangkan membandingkan terhadap ground truth independen (misal meteran/motion capture) di sesi pengujian terpisah.

#### `viz.py`
Dashboard `matplotlib` dengan `FuncAnimation` (interval 50ms). Punya 3 layout berbeda sesuai mode (`_create_full_layout`, `_create_distance_layout`, `_create_velocity_layout`) — masing-masing menampilkan grid grafik per target (3 baris) + panel metrik ringkas di sisi kanan tiap baris. Setiap frame data baru juga otomatis dicatat ke `Logger` (`radar_logger.py`) melalui `append_data()`.

#### `radar_logger.py`
Kelas `Logger` sederhana yang mengumpulkan setiap baris data (raw, UKF, PF, + semua metrik RMSE/MAE/MBE) ke dalam list of dict, lalu `save_excel()` mengonversinya ke `pandas.DataFrame` dan ekspor ke `.xlsx` di folder `logs/` (dibuat otomatis jika belum ada), dengan nama file `radar_log_<YYYYMMDD_HHMMSS>.xlsx`.

---

## 🔬 Detail Algoritma

### Unscented Kalman Filter (UKF) — `ukf.py`

UKF dipilih karena mampu menangani non-linearitas lebih baik daripada EKF tanpa perlu menghitung Jacobian, menggunakan pendekatan **sigma points**:

1. **Sigma points generation** — `2n+1 = 5` titik (n=2 dimensi state) dibangkitkan dari mean & kovarian state via Cholesky decomposition.
2. **Predict** — setiap sigma point dipropagasi lewat model constant-velocity `f(x,dt)`, lalu direkonstruksi jadi mean & kovarian prediksi (`x_pred`, `P_pred`), ditambah noise proses `Q`.
3. **Update** — sigma points hasil predict ditransformasi lewat model pengukuran `h(x)`, dihitung cross-covariance `Pxz` dan innovation covariance `Pzz` (+ noise pengukuran `R`), Kalman gain `K = Pxz @ inv(Pzz)`, lalu state & kovarian dikoreksi dengan pengukuran baru `z`.

### Particle Filter (PF) — `pf.py`

Pendekatan Monte Carlo yang merepresentasikan distribusi posterior state dengan **1000 partikel** berbobot:

1. **Predict** — setiap partikel dipropagasi dengan model constant-velocity + noise proses Gaussian (mensimulasikan ketidakpastian gerak).
2. **Update** — bobot tiap partikel diperbarui berdasarkan **likelihood Gaussian** seberapa dekat prediksi partikel dengan pengukuran baru; bobot dinormalisasi.
3. **Resampling** — jika Effective Sample Size (ESS) turun di bawah threshold (50% dari N), lakukan **systematic resampling** untuk mengganti partikel berbobot rendah dengan duplikat partikel berbobot tinggi (mencegah degenerasi).
4. **Estimate** — output akhir adalah rata-rata tertimbang seluruh partikel.

### Data Association — `data_association.py`

Menggunakan pendekatan **greedy nearest-neighbor dengan cost gabungan** (bukan Hungarian Algorithm/Munkres, jadi tidak menjamin solusi assignment global optimal, tapi jauh lebih ringan secara komputasi untuk real-time):

```
cost(track, deteksi) = ALPHA * jarak_ternormalisasi
                      + BETA  * heading_error
                      + GAMMA * momentum_x_error
```

Semua pasangan (track, deteksi) diurutkan dari cost terkecil, lalu diambil greedy (pasangan cost terendah lebih dulu, track/deteksi yang sudah dipakai di-skip) — mirip pendekatan **Global Nearest Neighbor (GNN)** yang disederhanakan.

---

## 📊 Output & Format Log Excel

File Excel disimpan di `logs/radar_log_<timestamp>.xlsx` saat kamu menekan `Q`. Setiap baris merepresentasikan **1 sample waktu untuk 1 target**, dengan kolom:

| Kolom | Keterangan |
|---|---|
| `timestamp` | Waktu Unix (detik, `time.time()`) saat sample dicatat |
| `target` | `t1` / `t2` / `t3` |
| `distance_raw`, `velocity_raw` | Data mentah (sudah melalui data association & kalibrasi jika diaktifkan) |
| `distance_ukf`, `velocity_ukf` | Hasil estimasi UKF |
| `distance_pf`, `velocity_pf` | Hasil estimasi PF |
| `rmse_dist_ukf`, `rmse_vel_ukf`, `rmse_dist_pf`, `rmse_vel_pf` | RMSE kumulatif hingga sample tersebut |
| `mae_dist_ukf`, `mae_vel_ukf`, `mae_dist_pf`, `mae_vel_pf` | MAE kumulatif |
| `mbe_dist_ukf`, `mbe_vel_ukf`, `mbe_dist_pf`, `mbe_vel_pf` | MBE kumulatif |

> Catatan: RMSE/MAE/MBE yang tercatat per baris adalah **nilai kumulatif** (dihitung dari seluruh histori sampai titik itu), bukan nilai instan per-sample — karena `metrics.py` menyimpan seluruh list histori dan menghitung ulang di setiap panggilan `get_metrics()`.

---

## 🎛 Parameter Tuning

Berikut parameter-parameter kunci yang paling sering perlu di-tuning saat pengembangan lanjutan:

### `data_association.py`

| Parameter | Default | Fungsi |
|---|---|---|
| `GATE_DIST` | 1.5 m | Radius maksimum asosiasi track↔deteksi |
| `LOST_TIMEOUT` | 10 frame | Toleransi berapa frame track boleh "hilang" sebelum di-free |
| `MIN_DIST_NEW` | 0.05 m | Jarak minimum agar deteksi dianggap valid (bukan noise dekat sensor) |
| `HISTORY_LEN` | 8 frame | Panjang riwayat posisi untuk hitung heading |
| `ALPHA / BETA / GAMMA` | 0.5 / 0.3 / 0.2 | Bobot komponen cost (jarak/heading/momentum-x) — total idealnya ≈ 1.0 |
| `GHOST_DIST` | 0.5 m | Radius dedup deteksi ganda dari 1 target fisik |
| `MERGE_DIST` | 0.3 m | Radius trigger mode "coasting" saat 2 track berdekatan |

### `ukf.py`

| Parameter | Default | Fungsi |
|---|---|---|
| `q_dist`, `q_vel` | 0.1, 0.1 | Noise proses (seberapa besar model dipercaya bisa "salah") |
| `r_dist`, `r_vel` | 0.8, 0.8 | Noise pengukuran (seberapa besar sensor dianggap noisy) — makin besar `R` relatif ke `Q`, filter makin "percaya" model & kurang responsif ke pengukuran baru |
| `alpha`, `beta`, `kappa` | 1e-3, 2.0, 0.0 | Parameter spread sigma points (standar UKF, biasanya tidak perlu diubah kecuali paham teori UKF) |

### `pf.py`

| Parameter | Default | Fungsi |
|---|---|---|
| `N` | 1000 | Jumlah partikel — makin besar makin akurat tapi makin berat komputasi |
| `process_std_dist/vel` | 0.2 | Noise proses (sebaran prediksi partikel) |
| `meas_std_dist/vel` | 0.005 | Noise pengukuran (ketajaman likelihood) — **sangat kecil**, pertimbangkan dinaikkan jika PF terlihat terlalu "kaku"/lambat merespons perubahan mendadak |
| `ess_threshold` | 0.5 | Batas ESS relatif (terhadap N) untuk trigger resampling |

---

## 🛠 Troubleshooting

| Gejala | Kemungkinan Penyebab & Solusi |
|---|---|
| `ESP32 Tidak Terdeteksi` saat `python main.py` | Cek kabel USB (harus data cable, bukan charging-only), cek driver CP210x/CH340 terpasang, cek Serial Monitor Arduino IDE sudah **ditutup** (port terkunci), cek hanya 1 board ESP32 tercolok |
| Serial terbuka tapi tidak ada data masuk / semua nilai 0 | Cek wiring TX/RX LD2450 ↔ ESP32 tidak terbalik/kendor, cek LD2450 dapat power yang cukup, cek tidak ada target yang benar-benar terdeteksi (coba gerakkan tangan di depan sensor) |
| Program crash saat start dengan error terkait Qt / backend matplotlib | Install binding Qt: `pip install PyQt6` (atau ganti baris `matplotlib.use("QtAgg")` di `main.py` ke backend lain seperti `"TkAgg"` jika tidak ingin pakai Qt) |
| Kalibrasi tidak mengumpulkan sampel sama sekali (selalu "Tidak ada Data") | Kemungkinan besar **mismatch format frame** — cek [peringatan 6 vs 9 kolom](#-format-data-esp32--laptop): firmware saat ini mengirim 6 kolom, tapi `calibration_system.py` mengekspektasikan 9 kolom |
| Dashboard terasa lag / FPS rendah | Turunkan `max_points` di `RadarVisualizer(..., max_points=100)` pada `main.py`, atau kurangi `N` (jumlah partikel) di `pf.py` |
| Target sering "loncat" ID antar t1/t2/t3 saat berpapasan | Aktifkan Data Association (`D`), lalu tuning `GATE_DIST`, `BETA` (heading weight), `MERGE_DIST` di `data_association.py` |
| `PermissionError` / `Access is denied` saat buka serial port | Port sedang dipakai aplikasi lain (Serial Monitor, skrip Python lain yang masih berjalan) — tutup semua aplikasi lain yang memakai port tersebut |
| File Excel gagal tersimpan saat tekan `Q` | Pastikan folder `logs/` bisa dibuat/ditulis (cek permission folder project), pastikan tidak ada file `.xlsx` dengan nama sama yang sedang terbuka di Excel |

---

## 🧩 Keterbatasan & Ide Pengembangan Lanjutan

Bagian ini khusus ditulis untuk **adik tingkat yang akan melanjutkan proyek ini**:

1. **Mismatch format parsing 6 vs 9 kolom** — `parsing.py` dan `calibration_system.py` punya definisi `parse_radar_frame()` berbeda (lihat [Format Data ESP32 → Laptop](#-format-data-esp32--laptop)). Sebaiknya disatukan jadi 1 modul parsing yang dipakai bersama (`import` dari satu sumber), dan pastikan firmware `.ino` konsisten dengan format yang diharapkan seluruh skrip Python.
2. **Hotkey `I` (Info) belum berfungsi** — di `main.py`, handler `on_key_press` untuk tombol `I` hanya memanggil `get_flags()` dan `viz.get_display_mode()` tanpa menampilkan apa pun ke user. Bisa dikembangkan jadi overlay info tambahan (misal jumlah track aktif, FPS, dsb).
3. **`SERIAL_PORT` hardcoded** di `calibration_system.py` (`'COM5'`) dan `calibration_quick_test.py` — sebaiknya diganti memakai `find_esp32_port()` dari `parsing.py` agar konsisten dengan `main.py` yang sudah auto-detect.
4. **Data association memakai posisi (mm) tapi `GATE_DIST`/`GHOST_DIST`/`MERGE_DIST` didefinisikan dalam meter** — perhatikan satuan saat membaca kode: posisi `posx`/`posy` di `data_association.py` berasal dari koordinat mentah ESP32 yang **belum dikonversi ke meter** (masih mm), sementara nilai gate seperti `GATE_DIST = 1.5` ditulis dengan komentar "meter". Cek ulang konsistensi satuan ini — ini titik rawan bug ketika tuning ulang parameter.
5. **Velocity dihitung dari selisih posisi antar-frame**, bukan dari data speed bawaan sensor LD2450 (yang sebenarnya tersedia di protokol biner tapi tidak dipakai/dibaca firmware). Membaca & memakai speed bawaan sensor bisa jadi eksperimen tambahan menarik untuk dibandingkan dengan pendekatan saat ini.
6. **Data association pakai greedy matching**, bukan optimal assignment (Hungarian Algorithm) — untuk skenario dengan banyak target berpapasan, algoritma Hungarian (`scipy.optimize.linear_sum_assignment`) berpotensi memberikan hasil asosiasi yang lebih stabil.
7. **Metrik (RMSE/MAE/MBE) dihitung terhadap data mentah radar**, bukan ground truth fisik independen — untuk laporan Tugas Akhir yang lebih kuat secara metodologis, pertimbangkan sesi pengujian terpisah dengan ground truth independen (meteran/marker/motion capture) untuk validasi akurasi absolut UKF vs PF.
8. **Belum ada file `requirements.txt`** resmi di repo — pertimbangkan menambahkannya (isi sudah dijabarkan di bagian [Instalasi Software](#-instalasi-software)) agar `pip install -r requirements.txt` langsung jalan.
9. **Noise pengukuran PF (`meas_std_dist/vel=0.005`) sangat kecil** dibanding noise pengukuran UKF (`r_dist/vel=0.8`) — kedua filter memakai skala noise yang jauh berbeda secara default, sehingga perbandingan performa UKF vs PF di laporan TA sebaiknya menyertakan justifikasi/eksperimen sensitivitas terhadap parameter noise ini agar perbandingan adil (apple-to-apple).
10. **Belum ada automated test** untuk modul-modul inti (`ukf.py`, `pf.py`, `data_association.py`, parsing). Menambahkan unit test (`pytest`) dengan data sintetis akan sangat membantu saat refactor ke depannya.

---

## ❓ FAQ Singkat

**Q: Apakah program bisa jalan tanpa sensor radar (mode simulasi)?**
A: Saat ini tidak ada mode simulasi bawaan — `main.py` selalu menunggu data dari `read_radar_data()` yang membutuhkan ESP32 tersambung. Untuk pengembangan/testing tanpa hardware, kamu bisa membuat generator dummy yang meniru format output `parsing.py` (`{'t1': {'distance':.., 'velocity':..}, ...}`).

**Q: Kenapa harus 2 filter (UKF & PF) sekaligus?**
A: Karena tujuan Tugas Akhir ini adalah **membandingkan** performa kedua metode, bukan memilih salah satu untuk dipakai secara final. Keduanya dijalankan paralel di setiap frame agar bisa dibandingkan langsung dari data yang identik.

**Q: Apakah bisa dipakai untuk lebih dari 3 target?**
A: Tidak langsung, karena keterbatasan hardware — LD2450 secara native hanya mendukung deteksi hingga **3 target** dalam satu waktu, dan seluruh pipeline (parsing, tracking, filter dictionary `t1/t2/t3`) dirancang mengikuti batas ini.

**Q: Di mana saya harus mulai kalau ingin mengembangkan fitur baru?**
A: Baca [Arsitektur & Alur Data](#-arsitektur--alur-data) dulu untuk memahami urutan pemrosesan, lalu lihat [Struktur Proyek](#-struktur-proyek--penjelasan-tiap-file) untuk tahu file mana yang perlu disentuh sesuai fitur yang ingin ditambahkan.

---

*Dokumentasi ini disusun untuk memudahkan transfer knowledge Tugas Akhir kepada pengembang berikutnya. Selamat mengembangkan! 🚀*
