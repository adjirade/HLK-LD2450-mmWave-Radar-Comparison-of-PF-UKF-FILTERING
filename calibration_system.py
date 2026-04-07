import serial
import time
import math
import numpy as np
import pickle
import json
from datetime import datetime

SERIAL_PORT = 'COM5'
BAUD_RATE = 115200
TIMEOUT = 0.1

# Pengumpul Data 
class CalibrationCollector:
    def __init__(self):
        self.distance_data = {'measured': [], 'ground_truth': []}
        self.velocity_data = {'measured': [], 'ground_truth': []}

    def add_distance_sample(self, measured, ground_truth):
        self.distance_data['measured'].append(measured)
        self.distance_data['ground_truth'].append(ground_truth)
        print(f"Sampe Jarak Ditambahkan: measured={measured:.3f}m, true={ground_truth:.3f}m")

    def add_velocity_sample(self, measured, ground_truth):
        self.velocity_data['measured'].append(measured)
        self.velocity_data['ground_truth'].append(ground_truth)
        print(f"Sampe Kecepatan Ditambahkan: measured={measured:.3f}m/s, true={ground_truth:.3f}m/s")

    def save_raw_data(self, filename='calibration_raw_data.json'):
        data = {
            'distance': self.distance_data,
            'velocity': self.velocity_data,
            'timestamp': datetime.now().isoformat()
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Data mentah disimpan ke {filename}")


# Model Linear
class LinearCalibration:
    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self.r2 = 0.0

    def fit(self, measured, ground_truth):
        measured = np.array(measured)
        ground_truth = np.array(ground_truth)

        if len(measured) < 2:
            print("Data tidak cukup untuk kalibrasi linear")
            self.a = 1.0
            self.b = 0.0
            return

        A = np.vstack([measured, np.ones(len(measured))]).T
        self.a, self.b = np.linalg.lstsq(A, ground_truth, rcond=None)[0]

        predicted = self.predict(measured)
        ss_res = np.sum((ground_truth - predicted) ** 2)
        ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
        self.r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        print(f"Model Linear: y = {self.a:.6f} * x + {self.b:.6f}, R² = {self.r2:.4f}")

    def predict(self, measured):
        return self.a * np.array(measured) + self.b

    def get_params(self):
        return {
            'type': 'linear',
            'a': float(self.a),
            'b': float(self.b),
            'r2': float(self.r2)
        }


# Model Logaritmik
class LogarithmicCalibration:
    def __init__(self):
        self.params = None
        self.r2 = 0.0

    def fit(self, measured, ground_truth):
        measured = np.array(measured)
        ground_truth = np.array(ground_truth)

        if len(measured) < 3:
            print("Data tidak cukup untuk kalibrasi logaritmik")
            print("Model linear akan digunakan...")
            if len(measured) >= 2:
                A = np.vstack([measured, np.ones(len(measured))]).T
                a, b = np.linalg.lstsq(A, ground_truth, rcond=None)[0]
                self.params = np.array([a, 0.0, b])
            else:
                self.params = np.array([1.0, 0.0, 0.0])
            return

        try:
            from scipy.optimize import curve_fit

            def log_func(x, a, b, c):
                return a * np.log(b * x + 1) + c

            p0 = [1.0, 1.0, 0.0]
            self.params, _ = curve_fit(log_func, measured, ground_truth, p0=p0, maxfev=10000)

            print(f"Model Logaritmik: y = {self.params[0]:.4f} * ln({self.params[1]:.4f} * x + 1) + {self.params[2]:.4f}")

            predicted = self.predict(measured)
            ss_res = np.sum((ground_truth - predicted) ** 2)
            ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
            self.r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            print(f"R² = {self.r2:.4f}")

        except Exception as e:
            print(f"Gagal Menyesuaikan logaritmik: {e}")
            print("Model linear akan digunakan...")
            if len(measured) >= 2:
                A = np.vstack([measured, np.ones(len(measured))]).T
                a, b = np.linalg.lstsq(A, ground_truth, rcond=None)[0]
                self.params = np.array([a, 0.0, b])
            else:
                self.params = np.array([1.0, 0.0, 0.0])

    def predict(self, measured):
        if self.params is None:
            return np.array(measured)

        measured_arr = np.array(measured)
        try:
            return self.params[0] * np.log(self.params[1] * measured_arr + 1) + self.params[2]
        except Exception as e:
            print(f"Gagal memprediksi: {e}")
            return measured_arr

    def get_params(self):
        return {
            'type': 'logarithmic',
            'params': self.params.tolist() if self.params is not None else None,
            'r2': float(self.r2)
        }


# Kalibrasi
class CalibrationManager:
    def __init__(self):
        self.distance_model = None
        self.velocity_model = None

    def calibrate(self, collector):
        print("\n" + "=" * 60)
        print("KALIBRASI JARAK (Logaritmik)")
        print("=" * 60)

        self.distance_model = LogarithmicCalibration()
        if len(collector.distance_data['measured']) >= 2:
            self.distance_model.fit(
                collector.distance_data['measured'],
                collector.distance_data['ground_truth']
            )
        else:
            print("Data jarak tidak cukup, menggunakan model identitas")
            self.distance_model = LogarithmicCalibration()
            self.distance_model.params = np.array([1.0, 0.0, 0.0])

        print("\n" + "=" * 60)
        print("KALIBRASI KECEPATAN (Linear)")
        print("=" * 60)

        self.velocity_model = LinearCalibration()
        if len(collector.velocity_data['measured']) >= 2:
            self.velocity_model.fit(
                collector.velocity_data['measured'],
                collector.velocity_data['ground_truth']
            )
        else:
            print("Data kecepatan tidak cukup, menggunakan model identitas")
            self.velocity_model = LinearCalibration()
            self.velocity_model.a = 1.0
            self.velocity_model.b = 0.0

    def apply_calibration(self, distance_raw, velocity_raw):
        if self.distance_model is None or self.velocity_model is None:
            print("Model belum dikalibrasi!")
            return distance_raw, velocity_raw

        try:
            distance_pred = self.distance_model.predict([distance_raw])
            if isinstance(distance_pred, np.ndarray):
                distance_corrected = float(distance_pred[0]) if distance_pred.size > 0 else distance_raw
            else:
                distance_corrected = float(distance_pred)

            velocity_pred = self.velocity_model.predict([velocity_raw])
            if isinstance(velocity_pred, np.ndarray):
                velocity_corrected = float(velocity_pred[0]) if velocity_pred.size > 0 else velocity_raw
            else:
                velocity_corrected = float(velocity_pred)

            return distance_corrected, velocity_corrected

        except Exception as e:
            print(f"Gagal mengkalibrasi: {e}, mengembalikan nilai mentah")
            return distance_raw, velocity_raw

    def save(self, filename='calibration_models.pkl'):
        data = {
            'distance': self.distance_model.get_params() if self.distance_model else None,
            'velocity': self.velocity_model.get_params() if self.velocity_model else None,
            'timestamp': datetime.now().isoformat()
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"\nModel kalibrasi disimpan ke {filename}")

    def load(self, filename='calibration_models.pkl'):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            print(f"Memuat kalibrasi dari {filename}...")

            dist_params = data.get('distance')
            if dist_params is None:
                self.distance_model = LogarithmicCalibration()
                self.distance_model.params = np.array([1.0, 0.0, 0.0])
            elif dist_params['type'] == 'logarithmic':
                self.distance_model = LogarithmicCalibration()
                params = dist_params.get('params')
                self.distance_model.params = np.array(params) if params and len(params) > 0 else np.array([1.0, 0.0, 0.0])
                self.distance_model.r2 = dist_params.get('r2', 0.0)
            else:
                print(f"Model jarak '{dist_params['type']}' tidak didukung, menggunakan identity.")
                self.distance_model = LogarithmicCalibration()
                self.distance_model.params = np.array([1.0, 0.0, 0.0])
            vel_params = data.get('velocity')
            if vel_params is None:
                self.velocity_model = LinearCalibration()
            elif vel_params['type'] == 'linear':
                self.velocity_model = LinearCalibration()
                self.velocity_model.a = vel_params.get('a', 1.0)
                self.velocity_model.b = vel_params.get('b', 0.0)
                self.velocity_model.r2 = vel_params.get('r2', 0.0)
            else:
                print(f"Model kecepatan '{vel_params['type']}' tidak didukung, menggunakan identity.")
                self.velocity_model = LinearCalibration()

            print(f"Model jarak '{dist_params['type'] if dist_params else 'none'}, "
                  f"R²={dist_params.get('r2', 0) if dist_params else 0:.4f}")
            print(f"Model kecepatan '{vel_params['type'] if vel_params else 'none'}, "
                  f"R²={vel_params.get('r2', 0) if vel_params else 0:.4f}")

        except FileNotFoundError:
            print(f"File kalibrasi '{filename}' tidak ditemukan.")
            self._set_identity_models()
        except Exception as e:
            print(f"Gagal memuat kalibrasi: {e}")
            self._set_identity_models()

    def _set_identity_models(self):
        self.distance_model = LogarithmicCalibration()
        self.distance_model.params = np.array([1.0, 0.0, 0.0])
        self.velocity_model = LinearCalibration()
        self.velocity_model.a = 1.0
        self.velocity_model.b = 0.0


# Mengambil data dari sensor
def parse_radar_frame(frame):
    try:
        values = [float(x) for x in frame.strip().split(',')]
        if len(values) != 9:
            return None
        targets = {
            't1': {'posx': values[0], 'posy': values[1], 'distance': values[2]},
            't2': {'posx': values[3], 'posy': values[4], 'distance': values[5]},
            't3': {'posx': values[6], 'posy': values[7], 'distance': values[8]},
        }
        return targets
    except ValueError:
        return None


def run_distance_calibration_interactive(serial_port=SERIAL_PORT):
    print("\n" + "=" * 60)
    print("KALIBRASI JARAK")
    print("=" * 60)
    print("Anda akan diminta untuk menempatkan target pada 6 jarak berbeda:")
    print("  1m, 2m, 3m, 4m, 5m, 6m")
    print("\nAnda bisa:")
    print("  - Lakukan semua kalibrasi")
    print("  - Skip beberapa titik")
    print("  - Skip seluruh kalibrasi jarak")
    print("=" * 60)

    skip_all = input("\nLewati SEMUA kalibrasi jarak? (y/n, default=n): ") or 'n'

    collector = CalibrationCollector()

    if skip_all.lower() == 'y':
        print("Melewati kalibrasi jarak. Akan menggunakan model identitas.")
        return collector

    try:
        ser = serial.Serial(serial_port, BAUD_RATE, timeout=TIMEOUT)
        time.sleep(2)
    except Exception as e:
        print(f"Tidak dapat membuka port serial: {e}")
        return collector

    suggested_distances = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    for idx, dist in enumerate(suggested_distances):
        print(f"\n--- Titik Kalibrasi {idx + 1}/{len(suggested_distances)} ---")
        action = input(f"▶ Jarak {dist}m: [c]ontinue / [s]kip / [q]uit calibration? (default=c): ") or 'c'

        if action.lower() == 'q':
            print("Kalibrasi jarak dihentikan oleh user.")
            break
        elif action.lower() == 's':
            print(f"Melewati {dist}m...")
            continue

        input(f"   Letakkan target pada jarak {dist}m, tekan Enter untuk mulai...")
        print(f"   Mengumpulkan data untuk {dist}m... (5 detik)")

        samples = []
        start_time = time.time()
        while time.time() - start_time < 5:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore')
                targets = parse_radar_frame(line)
                if targets and targets['t1']['distance'] > 0:
                    samples.append(targets['t1']['distance'])

        if len(samples) > 0:
            measured_mean = np.mean(samples)
            measured_std = np.std(samples)
            print(f"Membaca Sensor: {measured_mean:.3f} ± {measured_std:.3f} m ({len(samples)} sampel)")

            confirm = input(f"   Gunakan jarak sebenarnya {dist}m? (y/n): ")
            if confirm.lower() == 'y':
                ground_truth = dist
            else:
                ground_truth_input = input("Input jarak sebenarnya (meter, atau 's' untuk Melewati): ")
                if ground_truth_input.lower() == 's':
                    print("Dilewati")
                    continue
                try:
                    ground_truth = float(ground_truth_input)
                except ValueError:
                    print("Input Tidak Valid")
                    continue

            collector.add_distance_sample(measured_mean, ground_truth)
        else:
            print("Tidak ada Data")
            retry = input("   Retry titik ini? (y/n): ")
            if retry.lower() == 'y':
                suggested_distances.append(dist)

    ser.close()

    num_samples = len(collector.distance_data['measured'])
    print(f"\nKalibrasi Jarak Selesai: {num_samples} Sampel")

    if num_samples == 0:
        print("Peringatan: Tidak ada sampel jarak yang terkumpul. Model identity akan digunakan.")
    elif num_samples < 3:
        print("Peringatan: Kurang dari 3 sampel (logarithmic butuh minimal 3). Akan fallback ke linear.")

    return collector


def run_velocity_calibration_interactive(serial_port=SERIAL_PORT):
    """Prosedur kalibrasi kecepatan interaktif"""
    print("\n" + "=" * 60)
    print("VELOCITY CALIBRATION PROCEDURE")
    print("=" * 60)
    print("Anda akan diminta untuk:")
    print("  1. Buat lintasan dengan panjang tertentu (misal 3 meter)")
    print("  2. Minta target berjalan melewati lintasan")
    print("  3. Ukur waktu tempuh dengan stopwatch")
    print("  4. Ulangi beberapa kali dengan kecepatan berbeda")
    print("=" * 60)

    skip_all = input("\nLewati SEMUA kalibrasi kecepatan? (y/n, default=n): ") or 'n'

    collector = CalibrationCollector()

    if skip_all.lower() == 'y':
        print("Melewati Kalibrasi Kecepatan")
        return collector

    try:
        ser = serial.Serial(serial_port, BAUD_RATE, timeout=TIMEOUT)
        time.sleep(2)
    except Exception as e:
        print(f"Tidak Bisa Membuka Port {e}")
        return collector

    track_length = float(input("\nPanjang lintasan (meter): "))
    num_trials = int(input("Jumlah percobaan yang ingin dilakukan: "))

    prev_pos = {'t1': None}
    prev_time = None
    completed_trials = 0

    for trial in range(num_trials):
        print(f"\nPercobaan {trial + 1}/{num_trials}")
        action = input(f"Percobaan {trial + 1}: [c]ontinue / [s]kip / [q]uit calibration? (default=c): ") or 'c'

        if action.lower() == 'q':
            print("Kalibrasi kecepatan dihentikan")
            break
        elif action.lower() == 's':
            print(f"Melewati percobaan {trial + 1}...")
            continue

        input(f"Siapkan target, tekan Enter untuk mulai...")
        print("Mengumpulkan data kecepatan... (10 detik)")

        velocities = []
        start_time = time.time()
        prev_pos['t1'] = None
        prev_time = None

        while time.time() - start_time < 10:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore')
                targets = parse_radar_frame(line)

                if targets:
                    now = time.time()
                    if prev_pos['t1'] is not None and prev_time is not None:
                        dt = now - prev_time
                        if dt > 0:
                            dx = targets['t1']['posx'] - prev_pos['t1']['posx']
                            dy = targets['t1']['posy'] - prev_pos['t1']['posy']
                            v = math.sqrt(dx ** 2 + dy ** 2) / dt / 1000.0
                            if v > 0.1:
                                velocities.append(v)

                    prev_pos['t1'] = {'posx': targets['t1']['posx'], 'posy': targets['t1']['posy']}
                    prev_time = now

        if len(velocities) > 0:
            measured_mean = np.mean(velocities)
            measured_std = np.std(velocities)
            print(f"Membaca Sensor: {measured_mean:.3f} ± {measured_std:.3f} m/s ({len(velocities)} sampel)")

            travel_time_input = input("Waktu tempuh sebenarnya (detik, atau 's' untuk Lewati): ")
            if travel_time_input.lower() == 's':
                print("Dilewati.")
                continue

            try:
                travel_time = float(travel_time_input)
                if travel_time <= 0:
                    print("Input tidak valid, ulangi percobaan ini.")
                    continue

                ground_truth_vel = track_length / travel_time
                print(f"Kecepatan Sebenarnya: {ground_truth_vel:.3f} m/s")
                collector.add_velocity_sample(measured_mean, ground_truth_vel)
                completed_trials += 1

            except ValueError:
                print("Input tidak valid, ulangi percobaan ini.")
                continue
        else:
            print("Tidak ada data valid dari sensor!")
            retry = input("   Ulangi percobaan ini? (y/n): ")
            if retry.lower() == 'y':
                num_trials += 1

    ser.close()

    print(f"\nKalibrasi Kecepatan Selesai: {completed_trials}/{num_trials}")
    if completed_trials == 0:
        print("Tidak ada data kalibrasi kecepatan. Model identity akan digunakan.")
    elif completed_trials < 2:
        print("Kurang dari 2 sampel. Model mungkin tidak akurat.")

    return collector


# Fungsi Utama
def main_calibration():
    print("\n" + "🎯" * 30)
    print("Sistem Kalibrasi Radar")
    print("  Model Jarak   : Logarithmic")
    print("  Model Kecepatan: Linear")
    print("🎯" * 30)
    print("\nTahap Kalibrasi:")
    print("  1. Kalibrasi Jarak")
    print("  2. Kalibrasi Kecepatan")
    print("\nSkip Jika Diperlukan")
    print("=" * 60)

    dist_collector = run_distance_calibration_interactive()
    vel_collector = run_velocity_calibration_interactive()

    collector = CalibrationCollector()
    collector.distance_data = dist_collector.distance_data
    collector.velocity_data = vel_collector.velocity_data
    collector.save_raw_data()

    has_distance_data = len(collector.distance_data['measured']) >= 2
    has_velocity_data = len(collector.velocity_data['measured']) >= 2

    if not has_distance_data and not has_velocity_data:
        print("\nPeringatan: Tidak ada data kalibrasi!")
        print("   Model identity (tanpa koreksi) akan digunakan.")
        proceed = input("\nLanjutkan? (y/n): ")
        if proceed.lower() != 'y':
            print("Kalibrasi dibatalkan.")
            return

    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    manager = CalibrationManager()
    manager.calibrate(collector)
    manager.save()

    print("\nKalibrasi Selesai")
    print("\nHasil:")
    print(f"  Sampel Jarak    : {len(collector.distance_data['measured'])}")
    print(f"  Sampel Kecepatan: {len(collector.velocity_data['measured'])}")
    print(f"  Model Jarak     : Logarithmic")
    print(f"  Model Kecepatan : Linear")
    print(f"  Disimpan ke     : calibration_models.pkl")


if __name__ == "__main__":
    main_calibration()