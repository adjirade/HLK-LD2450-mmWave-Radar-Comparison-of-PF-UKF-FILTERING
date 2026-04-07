import serial
import serial.tools.list_ports
import time
import math
from calibration_system import CalibrationManager
from data_association import DataAssociator

# Parameter Awal
BAUD_RATE   = 115200
TIMEOUT     = 0.1
DIST_OFFSET = 0
VEL_OFFSET  = 0

def find_esp32_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if port.vid in (0x10C4, 0x1A86, 0x303A):
            return port.device
    return None

# Mengambil Model Kalibrasi
calib_manager = CalibrationManager()
try:
    calib_manager.load('calibration_models.pkl')
    print("Model Kalibrasi Ada")
except:
    print("Model Kalibrasi Tidak Ada")

# Asosiasi Data
associator = DataAssociator()


# Parsing Data
def parse_radar_frame(frame):
    try:
        values = [float(x) for x in frame.strip().split(',')]
        if len(values) != 9:
            return None

        detections = [
            {'posx': values[0], 'posy': values[1], 'distance': values[2]},
            {'posx': values[3], 'posy': values[4], 'distance': values[5]},
            {'posx': values[6], 'posy': values[7], 'distance': values[8]},
        ]
        return detections
    except ValueError:
        return None


# Fungsi Utama
def read_radar_data():
    port = find_esp32_port()
    if port is None:
        print("ESP32 Tidak Terdeteksi")
        return

    print(f"ESP32 Terhubung ke {port}")
    ser = serial.Serial(port, BAUD_RATE, timeout=TIMEOUT)
    time.sleep(2)

    prev_time = None

    while True:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore')
            detections = parse_radar_frame(line)

            if detections:
                now  = time.time()
                dt   = 0.0 if prev_time is None else now - prev_time
                prev_time = now


                associated = associator.update(detections, dt)

                # Tahap Kalibrasi
                output = {}
                for t in ['t1', 't2', 't3']:
                    dist_raw = associated[t]['distance']
                    vel_raw  = associated[t]['velocity']

                    if dist_raw == 0.0 and vel_raw == 0.0:
                        output[t] = {'distance': 0.0, 'velocity': 0.0}
                        continue

                    dist_cal, vel_cal = calib_manager.apply_calibration(dist_raw, vel_raw)

                    dist_cal += DIST_OFFSET
                    vel_cal  += VEL_OFFSET

                    dist_cal = max(dist_cal, 0.0)
                    vel_cal  = max(vel_cal,  0.0)

                    output[t] = {
                        'distance': dist_cal,
                        'velocity': vel_cal
                    }

                yield output

        else:
            time.sleep(0.005)