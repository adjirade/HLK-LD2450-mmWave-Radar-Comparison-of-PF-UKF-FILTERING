import serial
import serial.tools.list_ports
import time
import math
from calibration_system import CalibrationManager
from data_association import DataAssociator

BAUD_RATE   = 115200
TIMEOUT     = 0.1
DIST_OFFSET = 0
VEL_OFFSET  = 0

USE_DATA_ASSOCIATION = False
USE_CALIBRATION      = False

def find_esp32_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if port.vid in (0x10C4, 0x1A86, 0x303A):
            return port.device
    return None

calib_manager = CalibrationManager()
try:
    calib_manager.load('calibration_models.pkl')
    print("Model Kalibrasi Ada")
except:
    print("Model Kalibrasi Tidak Ada")

associator = DataAssociator()

def toggle_data_association():
    global USE_DATA_ASSOCIATION
    USE_DATA_ASSOCIATION = not USE_DATA_ASSOCIATION
    status = "ON" if USE_DATA_ASSOCIATION else "OFF"
    print(f"\n[TOGGLE] Data Association: {status}")
    return USE_DATA_ASSOCIATION

def toggle_calibration():
    global USE_CALIBRATION
    USE_CALIBRATION = not USE_CALIBRATION
    status = "ON" if USE_CALIBRATION else "OFF"
    print(f"[TOGGLE] Calibration: {status}")
    return USE_CALIBRATION

def get_flags():
    return USE_DATA_ASSOCIATION, USE_CALIBRATION

def parse_radar_frame(frame):
    try:
        values = [float(x) for x in frame.strip().split(',')]
        if len(values) != 6:
            return None

        detections = []
        for i in range(3):
            x = values[i * 2]
            y = values[i * 2 + 1]
            
            if x == 0 and y == 0:
                dist = 0.0
            else:
                dist = math.sqrt(x**2 + y**2)
            
            detections.append({
                'posx': x,
                'posy': y,
                'distance': max(0.0, dist)
            })
        
        return detections
    except ValueError:
        return None

def read_radar_data():
    port = find_esp32_port()
    if port is None:
        print("ESP32 Tidak Terdeteksi")
        return

    print(f"ESP32 Terhubung ke {port}")
    ser = serial.Serial(port, BAUD_RATE, timeout=TIMEOUT)
    time.sleep(2)

    prev_time = None
    prev_pos = {'t1': None, 't2': None, 't3': None}

    while True:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore')
            detections = parse_radar_frame(line)

            if detections:
                now  = time.time()
                dt   = 0.0 if prev_time is None else now - prev_time
                prev_time = now

                if USE_DATA_ASSOCIATION:
                    associated = associator.update(detections, dt)
                else:
                    associated = {}
                    for i, t in enumerate(['t1', 't2', 't3']):
                        det = detections[i]
                        
                        if prev_pos[t] is not None and dt > 0:
                            dx = det['posx'] - prev_pos[t]['posx']
                            dy = det['posy'] - prev_pos[t]['posy']
                            distance_mm = math.sqrt(dx**2 + dy**2)
                            velocity = distance_mm / dt / 1000.0
                        else:
                            velocity = 0.0
                        
                        associated[t] = {
                            'distance': det['distance'] / 1000.0,
                            'velocity': max(velocity, 0.0)
                        }
                        prev_pos[t] = {'posx': det['posx'], 'posy': det['posy']}

                output = {}
                for t in ['t1', 't2', 't3']:
                    dist_raw = associated[t]['distance']
                    vel_raw  = associated[t]['velocity']

                    if dist_raw == 0.0 and vel_raw == 0.0:
                        output[t] = {'distance': 0.0, 'velocity': 0.0}
                        continue

                    if USE_CALIBRATION:
                        dist_cal, vel_cal = calib_manager.apply_calibration(dist_raw, vel_raw)
                        dist_cal += DIST_OFFSET
                        vel_cal  += VEL_OFFSET
                    else:
                        dist_cal = dist_raw
                        vel_cal  = vel_raw

                    dist_cal = max(dist_cal, 0.0)
                    vel_cal  = max(vel_cal,  0.0)

                    output[t] = {
                        'distance': dist_cal,
                        'velocity': vel_cal
                    }

                yield output

        else:
            time.sleep(0.005)