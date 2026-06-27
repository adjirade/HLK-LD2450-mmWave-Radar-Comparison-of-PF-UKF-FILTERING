import serial
import time
import numpy as np
from calibration_system import CalibrationManager, parse_radar_frame

SERIAL_PORT = 'COM5'
BAUD_RATE = 115200
TIMEOUT = 0.1

def quick_test(duration=10):
    """Test kalibrasi selama beberapa detik"""
    
    # Load calibration
    calib = CalibrationManager()
    try:
        calib.load('calibration_models.pkl')
        print("✅ Calibration loaded!\n")
    except Exception as e:
        print(f"❌ Error loading calibration: {e}")
        print("   Run 'python run_calibration.py' first!\n")
        return
    
    # Validate models
    if calib.distance_model is None or calib.velocity_model is None:
        print("❌ Calibration models not properly initialized!")
        return
    
    print(f"Testing for {duration} seconds...")
    print("Place target at known distance (e.g., 2.0m) and observe:")
    print("-" * 60)
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
        time.sleep(2)
    except Exception as e:
        print(f"❌ Cannot open serial port: {e}")
        return
    
    start_time = time.time()
    samples_raw = []
    samples_calibrated = []
    
    while time.time() - start_time < duration:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore')
            targets = parse_radar_frame(line)
            
            if targets and targets['t1']['distance'] > 0:
                raw_dist = targets['t1']['distance']
                
                try:
                    calib_dist, _ = calib.apply_calibration(raw_dist, 0)
                    
                    samples_raw.append(raw_dist)
                    samples_calibrated.append(calib_dist)
                    
                    print(f"Raw: {raw_dist:.3f}m  →  Calibrated: {calib_dist:.3f}m  "
                          f"(Δ = {calib_dist - raw_dist:+.3f}m)")
                except Exception as e:
                    print(f"⚠️ Error applying calibration: {e}")
                    continue
    
    ser.close()
    
    if len(samples_raw) == 0:
        print("\n⚠️ No valid samples collected!")
        return
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Samples collected:   {len(samples_raw)}")
    print(f"Raw Distance:        {np.mean(samples_raw):.3f} ± {np.std(samples_raw):.3f}m")
    print(f"Calibrated Distance: {np.mean(samples_calibrated):.3f} ± {np.std(samples_calibrated):.3f}m")
    print(f"Average Correction:  {np.mean(samples_calibrated) - np.mean(samples_raw):+.3f}m")
    print(f"Std Deviation:       {np.std(samples_calibrated):.4f}m")
    print("="*60)

if __name__ == "__main__":
    quick_test(duration=10)