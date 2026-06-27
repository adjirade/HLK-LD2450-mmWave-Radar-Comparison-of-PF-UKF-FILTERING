import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

import time
import numpy as np
import threading
from parsing import read_radar_data, toggle_data_association, toggle_calibration, get_flags
from ukf import UKF
from pf import ParticleFilter
from metrics import Metrics
from viz import RadarVisualizer
from mode_selector import select_mode_gui

selected_mode = select_mode_gui()

print(f"\n{'='*70}")
print(f"✅ Mode Selected: {selected_mode}")
if selected_mode == 'FULL':
    print("   → Excel output: Complete data (distance + velocity)")
elif selected_mode == 'DISTANCE':
    print("   → Excel output: Distance")
elif selected_mode == 'VELOCITY':
    print("   → Excel output: Velocity")
print(f"{'='*70}\n")
print("Starting system...\n")

targets = ['t1', 't2', 't3']

metrics = Metrics()

viz = RadarVisualizer(metrics, max_points=100, display_mode=selected_mode)

radar_gen = read_radar_data()

ukfs = {t: UKF([0.0, 0.0]) for t in targets}
pfs  = {t: ParticleFilter([0.0, 0.0]) for t in targets}

def on_key_press(event):
    if event.key == 'd' or event.key == 'D':
        toggle_data_association()
    elif event.key == 'c' or event.key == 'C':
        toggle_calibration()
    elif event.key == 'i' or event.key == 'I':
        use_da, use_cal = get_flags()
        display_mode = viz.get_display_mode()

def data_loop():
    dt_prev = time.time()
    try:
        for data in radar_gen:
            now     = time.time()
            dt      = now - dt_prev
            dt_prev = now

            dist_ukf = {}
            vel_ukf  = {}
            dist_pf  = {}
            vel_pf   = {}

            for t in targets:
                raw_dist = data[t]['distance']
                raw_vel  = data[t]['velocity']

                if raw_dist == 0.0 and raw_vel == 0.0:
                    dist_ukf[t] = 0.0
                    vel_ukf[t]  = 0.0
                    dist_pf[t]  = 0.0
                    vel_pf[t]   = 0.0
                    if ukfs[t].state[0] != 0.0 or ukfs[t].state[1] != 0.0:
                        ukfs[t].reset([0.0, 0.0])
                        pfs[t].reset([0.0, 0.0])
                        metrics.reset(t)
                    continue

                z = [raw_dist, raw_vel]

                ukfs[t].predict(dt)
                est_ukf = ukfs[t].update(z)
                dist_ukf[t] = max(est_ukf[0], 0.0)
                vel_ukf[t] = max(est_ukf[1], 0.0)

                pfs[t].predict(dt)
                est_pf = pfs[t].update(np.array(z))
                dist_pf[t] = max(est_pf[0], 0.0)
                vel_pf[t] = max(est_pf[1], 0.0)

                metrics.update(
                    t,
                    {'distance': raw_dist,    'velocity': raw_vel},
                    {'distance': dist_ukf[t], 'velocity': vel_ukf[t]},
                    {'distance': dist_pf[t],  'velocity': vel_pf[t]}
                )

            viz.append_data(data, dist_ukf, vel_ukf, dist_pf, vel_pf)

    except Exception as e:
        import traceback
        print(f"\n[ERROR data_loop] {e}")
        traceback.print_exc()

viz.fig.canvas.mpl_connect('key_press_event', on_key_press)

thread = threading.Thread(target=data_loop, daemon=True)
thread.start()


print("  [Q] Quit & Save to Excel")
print("="*70)
print(f"\n📊 Active Mode: {selected_mode}")
if selected_mode == 'FULL':
    print("   → Excel output: Complete data (distance + velocity)")
elif selected_mode == 'DISTANCE':
    print("   → Excel output: Distance")
elif selected_mode == 'VELOCITY':
    print("   → Excel output: Velocity")
print("="*70 + "\n")

try:
    plt.show()
except KeyboardInterrupt:
    pass

print("Program ditutup")