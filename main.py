import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

import time
import numpy as np
import threading
from parsing import read_radar_data
from ukf import UKF
from pf import ParticleFilter
from metrics import Metrics
from viz import RadarVisualizer

targets = ['t1', 't2', 't3']

metrics = Metrics()
viz     = RadarVisualizer(metrics, max_points=100)

radar_gen = read_radar_data()

ukfs = {t: UKF([0.0, 0.0]) for t in targets}
pfs  = {t: ParticleFilter([0.0, 0.0]) for t in targets}

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
                    # Fungsi Reset
                    if ukfs[t].state[0] != 0.0 or ukfs[t].state[1] != 0.0:
                        ukfs[t].reset([0.0, 0.0])
                        pfs[t].reset([0.0, 0.0])
                        metrics.reset(t)
                    continue

                z = [raw_dist, raw_vel]

                ukfs[t].predict(dt)
                est_ukf = ukfs[t].update(z)
                dist_ukf[t], vel_ukf[t] = est_ukf

                pfs[t].predict(dt)
                est_pf = pfs[t].update(np.array(z))
                dist_pf[t], vel_pf[t] = est_pf

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

thread = threading.Thread(target=data_loop, daemon=True)
thread.start()

try:
    plt.show()
except KeyboardInterrupt:
    pass

print("Program ditutup")