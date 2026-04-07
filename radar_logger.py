import pandas as pd
from datetime import datetime

class Logger:
    def __init__(self):
        self.data = []

    def append(self, timestamp, target, distance_raw, velocity_raw,
               distance_ukf, velocity_ukf, distance_pf, velocity_pf,
               rmse_dist_ukf, rmse_vel_ukf, rmse_dist_pf, rmse_vel_pf,
               mae_dist_ukf, mae_vel_ukf, mae_dist_pf, mae_vel_pf,
               mbe_dist_ukf, mbe_vel_ukf, mbe_dist_pf, mbe_vel_pf):
        self.data.append({
            'timestamp': timestamp,
            'target': target,
            'distance_raw': distance_raw,
            'velocity_raw': velocity_raw,
            'distance_ukf': distance_ukf,
            'velocity_ukf': velocity_ukf,
            'distance_pf': distance_pf,
            'velocity_pf': velocity_pf,
            'rmse_dist_ukf': rmse_dist_ukf,
            'rmse_vel_ukf': rmse_vel_ukf,
            'rmse_dist_pf': rmse_dist_pf,
            'rmse_vel_pf': rmse_vel_pf,
            'mae_dist_ukf': mae_dist_ukf,
            'mae_vel_ukf': mae_vel_ukf,
            'mae_dist_pf': mae_dist_pf,
            'mae_vel_pf': mae_vel_pf,
            'mbe_dist_ukf': mbe_dist_ukf,
            'mbe_vel_ukf': mbe_vel_ukf,
            'mbe_dist_pf': mbe_dist_pf,
            'mbe_vel_pf': mbe_vel_pf
        })

    def save_excel(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/radar_log_{timestamp}.xlsx"
        df = pd.DataFrame(self.data)
        df.to_excel(filename, index=False)
        print(f"[Logger] Data Disimpan di {filename}")