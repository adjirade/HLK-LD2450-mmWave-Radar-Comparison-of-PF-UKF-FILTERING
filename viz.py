import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
from radar_logger import Logger
from time import time

# Styling
plt.rcParams.update({
    "font.family"         : "DejaVu Sans",
    "font.size"           : 10,
    "axes.titlesize"      : 12,
    "axes.titleweight"    : "bold",
    "axes.labelsize"      : 10,
    "axes.spines.top"     : False,
    "axes.spines.right"   : False,
    "figure.dpi"          : 110,
    "figure.facecolor"    : "white",
    "axes.facecolor"      : "#FAFAFA",
    "axes.grid"           : True,
    "grid.alpha"          : 0.3,
    "grid.linestyle"      : "--",
    "legend.fontsize"     : 9,
    "legend.framealpha"   : 0.85,
})

C_RAW = "#E74C3C"
C_UKF = "#2980B9"
C_PF  = "#27AE60"

T_COLORS = {"t1": "#E74C3C", "t2": "#2980B9", "t3": "#F39C12"}
T_LABELS = {"t1": "Target 1", "t2": "Target 2", "t3": "Target 3"}


class RadarVisualizer:
    def __init__(self, metrics, max_points=100):
        self.metrics    = metrics
        self.max_points = max_points

        #Buffer
        self.data_buffer = {
            t: {
                key: deque(maxlen=max_points)
                for key in ['time',
                            'distance_raw', 'velocity_raw',
                            'distance_ukf', 'velocity_ukf',
                            'distance_pf',  'velocity_pf']
            }
            for t in ['t1', 't2', 't3']
        }

        self.logger       = Logger()
        self.start_time   = time()
        self._frame_count = 0

        # ── Layout figure ────────────────────────────────────────────────────
        self.fig = plt.figure(figsize=(17, 9), constrained_layout=True)
        self.fig.patch.set_facecolor("white")

        # Judul utama
        self.fig.suptitle(
            "Radar mmWave  |  UKF vs PF",
            fontsize=13, fontweight="bold", y=0.995, color="#1F4E79"
        )

        # Ukuran Grid
        self.fig.set_constrained_layout_pads(
            w_pad=0.04, h_pad=0.04,
            hspace=0.12, wspace=0.10
        )
        gs = gridspec.GridSpec(
            3, 3,
            figure=self.fig,
            width_ratios=[3.2, 3.2, 1.05],
            hspace=0.38,
            wspace=0.28
        )

        self.axes  = {}
        self.lines = {}
        self.texts = {}

        for i, t in enumerate(['t1', 't2', 't3']):
            tc = T_COLORS[t]

            # Plot Kecepatan
            ax_v = self.fig.add_subplot(gs[i, 0])
            self.lines[f'{t}_vel_raw'], = ax_v.plot(
                [], [], label='CRL', linestyle='--',
                color=C_RAW, linewidth=1.4, alpha=0.8)
            self.lines[f'{t}_vel_ukf'], = ax_v.plot(
                [], [], label='UKF',
                color=C_UKF, linewidth=1.8)
            self.lines[f'{t}_vel_pf'],  = ax_v.plot(
                [], [], label='PF',
                color=C_PF,  linewidth=1.8)

            ax_v.set_title(
                f"{T_LABELS[t]} — Kecepatan",
                color=tc, pad=6)
            ax_v.set_xlabel('Waktu (s)')
            ax_v.set_ylabel('Kecepatan (m/s)')
            ax_v.set_ylim(-0.5, 5.5)
            ax_v.legend(loc='upper right', ncol=3)
            ax_v.yaxis.grid(True, alpha=0.3)
            ax_v.xaxis.grid(False)
            self.axes[f'{t}_vel'] = ax_v

            # Plot Jarak
            ax_d = self.fig.add_subplot(gs[i, 1])
            self.lines[f'{t}_dist_raw'], = ax_d.plot(
                [], [], label='CRL', linestyle='--',
                color=C_RAW, linewidth=1.4, alpha=0.8)
            self.lines[f'{t}_dist_ukf'], = ax_d.plot(
                [], [], label='UKF',
                color=C_UKF, linewidth=1.8)
            self.lines[f'{t}_dist_pf'],  = ax_d.plot(
                [], [], label='PF',
                color=C_PF,  linewidth=1.8)

            ax_d.set_title(
                f"{T_LABELS[t]} — Jarak",
                color=tc, pad=6)
            ax_d.set_xlabel('Waktu (s)')
            ax_d.set_ylabel('Jarak (m)')
            ax_d.set_ylim(-0.2, 10.5)
            ax_d.legend(loc='upper right', ncol=3)
            ax_d.yaxis.grid(True, alpha=0.3)
            ax_d.xaxis.grid(False)
            self.axes[f'{t}_dist'] = ax_d

            # Panel Matriks
            ax_txt = self.fig.add_subplot(gs[i, 2])
            ax_txt.axis('off')

            ax_txt.text(
                0.5, 1.02, T_LABELS[t].upper(),
                transform=ax_txt.transAxes,
                ha='center', va='bottom',
                fontsize=9, fontweight='bold',
                color='white',
                bbox=dict(facecolor=tc, edgecolor='none',
                          boxstyle='round,pad=0.35', alpha=0.9)
            )

            txt = ax_txt.text(
                0.03, 0.50, "",
                transform=ax_txt.transAxes,
                va='center', ha='left',
                fontsize=8.5,
                family='monospace',
                linespacing=1.55,
                bbox=dict(facecolor='#F4F6F9', edgecolor='#CCCCCC',
                          boxstyle='round,pad=0.55', linewidth=0.8)
            )
            self.texts[t] = txt

        # Indikator Live
        self.live_text = self.fig.text(
            0.985, 0.985, "● Langsung",
            ha='right', va='top',
            fontsize=10, fontweight='bold',
            color='#27AE60',
            transform=self.fig.transFigure
        )

        # Legenda
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0],[0], color=C_RAW, linestyle='--', linewidth=1.6, label='Raw'),
            Line2D([0],[0], color=C_UKF, linewidth=2.0, label='UKF'),
            Line2D([0],[0], color=C_PF,  linewidth=2.0, label='PF'),
        ]
        self.fig.legend(
            handles=legend_elements,
            loc='lower center', ncol=3,
            fontsize=10, framealpha=0.9,
            bbox_to_anchor=(0.46, 0.005),
            edgecolor='#CCCCCC'
        )

        # Animasi
        self.ani = FuncAnimation(
            self.fig,
            self._animate,
            interval=50,
            blit=False,
            cache_frame_data=False
        )

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_key(self, event):
        if event.key == 'q':
            self.logger.save_excel()
            plt.close(self.fig)
            print("Program ditutup Pengguna.")

    # Loop Data
    def append_data(self, data, dist_ukf, vel_ukf, dist_pf, vel_pf):
        timestamp = time()

        for t in ['t1', 't2', 't3']:
            buf = self.data_buffer[t]

            buf['time'].append(timestamp)
            buf['distance_raw'].append(data[t]['distance'])
            buf['velocity_raw'].append(data[t]['velocity'])
            buf['distance_ukf'].append(dist_ukf[t])
            buf['velocity_ukf'].append(vel_ukf[t])
            buf['distance_pf'].append(dist_pf[t])
            buf['velocity_pf'].append(vel_pf[t])

            m = self.metrics.get_metrics()[t]

            self.logger.append(
                timestamp, t,
                buf['distance_raw'][-1], buf['velocity_raw'][-1],
                buf['distance_ukf'][-1], buf['velocity_ukf'][-1],
                buf['distance_pf'][-1],  buf['velocity_pf'][-1],
                m.get('rmse_distance_ukf', 0), m.get('rmse_velocity_ukf', 0),
                m.get('rmse_distance_pf', 0),  m.get('rmse_velocity_pf', 0),
                m.get('mae_distance_ukf', 0),  m.get('mae_velocity_ukf', 0),
                m.get('mae_distance_pf', 0),   m.get('mae_velocity_pf', 0),
                m.get('mbe_distance_ukf', 0),  m.get('mbe_velocity_ukf', 0),
                m.get('mbe_distance_pf', 0),   m.get('mbe_velocity_pf', 0)
            )

    # Animasi Frame
    def _animate(self, frame):
        window = 5
        self._frame_count += 1

        try:
            if self._frame_count % 10 < 5:
                self.live_text.set_text('● Langsung')
                self.live_text.set_color('#27AE60')
            else:
                self.live_text.set_text('○ Langsung')
                self.live_text.set_color('#AAAAAA')

            for t in ['t1', 't2', 't3']:
                buf = self.data_buffer[t]
                if len(buf['time']) == 0:
                    continue

                times    = np.array(buf['time']) - self.start_time
                vel_raw  = list(buf['velocity_raw'])
                vel_ukf  = list(buf['velocity_ukf'])
                vel_pf   = list(buf['velocity_pf'])
                dist_raw = list(buf['distance_raw'])
                dist_ukf = list(buf['distance_ukf'])
                dist_pf  = list(buf['distance_pf'])

                self.lines[f'{t}_vel_raw'].set_data(times, vel_raw)
                self.lines[f'{t}_vel_ukf'].set_data(times, vel_ukf)
                self.lines[f'{t}_vel_pf'].set_data(times,  vel_pf)

                self.lines[f'{t}_dist_raw'].set_data(times, dist_raw)
                self.lines[f'{t}_dist_ukf'].set_data(times, dist_ukf)
                self.lines[f'{t}_dist_pf'].set_data(times,  dist_pf)

                t_now = times[-1]
                self.axes[f'{t}_vel'].set_xlim(max(0, t_now - window), t_now + 0.1)
                self.axes[f'{t}_dist'].set_xlim(max(0, t_now - window), t_now + 0.1)

                m = self.metrics.get_metrics()[t]

                # Styling Panel Matriks
                txt = (
                    f"{'':─<28}\n"
                    f" Val  CRL : {dist_raw[-1]:6.3f} m  {vel_raw[-1]:5.3f} m/s\n"
                    f" Val  UKF : {dist_ukf[-1]:6.3f} m  {vel_ukf[-1]:5.3f} m/s\n"
                    f" Val  PF  : {dist_pf[-1]:6.3f} m  {vel_pf[-1]:5.3f} m/s\n"
                    f"{'':─<28}\n"
                    f" RMSE UKF : {m.get('rmse_distance_ukf',0):.4f} / {m.get('rmse_velocity_ukf',0):.4f}\n"
                    f" RMSE PF  : {m.get('rmse_distance_pf',0):.4f} / {m.get('rmse_velocity_pf',0):.4f}\n"
                    f"{'':─<28}\n"
                    f" MAE  UKF : {m.get('mae_distance_ukf',0):.4f} / {m.get('mae_velocity_ukf',0):.4f}\n"
                    f" MAE  PF  : {m.get('mae_distance_pf',0):.4f} / {m.get('mae_velocity_pf',0):.4f}\n"
                    f"{'':─<28}\n"
                    f" MBE  UKF : {m.get('mbe_distance_ukf',0):+.4f} / {m.get('mbe_velocity_ukf',0):+.4f}\n"
                    f" MBE  PF  : {m.get('mbe_distance_pf',0):+.4f} / {m.get('mbe_velocity_pf',0):+.4f}"
                )
                self.texts[t].set_text(txt)

            self.fig.canvas.draw_idle()

        except Exception as e:
            import traceback
            print(f"[ERROR _animate frame={frame}] {e}")
            traceback.print_exc()