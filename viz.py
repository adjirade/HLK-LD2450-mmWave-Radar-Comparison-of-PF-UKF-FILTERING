import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
from radar_logger import Logger
from time import time

plt.rcParams.update({
    "font.family"         : "DejaVu Sans",
    "font.size"           : 9,
    "axes.titlesize"      : 11,
    "axes.titleweight"    : "bold",
    "axes.labelsize"      : 9,
    "axes.spines.top"     : False,
    "axes.spines.right"   : False,
    "figure.dpi"          : 110,
    "figure.facecolor"    : "white",
    "axes.facecolor"      : "#FAFAFA",
    "axes.grid"           : True,
    "grid.alpha"          : 0.25,
    "grid.linestyle"      : "--",
    "legend.fontsize"     : 8,
    "legend.framealpha"   : 0.9,
})

C_RAW = "#E74C3C"
C_UKF = "#2980B9"
C_PF  = "#27AE60"

T_COLORS = {"t1": "#E74C3C", "t2": "#2980B9", "t3": "#F39C12"}
T_LABELS = {"t1": "Target 1", "t2": "Target 2", "t3": "Target 3"}

DISPLAY_MODES = ['FULL', 'DISTANCE', 'VELOCITY']


class RadarVisualizer:
    def __init__(self, metrics, max_points=100, display_mode='FULL'):
        self.metrics    = metrics
        self.max_points = max_points
        self.display_mode = display_mode

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

        self._create_layout()
        
        self.ani = FuncAnimation(
            self.fig,
            self._animate,
            interval=50,
            blit=False,
            cache_frame_data=False
        )

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
    
    def get_display_mode(self):
        return self.display_mode

    def _create_layout(self):
        if self.display_mode == 'FULL':
            self.fig = plt.figure(figsize=(19, 10))
        else:
            self.fig = plt.figure(figsize=(18, 10))
        
        self.fig.patch.set_facecolor("white")

        self.status_text = self.fig.text(
            0.01, 0.005,
            "",
            ha='left', va='bottom',
            fontsize=6, 
            color='#999999',
            alpha=0.4,
            transform=self.fig.transFigure,
            family='monospace'
        )
        
        self.help_text = self.fig.text(
            0.99, 0.005, 
            "Hotkeys: [D] Data Assoc | [C] Calibration | [I] Info | [Q] Quit & Save",
            ha='right', va='bottom', fontsize=6,
            color='#999999', transform=self.fig.transFigure,
            style='italic'
        )

        if self.display_mode == 'FULL':
            self._create_full_layout()
        elif self.display_mode == 'DISTANCE':
            self._create_distance_layout()
        elif self.display_mode == 'VELOCITY':
            self._create_velocity_layout()

    def _create_full_layout(self):
        """Full layout: Distance + Velocity + Metrics - PIZO OPTIMIZED"""
        # PIZO Style: Minimal padding untuk maksimalkan grafik
        self.fig.subplots_adjust(
            left=0.04, right=0.99, top=0.97, bottom=0.04,
            hspace=0.25, wspace=0.20
        )
        
        gs = gridspec.GridSpec(
            3, 3,
            figure=self.fig,
            width_ratios=[4.0, 4.0, 1.0],  # PIZO: Grafik lebih besar, info lebih kecil
            hspace=0.25, wspace=0.20
        )

        self.axes  = {}
        self.lines = {}
        self.texts = {}

        for i, t in enumerate(['t1', 't2', 't3']):
            tc = T_COLORS[t]

            # ── VELOCITY PLOT (PIZO Optimized) ────────────────────────
            ax_v = self.fig.add_subplot(gs[i, 0])
            self.lines[f'{t}_vel_raw'], = ax_v.plot(
                [], [], label='Raw', linestyle='-',  # PIZO: solid line lebih clean
                color=C_RAW, linewidth=1.2, alpha=0.7)
            self.lines[f'{t}_vel_ukf'], = ax_v.plot(
                [], [], label='UKF',
                color=C_UKF, linewidth=1.8, alpha=0.95)
            self.lines[f'{t}_vel_pf'],  = ax_v.plot(
                [], [], label='PF',
                color=C_PF,  linewidth=1.8, alpha=0.95)

            ax_v.set_title(
                f"{T_LABELS[t]} — Velocity",
                color=tc, pad=3, fontsize=10)  # PIZO: pad minimal
            ax_v.set_xlabel('Time (s)', fontsize=8)
            ax_v.set_ylabel('Velocity (m/s)', fontsize=8)
            ax_v.set_ylim(-0.2, 5.0)  # PIZO: range lebih ketat
            ax_v.legend(loc='upper right', ncol=3, fontsize=7, frameon=False)  # PIZO: no frame
            ax_v.tick_params(labelsize=8)
            ax_v.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)  # PIZO: subtle grid
            self.axes[f'{t}_vel'] = ax_v

            # ── DISTANCE PLOT (PIZO Optimized) ────────────────────────
            ax_d = self.fig.add_subplot(gs[i, 1])
            self.lines[f'{t}_dist_raw'], = ax_d.plot(
                [], [], label='Raw', linestyle='-',
                color=C_RAW, linewidth=1.2, alpha=0.7)
            self.lines[f'{t}_dist_ukf'], = ax_d.plot(
                [], [], label='UKF',
                color=C_UKF, linewidth=1.8, alpha=0.95)
            self.lines[f'{t}_dist_pf'],  = ax_d.plot(
                [], [], label='PF',
                color=C_PF,  linewidth=1.8, alpha=0.95)

            ax_d.set_title(
                f"{T_LABELS[t]} — Distance",
                color=tc, pad=3, fontsize=10)
            ax_d.set_xlabel('Time (s)', fontsize=8)
            ax_d.set_ylabel('Distance (m)', fontsize=8)
            ax_d.set_ylim(-0.2, 10.0)  # PIZO: range lebih ketat
            ax_d.legend(loc='upper right', ncol=3, fontsize=7, frameon=False)
            ax_d.tick_params(labelsize=8)
            ax_d.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
            self.axes[f'{t}_dist'] = ax_d

            # ── METRICS PANEL (PIZO: Super compact!) ──────────────────
            ax_txt = self.fig.add_subplot(gs[i, 2])
            ax_txt.axis('off')

            # Target label (smaller)
            ax_txt.text(
                0.5, 0.98, T_LABELS[t].upper(),
                transform=ax_txt.transAxes,
                ha='center', va='top',
                fontsize=8, fontweight='bold',
                color='white',
                bbox=dict(facecolor=tc, edgecolor='none',
                          boxstyle='round,pad=0.25', alpha=0.85)
            )

            # Metrics text (compact)
            txt = ax_txt.text(
                0.05, 0.50, "",
                transform=ax_txt.transAxes,
                va='center', ha='left',
                fontsize=7,  # PIZO: smaller
                family='monospace',
                linespacing=1.35,  # PIZO: tighter
                bbox=dict(facecolor='#F8F9FA', edgecolor='#DDD',
                          boxstyle='round,pad=0.4', linewidth=0.5)
            )
            self.texts[t] = txt

        # Live indicator (pojok kanan atas)
        self.live_text = self.fig.text(
            0.99, 0.99, "● LIVE",
            ha='right', va='top',
            fontsize=8, fontweight='bold',  # PIZO: smaller
            color='#27AE60',
            transform=self.fig.transFigure
        )

        # Footer legend (PIZO: frameon=False)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0],[0], color=C_RAW, linestyle='-', linewidth=1.2, alpha=0.7, label='Raw'),
            Line2D([0],[0], color=C_UKF, linewidth=1.8, label='UKF'),
            Line2D([0],[0], color=C_PF,  linewidth=1.8, label='PF'),
        ]
        self.fig.legend(
            handles=legend_elements,
            loc='lower center', ncol=3,
            fontsize=8, frameon=False,  # PIZO: no frame
            bbox_to_anchor=(0.5, 0.015)  # PIZO: slightly higher
        )

    def _create_distance_layout(self):
        """Distance only layout - PIZO OPTIMIZED: Grafik sangat lebar"""
        self.fig.subplots_adjust(
            left=0.04, right=0.99, top=0.97, bottom=0.04,
            hspace=0.25, wspace=0.18
        )
        
        gs = gridspec.GridSpec(
            3, 2,
            figure=self.fig,
            width_ratios=[5.5, 1.0],  # PIZO: Grafik sangat lebar
            hspace=0.25, wspace=0.18
        )

        self.axes  = {}
        self.lines = {}
        self.texts = {}

        for i, t in enumerate(['t1', 't2', 't3']):
            tc = T_COLORS[t]

            # ── DISTANCE PLOT (EXTRA WIDE) ────────────────────────────
            ax_d = self.fig.add_subplot(gs[i, 0])
            self.lines[f'{t}_dist_raw'], = ax_d.plot(
                [], [], label='Raw', linestyle='-',
                color=C_RAW, linewidth=1.4, alpha=0.8)
            self.lines[f'{t}_dist_ukf'], = ax_d.plot(
                [], [], label='UKF',
                color=C_UKF, linewidth=1.8)
            self.lines[f'{t}_dist_pf'],  = ax_d.plot(
                [], [], label='PF',
                color=C_PF,  linewidth=1.8)

            ax_d.set_title(
                f"{T_LABELS[t]} — Distance",
                color=tc, pad=3, fontsize=10)
            ax_d.set_xlabel('Time (s)', fontsize=8)
            ax_d.set_ylabel('Distance (m)', fontsize=8)
            ax_d.set_ylim(-0.2, 10.0)
            ax_d.legend(loc='upper right', ncol=3, fontsize=7, frameon=False)
            ax_d.tick_params(labelsize=8)
            ax_d.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
            self.axes[f'{t}_dist'] = ax_d

            # Dummy velocity lines (hidden)
            self.lines[f'{t}_vel_raw'], = ax_d.plot([], [], visible=False)
            self.lines[f'{t}_vel_ukf'], = ax_d.plot([], [], visible=False)
            self.lines[f'{t}_vel_pf'],  = ax_d.plot([], [], visible=False)

            # ── METRICS PANEL (PIZO Compact) ──────────────────────────
            ax_txt = self.fig.add_subplot(gs[i, 1])
            ax_txt.axis('off')

            ax_txt.text(
                0.5, 0.98, T_LABELS[t].upper(),
                transform=ax_txt.transAxes,
                ha='center', va='top',
                fontsize=8, fontweight='bold',
                color='white',
                bbox=dict(facecolor=tc, edgecolor='none',
                          boxstyle='round,pad=0.25', alpha=0.85)
            )

            txt = ax_txt.text(
                0.05, 0.50, "",
                transform=ax_txt.transAxes,
                va='center', ha='left',
                fontsize=7,
                family='monospace',
                linespacing=1.35,
                bbox=dict(facecolor='#F8F9FA', edgecolor='#DDD',
                          boxstyle='round,pad=0.4', linewidth=0.5)
            )
            self.texts[t] = txt

        # Live indicator
        self.live_text = self.fig.text(
            0.99, 0.99, "● LIVE",
            ha='right', va='top',
            fontsize=8, fontweight='bold',
            color='#27AE60',
            transform=self.fig.transFigure
        )

        # Footer legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0],[0], color=C_RAW, linestyle='-', linewidth=1.4, alpha=0.8, label='Raw'),
            Line2D([0],[0], color=C_UKF, linewidth=1.8, label='UKF'),
            Line2D([0],[0], color=C_PF,  linewidth=1.8, label='PF'),
        ]
        self.fig.legend(
            handles=legend_elements,
            loc='lower center', ncol=3,
            fontsize=8, frameon=False,
            bbox_to_anchor=(0.5, 0.015)
        )

    def _create_velocity_layout(self):
        """Velocity only layout - PIZO OPTIMIZED: Grafik sangat lebar"""
        self.fig.subplots_adjust(
            left=0.04, right=0.99, top=0.97, bottom=0.04,
            hspace=0.25, wspace=0.18
        )
        
        gs = gridspec.GridSpec(
            3, 2,
            figure=self.fig,
            width_ratios=[5.5, 1.0],  # PIZO: Grafik sangat lebar
            hspace=0.25, wspace=0.18
        )

        self.axes  = {}
        self.lines = {}
        self.texts = {}

        for i, t in enumerate(['t1', 't2', 't3']):
            tc = T_COLORS[t]

            # ── VELOCITY PLOT (EXTRA WIDE) ────────────────────────────
            ax_v = self.fig.add_subplot(gs[i, 0])
            self.lines[f'{t}_vel_raw'], = ax_v.plot(
                [], [], label='Raw', linestyle='-',
                color=C_RAW, linewidth=1.4, alpha=0.8)
            self.lines[f'{t}_vel_ukf'], = ax_v.plot(
                [], [], label='UKF',
                color=C_UKF, linewidth=1.8)
            self.lines[f'{t}_vel_pf'],  = ax_v.plot(
                [], [], label='PF',
                color=C_PF,  linewidth=1.8)

            ax_v.set_title(
                f"{T_LABELS[t]} — Velocity",
                color=tc, pad=3, fontsize=10)
            ax_v.set_xlabel('Time (s)', fontsize=8)
            ax_v.set_ylabel('Velocity (m/s)', fontsize=8)
            ax_v.set_ylim(-0.2, 5.0)
            ax_v.legend(loc='upper right', ncol=3, fontsize=7, frameon=False)
            ax_v.tick_params(labelsize=8)
            ax_v.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
            self.axes[f'{t}_vel'] = ax_v

            # Dummy distance lines (hidden)
            self.lines[f'{t}_dist_raw'], = ax_v.plot([], [], visible=False)
            self.lines[f'{t}_dist_ukf'], = ax_v.plot([], [], visible=False)
            self.lines[f'{t}_dist_pf'],  = ax_v.plot([], [], visible=False)

            # ── METRICS PANEL (PIZO Compact) ──────────────────────────
            ax_txt = self.fig.add_subplot(gs[i, 1])
            ax_txt.axis('off')

            ax_txt.text(
                0.5, 0.98, T_LABELS[t].upper(),
                transform=ax_txt.transAxes,
                ha='center', va='top',
                fontsize=8, fontweight='bold',
                color='white',
                bbox=dict(facecolor=tc, edgecolor='none',
                          boxstyle='round,pad=0.25', alpha=0.85)
            )

            txt = ax_txt.text(
                0.05, 0.50, "",
                transform=ax_txt.transAxes,
                va='center', ha='left',
                fontsize=7,
                family='monospace',
                linespacing=1.35,
                bbox=dict(facecolor='#F8F9FA', edgecolor='#DDD',
                          boxstyle='round,pad=0.4', linewidth=0.5)
            )
            self.texts[t] = txt

        # Live indicator
        self.live_text = self.fig.text(
            0.99, 0.99, "● LIVE",
            ha='right', va='top',
            fontsize=8, fontweight='bold',
            color='#27AE60',
            transform=self.fig.transFigure
        )

        # Footer legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0],[0], color=C_RAW, linestyle='-', linewidth=1.4, alpha=0.8, label='Raw'),
            Line2D([0],[0], color=C_UKF, linewidth=1.8, label='UKF'),
            Line2D([0],[0], color=C_PF,  linewidth=1.8, label='PF'),
        ]
        self.fig.legend(
            handles=legend_elements,
            loc='lower center', ncol=3,
            fontsize=8, frameon=False,
            bbox_to_anchor=(0.5, 0.015)
        )

    def _on_key(self, event):
        if event.key == 'q':
            self.logger.save_excel()
            plt.close(self.fig)
            print("\n" + "="*60)
            print(f"✅ Data disimpan ke Excel!")
            print(f"   Mode: {self.display_mode}")
            if self.display_mode == 'FULL':
                print("   Output: Semua data (jarak + kecepatan)")
            elif self.display_mode == 'DISTANCE':
                print("   Output: Jarak lengkap, velocity = 0")
            elif self.display_mode == 'VELOCITY':
                print("   Output: Velocity lengkap, distance = 0")
            print("="*60)
            print("Program ditutup Pengguna.")
        elif event.key == 'm' or event.key == 'M':
            # Warn user: mode sudah dipilih di awal
            print(f"\n⚠️  Mode sudah dipilih saat startup: {self.display_mode}")
            print("   Restart program untuk ganti mode.")
        # Note: D, C, I hotkeys dihandle di main.py

    # Loop Data
    def append_data(self, data, dist_ukf, vel_ukf, dist_pf, vel_pf):
        timestamp = time()

        for t in ['t1', 't2', 't3']:
            buf = self.data_buffer[t]

            # Apply display mode filtering
            if self.display_mode == 'FULL':
                dist_raw = data[t]['distance']
                vel_raw = data[t]['velocity']
                dist_ukf_val = dist_ukf[t]
                vel_ukf_val = vel_ukf[t]
                dist_pf_val = dist_pf[t]
                vel_pf_val = vel_pf[t]
            elif self.display_mode == 'DISTANCE':
                dist_raw = data[t]['distance']
                vel_raw = 0.0  # Zero out velocity
                dist_ukf_val = dist_ukf[t]
                vel_ukf_val = 0.0
                dist_pf_val = dist_pf[t]
                vel_pf_val = 0.0
            elif self.display_mode == 'VELOCITY':
                dist_raw = 0.0  # Zero out distance
                vel_raw = data[t]['velocity']
                dist_ukf_val = 0.0
                vel_ukf_val = vel_ukf[t]
                dist_pf_val = 0.0
                vel_pf_val = vel_pf[t]

            buf['time'].append(timestamp)
            buf['distance_raw'].append(dist_raw)
            buf['velocity_raw'].append(vel_raw)
            buf['distance_ukf'].append(dist_ukf_val)
            buf['velocity_ukf'].append(vel_ukf_val)
            buf['distance_pf'].append(dist_pf_val)
            buf['velocity_pf'].append(vel_pf_val)

            m = self.metrics.get_metrics()[t]

            self.logger.append(
                timestamp, t,
                dist_raw, vel_raw,
                dist_ukf_val, vel_ukf_val,
                dist_pf_val, vel_pf_val,
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
            # ── Update live indicator ──────────────────────────────────
            if self._frame_count % 10 < 5:
                self.live_text.set_text('● LIVE')
                self.live_text.set_color('#27AE60')
            else:
                self.live_text.set_text('○ LIVE')
                self.live_text.set_color('#AAAAAA')

            # ── Update status indicators (ULTRA SUBTLE - pojok kiri bawah) ──
            from parsing import get_flags
            use_da, use_cal = get_flags()
            
            da_status = "on" if use_da else "off"
            cal_status = "on" if use_cal else "off"
            mode_short = self.display_mode.lower()
            
            # PIZO Style: ultra subtle, all in one line
            self.status_text.set_text(f"mode:{mode_short} | da:{da_status} cal:{cal_status}")

            # ── Update plots ───────────────────────────────────────────
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

                # Update line data
                self.lines[f'{t}_vel_raw'].set_data(times, vel_raw)
                self.lines[f'{t}_vel_ukf'].set_data(times, vel_ukf)
                self.lines[f'{t}_vel_pf'].set_data(times,  vel_pf)

                self.lines[f'{t}_dist_raw'].set_data(times, dist_raw)
                self.lines[f'{t}_dist_ukf'].set_data(times, dist_ukf)
                self.lines[f'{t}_dist_pf'].set_data(times,  dist_pf)

                # Update x-axis limits
                t_now = times[-1]
                if self.display_mode == 'FULL':
                    if f'{t}_vel' in self.axes:
                        self.axes[f'{t}_vel'].set_xlim(max(0, t_now - window), t_now + 0.1)
                    if f'{t}_dist' in self.axes:
                        self.axes[f'{t}_dist'].set_xlim(max(0, t_now - window), t_now + 0.1)
                elif self.display_mode == 'DISTANCE':
                    if f'{t}_dist' in self.axes:
                        self.axes[f'{t}_dist'].set_xlim(max(0, t_now - window), t_now + 0.1)
                elif self.display_mode == 'VELOCITY':
                    if f'{t}_vel' in self.axes:
                        self.axes[f'{t}_vel'].set_xlim(max(0, t_now - window), t_now + 0.1)

                m = self.metrics.get_metrics()[t]

                # Update metrics panel based on display mode (PIZO: Super Compact!)
                if self.display_mode == 'FULL':
                    # PIZO Style: Hanya current value + RMSE (compact!)
                    txt = (
                        f" Raw: {dist_raw[-1]:5.2f}m {vel_raw[-1]:4.2f}m/s\n"
                        f" UKF: {dist_ukf[-1]:5.2f}m {vel_ukf[-1]:4.2f}m/s\n"
                        f" PF : {dist_pf[-1]:5.2f}m {vel_pf[-1]:4.2f}m/s\n"
                        f"{'':─<24}\n"
                        f" RMSE (d|v)\n"
                        f" UKF: {m.get('rmse_distance_ukf',0):.3f}|{m.get('rmse_velocity_ukf',0):.3f}\n"
                        f" PF : {m.get('rmse_distance_pf',0):.3f}|{m.get('rmse_velocity_pf',0):.3f}"
                    )
                elif self.display_mode == 'DISTANCE':
                    # Info hanya distance (PIZO format)
                    txt = (
                        f"{'':─<22}\n"
                        f" Distance (m)\n"
                        f"{'':─<22}\n"
                        f" Raw : {dist_raw[-1]:6.3f}\n"
                        f" UKF : {dist_ukf[-1]:6.3f}\n"
                        f" PF  : {dist_pf[-1]:6.3f}\n"
                        f"{'':─<22}\n"
                        f" RMSE\n"
                        f" UKF : {m.get('rmse_distance_ukf',0):.4f}\n"
                        f" PF  : {m.get('rmse_distance_pf',0):.4f}\n"
                        f"{'':─<22}\n"
                        f" MAE\n"
                        f" UKF : {m.get('mae_distance_ukf',0):.4f}\n"
                        f" PF  : {m.get('mae_distance_pf',0):.4f}\n"
                        f"{'':─<22}\n"
                        f" MBE\n"
                        f" UKF : {m.get('mbe_distance_ukf',0):+.4f}\n"
                        f" PF  : {m.get('mbe_distance_pf',0):+.4f}"
                    )
                elif self.display_mode == 'VELOCITY':
                    # Info hanya velocity (PIZO format)
                    txt = (
                        f"{'':─<22}\n"
                        f" Velocity (m/s)\n"
                        f"{'':─<22}\n"
                        f" Raw : {vel_raw[-1]:6.3f}\n"
                        f" UKF : {vel_ukf[-1]:6.3f}\n"
                        f" PF  : {vel_pf[-1]:6.3f}\n"
                        f"{'':─<22}\n"
                        f" RMSE\n"
                        f" UKF : {m.get('rmse_velocity_ukf',0):.4f}\n"
                        f" PF  : {m.get('rmse_velocity_pf',0):.4f}\n"
                        f"{'':─<22}\n"
                        f" MAE\n"
                        f" UKF : {m.get('mae_velocity_ukf',0):.4f}\n"
                        f" PF  : {m.get('mae_velocity_pf',0):.4f}\n"
                        f"{'':─<22}\n"
                        f" MBE\n"
                        f" UKF : {m.get('mbe_velocity_ukf',0):+.4f}\n"
                        f" PF  : {m.get('mbe_velocity_pf',0):+.4f}"
                    )
                
                self.texts[t].set_text(txt)

            self.fig.canvas.draw_idle()

        except Exception as e:
            import traceback
            print(f"[ERROR _animate frame={frame}] {e}")
            traceback.print_exc()