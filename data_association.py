import math
import time
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
# Konstanta / tuning
# ─────────────────────────────────────────────────────────────────────────────
GATE_DIST         = 1.5   # meter — radius maksimum asosiasi
LOST_TIMEOUT      = 10    # frame — toleransi hilang sebelum track di-free
MIN_DIST_NEW      = 0.05  # meter — minimum distance deteksi valid (filter 0,0,0)
HISTORY_LEN       = 8     # frame — panjang riwayat posisi untuk heading

ALPHA             = 0.5   # bobot jarak dalam cost gabungan
BETA              = 0.3   # bobot heading error
GAMMA             = 0.2   # bobot momentum X (gerak horizontal)
# ALPHA + BETA + GAMMA = 1.0

GHOST_DIST        = 0.5   # meter — dua deteksi dalam radius ini = ghost dari orang sama
                           # Perkecil (mis. 0.3) jika 2 orang nyata sering berjarak dekat
MERGE_DIST        = 0.3   # meter — dua track dalam radius ini → coasting mode


# ─────────────────────────────────────────────────────────────────────────────
# Helper geometri
# ─────────────────────────────────────────────────────────────────────────────
def _euclidean(a, b):
    """Jarak Euclidean 2D (meter)."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _heading_vector(history):
    """
    Hitung heading unit vector dari riwayat posisi.
    Rata-rata displacement antar frame → lebih tahan noise sesaat.
    Returns (hx, hy) unit vector, atau (0,0) jika belum cukup data.
    """
    if len(history) < 2:
        return (0.0, 0.0)

    positions = list(history)
    dx_total = sum(positions[i][0] - positions[i-1][0] for i in range(1, len(positions)))
    dy_total = sum(positions[i][1] - positions[i-1][1] for i in range(1, len(positions)))
    dx_avg   = dx_total / (len(positions) - 1)
    dy_avg   = dy_total / (len(positions) - 1)

    mag = math.sqrt(dx_avg**2 + dy_avg**2)
    if mag < 1e-6:
        return (0.0, 0.0)
    return (dx_avg / mag, dy_avg / mag)


def _heading_error(hv, from_pos, to_pos):
    """
    Seberapa berlawanan arah deteksi dibanding heading track.
    Returns 0.0 (searah) s.d. 1.0 (berlawanan).
    Jika heading belum ada → return 0.0 (tidak ada penalti).
    """
    if hv == (0.0, 0.0):
        return 0.0

    dx  = to_pos[0] - from_pos[0]
    dy  = to_pos[1] - from_pos[1]
    mag = math.sqrt(dx**2 + dy**2)
    if mag < 1e-6:
        return 0.0

    ux, uy = dx / mag, dy / mag
    dot = max(-1.0, min(1.0, hv[0] * ux + hv[1] * uy))
    return (1.0 - dot) / 2.0


def _x_momentum_error(history, det_posx):
    """
    Penalti jika deteksi berlawanan dengan momentum horizontal (X) track.
    Kunci untuk skenario dua orang berpapasan di garis horizontal.

    Returns 0.0 (searah) s.d. 1.0 (berlawanan arah X).
    """
    if len(history) < 2:
        return 0.0

    positions = list(history)
    dx_total  = sum(positions[i][0] - positions[i-1][0]
                    for i in range(1, len(positions)))
    dx_avg    = dx_total / (len(positions) - 1)

    # Threshold: target hampir diam di X → tidak ada penalti
    # Satuan: mm (koordinat radar dalam mm)
    if abs(dx_avg) < 5.0:   # < 5 mm/frame dianggap diam
        return 0.0

    dx_to_det = det_posx - positions[-1][0]

    if dx_avg * dx_to_det >= 0:
        return 0.0  # searah → tidak ada penalti
    else:
        # Berlawanan → penalti proporsional, normalisasi terhadap gate (dalam mm)
        gate_mm   = GATE_DIST * 1000.0
        magnitude = min(abs(dx_to_det) / (gate_mm + 1e-6), 1.0)
        return magnitude


def _is_valid(det):
    """Deteksi valid = bukan (0,0,0) dan distance > MIN_DIST_NEW."""
    return not (det['posx'] == 0.0 and
                det['posy'] == 0.0 and
                det['distance'] == 0.0) \
           and det['distance'] > MIN_DIST_NEW


def _deduplicate(detections):
    """
    Buang ghost detection — pantulan multipath dari 1 orang yang muncul
    di beberapa slot radar sekaligus.

    Ini adalah PENYEBAB UTAMA 1 orang terdeteksi sebagai 2-3 target.

    Logika:
      - Urutkan dari distance terkecil (deteksi LOS paling akurat duluan)
      - Jika dua deteksi berjarak XY < GHOST_DIST → ghost, buang yang lebih jauh
      - Pertahankan hanya deteksi yang tidak berdekatan satu sama lain

    Catatan tuning:
      - Perkecil GHOST_DIST jika 2 orang nyata yang berdekatan ikut terbuang
      - Perbesar GHOST_DIST jika ghost masih lolos
    """
    if len(detections) <= 1:
        return detections

    # Urutkan distance terkecil dulu (LOS > multipath)
    sorted_dets = sorted(detections, key=lambda d: d['distance'])

    kept = []
    for det in sorted_dets:
        is_ghost = any(
            _euclidean((det['posx'], det['posy']),
                       (k['posx'],  k['posy'])) < GHOST_DIST
            for k in kept
        )
        if not is_ghost:
            kept.append(det)

    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Track — satu slot target
# ─────────────────────────────────────────────────────────────────────────────
class Track:
    """
    Satu slot target dengan riwayat posisi untuk trajectory-aware association.

    Atribut:
        history   : deque (x,y) N frame terakhir
        heading   : unit vector arah gerak
        predicted : prediksi posisi frame berikutnya (ekstrapolasi linear)
    """

    def __init__(self, tid):
        self.tid         = tid
        self.posx        = 0.0
        self.posy        = 0.0
        self.distance    = 0.0
        self.velocity    = 0.0
        self.status      = 'FREE'    # 'FREE' | 'ACTIVE' | 'LOST'
        self.lost_frames = 0
        self.last_seen   = None

        self.history   = deque(maxlen=HISTORY_LEN)
        self.heading   = (0.0, 0.0)
        self.predicted = None

    @property
    def pos(self):
        return (self.posx, self.posy)

    def assign(self, det, velocity=0.0):
        """Update track dengan deteksi baru → perbarui heading & prediksi."""
        self.posx        = det['posx']
        self.posy        = det['posy']
        self.distance    = det['distance']
        self.velocity    = velocity
        self.status      = 'ACTIVE'
        self.lost_frames = 0
        self.last_seen   = time.time()

        self.history.append((self.posx, self.posy))
        self.heading = _heading_vector(self.history)
        self._update_predicted()

    def _update_predicted(self):
        """Prediksi posisi = posisi kini + displacement rata-rata terakhir."""
        if len(self.history) < 2:
            self.predicted = self.pos
            return
        positions = list(self.history)
        dx = positions[-1][0] - positions[-2][0]
        dy = positions[-1][1] - positions[-2][1]
        self.predicted = (self.posx + dx, self.posy + dy)

    def mark_lost(self):
        """Target tidak terdeteksi frame ini — pertahankan & ekstrapolasi prediksi."""
        self.lost_frames += 1
        if self.lost_frames > LOST_TIMEOUT:
            self.status = 'FREE'
            self._reset()
        else:
            self.status = 'LOST'
            # Lanjutkan ekstrapolasi posisi selama LOST
            if self.predicted is not None:
                hx, hy = self.heading
                self.predicted = (self.predicted[0] + hx,
                                  self.predicted[1] + hy)

    def _reset(self):
        self.posx        = 0.0
        self.posy        = 0.0
        self.distance    = 0.0
        self.velocity    = 0.0
        self.lost_frames = 0
        self.last_seen   = None
        self.history.clear()
        self.heading     = (0.0, 0.0)
        self.predicted   = None

    def to_dict(self):
        return {
            'posx'    : self.posx,
            'posy'    : self.posy,
            'distance': self.distance,
            'velocity': self.velocity,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DataAssociator — mesin utama
# ─────────────────────────────────────────────────────────────────────────────
class DataAssociator:
    """
    Asosiasi deteksi radar ke track T1/T2/T3 secara konsisten.

    Alur tiap frame:
        1. Filter valid    — buang slot (0,0,0) dan distance < MIN_DIST_NEW
        2. Deduplicate     — buang ghost/multipath (1 orang muncul di banyak slot)
        3. Cost matrix     — hitung cost gabungan: jarak + heading + momentum X
        4. Greedy match    — pasangkan track ↔ deteksi dari cost terkecil
        5. Mark lost       — track tanpa pasangan → LOST / FREE
        6. New track       — deteksi sisa → slot FREE by-order
        7. Merge zone      — dua track terlalu dekat → coasting dari prediksi
        8. Update & output
    """

    def __init__(self):
        self.tracks = {
            't1': Track('t1'),
            't2': Track('t2'),
            't3': Track('t3'),
        }
        self._prev_pos = {'t1': None, 't2': None, 't3': None}

    def update(self, raw_detections, dt=0.0):
        """
        Proses satu frame.

        Parameters
        ----------
        raw_detections : list of dict  [{posx, posy, distance}, ×3]
        dt             : float — delta-t detik

        Returns
        -------
        dict {t1,t2,t3} → {posx, posy, distance, velocity}
        """

        # ── 1. Filter valid ───────────────────────────────────────────────
        valid_dets = [d for d in raw_detections if _is_valid(d)]

        # ── 2. Deduplicate (buang ghost) ──────────────────────────────────
        # Ini menangani masalah: 1 orang muncul di T1, T2, T3 sekaligus
        valid_dets = _deduplicate(valid_dets)

        # ── 3 & 4. Cost matrix + Greedy matching ──────────────────────────
        matchable      = [tid for tid, tr in self.tracks.items()
                          if tr.status in ('ACTIVE', 'LOST')]
        matched_tracks = {}
        matched_dets   = set()

        if matchable and valid_dets:
            cost = {}
            for tid in matchable:
                tr      = self.tracks[tid]
                ref_pos = tr.predicted if tr.predicted is not None else tr.pos

                for di, det in enumerate(valid_dets):
                    det_pos   = (det['posx'], det['posy'])
                    dist      = _euclidean(ref_pos, det_pos)
                    dist_norm = min(dist / GATE_DIST, 1.0)
                    h_err     = _heading_error(tr.heading, tr.pos, det_pos)
                    x_err     = _x_momentum_error(tr.history, det['posx'])
                    combined  = ALPHA * dist_norm + BETA * h_err + GAMMA * x_err
                    cost[(tid, di)] = (combined, dist)

            sorted_pairs = sorted(cost.items(), key=lambda x: x[1][0])
            used_tracks  = set()
            used_dets    = set()

            for (tid, di), (combined_cost, dist) in sorted_pairs:
                if tid in used_tracks or di in used_dets:
                    continue
                if dist > GATE_DIST:
                    continue
                matched_tracks[tid] = valid_dets[di]
                matched_dets.add(di)
                used_tracks.add(tid)
                used_dets.add(di)

        # ── 5. Mark lost ──────────────────────────────────────────────────
        for tid in matchable:
            if tid not in matched_tracks:
                self.tracks[tid].mark_lost()

        # ── 6. Deteksi sisa → slot FREE by-order ─────────────────────────
        unmatched_dets = [det for di, det in enumerate(valid_dets)
                          if di not in matched_dets]
        free_slots     = [tid for tid, tr in self.tracks.items()
                          if tr.status == 'FREE']
        for det, tid in zip(unmatched_dets, free_slots):
            matched_tracks[tid] = det

        # ── 7. Merge zone coasting ────────────────────────────────────────
        # Cek SEBELUM update: gunakan posisi KINI (sebelum assign baru)
        # Jika dua track yang akan diupdate saling berdekatan → pakai prediksi
        self._handle_merge_zone(matched_tracks)

        # ── 8. Update track ───────────────────────────────────────────────
        for tid, det in matched_tracks.items():
            vel = self._calc_velocity(tid, det, dt)
            self.tracks[tid].assign(det, velocity=vel)

        # ── Output ────────────────────────────────────────────────────────
        output = {}
        for tid, tr in self.tracks.items():
            if tr.status == 'FREE':
                output[tid] = {
                    'posx': 0.0, 'posy': 0.0,
                    'distance': 0.0, 'velocity': 0.0
                }
            else:
                output[tid] = tr.to_dict()

        return output

    # ── Private ───────────────────────────────────────────────────────────────
    def _calc_velocity(self, tid, det, dt):
        """Hitung kecepatan dari selisih posisi XY (mm → m/s)."""
        if self._prev_pos[tid] is None or dt <= 0:
            self._prev_pos[tid] = (det['posx'], det['posy'])
            return 0.0
        dx  = det['posx'] - self._prev_pos[tid][0]
        dy  = det['posy'] - self._prev_pos[tid][1]
        vel = math.sqrt(dx**2 + dy**2) / dt / 1000.0
        self._prev_pos[tid] = (det['posx'], det['posy'])
        return max(vel, 0.0)

    def _handle_merge_zone(self, matched_tracks):
        """
        Saat dua track yang akan diupdate saling berdekatan (< MERGE_DIST),
        radar tidak bisa membedakan keduanya → gunakan prediksi momentum
        masing-masing (coasting) agar identitas tidak tertukar.

        Pengecekan dilakukan pada posisi KINI (sebelum assign),
        bukan setelah, agar tidak terlambat mendeteksi overlap.
        """
        tids_to_update = list(matched_tracks.keys())

        for i in range(len(tids_to_update)):
            for j in range(i + 1, len(tids_to_update)):
                tid_a = tids_to_update[i]
                tid_b = tids_to_update[j]

                # Posisi deteksi yang akan di-assign ke masing-masing track
                pos_a = (matched_tracks[tid_a]['posx'], matched_tracks[tid_a]['posy'])
                pos_b = (matched_tracks[tid_b]['posx'], matched_tracks[tid_b]['posy'])

                if _euclidean(pos_a, pos_b) < MERGE_DIST:
                    # Override dengan prediksi momentum masing-masing
                    tr_a = self.tracks[tid_a]
                    tr_b = self.tracks[tid_b]

                    if tr_a.predicted is not None:
                        matched_tracks[tid_a] = {
                            'posx'    : tr_a.predicted[0],
                            'posy'    : tr_a.predicted[1],
                            'distance': tr_a.distance,
                        }
                    if tr_b.predicted is not None:
                        matched_tracks[tid_b] = {
                            'posx'    : tr_b.predicted[0],
                            'posy'    : tr_b.predicted[1],
                            'distance': tr_b.distance,
                        }

    def reset(self):
        """Reset semua track."""
        for tr in self.tracks.values():
            tr.status = 'FREE'
            tr._reset()
        self._prev_pos = {'t1': None, 't2': None, 't3': None}

    def get_status(self):
        """Debug — status, heading, dan prediksi tiap track."""
        return {
            tid: {
                'status'   : tr.status,
                'heading'  : tr.heading,
                'predicted': tr.predicted,
                'pos'      : tr.pos,
            }
            for tid, tr in self.tracks.items()
        }