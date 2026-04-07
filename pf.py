import numpy as np

class ParticleFilter:
    def __init__(self, init_state=None, N=1000, process_std_dist=0.2, process_std_vel=0.2,
                 meas_std_dist=0.005, meas_std_vel=0.005, ess_threshold=0.5):
        self.N = N
        self.ess_threshold = ess_threshold

        # Noise Proses
        self.process_std = np.array([process_std_dist, process_std_vel])

        # Noise pengukuran
        self.meas_std = np.array([meas_std_dist, meas_std_vel])

        # Inisialisasi Partikel
        if init_state is None:
            init_state = [0.0, 0.0]
        self.particles = np.random.normal(
            np.array(init_state),
            self.process_std,
            size=(N, 2)
        )
        self.weights = np.ones(N) / N
        self.ess_history = []

    def predict(self, dt=1.0): # Tahap Prediksi
        self.particles[:, 0] += self.particles[:, 1] * dt
        noise = np.random.normal(0, self.process_std, size=(self.N, 2))
        self.particles += noise

    def update(self, z): # Tahap Update
        z = np.atleast_1d(z)

        # Likelihood: produk Gaussian per dimensi
        diff = self.particles - z
        log_likelihood = -0.5 * np.sum((diff / self.meas_std) ** 2, axis=1)
        likelihood = np.exp(log_likelihood) + 1e-300

        self.weights *= likelihood
        total = np.sum(self.weights)
        if total == 0 or np.isnan(total):
            self.weights.fill(1.0 / self.N)
        else:
            self.weights /= total

        # Resampling
        self.resample_if_needed()

        return self.estimate()

    def ess(self):
        return 1.0 / np.sum(self.weights ** 2)

    def _systematic_resample(self):
        positions = (np.arange(self.N) + np.random.random()) / self.N
        cumulative = np.cumsum(self.weights)
        i, j = 0, 0
        indexes = np.zeros(self.N, dtype=int)
        while i < self.N:
            if positions[i] < cumulative[j]:
                indexes[i] = j
                i += 1
            else:
                j = min(j + 1, self.N - 1)
        self.particles = self.particles[indexes]
        self.weights.fill(1.0 / self.N)

    def resample_if_needed(self): # Tahap Resampling
        current_ess = self.ess()
        self.ess_history.append(current_ess)
        if current_ess < self.N * self.ess_threshold:
            self._systematic_resample()
            return True
        return False

    def estimate(self): # Tahap Estimasi
        return np.average(self.particles, weights=self.weights, axis=0)

    def reset(self, init_state):
        self.particles = np.random.normal(
            np.array(init_state),
            self.process_std,
            size=(self.N, 2)
        )
        self.weights = np.ones(self.N) / self.N
        self.ess_history = []