import numpy as np

class UKF:
    def __init__(self, init_state, q_dist=0.1, q_vel=0.1, r_dist=0.8, r_vel=0.8):
        self.n = 2
        self.state = np.array(init_state, dtype=float)
        self.P = np.eye(self.n)

        # Array Proses Noise (Q)
        self.Q = np.array([[q_dist, 0.0  ],
                           [0.0,    q_vel]])

        # Array Kovarian Noise (R)
        self.R = np.array([[r_dist, 0.0  ],
                           [0.0,    r_vel]])

        # Parameter UKF
        self.alpha = 1e-3
        self.beta  = 2.0
        self.kappa = 0.0
        self.lam   = self.alpha**2 * (self.n + self.kappa) - self.n

        # Bobot Poin Sigma
        self.Wm = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.lam)))
        self.Wc = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.lam)))
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.lam / (self.n + self.lam) + (1 - self.alpha**2 + self.beta)

    def _sigma_points(self): # Poin Sigma
        sigma = np.zeros((2 * self.n + 1, self.n))
        sigma[0] = self.state

        P_reg = self.P + np.eye(self.n) * 1e-9
        try:
            L = np.linalg.cholesky((self.n + self.lam) * P_reg)
        except np.linalg.LinAlgError:
            U, s, _ = np.linalg.svd(P_reg)
            L = U @ np.diag(np.sqrt(np.maximum(s, 0))) * np.sqrt(self.n + self.lam)

        for i in range(self.n):
            sigma[i + 1]          = self.state + L[:, i]
            sigma[i + 1 + self.n] = self.state - L[:, i]

        return sigma

    def _f(self, x, dt):
        return np.array([
            x[0] + x[1] * dt,
            x[1]
        ])

    def _h(self, x): # Model Pengukuran
        return x.copy()

    def predict(self, dt): # Tahap Prediksi
        sigmas = self._sigma_points()

        sigmas_pred = np.array([self._f(s, dt) for s in sigmas])

        x_pred = np.sum(self.Wm[:, None] * sigmas_pred, axis=0)

        diff = sigmas_pred - x_pred
        P_pred = np.zeros((self.n, self.n))
        for i in range(2 * self.n + 1):
            P_pred += self.Wc[i] * np.outer(diff[i], diff[i])
        P_pred += self.Q

        self.state = x_pred
        self.P = (P_pred + P_pred.T) / 2
        self._sigmas_pred = sigmas_pred

    def update(self, z): # Tahap Update dan Koreksi
        z = np.atleast_1d(z)

        Z = np.array([self._h(s) for s in self._sigmas_pred])
        z_mean = np.sum(self.Wm[:, None] * Z, axis=0)

        diff_z = Z - z_mean
        diff_x = self._sigmas_pred - self.state
        Pzz = np.zeros((self.n, self.n))
        Pxz = np.zeros((self.n, self.n))
        for i in range(2 * self.n + 1):
            Pzz += self.Wc[i] * np.outer(diff_z[i], diff_z[i])
            Pxz += self.Wc[i] * np.outer(diff_x[i], diff_z[i])
        Pzz += self.R

        K = Pxz @ np.linalg.inv(Pzz)

        self.state = self.state + K @ (z - z_mean)
        self.P = self.P - K @ Pzz @ K.T
        self.P = (self.P + self.P.T) / 2

        return self.state.copy()

    def reset(self, init_state): # Tahap Reset Filter
        self.state = np.array(init_state, dtype=float)
        self.P = np.eye(self.n)