# wave_sim_stabilizer_2d.py
# 2D nonlinear wave simulation with suppression mechanism

import numpy as np
import matplotlib.pyplot as plt

class WaveSimStabilizer2D:
    def __init__(self, L=20, N=256, dt=0.001, T=0.5, beta=1.0, g=1.0, theta=0.05, lambda_=1.0, epsilon=1e-2):
        self.L = L
        self.N = N
        self.dx = L / N
        self.dt = dt
        self.T = T
        self.steps = int(T / dt)
        self.beta = beta
        self.g = g
        self.theta = theta
        self.lambda_ = lambda_
        self.epsilon = epsilon

        self.x = np.linspace(-L/2, L/2, N)
        self.y = np.linspace(-L/2, L/2, N)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        kx = 2 * np.pi * np.fft.fftfreq(N, d=self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(N, d=self.dx)
        self.kx2, self.ky2 = np.meshgrid(kx**2, ky**2)
        self.k2 = self.kx2 + self.ky2

    def initial_condition(self):
        return np.exp(-self.X**2 - self.Y**2) * (1 + 0.2 * np.sin(4 * self.X) * np.sin(4 * self.Y))

    def S_epsilon(self, grad):
        return 1 / (1 + np.exp(-(np.abs(grad) - self.theta) / self.epsilon))

    def simulate(self, psi0=None, apply_suppression=True):
        if psi0 is None:
            psi = self.initial_condition()
        else:
            psi = psi0

        psi = psi.astype(complex)
        psi_t = np.copy(psi)
        snapshots = []

        for step in range(self.steps):
            gradx, grady = np.gradient(psi_t, self.dx)
            grad_mag = np.sqrt(np.abs(gradx)**2 + np.abs(grady)**2)

            suppression_term = self.lambda_ * self.S_epsilon(grad_mag) * np.abs(psi_t)**2 if apply_suppression else 0
            nonlinear = np.exp(-1j * self.dt * (self.g * np.abs(psi_t)**2 + suppression_term))
            psi_t = psi_t * nonlinear

            psi_k = np.fft.fft2(psi_t)
            psi_k *= np.exp(-1j * self.beta * self.k2 * self.dt)
            psi_t = np.fft.ifft2(psi_k)

            if step % 50 == 0:
                snapshots.append(np.abs(psi_t)**2)

        return np.array(snapshots), psi_t

    def plot_snapshot(self, frame, title="|psi|^2 Snapshot"):
        plt.imshow(frame, extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]], origin='lower', cmap='magma')
        plt.colorbar(label='|psi|^2')
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

if __name__ == "__main__":
    sim2d = WaveSimStabilizer2D()
    snapshots_with, _ = sim2d.simulate(apply_suppression=True)
    snapshots_without, _ = sim2d.simulate(apply_suppression=False)

    sim2d.plot_snapshot(snapshots_with[-1], title="With Suppression (Final Snapshot)")
    sim2d.plot_snapshot(snapshots_without[-1], title="Without Suppression (Final Snapshot)")