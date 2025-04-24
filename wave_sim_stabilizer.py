# Wave Sim Stabilizer
# Nonlinear Schrödinger Equation (NLSE) solver with optional suppression mechanism

import numpy as np
import matplotlib.pyplot as plt

class WaveSimStabilizer:
    def __init__(self, L=20, N=1024, dt=0.001, T=1.0, beta=1.0, g=1.0, theta=0.05, lambda_=1.0, epsilon=1e-2):
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
        self.k = 2 * np.pi * np.fft.fftfreq(N, d=self.dx)
        self.k2 = self.k**2

    def initial_condition(self):
        return np.exp(-self.x**2) * (1 + 0.1 * np.cos(5 * self.x))

    def S_epsilon(self, grad):
        return 1 / (1 + np.exp(-(np.abs(grad) - self.theta) / self.epsilon))

    def simulate(self, psi0=None, apply_suppression=True):
        if psi0 is None:
            psi = self.initial_condition()
        else:
            psi = psi0

        psi = psi.astype(complex)
        psi_t = np.copy(psi)
        history = []

        for step in range(self.steps):
            grad_psi = np.gradient(psi_t, self.dx)
            suppression_term = self.lambda_ * self.S_epsilon(grad_psi) * np.abs(psi_t)**2 if apply_suppression else 0
            nonlinear = np.exp(-1j * self.dt * (self.g * np.abs(psi_t)**2 + suppression_term))
            psi_t = psi_t * nonlinear

            psi_k = np.fft.fft(psi_t)
            psi_k *= np.exp(-1j * self.beta * self.k2 * self.dt)
            psi_t = np.fft.ifft(psi_k)

            if step % 100 == 0:
                history.append(np.abs(psi_t)**2)

        return np.array(history), psi_t

    def plot_history(self, history, title="Wave Evolution"):
        plt.imshow(history, extent=[self.x[0], self.x[-1], 0, self.T], aspect='auto', origin='lower', cmap='magma')
        plt.colorbar(label='|psi|^2')
        plt.xlabel('x')
        plt.ylabel('time')
        plt.title(title)
        plt.show()

if __name__ == "__main__":
    sim = WaveSimStabilizer()
    history_with, _ = sim.simulate(apply_suppression=True)
    history_without, _ = sim.simulate(apply_suppression=False)

    sim.plot_history(history_with, title="With Suppression")
    sim.plot_history(history_without, title="Without Suppression")