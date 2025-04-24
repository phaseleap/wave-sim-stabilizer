# rogue_wave_test.py
# Demonstration of percolation suppression on a rogue-wave-like high-gradient initial condition

from wave_sim_stabilizer import WaveSimStabilizer
import numpy as np
import matplotlib.pyplot as plt

# --- Suppression Charge Function ---
def suppression_charge(psi, grad_psi, dx, lambda_, theta):
    return np.sum(lambda_ * (np.abs(grad_psi) > theta) * np.abs(psi)**2) * dx

# --- Simulation Wrapper with Metric Tracking ---
def simulate_with_metrics(sim: WaveSimStabilizer, psi0, apply_suppression=True):
    psi = psi0.astype(complex)
    psi_t = np.copy(psi)
    max_vals, variances, charges = [], [], []

    for step in range(sim.steps):
        grad_psi = np.gradient(psi_t, sim.dx)
        suppression_term = sim.lambda_ * sim.S_epsilon(grad_psi) * np.abs(psi_t)**2 if apply_suppression else 0
        nonlinear = np.exp(-1j * sim.dt * (sim.g * np.abs(psi_t)**2 + suppression_term))
        psi_t = psi_t * nonlinear

        psi_k = np.fft.fft(psi_t)
        psi_k *= np.exp(-1j * sim.beta * sim.k2 * sim.dt)
        psi_t = np.fft.ifft(psi_k)

        if step % 100 == 0:
            abs_psi2 = np.abs(psi_t)**2
            max_vals.append(np.max(abs_psi2))
            variances.append(np.var(abs_psi2))
            charges.append(suppression_charge(psi_t, grad_psi, sim.dx, sim.lambda_, sim.theta))

    return np.array(max_vals), np.array(variances), np.array(charges)

# --- Define the Simulation ---
sim = WaveSimStabilizer(g=3.5, lambda_=1.5, theta=0.1)
psi0 = np.exp(-sim.x**2) * (1 + 0.9 * np.sin(10 * sim.x))  # High-energy input

# --- Run Both Versions ---
max_w, var_w, Q_w = simulate_with_metrics(sim, psi0, apply_suppression=True)
max_wo, var_wo, Q_wo = simulate_with_metrics(sim, psi0, apply_suppression=False)

# --- Report ---
print("===== Final Metrics =====")
print(f"Max |psi|^2 WITH suppression: {max_w[-1]:.4f}")
print(f"Max |psi|^2 WITHOUT suppression: {max_wo[-1]:.4f}")
print(f"Variance WITH suppression: {var_w[-1]:.4f}")
print(f"Variance WITHOUT suppression: {var_wo[-1]:.4f}")
print(f"Delta Q WITH: {abs(Q_w[-1] - Q_w[0]):.4e}")
print(f"Delta Q WITHOUT: {abs(Q_wo[-1] - Q_wo[0]):.4e}")

# --- Plot Results ---
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

axs[0].plot(max_w, label='With Suppression')
axs[0].plot(max_wo, label='Without Suppression')
axs[0].set_ylabel("Max |psi|^2")
axs[0].legend()

axs[1].plot(var_w, label='With Suppression')
axs[1].plot(var_wo, label='Without Suppression')
axs[1].set_ylabel("Variance")

axs[2].plot(Q_w, label='Q With Suppression')
axs[2].plot(Q_wo, label='Q Without Suppression', linestyle='--')
axs[2].set_ylabel("Suppression Charge Q")
axs[2].set_xlabel("Step (x100)")

plt.tight_layout()
plt.show()