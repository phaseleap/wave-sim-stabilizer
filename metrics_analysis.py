## metric_analysis.py
from wave_sim_stabilizer import WaveSimStabilizer
import numpy as np
import matplotlib.pyplot as plt

# --- Utilities ---
def suppression_charge(psi, grad_psi, dx, lambda_, theta):
    return np.sum(lambda_ * (np.abs(grad_psi) > theta) * np.abs(psi)**2) * dx

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

# --- Julia's high-energy test ---
sim = WaveSimStabilizer(g=2.5, lambda_=1.2, theta=0.1)
psi_custom = np.exp(-sim.x**2) * (1 + 0.4 * np.sin(10 * sim.x))

# Run both
max_w, var_w, Q_w = simulate_with_metrics(sim, psi_custom, apply_suppression=True)
max_wo, var_wo, Q_wo = simulate_with_metrics(sim, psi_custom, apply_suppression=False)

# --- Print metrics ---
print("----- FINAL NUMERIC METRICS -----")
print(f"Final Max |psi|^2 (With Suppression):     {max_w[-1]:.6f}")
print(f"Final Max |psi|^2 (Without Suppression):  {max_wo[-1]:.6f}")
print(f"Final Variance (With Suppression):        {var_w[-1]:.6f}")
print(f"Final Variance (Without Suppression):     {var_wo[-1]:.6f}")
print(f"Suppression Charge ΔQ (With):             {abs(Q_w[-1] - Q_w[0]):.6f}")
print(f"Suppression Charge ΔQ (Without):          {abs(Q_wo[-1] - Q_wo[0]):.6f}")

# --- Plot metrics ---
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

axs[0].plot(max_w, label='With Suppression')
axs[0].plot(max_wo, label='Without Suppression')
axs[0].set_ylabel("Max |psi|^2")
axs[0].legend()

axs[1].plot(var_w, label='With Suppression')
axs[1].plot(var_wo, label='Without Suppression')
axs[1].set_ylabel("Variance of |psi|^2")

axs[2].plot(Q_w, label='Suppression Charge Q (With)')
axs[2].plot(Q_wo, label='Q (Without)', linestyle='--')
axs[2].set_ylabel("Suppression Charge Q")
axs[2].set_xlabel("Time step")

plt.tight_layout()
plt.show()
