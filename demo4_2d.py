import numpy as np
from wave_sim_stabilizer_2d import WaveSimStabilizer2D

# Run the 2D simulation with and without suppression
sim2d = WaveSimStabilizer2D(g=2.0, lambda_=1.2, theta=0.1)

snapshots_with, final_with = sim2d.simulate(apply_suppression=True)
snapshots_without, final_without = sim2d.simulate(apply_suppression=False)

# Compute final metrics
def compute_metrics(field):
    abs_psi2 = np.abs(field)**2
    max_val = np.max(abs_psi2)
    var_val = np.var(abs_psi2)
    return max_val, var_val

max_with, var_with = compute_metrics(final_with)
max_wo, var_wo = compute_metrics(final_without)

# Output numerical comparison
{
    "Max |psi|^2 (With Suppression)": max_with,
    "Max |psi|^2 (Without Suppression)": max_wo,
    "Variance (With Suppression)": var_with,
    "Variance (Without Suppression)": var_wo
}
