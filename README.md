# Wave Sim Stabilizer

A nonlinear wave simulation toolkit with built-in **threshold-based suppression**. Prevents chaotic blow-up in wave equations like the nonlinear Schrödinger equation (NLSE).

## Features
- ✅ Stabilizes rogue waves and solitons
- ✅ One-line toggle for suppression
- ✅ Built-in visualization
- ✅ Zero-tuning threshold model

## Usage
from wave_sim_stabilizer import WaveSimStabilizer

sim = WaveSimStabilizer()
history, _ = sim.simulate(apply_suppression=True)
sim.plot_history(history, title="With Suppression")

## or use the demo notebook:
jupyter notebook demo.ipynb

## Who It's For?
Who It's For

Researchers modeling nonlinear waves (optics, BEC, plasmas)
Physics & engineering students
Sim tool devs who want stable dynamics

---

### ✅ **Step 4: Write the Demo Notebook**

Create `demo.ipynb` and add a simple test case:
```python
from wave_sim_stabilizer import WaveSimStabilizer

sim = WaveSimStabilizer()
history_with, _ = sim.simulate(apply_suppression=True)
sim.plot_history(history_with, title="With Suppression")

history_without, _ = sim.simulate(apply_suppression=False)
sim.plot_history(history_without, title="Without Suppression")

## Installation
```bash
pip install -r requirements.txt

