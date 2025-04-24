# Wave Sim Stabilizer

A lightweight nonlinear wave simulation tool for the nonlinear Schrödinger equation (NLSE), featuring a threshold-based suppression mechanism to improve stability in high-gradient wave systems.

This method is designed for users modeling nonlinear dynamics where standard simulations may exhibit blow-up or unstable evolution. The suppression term is optional and configurable.

---

## 🔧 Features

- Simulates 1D NLSE-type systems using a split-step Fourier method
- Optional suppression mechanism activates beyond a user-defined gradient threshold
- Tracks key metrics: peak amplitude, variance, suppression charge
- Simple API for running custom initial waveforms
- Includes plotting tools and analysis-ready outputs

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Basic Usage

```python
from wave_sim_stabilizer import WaveSimStabilizer

sim = WaveSimStabilizer()
history, _ = sim.simulate(apply_suppression=True)
sim.plot_history(history, title="With Suppression")
```

Or run the interactive notebook:

```bash
jupyter notebook demo.ipynb
```

---

## 📘 Test Case: Rogue Wave Stability

Run `rogue_wave_test.py` to compare suppression ON vs OFF under a high-energy initial condition:

```bash
python rogue_wave_test.py
```

This script outputs:
- Peak wave intensity over time
- Variance of the field
- Evolution of suppression charge \( Q \)

---

## 👤 Who Is This For?

- Researchers modeling rogue waves, solitons, or BECs
- Physics students exploring nonlinear PDE behavior
- Developers building wave simulation tools or teaching materials

---

## 📂 Project Structure

```
wave-sim-stabilizer/
├── wave_sim_stabilizer.py    # Core solver
├── demo.ipynb                # Interactive walkthrough
├── rogue_wave_test.py        # Full comparison test
├── README.md                 # You’re here
├── requirements.txt          # Dependencies
```

---

## 🧠 Notes

- This implementation is in 1D and intended for exploratory or educational use.
- Suppression parameters (threshold, strength, smoothness) can be adjusted as needed.
- The code can be extended to higher dimensions or adapted to other nonlinear systems.

---

## 📫 Feedback / Collaboration

Open to feedback, contributions, or collaborative extensions (2D, experimental comparisons, ML applications).

```