
import numpy as np

# Physical constants
c = 299792458.0             # m/s
mp = 1.67262192369e-27      # kg
e = 1.602176634e-19         # C

# Proton kinetic energies in MeV
energy_MeV = np.array([100, 316, 1000])

# Convert kinetic energy to joules
K = energy_MeV * 1e6 * e

# Relativistic momentum
p = np.sqrt(K * (K + 2 * mp * c**2)) / c

# Rigidity in volts
R = p * c / e

# Convert to GV
R_GV = R / 1e9

for E, rigidity in zip(energy_MeV, R_GV):
    print(f"{E:4d} MeV: {rigidity:.4f} GV")
