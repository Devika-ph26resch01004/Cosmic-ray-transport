
import numpy as np

# ============================================================
# Analytical parallel diffusion coefficient: QLT
# Run 7 parameters — Giacalone & Jokipii (1999)
# ============================================================

# Physical constants (SI)
c = 2.99792458e8             # Speed of light [m/s]
e = 1.602176634e-19          # Elementary charge [C]
mp = 1.67262192369e-27       # Proton mass [kg]
AU = 1.495978707e11          # Astronomical unit [m]

# ------------------------------------------------------------
# Simulation parameters
# ------------------------------------------------------------
K_MeV = 316.0                # Proton kinetic energy [MeV]
B0_nT = 5.0                  # Mean magnetic field [nT]
Brms_nT = 5.0                # Turbulent magnetic-field RMS [nT]
Lc_AU = 0.01                 # Correlation length [AU]

# Simulation diffusion coefficient (realization 1)
#kappa_sim = 1.524437e+17  # [m^2/s]
kappa_sim = 3.221147e+17
# Reference value reported for Run 7
kappa_reference = 5e17    # [m^2/s]

# ------------------------------------------------------------
# Unit conversions
# ------------------------------------------------------------
K_J = K_MeV * 1e6 * e
B0 = B0_nT * 1e-9
Brms = Brms_nT * 1e-9
Lc = Lc_AU * AU

# ------------------------------------------------------------
# Relativistic particle properties
# ------------------------------------------------------------
gamma = 1.0 + K_J / (mp * c**2)
v = c * np.sqrt(1.0 - 1.0 / gamma**2)

# Relativistic gyrofrequency [rad/s; numerically s^-1]
Omega0 = e * B0 / (gamma * mp)

# Turbulence variance convention:
# sigma^2 = <delta B^2>, with units T^2
sigma = Brms

# ------------------------------------------------------------
# Derived QLT equation
#
# kappa_parallel =
# 3 v^3 / (20 Lc Omega0^2)
# * (B0/sigma)^2
# * csc(3 pi/5)
# * [1 + (72/7)(Omega0 Lc/v)^(5/3)]
# ------------------------------------------------------------
cosecant = 1.0 / np.sin(3.0 * np.pi / 5.0)
x = Omega0 * Lc / v

kappa_QLT = (
    (3.0 * v**3) / (20.0 * Lc * Omega0**2)
    * (B0 / sigma)**2
    * cosecant
    * (1.0 + (72.0 / 7.0) * x**(5.0 / 3.0))
)

'''
kappa_QLT = (
    (3.0 * v**3 * Omega0**2) / (20.0 * Lc)
    * (B0 / sigma)**2
    * cosecant
    * (1.0 + (72.0 / 7.0) * x**(5.0 / 3.0))
)
'''
# ------------------------------------------------------------
# Comparisons
# ------------------------------------------------------------
def percent_difference(value, reference):
    return 100.0 * (value - reference) / reference

print("=" * 65)
print("ANALYTICAL PARALLEL DIFFUSION COEFFICIENT")
print("=" * 65)

print(f"Kinetic energy             : {K_MeV:.3f} MeV")
print(f"Lorentz factor, gamma      : {gamma:.6f}")
print(f"Particle speed             : {v:.6e} m/s")
print(f"Mean magnetic field        : {B0_nT:.3f} nT")
print(f"Turbulent RMS field        : {Brms_nT:.3f} nT")
print(f"Correlation length         : {Lc_AU:.6f} AU")
print(f"Relativistic gyrofrequency : {Omega0:.6e} s^-1")
print(f"Omega0 * Lc / v            : {x:.6f}")
print("-" * 65)

print(f"Analytical QLT kappa       : {kappa_QLT:.6e} m^2/s")
print(f"Simulation kappa           : {kappa_sim:.6e} m^2/s")
print(f"Paper reference kappa      : {kappa_reference:.6e} m^2/s")
print("-" * 65)

print(
    "Simulation vs QLT          : "
    f"{percent_difference(kappa_sim, kappa_QLT):+.2f}%"
)
print(
    "QLT vs paper reference     : "
    f"{percent_difference(kappa_QLT, kappa_reference):+.2f}%"
)
print(
    "Simulation vs paper ref.   : "
    f"{percent_difference(kappa_sim, kappa_reference):+.2f}%"
)
print("=" * 65)
