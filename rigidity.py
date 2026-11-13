
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Simulation data
# --------------------------------------------------

# Proton kinetic energies (MeV)
energy = np.array([100, 316, 1000])

# Proton rigidities (GV)
rigidity = np.array([0.445, 0.832, 1.696])

# Parallel diffusion coefficient (10^17 m^2/s)
kappa = np.array([1.5244, 3.2253, 6.9536])

# Standard deviation across 3 field realisations
kappa_std = np.array([0.0477, 0.3774, 0.9284])

# --------------------------------------------------
# Power-law fit: kappa = C * R^alpha
# --------------------------------------------------

log_R = np.log10(rigidity)
log_kappa = np.log10(kappa)

# Linear fit in log-log space
alpha, log_C = np.polyfit(log_R, log_kappa, 1)

C = 10**log_C

# Predicted values
kappa_fit = C * rigidity**alpha

# R-squared
residuals = log_kappa - np.log10(kappa_fit)
SS_res = np.sum(residuals**2)
SS_tot = np.sum((log_kappa - np.mean(log_kappa))**2)
R2 = 1 - SS_res / SS_tot

# --------------------------------------------------
# Kolmogorov QLT prediction
# --------------------------------------------------

alpha_QLT = 1/3

# Normalise theoretical curve at first simulation point
C_QLT = kappa[0] / rigidity[0]**alpha_QLT

R_smooth = np.logspace(
    np.log10(rigidity.min()) - 0.05,
    np.log10(rigidity.max()) + 0.05,
    200
)

kappa_QLT = C_QLT * R_smooth**alpha_QLT
kappa_smooth = C * R_smooth**alpha

# --------------------------------------------------
# Print results
# --------------------------------------------------

print("Power-law fit: kappa_parallel = C * R^alpha")
print(f"Fitted exponent alpha = {alpha:.4f}")
print(f"Fitting constant C = {C:.4f}")
print(f"R-squared = {R2:.5f}")
print(f"Kolmogorov QLT exponent = {alpha_QLT:.4f}")

# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

ax.errorbar(
    rigidity,
    kappa,
    yerr=kappa_std,
    fmt='o',
    markersize=8,
    capsize=5,
    linewidth=2,
    label='Simulation'
)

ax.plot(
    R_smooth,
    kappa_smooth,
    linewidth=2,
    label=fr'Power-law fit ($\alpha={alpha:.3f}$)'
)

ax.plot(
    R_smooth,
    kappa_QLT,
    '--',
    linewidth=2,
    label=r'Kolmogorov QLT ($\alpha=1/3$)'
)

ax.set_xscale('log')
ax.set_yscale('log')

ax.yaxis.set_major_formatter(plt.ScalarFormatter())
ax.ticklabel_format(axis='y', style='plain')

# Axis titles
ax.set_xlabel('Rigidity (GV)', fontsize=18)
ax.set_ylabel(r'$\kappa_{\parallel}$ ($10^{17}\,\mathrm{m^2\,s^{-1}}$)',
              fontsize=18)

# Tick values
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=14)

# Legend and grid
ax.legend(fontsize=14)
ax.grid(True, which='both', alpha=0.3)
ax.set_xlabel('Rigidity (GV)', fontsize=14)
ax.set_ylabel(
    r'$\kappa_\parallel$ ($10^{17}$ m$^2$ s$^{-1}$)',
    fontsize=14
)



ax.tick_params(axis='x', labelsize=14)  # X-axis values
ax.tick_params(axis='y', labelsize=14)  # Y-axis values

ax.legend(fontsize=14)
ax.grid(True, which='both', alpha=0.3)

# Display fitted exponent and R-squared
ax.text(
    0.05, 0.95,
    fr'$\alpha = {alpha:.3f}$' + '\n' + fr'$R^2 = {R2:.4f}$',
    transform=ax.transAxes,
    fontsize=14,
    verticalalignment='top'
)

plt.tight_layout()
plt.savefig('rigidity_diffusion_fit.png', dpi=300)
plt.show()
