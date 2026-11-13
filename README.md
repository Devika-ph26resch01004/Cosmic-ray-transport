# Cosmic-Ray Transport in Turbulent Magnetic Fields

## 1. Project Overview

This project investigates the parallel diffusion of energetic cosmic-ray protons in turbulent magnetic fields using numerical test-particle simulations.

The primary objective is to study the dependence of the parallel diffusion coefficient on particle energy and rigidity. The simulation approach is inspired by the work of Giacalone and Jokipii (1999), *The Transport of Cosmic Rays across a Turbulent Magnetic Field*.

Particle trajectories are obtained by numerically integrating the relativistic Lorentz equation in a prescribed, static, three-dimensional turbulent magnetic field. The parallel diffusion coefficient is estimated from the time evolution of the ensemble-averaged mean-square displacement along the direction of the mean magnetic field.

## 2. Project Objectives

- Investigate the parallel transport of energetic protons in turbulent magnetic fields.
- Calculate the parallel diffusion coefficient using the mean-square displacement method.
- Examine the dependence of the diffusion coefficient on particle kinetic energy and rigidity.
- Compare the numerical results with quasi-linear theory (QLT) and published results from Giacalone and Jokipii (1999).

## 3. Simulation Setup

The simulations use a uniform background magnetic field along the z-direction, with superimposed three-dimensional magnetic fluctuations.

| Parameter | Value |
|---|---|
| Background magnetic field, \(B_0\) | 5 nT |
| RMS turbulent magnetic field, \(\delta B_{\rm rms}\) | 5 nT |
| Relative fluctuation amplitude, \(\delta B_{\rm rms}/B_0\) | 1 |
| Correlation length, \(L_c\) | 0.01 AU |
| Number of Fourier modes | 512 |
| Particle energies | 100, 316, and 1000 MeV |
| Particles per realisation | 300 |
| Independent turbulent-field realisations per energy | 3 |
| Magnetic field | Static, three-dimensional |

The turbulent magnetic field is constructed using a Fourier-mode representation with isotropically distributed wavevector orientations. The adopted spectral weighting is

\[
W(k)=\frac{1}{1+(kL_c)^{11/3}}.
\]

Particle trajectories are calculated by numerically integrating the relativistic Lorentz equation using the DOP853 integration method.

## 4. Calculation of the Parallel Diffusion Coefficient

The ensemble-averaged mean-square displacement along the mean magnetic field is calculated as

\[
\left\langle \Delta z^2(t)\right\rangle
=
\frac{1}{N}\sum_{i=1}^{N}
[z_i(t)-z_i(0)]^2.
\]

In the diffusive regime, the parallel diffusion coefficient is obtained from

\[
\kappa_\parallel
=
\frac{1}{2}
\frac{d\left\langle\Delta z^2(t)\right\rangle}{dt}.
\]

A linear fit to the mean-square displacement in the identified diffusive time interval is used to estimate the diffusion coefficient.

## 5. Repository Structure

```text
Cosmic-ray-transport/
│
├── simulation_codes/
│   ├── turbulent_field.py
│   └── ensemble_300.py
│
├── analysis_codes/
│   ├── calculate_kappa_parallel.py
│   ├── rigidity_analysis.py
│   └── plot_results.py
│
├── simulation_results/
│   ├── Ensemble MSD data files
│   └── Diffusion coefficient summary
│
├── requirements.txt
└── README.md
```

The directory contains the simulation scripts, analysis scripts, and numerical results used in this project.

## 6. Simulation Results

The simulations were performed for three proton kinetic energies, with three independent turbulent-field realisations for each energy.

The resulting mean parallel diffusion coefficients are:

| Proton kinetic energy | Mean \(\kappa_\parallel\) (m²/s) |
|---|---:|
| 100 MeV | \(1.5244\times10^{17}\) |
| 316 MeV | \(3.2253\times10^{17}\) |
| 1000 MeV | \(6.9536\times10^{17}\) |

The uncertainties are estimated using the sample standard deviation across the three independent turbulent-field realisations.

The results indicate an increase in the parallel diffusion coefficient with increasing particle energy and rigidity.

## 7. Software Requirements

The simulations and analysis use Python and the following packages:

- NumPy
- SciPy
- MPI4Py
- Numba
- Pandas
- Matplotlib

The package dependencies are listed in `requirements.txt`.

To install the Python dependencies, run:

```bash
pip install -r requirements.txt
```

An MPI implementation, such as Open MPI or MPICH, must also be installed separately for parallel execution using MPI.

## 8. References

1. Giacalone, J., & Jokipii, J. R. (1999). The transport of cosmic rays across a turbulent magnetic field. *The Astrophysical Journal*, 520, 204–214. https://doi.org/10.1086/307452

2. Jokipii, J. R. (1966). Cosmic-ray propagation. I. Charged particles in a random magnetic field. *The Astrophysical Journal*, 146, 480–487. https://doi.org/10.1086/148912

3. Casse, F., Lemoine, M., & Pelletier, G. (2002). Transport of cosmic rays in chaotic magnetic fields. *Physical Review D*, 65, 023002.

