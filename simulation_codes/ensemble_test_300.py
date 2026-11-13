import sys
import numpy as np
from scipy.integrate import solve_ivp
from turbulent_field1 import GJ99Turbulence
sys.path.insert(0, "/home/devika/.local/lib/python3.11/site-packages")
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Physical constants (SI)
c = 299792458.0
mp = 1.67262192369e-27
qe = 1.602176634e-19
AU = 1.495978707e11

# Simulation parameters
N = 300
K0 = 1000e6 * qe
B0 = 5e-9
Brms = 5e-9

lmin_AU = 1.01e-4
lmax_AU = 1.006
Lc_AU = 0.01
Nm = 512

field_seed = 12345
particle_seed = 24680

# Relativistic initial momentum and gyroperiod
gamma0 = 1.0 + K0 / (mp * c**2)
p0 = np.sqrt(K0 * (K0 + 2.0 * mp * c**2)) / c
Tgyro = 2.0 * np.pi * gamma0 * mp / (qe * B0)

# Identical turbulent field realization on every rank
turbulence = GJ99Turbulence(
    Brms=Brms,
    lmin_AU=lmin_AU,
    lmax_AU=lmax_AU,
    Lc_AU=Lc_AU,
    Nm=Nm,
    seed=field_seed,
)

B_mean = np.array([0.0, 0.0, B0])

def magnetic_field(x):
    return turbulence.field(x) + B_mean

def rhs(t, y):
    x = y[:3]
    p = y[3:]

    gamma = np.sqrt(1.0 + np.dot(p, p) / (mp**2 * c**2))
    v = p / (gamma * mp)

    B = magnetic_field(x)
    dpdt = qe * np.cross(v, B)

    return np.concatenate((v, dpdt))

# Integration duration and output cadence
t_end = 200.0 * Tgyro
n_times = 1001
times = np.linspace(0.0, t_end, n_times)

# Reproducible isotropic initial directions
rng = np.random.default_rng(particle_seed)
mu = rng.uniform(-1.0, 1.0, N)
phi = rng.uniform(0.0, 2.0 * np.pi, N)
sin_theta = np.sqrt(1.0 - mu**2)

directions = np.column_stack((
    sin_theta * np.cos(phi),
    sin_theta * np.sin(phi),
    mu,
))

# Distribute particle indices across MPI ranks
particle_indices = np.array_split(np.arange(N), size)
my_indices = particle_indices[rank]

local_z = np.empty((len(my_indices), n_times))
local_energy_errors = np.empty(len(my_indices))

print(
    f"Rank {rank}/{size}: assigned {len(my_indices)} particles",
    flush=True,
)

# Trigger Numba compilation before timing particle integrations
turbulence.field(np.zeros(3))

local_start = MPI.Wtime()

for j, i in enumerate(my_indices):
    y0 = np.concatenate((np.zeros(3), p0 * directions[i]))

    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        y0,
        method="DOP853",
        t_eval=times,
        rtol=1e-10,
        atol=1e-22,
        max_step=Tgyro / 200.0,
    )

    if not sol.success or sol.y.shape[1] != n_times:
        raise RuntimeError(
            f"Rank {rank}, particle {i} failed: {sol.message}"
        )

    local_z[j] = sol.y[2]

    momenta = sol.y[3:6].T
    p_squared = np.sum(momenta**2, axis=1)
    gamma = np.sqrt(1.0 + p_squared / (mp**2 * c**2))
    kinetic_energy = (gamma - 1.0) * mp * c**2

    local_energy_errors[j] = (
        np.max(np.abs(kinetic_energy - K0)) / K0
    )

    if (j + 1) % 5 == 0 or j + 1 == len(my_indices):
        print(
            f"Rank {rank}: completed {j + 1}/{len(my_indices)}",
            flush=True,
        )

local_elapsed = MPI.Wtime() - local_start

# Gather results to rank 0; only rank 0 writes the output files
gathered = comm.gather(
    (my_indices, local_z, local_energy_errors, local_elapsed),
    root=0,
)

if rank == 0:
    z_trajectories = np.empty((N, n_times))
    energy_errors = np.empty(N)
    rank_times = np.empty(size)

    for indices, z_part, errors_part, elapsed_part in gathered:
        z_trajectories[indices] = z_part
        energy_errors[indices] = errors_part

    for r, item in enumerate(gathered):
        rank_times[r] = item[3]

    # Ensemble parallel mean-square displacement
    msd_z = np.mean(z_trajectories**2, axis=0)

    trajectory_file = "particle_z_trajectories_N300_Lc_0.01.csv"
    msd_file = "ensemble_msd_N300_Lc_0.01.csv"

    np.savetxt(
        trajectory_file,
        np.column_stack((times, z_trajectories.T / AU)),
        delimiter=",",
        header="time_s," + ",".join(
            f"z_particle_{i}_AU" for i in range(N)
        ),
        comments="",
    )

    np.savetxt(
        msd_file,
        np.column_stack((times, msd_z / AU**2)),
        delimiter=",",
        header="time_s,msd_z_AU2",
        comments="",
    )

    print("\n--- N300 ensemble complete ---")
    print(f"MPI ranks: {size}")
    print(f"Particles: {N}")
    print(f"Simulated duration [s]: {t_end:.3f}")
    print(f"Final MSD [AU^2]: {msd_z[-1] / AU**2:.8e}")
    print(
        "Maximum relative energy error: "
        f"{np.max(energy_errors):.3e}"
    )
    print(f"Slowest rank integration time [s]: {np.max(rank_times):.2f}")
    print(f"Saved: {trajectory_file}")
    print(f"Saved: {msd_file}")

comm.Barrier()

