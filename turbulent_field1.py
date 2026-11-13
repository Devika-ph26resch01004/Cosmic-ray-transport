
import numpy as np
from numba import njit

AU = 1.495978707e11  # m


@njit
def _field_numba(x, k, khat, xprime, yprime, alpha, beta, amplitudes):
    """Compute the turbulent magnetic field in tesla."""
    bx = 0.0
    by = 0.0
    bz = 0.0

    for n in range(len(k)):
        phase = (
            k[n] * (
                khat[n, 0] * x[0]
                + khat[n, 1] * x[1]
                + khat[n, 2] * x[2]
            )
            + beta[n]
        )

        ca = np.cos(alpha[n])
        sa = np.sin(alpha[n])
        cp = np.cos(phase)
        sp = np.sin(phase)

        factor_x = amplitudes[n] * ca * cp
        factor_y = -amplitudes[n] * sa * sp

        bx += factor_x * xprime[n, 0] + factor_y * yprime[n, 0]
        by += factor_x * xprime[n, 1] + factor_y * yprime[n, 1]
        bz += factor_x * xprime[n, 2] + factor_y * yprime[n, 2]

    return np.array([bx, by, bz])


class GJ99Turbulence:
    """
    GJ99-inspired isotropic turbulence model.

    Coordinates passed to field(x) are in metres.
    Wavelengths and Lc are specified in AU at initialization.
    """

    def __init__(
        self,
        Brms,
        lmin_AU,
        lmax_AU,
        Lc_AU,
        Nm=512,
        seed=12345,
    ):
        self.Brms = float(Brms)
        self.Lc = Lc_AU * AU
        self.Nm = int(Nm)
        self.rng = np.random.default_rng(seed)

        kmin = 2.0 * np.pi / (lmax_AU * AU)
        kmax = 2.0 * np.pi / (lmin_AU * AU)
        self.k = np.geomspace(kmin, kmax, self.Nm)

        logk = np.log(self.k)
        log_edges = np.empty(self.Nm + 1)
        log_edges[1:-1] = 0.5 * (logk[:-1] + logk[1:])
        log_edges[0] = logk[0] - 0.5 * (logk[1] - logk[0])
        log_edges[-1] = logk[-1] + 0.5 * (logk[-1] - logk[-2])

        edges = np.exp(log_edges)
        dk = edges[1:] - edges[:-1]
        shell_volume = 4.0 * np.pi * self.k**2 * dk

        gamma = 11.0 / 3.0
        weights = shell_volume / (1.0 + (self.k * self.Lc)**gamma)

        # Preserve the existing real-field RMS normalization.
        self.A = np.sqrt(2.0) * self.Brms * np.sqrt(
            weights / np.sum(weights)
        )

        cos_theta = self.rng.uniform(-1.0, 1.0, self.Nm)
        phi = self.rng.uniform(0.0, 2.0 * np.pi, self.Nm)
        sin_theta = np.sqrt(1.0 - cos_theta**2)

        self.khat = np.column_stack((
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta,
        ))

        self.xprime = np.column_stack((
            cos_theta * np.cos(phi),
            cos_theta * np.sin(phi),
            -sin_theta,
        ))

        self.yprime = np.column_stack((
            -np.sin(phi),
            np.cos(phi),
            np.zeros(self.Nm),
        ))

        self.alpha = self.rng.uniform(0.0, 2.0 * np.pi, self.Nm)
        self.beta = self.rng.uniform(0.0, 2.0 * np.pi, self.Nm)

    def field(self, x):
        """Return delta B(x) in tesla; x must be a 3-vector in metres."""
        x = np.asarray(x, dtype=np.float64)
        return _field_numba(
            x,
            self.k,
            self.khat,
            self.xprime,
            self.yprime,
            self.alpha,
            self.beta,
            self.A,
        )
