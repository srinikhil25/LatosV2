"""Gaussian-process surrogate, acquisition functions and leave-one-out evaluation.

Generic numerics, independent of any experiment. Inputs arrive scaled to the
unit box [0, 1]^d; responses arrive in their own units.

The model follows the form Latos V1 used with scikit-learn, with the noise made
explicit rather than assumed as a percentage:

* Kernel: amplitude² × Matérn(ν = 5/2), one length-scale per input (ARD).
  Amplitude and length-scales are fitted by maximum marginal likelihood with
  seeded restarts, so a fit is reproducible from its seed.
* Length-scales are bounded to [0.25, 5] times each input's range. With a
  handful of specimens, structure finer than a quarter of the range cannot be
  identified; reaching a bound is reported, not hidden.
* Responses are centred and scaled by their standard deviation for fitting
  (scikit-learn's ``normalize_y``) and predictions are returned in their units.
* Observation noise is fixed, not fitted: point i has variance
  SEM_i² + σ_add², where SEM_i is the specimen's measured within-specimen
  standard error and σ_add an additional level the researcher chooses. SEM_i
  alone is a lower bound on specimen-to-specimen variation.

Predicted standard deviations are those of the latent response (the model's
uncertainty about the mean response), not of a new measurement, unless stated.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

__all__ = [
    "AMPLITUDE_BOUNDS",
    "LENGTH_SCALE_BOUNDS",
    "RESTARTS",
    "Z95",
    "FittedGP",
    "LooResult",
    "expected_improvement",
    "expected_target_improvement",
    "fit_gp",
    "leave_one_out",
]

LENGTH_SCALE_BOUNDS = (0.25, 5.0)  # in units of each input's range
AMPLITUDE_BOUNDS = (1e-2, 1e2)  # amplitude² in units of the response variance
RESTARTS = 8
Z95 = 1.959963984540054
_INITIAL_LENGTH_SCALE = 0.5
# Added to the scaled noise variance so the covariance stays factorisable when
# a point carries no noise at all; far below any measured uncertainty.
_JITTER = 1e-10
_SD_FLOOR = 1e-12


@dataclass(frozen=True)
class FittedGP:
    """A fitted surrogate and the assumptions it was fitted under."""

    model: GaussianProcessRegressor
    x: np.ndarray  # (n, d) unit-box inputs
    y: np.ndarray  # (n,) responses
    noise_sd: np.ndarray  # (n,) assumed observation sd, response units
    additional_noise: float
    seed: int

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Latent mean and standard deviation at unit-box points (m, d)."""
        mean, sd = self.model.predict(np.atleast_2d(np.asarray(x, dtype=float)), return_std=True)
        return np.asarray(mean, dtype=float), np.maximum(np.asarray(sd, dtype=float), 0.0)

    @property
    def length_scales(self) -> tuple[float, ...]:
        return tuple(float(v) for v in np.atleast_1d(self.model.kernel_.k2.length_scale))

    @property
    def amplitude(self) -> float:
        """Prior standard deviation of the response, in response units."""
        return float(np.sqrt(self.model.kernel_.k1.constant_value) * _scale(self.y))

    @property
    def at_bounds(self) -> list[str]:
        """Hyperparameters that ended on a bound of their search range."""
        out = []
        for i, ls in enumerate(self.length_scales):
            for bound, side in zip(LENGTH_SCALE_BOUNDS, ("lower", "upper")):
                if np.isclose(ls, bound, rtol=1e-3):
                    out.append(f"length-scale {i + 1} at its {side} bound ({bound:g} × range)")
        amp2 = float(self.model.kernel_.k1.constant_value)
        for bound, side in zip(AMPLITUDE_BOUNDS, ("lower", "upper")):
            if np.isclose(amp2, bound, rtol=1e-3):
                out.append(f"amplitude at its {side} bound")
        return out

    @property
    def log_marginal_likelihood(self) -> float:
        return float(self.model.log_marginal_likelihood_value_)


def _scale(y: np.ndarray) -> float:
    """The standard deviation scikit-learn normalises by (1 when it is zero)."""
    s = float(np.std(y))
    return s if s > 0 else 1.0


def fit_gp(
    x: np.ndarray,
    y: np.ndarray,
    sem: np.ndarray | None = None,
    additional_noise: float = 0.0,
    seed: int = 0,
) -> FittedGP:
    """Fit the surrogate to unit-box inputs `x` (n, d) and responses `y` (n,).

    Args:
        sem: Measured standard error of each response; NaN or None where none
            was measured (that point then carries only the additional noise).
        additional_noise: σ_add, in response units, added in quadrature to
            every point's SEM.
        seed: Seeds the restarts of the hyperparameter search.
    """
    x = np.asarray(x, dtype=float).reshape(len(y), -1)
    y = np.asarray(y, dtype=float).ravel()
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("inputs and responses must be finite")
    if additional_noise < 0 or not np.isfinite(additional_noise):
        raise ValueError("the additional noise must be a finite, non-negative number")
    sem = np.zeros(y.size) if sem is None else np.asarray(sem, dtype=float).ravel()
    sem = np.where(np.isfinite(sem), sem, 0.0)
    if np.any(sem < 0):
        raise ValueError("a standard error cannot be negative")
    noise_sd = np.sqrt(sem**2 + additional_noise**2)
    d = x.shape[1]
    kernel = ConstantKernel(1.0, AMPLITUDE_BOUNDS) * Matern(
        length_scale=np.full(d, _INITIAL_LENGTH_SCALE), length_scale_bounds=LENGTH_SCALE_BOUNDS, nu=2.5
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=(noise_sd / _scale(y)) ** 2 + _JITTER,
        normalize_y=True,
        n_restarts_optimizer=RESTARTS,
        random_state=seed,
    )
    with warnings.catch_warnings():
        # A bounded hyperparameter ending on its bound is reported by `at_bounds`.
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x, y)
    return FittedGP(model, x, y, noise_sd, float(additional_noise), int(seed))


def expected_improvement(mean: np.ndarray, sd: np.ndarray, best: float, xi: float = 0.0) -> np.ndarray:
    """E[max(0, Y − best − ξ)] for Y ~ N(mean, sd²): improvement when maximizing.

    Never NaN and never negative; a point with no finite posterior has none.
    """
    mean = np.asarray(mean, dtype=float)
    sd = np.maximum(np.where(np.isfinite(sd), sd, 0.0), _SD_FLOOR)
    gain = mean - best - xi
    z = gain / sd
    ei = gain * norm.cdf(z) + sd * norm.pdf(z)
    return np.where(np.isfinite(ei), np.maximum(ei, 0.0), 0.0)


def expected_target_improvement(mean: np.ndarray, sd: np.ndarray, target: float, best_distance: float) -> np.ndarray:
    """E[max(0, d* − |Y − t|)] for Y ~ N(mean, sd²): how much closer to the target.

    `best_distance` d* is the smallest distance to the target achieved so far.
    Closed form: with U = Y − t ~ N(m, s²), the expectation is
    d·P(−d < U < d) + ∫₋d⁰ u φ du − ∫₀ᵈ u φ du.
    """
    d = float(best_distance)
    m = np.asarray(mean, dtype=float) - float(target)
    if not d > 0:
        return np.zeros_like(m)
    s = np.maximum(np.where(np.isfinite(sd), sd, 0.0), _SD_FLOOR)

    def mass(a: float, b: float) -> np.ndarray:
        return norm.cdf((b - m) / s) - norm.cdf((a - m) / s)

    def first_moment(a: float, b: float) -> np.ndarray:
        alpha, beta = (a - m) / s, (b - m) / s
        return m * (norm.cdf(beta) - norm.cdf(alpha)) - s * (norm.pdf(beta) - norm.pdf(alpha))

    value = d * mass(-d, d) + first_moment(-d, 0.0) - first_moment(0.0, d)
    return np.where(np.isfinite(value), np.maximum(value, 0.0), 0.0)


@dataclass(frozen=True)
class LooResult:
    """Leave-one-out predictions: each specimen predicted by a model that never saw it."""

    observed: np.ndarray
    predicted: np.ndarray
    latent_sd: np.ndarray
    predictive_sd: np.ndarray  # latent sd combined with the point's own assumed noise

    @property
    def errors(self) -> np.ndarray:
        return self.observed - self.predicted

    @property
    def z(self) -> np.ndarray:
        return self.errors / np.maximum(self.predictive_sd, _SD_FLOOR)

    @property
    def inside(self) -> np.ndarray:
        """Whether each specimen fell inside its nominal 95 % predictive interval."""
        return np.abs(self.z) <= Z95

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.errors**2)))


def leave_one_out(
    x: np.ndarray, y: np.ndarray, sem: np.ndarray | None, additional_noise: float, seed: int
) -> LooResult | None:
    """Refit without each point in turn, hyperparameters included, and predict it.

    Refitting the hyperparameters in every fold keeps the held-out point from
    informing its own prediction. None with fewer than three points.
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).reshape(y.size, -1)
    n = y.size
    if n < 3:
        return None
    sem = np.zeros(n) if sem is None else np.where(np.isfinite(sem), np.asarray(sem, dtype=float), 0.0)
    predicted, latent, predictive = np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        fold = fit_gp(x[keep], y[keep], sem[keep], additional_noise, seed)
        mean, sd = fold.predict(x[i : i + 1])
        predicted[i], latent[i] = mean[0], sd[0]
        predictive[i] = np.sqrt(sd[0] ** 2 + sem[i] ** 2 + additional_noise**2)
    return LooResult(y.copy(), predicted, latent, predictive)
