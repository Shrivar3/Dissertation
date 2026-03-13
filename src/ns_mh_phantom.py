# ============================================================
# NESTED SAMPLING (MH constrained kernel) + PHANTOM STORAGE
#   - Logistic regression toy model
#   - Deterministic shrinkage NS (static n_live)
#   - Constrained RW-MH replacement kernel (in whitened space z)
#   - PER-ITERATION STEP-SIZE TUNING (warmup -> freeze -> production)
#   - Stores PHANTOM chain states (theta and logL) for each NS iteration
#   - Builds candidate pool of logL values from dead + phantom
#
# This file contains:
#   - Definitions only (no plots, no "run end-to-end" at import time)
# ============================================================

from __future__ import annotations

import numpy as np
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, List


# ============================================================
# Basic utilities
# ============================================================

def set_seed(s: Optional[int] = None) -> None:
    if s is not None:
        np.random.seed(int(s))


def ar1_cov(p: int, rho: float) -> np.ndarray:
    idx = np.arange(p)
    return rho ** np.abs(np.subtract.outer(idx, idx))


def standardize_columns(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = X.mean(axis=0)
    sds = X.std(axis=0, ddof=1)
    sds[sds == 0] = 1.0
    Z = (X - means) / sds
    return Z, means, sds


def sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def _logsumexp2(a: float, b: float) -> float:
    m = max(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def _logdiffexp(a: float, b: float) -> float:
    """
    log(exp(a) - exp(b)) with a > b.
    """
    if b >= a:
        return -np.inf
    return a + np.log(1.0 - np.exp(b - a))


def _logmeanexp(arr: np.ndarray) -> float:
    """
    Stable log(mean(exp(arr))).
    """
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return -np.inf
    m = float(np.max(arr))
    if not np.isfinite(m):
        return -np.inf
    return m + np.log(np.mean(np.exp(arr - m)))


# ============================================================
# Simulation
# ============================================================

@dataclass
class SimResult:
    X: np.ndarray
    y: np.ndarray
    beta0: float
    beta: np.ndarray
    means: np.ndarray
    sds: np.ndarray
    settings: dict


def simulate_logistic_data(
    n: int,
    p: int,
    use_correlated_X: bool = False,
    rho: float = 0.3,
    sigma_beta: float = 0.75,
    sparsity: float = 0.0,
    include_intercept: bool = True,
    seed: Optional[int] = None,
) -> SimResult:
    set_seed(seed)

    if use_correlated_X:
        cov = ar1_cov(p, rho)
        L = np.linalg.cholesky(cov + 1e-12 * np.eye(p))
        X_raw = np.random.randn(n, p) @ L.T
    else:
        X_raw = np.random.randn(n, p)

    X, means, sds = standardize_columns(X_raw)

    beta = sigma_beta * np.random.randn(p)
    if sparsity > 0:
        k_zero = int(np.floor(sparsity * p))
        if k_zero > 0:
            zero_idx = np.random.choice(p, size=k_zero, replace=False)
            beta[zero_idx] = 0.0

    beta0 = np.random.randn() if include_intercept else 0.0
    eta = beta0 + X @ beta
    pi = sigmoid(eta)
    y = np.random.binomial(1, pi, size=n)

    return SimResult(
        X=X,
        y=y,
        beta0=beta0,
        beta=beta,
        means=means,
        sds=sds,
        settings=dict(
            n=n,
            p=p,
            rho=rho,
            sigma_beta=sigma_beta,
            sparsity=sparsity,
            use_correlated_X=use_correlated_X,
            include_intercept=include_intercept,
            seed=seed,
        ),
    )


# ============================================================
# Preconditioner and transforms
# ============================================================

def build_preconditioner(X: np.ndarray, tau0: float, include_intercept: bool) -> np.ndarray:
    """
    Build a lower-triangular L such that theta = L z, with z ~ N(0, I) under the prior.
    For betas: covariance is Sigma_beta = n * (X'X + jitter I)^{-1}.
    For intercept (if included): independent N(0, tau0^2).
    """
    n_obs, p = X.shape
    XtX = X.T @ X
    I = np.eye(p)

    jitter = 1e-10
    Sigma_beta = n_obs * np.linalg.inv(XtX + jitter * I)
    try:
        L_beta = np.linalg.cholesky(Sigma_beta)
    except np.linalg.LinAlgError:
        Sigma_beta = n_obs * np.linalg.inv(XtX + 1e-6 * I)
        L_beta = np.linalg.cholesky(Sigma_beta)

    if include_intercept:
        L = np.zeros((p + 1, p + 1))
        L[0, 0] = tau0
        L[1:, 1:] = L_beta
        return L

    return L_beta


def theta_to_z(theta: np.ndarray, L: np.ndarray) -> np.ndarray:
    return np.linalg.solve(L, theta)


def z_to_theta(z: np.ndarray, L: np.ndarray) -> np.ndarray:
    return L @ z


# ============================================================
# Prior sampler (for NS initialisation)
# ============================================================

def _prior_sampler(d_beta: int, X: np.ndarray, tau0: float, include_intercept: bool) -> Callable[[], np.ndarray]:
    """
    Prior:
      beta ~ N(0, Sigma_beta)  where Sigma_beta = n * (X'X + jitter I)^{-1}
      beta0 ~ N(0, tau0^2) if include_intercept
    """
    n_obs = X.shape[0]
    XtX = X.T @ X
    jitter = 1e-10

    Sigma_beta = n_obs * np.linalg.inv(XtX + jitter * np.eye(d_beta))
    try:
        L_beta = np.linalg.cholesky(Sigma_beta)
    except np.linalg.LinAlgError:
        Sigma_beta = n_obs * np.linalg.inv(XtX + 1e-6 * np.eye(d_beta))
        L_beta = np.linalg.cholesky(Sigma_beta)

    def draw() -> np.ndarray:
        z = np.random.randn(d_beta)
        beta = L_beta @ z
        if include_intercept:
            beta0 = tau0 * np.random.randn()
            return np.concatenate(([beta0], beta))
        return beta.copy()

    return draw


# ============================================================
# Log-likelihood + log-prior for logistic regression
# ============================================================

def make_payload(sim: SimResult, include_intercept: bool, tau0_for_intercept: float) -> Dict[str, object]:
    X, y = sim.X, sim.y
    n_obs, p_dim = X.shape

    XtX = X.T @ X
    jitter = 1e-10
    Sigma_beta = n_obs * np.linalg.inv(XtX + jitter * np.eye(p_dim))
    sign, logdet = np.linalg.slogdet(Sigma_beta)
    if sign <= 0:
        Sigma_beta = n_obs * np.linalg.inv(XtX + 1e-6 * np.eye(p_dim))
        sign, logdet = np.linalg.slogdet(Sigma_beta)

    def _unpack(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        if include_intercept:
            return float(theta[0]), theta[1:]
        return 0.0, theta

    def _loglik(theta: np.ndarray) -> float:
        beta0, beta = _unpack(theta)
        eta = beta0 + X @ beta
        softplus = np.where(eta > 0, eta + np.log1p(np.exp(-eta)), np.log1p(np.exp(eta)))
        return float(np.sum(y * eta - softplus))

    def _logprior(theta: np.ndarray) -> float:
        beta0, beta = _unpack(theta)

        quad_beta = beta @ np.linalg.solve(Sigma_beta, beta)
        d = beta.size
        logp_beta = -0.5 * quad_beta - 0.5 * d * np.log(2 * np.pi) - 0.5 * logdet

        if include_intercept:
            var0 = float(tau0_for_intercept ** 2)
            logp_b0 = -0.5 * (beta0 ** 2) / var0 - 0.5 * np.log(2 * np.pi * var0)
            return float(logp_b0 + logp_beta)

        return float(logp_beta)

    return {
        "X": X,
        "y": y,
        "include_intercept": include_intercept,
        "loglik": _loglik,
        "logprior": _logprior,
    }


# ============================================================
# Constrained RW-MH kernel WITH phantom trace storage
# ============================================================

def mh_constrained_kernel_precond_with_trace(
    prior_logpdf: Callable[[np.ndarray], float],
    loglik: Callable[[np.ndarray], float],
    x0: np.ndarray,
    L: np.ndarray,
    Lstar: float,
    step_size: float,
    n_steps: int,
    warmup_steps: int,
    target_accept: float,
    adapt_rate: float,
    step_size_min: float,
    step_size_max: float,
    store_warmup: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float, int, np.ndarray, np.ndarray, float]:
    """
    Constrained RW-MH in whitened space z = L^{-1} theta.
    """
    set_seed(seed)

    d = int(x0.size)
    z = theta_to_z(x0, L)
    theta = z_to_theta(z, L)

    ll0 = float(loglik(theta))
    if ll0 <= Lstar:
        total_steps = (int(warmup_steps) + int(n_steps)) if store_warmup else int(n_steps)
        tr_theta = np.tile(theta[None, :], (total_steps, 1))
        tr_logL = np.full(total_steps, ll0, dtype=float)
        return theta.copy(), 0.0, 0, tr_theta, tr_logL, float(step_size)

    logp_x = float(prior_logpdf(theta))

    s = float(step_size)
    s = min(max(s, float(step_size_min)), float(step_size_max))

    warmup_steps = int(max(0, warmup_steps))
    n_steps = int(max(0, n_steps))

    accepts = 0
    tries = 0

    total_store = (warmup_steps + n_steps) if store_warmup else n_steps
    tr_theta = np.empty((total_store, d), dtype=float)
    tr_logL = np.empty(total_store, dtype=float)

    write_idx = 0

    for _ in range(warmup_steps):
        tries += 1

        z_prop = z + s * np.random.randn(d)
        theta_prop = z_to_theta(z_prop, L)

        if float(loglik(theta_prop)) > Lstar:
            logp_prop = float(prior_logpdf(theta_prop))
            if np.log(np.random.rand()) < (logp_prop - logp_x):
                z = z_prop
                theta = theta_prop
                logp_x = logp_prop
                accepts += 1

        acc_rate_running = max(1e-12, accepts / tries)
        s *= np.exp(float(adapt_rate) * (acc_rate_running - float(target_accept)))
        s = min(max(s, float(step_size_min)), float(step_size_max))

        if store_warmup:
            tr_theta[write_idx, :] = theta
            tr_logL[write_idx] = float(loglik(theta))
            write_idx += 1

    tuned_s = float(s)

    for _ in range(n_steps):
        tries += 1

        z_prop = z + tuned_s * np.random.randn(d)
        theta_prop = z_to_theta(z_prop, L)

        if float(loglik(theta_prop)) > Lstar:
            logp_prop = float(prior_logpdf(theta_prop))
            if np.log(np.random.rand()) < (logp_prop - logp_x):
                z = z_prop
                theta = theta_prop
                logp_x = logp_prop
                accepts += 1

        tr_theta[write_idx, :] = theta
        tr_logL[write_idx] = float(loglik(theta))
        write_idx += 1

    acc_rate = accepts / max(1, tries)
    return theta.copy(), float(acc_rate), int(tries), tr_theta, tr_logL, tuned_s


# ============================================================
# Nested Sampling core
# ============================================================

def nested_sampling_toy(
    *,
    payload: Dict[str, object],
    n_live: int,
    n_iter_max: int,
    tau: float,
    step_size: float,
    ns_mcmc_steps: int,
    seed: Optional[int],
    L_full: np.ndarray,
    tol_logZ: float,
    tol_tail: float,
    patience: int,
    verbose: bool,
    verbose_interval: int,
    stable_repeats: int,
    mh_target_accept: float,
    mh_adapt_rate: float,
    mh_warmup_steps: int,
    mh_step_size_min: float,
    mh_step_size_max: float,
    mh_store_warmup: bool,
) -> Dict[str, object]:
    """
    Nested Sampling with deterministic shrinkage and constrained RW-MH replacement kernel.
    Tail correction uses mean remaining live likelihood.
    """
    set_seed(seed)

    X = payload["X"]
    loglik_fn = payload["loglik"]
    prior_logpdf = payload["logprior"]
    include_intercept = payload["include_intercept"]

    d_beta = X.shape[1]

    prior_draw = _prior_sampler(
        d_beta=d_beta,
        X=X,
        tau0=tau,
        include_intercept=include_intercept,
    )

    thetas = np.array([prior_draw() for _ in range(int(n_live))])
    logLs = np.array([float(loglik_fn(th)) for th in thetas], dtype=float)

    init_birth_level = -1e30
    birth_logl_live = np.full(int(n_live), init_birth_level, dtype=float)

    logX_prev = 0.0
    logZ = -np.inf
    H = 0.0

    dead_thetas: List[np.ndarray] = []
    dead_logLs: List[float] = []
    dead_birth_logl: List[float] = []

    phantom_bins_theta: List[np.ndarray] = []
    phantom_bins_logL: List[np.ndarray] = []

    step_sizes_used: List[float] = []
    acc_rates: List[float] = []
    logZ_trace: List[float] = []

    logZ_window: List[float] = []
    stable_counter = 0

    n_live_int = int(n_live)

    for i in range(1, int(n_iter_max) + 1):
        j = int(np.argmin(logLs))
        th_worst = thetas[j].copy()
        logL_worst = float(logLs[j])
        birth_worst = float(birth_logl_live[j])

        # Deterministic shrinkage
        logX_new = -i / float(n_live_int)
        log_w = _logdiffexp(logX_prev, logX_new)

        logZ_old = float(logZ)
        logZ_new = _logsumexp2(logZ_old, log_w + logL_worst)

        # Information update
        if np.isfinite(logZ_new):
            w_new = np.exp(log_w + logL_worst - logZ_new)
            if np.isfinite(logZ_old):
                w_old = np.exp(logZ_old - logZ_new)
                delta = logZ_old - logZ_new
            else:
                w_old = 0.0
                delta = 0.0

            term_old = H + delta
            if not np.isfinite(term_old):
                term_old = 0.0
            term_new = logL_worst - logZ_new
            if not np.isfinite(term_new):
                term_new = 0.0
            H = w_old * term_old + w_new * term_new

        delta_logZ = abs(logZ_new - logZ) if np.isfinite(logZ) else np.inf
        logZ = float(logZ_new)
        logX_prev = float(logX_new)

        dead_thetas.append(th_worst)
        dead_logLs.append(logL_worst)
        dead_birth_logl.append(birth_worst)
        logZ_trace.append(float(logZ))

        # Replacement seed from survivors
        k = np.random.randint(0, n_live_int - 1)
        if k >= j:
            k += 1
        seed_point = thetas[k].copy()

        th_new, acc_rate, _, tr_theta, tr_logL, tuned_s = mh_constrained_kernel_precond_with_trace(
            prior_logpdf=lambda th: float(prior_logpdf(th)),
            loglik=lambda th: float(loglik_fn(th)),
            x0=seed_point,
            L=L_full,
            Lstar=logL_worst,
            step_size=float(step_size),
            n_steps=int(ns_mcmc_steps),
            warmup_steps=int(mh_warmup_steps),
            target_accept=float(mh_target_accept),
            adapt_rate=float(mh_adapt_rate),
            step_size_min=float(mh_step_size_min),
            step_size_max=float(mh_step_size_max),
            store_warmup=bool(mh_store_warmup),
            seed=(int(seed) + i) if seed is not None else None,
        )

        phantom_bins_theta.append(tr_theta)
        phantom_bins_logL.append(tr_logL)
        step_sizes_used.append(float(tuned_s))

        acc_rates.append(float(acc_rate))
        thetas[j] = th_new
        logLs[j] = float(loglik_fn(th_new))
        birth_logl_live[j] = logL_worst

        # Convergence checks
        logZ_window.append(float(logZ))
        if len(logZ_window) > int(patience):
            logZ_window.pop(0)

        if bool(verbose) and (i % int(verbose_interval) == 0):
            recent = acc_rates[-50:] if len(acc_rates) >= 50 else acc_rates
            recent_s = step_sizes_used[-50:] if len(step_sizes_used) >= 50 else step_sizes_used
            print(
                f"Iter {i:5d}: logZ={logZ:.3f}, ΔlogZ={delta_logZ:.2e}, "
                f"acc≈{np.mean(recent):.3f}, step≈{np.mean(recent_s):.4f}"
            )

        if len(logZ_window) == int(patience):
            window_std = float(np.std(logZ_window))

            log_mean_live_tail = _logmeanexp(logLs)
            log_tail = float(logX_prev + log_mean_live_tail)
            tail_gap = log_tail - float(logZ)

            if window_std < float(tol_logZ) and tail_gap < float(tol_tail):
                stable_counter += 1
            else:
                stable_counter = 0

            if stable_counter >= int(stable_repeats):
                if bool(verbose):
                    print(f"Stopping at iter {i} (ΔlogZ<{tol_logZ}, tail<{tol_tail})")
                break

    # Final tail contribution using mean remaining live likelihood
    log_mean_live = _logmeanexp(logLs)
    logZ = _logsumexp2(float(logZ), float(logX_prev + log_mean_live))

    dead_thetas_arr = np.asarray(dead_thetas, dtype=float)
    dead_logLs_arr = np.asarray(dead_logLs, dtype=float)
    dead_birth_logl_arr = np.asarray(dead_birth_logl, dtype=float)

    return {
        "logZ": float(logZ),
        "tail_logL": float(log_mean_live),
        "H": float(H),

        "dead_thetas": dead_thetas_arr,
        "dead_logLs": dead_logLs_arr,
        "dead_birth_logl": dead_birth_logl_arr,

        "live_thetas": np.asarray(thetas, dtype=float),
        "live_logLs": np.asarray(logLs, dtype=float),

        "trace_logZ": np.asarray(logZ_trace, dtype=float),
        "mean_acc_rate_constrained": float(np.mean(acc_rates) if acc_rates else np.nan),

        "phantom_bins_theta": phantom_bins_theta,
        "phantom_bins_logL": phantom_bins_logL,
        "step_sizes_used": np.asarray(step_sizes_used, dtype=float),

        "settings": dict(
            n_live=int(n_live),
            tau=float(tau),
            step_size=float(step_size),
            ns_mcmc_steps=int(ns_mcmc_steps),
            tol_logZ=float(tol_logZ),
            tol_tail=float(tol_tail),
            patience=int(patience),
            seed=seed,
        ),
    }


# ============================================================
# Candidate pool builder (dead + phantom)
# ============================================================

def build_global_candidate_pool_logL_unique(
    ns_out: Dict[str, object],
    mode: str = "tol",
    decimals: int = 14,
    tol: float = 1e-12,
) -> np.ndarray:
    dead_logL = np.asarray(ns_out["dead_logLs"], dtype=float).ravel()
    pool_parts = [dead_logL]

    pb = ns_out.get("phantom_bins_logL", None)
    if pb is not None:
        for arr in pb:
            pool_parts.append(np.asarray(arr, dtype=float).ravel())

    pool = np.concatenate(pool_parts)
    pool = pool[np.isfinite(pool)]
    if pool.size == 0:
        raise RuntimeError("Candidate pool is empty (no finite dead/phantom logLs).")

    mode = str(mode).lower()
    if mode == "exact":
        _, idx = np.unique(pool, return_index=True)
        cand = pool[idx]
        cand.sort()
        return cand

    if mode == "round":
        pool_r = np.round(pool, decimals=int(decimals))
        _, idx = np.unique(pool_r, return_index=True)
        cand = pool[idx]
        cand = cand[np.isfinite(cand)]
        cand.sort()
        return cand

    if mode == "tol":
        pool_sorted = np.sort(pool)
        keep = np.ones(pool_sorted.size, dtype=bool)
        keep[1:] = np.abs(np.diff(pool_sorted)) > float(tol)
        return pool_sorted[keep]

    raise ValueError("mode must be one of {'tol','round','exact'}")


def report_storage(ns_out: Dict[str, object], ns_mcmc_steps: int, unique_tol: float = 1e-12) -> None:
    N_dead = int(np.asarray(ns_out["dead_logLs"]).size)
    pb = ns_out.get("phantom_bins_logL", [])
    phantom_counts = [int(np.asarray(a).ravel().size) for a in pb]
    N_phantom = int(np.sum(phantom_counts)) if len(phantom_counts) > 0 else 0
    N_total = N_dead + N_phantom

    print("\n=== STORAGE REPORT ===")
    print(f"Dead points stored:          {N_dead}")
    print(f"Phantom points stored:       {N_phantom}  (expected ≈ {N_dead * int(ns_mcmc_steps)})")
    print(f"Total stored (dead+phantom): {N_total}")

    cand = build_global_candidate_pool_logL_unique(ns_out, mode="tol", tol=float(unique_tol))
    print(f"Unique logL pool size (tol={float(unique_tol):g}): {cand.size}")

    if len(phantom_counts) > 0:
        print(
            "Phantom bin length (min/median/max): "
            f"{int(np.min(phantom_counts))} / {int(np.median(phantom_counts))} / {int(np.max(phantom_counts))}"
        )


# ============================================================
# Clean end-to-end wrapper
# ============================================================

def run_ns_mh_phantom(
    *,
    n: int = 600,
    p: int = 12,
    use_correlated_X: bool = False,
    rho: float = 1.0,
    sigma_beta: float = 1.0,
    sparsity: float = 0.0,
    include_intercept: bool = False,
    data_seed: Optional[int] = 415,

    tau_prior: float = 1.0,

    n_live: int = 100,
    ns_mcmc_steps: int = 20,
    n_iter_max: int = 1_000_000,
    ns_seed: Optional[int] = 415,
    tol_logZ: float = 1e-3,
    tol_tail: float = 1e-2,
    patience: int = 50,
    stable_repeats: int = 3,
    verbose: bool = True,
    verbose_interval: int = 500,

    mh_step_size: float = 0.103,
    mh_target_accept: float = 0.234,
    mh_adapt_rate: float = 0.05,
    mh_warmup_steps: int = 20,
    mh_step_size_min: float = 1e-6,
    mh_step_size_max: float = 10.0,
    mh_store_warmup: bool = False,

    attach_sim: bool = True,
) -> Dict[str, object]:
    """
    End-to-end wrapper: simulate data -> payload -> preconditioner -> nested sampling.
    """
    sim = simulate_logistic_data(
        n=int(n),
        p=int(p),
        use_correlated_X=bool(use_correlated_X),
        rho=float(rho),
        sigma_beta=float(sigma_beta),
        sparsity=float(sparsity),
        include_intercept=bool(include_intercept),
        seed=data_seed,
    )

    payload = make_payload(
        sim,
        include_intercept=bool(include_intercept),
        tau0_for_intercept=float(tau_prior),
    )
    L_full = build_preconditioner(sim.X, float(tau_prior), bool(include_intercept))

    ns_out = nested_sampling_toy(
        payload=payload,
        n_live=int(n_live),
        n_iter_max=int(n_iter_max),
        tau=float(tau_prior),
        step_size=float(mh_step_size),
        ns_mcmc_steps=int(ns_mcmc_steps),
        seed=ns_seed,
        L_full=L_full,
        tol_logZ=float(tol_logZ),
        tol_tail=float(tol_tail),
        patience=int(patience),
        verbose=bool(verbose),
        verbose_interval=int(verbose_interval),
        stable_repeats=int(stable_repeats),

        mh_target_accept=float(mh_target_accept),
        mh_adapt_rate=float(mh_adapt_rate),
        mh_warmup_steps=int(mh_warmup_steps),
        mh_step_size_min=float(mh_step_size_min),
        mh_step_size_max=float(mh_step_size_max),
        mh_store_warmup=bool(mh_store_warmup),
    )

    if bool(attach_sim):
        ns_out["sim"] = sim
        ns_out["payload"] = payload

    return ns_out


# ============================================================
# Optional timing helper
# ============================================================

def timed_run_ns_mh_phantom(**kwargs) -> Tuple[Dict[str, object], float]:
    t0 = time.perf_counter()
    ns_out = run_ns_mh_phantom(**kwargs)
    t1 = time.perf_counter()
    return ns_out, float(t1 - t0)
