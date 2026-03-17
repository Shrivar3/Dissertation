from __future__ import annotations

import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any, List

from ns_mh_phantom import run_ns_mh_phantom  # your canonical implementation


def run_multi_ns_and_save(
    *,
    out_dir: Path,
    out_name: str,

    # run control
    n_runs: int,
    base_seed: int,
    regenerate_data_each_run: bool = False,

    # ALL tunables forwarded to run_ns_mh_phantom
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
    verbose: bool = False,
    verbose_interval: int = 500,

    mh_step_size: float = 0.103,
    mh_target_accept: float = 0.234,
    mh_adapt_rate: float = 0.05,
    mh_warmup_steps: int = 10,
    mh_step_size_min: float = 1e-6,
    mh_step_size_max: float = 10.0,
    mh_store_warmup: bool = False,

    # optional: choose whether to store traces (big)
    store_trace_logZ: bool = True,
) -> Path:
    """
    Runs NS n_runs times (varying ns_seed each run), saves a single .npz.
    Uses the canonical implementation in ns_mh_phantom.py (no notebook forks).
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / out_name

    rng = np.random.RandomState(int(base_seed))
    run_seeds = rng.randint(0, 2**31 - 1, size=int(n_runs), dtype=np.int64)

    fixed_data_seed = int(data_seed) if (not regenerate_data_each_run and data_seed is not None) else None

    logZs = np.empty(n_runs, dtype=float)
    Hs = np.empty(n_runs, dtype=float)
    mean_accs = np.empty(n_runs, dtype=float)
    n_dead = np.empty(n_runs, dtype=int)

    dead_logLs_list: List[np.ndarray] = []
    trace_logZ_list: List[np.ndarray] = []
    step_sizes_list: List[np.ndarray] = []

    t0 = time.perf_counter()
    for r in range(n_runs):
        rs = int(run_seeds[r])

        ns_out = run_ns_mh_phantom(
            # data
            n=n,
            p=p,
            use_correlated_X=use_correlated_X,
            rho=rho,
            sigma_beta=sigma_beta,
            sparsity=sparsity,
            include_intercept=include_intercept,
            data_seed=(rs if regenerate_data_each_run else fixed_data_seed),

            # prior
            tau_prior=tau_prior,

            # ns
            n_live=n_live,
            ns_mcmc_steps=ns_mcmc_steps,
            n_iter_max=n_iter_max,
            ns_seed=rs,  # <-- vary NS randomness across runs
            tol_logZ=tol_logZ,
            tol_tail=tol_tail,
            patience=patience,
            stable_repeats=stable_repeats,
            verbose=verbose,
            verbose_interval=verbose_interval,

            # mh tuning
            mh_step_size=mh_step_size,
            mh_target_accept=mh_target_accept,
            mh_adapt_rate=mh_adapt_rate,
            mh_warmup_steps=mh_warmup_steps,
            mh_step_size_min=mh_step_size_min,
            mh_step_size_max=mh_step_size_max,
            mh_store_warmup=mh_store_warmup,
        )

        logZs[r] = float(ns_out["logZ"])
        Hs[r] = float(ns_out.get("H", np.nan))
        mean_accs[r] = float(ns_out.get("mean_acc_rate_constrained", np.nan))

        dead = np.asarray(ns_out["dead_logLs"], dtype=float)
        dead_logLs_list.append(dead)
        n_dead[r] = int(dead.size)

        if store_trace_logZ and ("trace_logZ" in ns_out):
            trace_logZ_list.append(np.asarray(ns_out["trace_logZ"], dtype=float))
        else:
            trace_logZ_list.append(np.array([], dtype=float))

        # step sizes: support either key name
        if "step_sizes_used" in ns_out:
            step_sizes_list.append(np.asarray(ns_out["step_sizes_used"], dtype=float))
        elif "mh_step_sizes_used" in ns_out:
            step_sizes_list.append(np.asarray(ns_out["mh_step_sizes_used"], dtype=float))
        else:
            step_sizes_list.append(np.array([], dtype=float))

        if (r + 1) % max(1, n_runs // 10) == 0:
            print(f"Completed {r+1}/{n_runs} runs...")

    elapsed = time.perf_counter() - t0
    print(f"\nDone. Total time: {elapsed:.2f}s")
    print(f"logZ mean ± sd: {logZs.mean():.3f} ± {logZs.std(ddof=1):.3f}")

    config = dict(
        n=n, p=p, use_correlated_X=use_correlated_X, rho=rho, sigma_beta=sigma_beta,
        sparsity=sparsity, include_intercept=include_intercept,
        tau_prior=tau_prior,
        n_live=n_live, ns_mcmc_steps=ns_mcmc_steps, n_iter_max=n_iter_max,
        tol_logZ=tol_logZ, tol_tail=tol_tail, patience=patience, stable_repeats=stable_repeats,
        mh_step_size=mh_step_size, mh_target_accept=mh_target_accept, mh_adapt_rate=mh_adapt_rate,
        mh_warmup_steps=mh_warmup_steps, mh_step_size_min=mh_step_size_min, mh_step_size_max=mh_step_size_max,
        mh_store_warmup=mh_store_warmup,
        regenerate_data_each_run=regenerate_data_each_run,
        store_trace_logZ=store_trace_logZ,
    )

    np.savez(
        save_path,
        # meta
        n_runs=int(n_runs),
        base_seed=int(base_seed),
        run_seeds=np.asarray(run_seeds, dtype=np.int64),
        config=np.array([config], dtype=object),

        # per-run scalars
        logZs=logZs,
        Hs=Hs,
        mean_accs=mean_accs,
        n_dead=n_dead,

        # ragged arrays
        dead_logLs=np.array(dead_logLs_list, dtype=object),
        trace_logZ=np.array(trace_logZ_list, dtype=object),
        step_sizes=np.array(step_sizes_list, dtype=object),
    )

    print(f"Saved: {save_path.resolve()}")
    return save_path
