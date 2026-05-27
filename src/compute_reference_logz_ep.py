from __future__ import annotations


import numpy as np

from numpy.polynomial.hermite import hermgauss

from scipy.linalg import solve_triangular

from scipy.special import expit, logsumexp

from scipy.stats import multivariate_normal, norm


# Import from your NS file

# Change this import to match your actual filename

from ns_mh_phantom import simulate_logistic_data



# ============================================================

# EP / IS reference logZ for YOUR setup:

#   - logistic regression

#   - no intercept

#   - prior beta ~ N(0, n (X'X + jitter I)^(-1))

# ============================================================


def compute_prior_factor(X: np.ndarray, jitter: float = 1e-10) -> np.ndarray:

    """

    Computes M such that M M^T = n (X^T X + jitter I)^(-1).

    This matches the prior used in your NS code.

    """

    n, p = X.shape

    XtX = X.T @ X

    A = XtX + jitter * np.eye(p)


    L_x = np.linalg.cholesky(A)

    M = solve_triangular(L_x.T, np.eye(p) * np.sqrt(n), lower=False)

    return M



def gaussian_ep(

    X: np.ndarray,

    compute_tilted_moments,

    max_iter: int = 100,

    tol: float = 1e-5,

    damping: float = 0.8,

    seed: int | None = None,

):

    """

    Sequential Gaussian EP assuming prior z ~ N(0, I).

    """

    if seed is not None:

        np.random.seed(seed)


    n, d = X.shape


    tau_tilde = np.zeros(n)

    nu_tilde = np.zeros(n)


    Sigma = np.eye(d)

    mu = np.zeros(d)


    for _ in range(max_iter):

        max_diff = 0.0

        indices = np.random.permutation(n)


        for i in indices:

            xi = X[i]


            vi = xi @ Sigma @ xi

            mi = xi @ mu


            tau_cav = 1.0 / vi - tau_tilde[i]

            if tau_cav <= 1e-12:

                continue


            v_cav = 1.0 / tau_cav

            m_cav = v_cav * (mi / vi - nu_tilde[i])


            m_tilted, v_tilted = compute_tilted_moments(m_cav, v_cav, i)


            tau_tilde_new = 1.0 / v_tilted - tau_cav

            nu_tilde_new = m_tilted / v_tilted - m_cav * tau_cav


            tau_tilde_update = (1.0 - damping) * tau_tilde[i] + damping * tau_tilde_new

            nu_tilde_update = (1.0 - damping) * nu_tilde[i] + damping * nu_tilde_new


            d_tau = tau_tilde_update - tau_tilde[i]

            d_nu = nu_tilde_update - nu_tilde[i]


            max_diff = max(max_diff, abs(d_tau), abs(d_nu))


            tau_tilde[i] = tau_tilde_update

            nu_tilde[i] = nu_tilde_update


            Sigma_xi = Sigma @ xi

            c = d_tau / (1.0 + d_tau * vi)


            Sigma -= c * np.outer(Sigma_xi, Sigma_xi)

            Sigma = 0.5 * (Sigma + Sigma.T)


            mu += ((d_nu - d_tau * mi) / (1.0 + d_tau * vi)) * Sigma_xi


        if max_diff < tol:

            break


    return mu, Sigma



def get_tilted_moments_function(y: np.ndarray, degree: int = 30):

    """

    Returns tilted moment function for Bernoulli-logit likelihood.

    """

    x_gh, w_gh = hermgauss(degree)

    w_gh = w_gh / np.sqrt(np.pi)

    s = 2 * y - 1


    def compute_tilted_moments(m_cav, v_cav, i):

        eta_nodes = m_cav + np.sqrt(2.0 * v_cav) * x_gh

        lik_nodes = expit(s[i] * eta_nodes)


        Z_i = np.sum(w_gh * lik_nodes)

        m_unnorm = np.sum(w_gh * lik_nodes * eta_nodes)

        m2_unnorm = np.sum(w_gh * lik_nodes * eta_nodes**2)


        m_tilted = m_unnorm / Z_i

        v_tilted = m2_unnorm / Z_i - m_tilted**2


        return m_tilted, v_tilted


    return compute_tilted_moments



def apply_ep_to_model_no_intercept(

    X: np.ndarray,

    y: np.ndarray,

    max_iter: int = 100,

    tol: float = 1e-5,

    damping: float = 0.8,

    ep_seed: int | None = None,

):

    """

    Applies EP to your no-intercept logistic regression model.

    Reparameterisation: beta = M z, z ~ N(0, I).

    """

    M = compute_prior_factor(X)

    X_tilde = X @ M

    moment_func = get_tilted_moments_function(y)


    mu_ep, Sigma_ep = gaussian_ep(

        X_tilde,

        moment_func,

        max_iter=max_iter,

        tol=tol,

        damping=damping,

        seed=ep_seed,

    )


    return mu_ep, Sigma_ep, X_tilde



def compute_log_weights(

    Z_samples: np.ndarray,

    X_tilde: np.ndarray,

    y: np.ndarray,

    mu_ep: np.ndarray,

    Sigma_ep: np.ndarray,

) -> np.ndarray:

    """

    Computes log importance weights:

        log p(y | z) + log p(z) - log q(z)

    with z ~ N(0, I) prior.

    """

    log_prior = np.sum(norm.logpdf(Z_samples), axis=1)


    eta = Z_samples @ X_tilde.T

    log_lik = np.sum(y * eta - np.logaddexp(0.0, eta), axis=1)


    log_prop = multivariate_normal.logpdf(Z_samples, mean=mu_ep, cov=Sigma_ep)


    return log_lik + log_prior - log_prop



def importance_sampling_evidence(

    X_tilde: np.ndarray,

    y: np.ndarray,

    mu_ep: np.ndarray,

    Sigma_ep: np.ndarray,

    S: int = 100_000,

    is_seed: int | None = None,

):

    """

    Importance sampling estimate of log evidence and ESS.

    """

    if is_seed is not None:

        np.random.seed(is_seed)


    Z_samples = np.random.multivariate_normal(mu_ep, Sigma_ep, size=S)

    log_w = compute_log_weights(Z_samples, X_tilde, y, mu_ep, Sigma_ep)


    log_Z_hat = logsumexp(log_w) - np.log(S)


    log_w_norm = log_w - logsumexp(log_w)

    ess = np.exp(-logsumexp(2.0 * log_w_norm))


    return log_Z_hat, ess



def compute_reference_logz_from_xy(

    X: np.ndarray,

    y: np.ndarray,

    S: int = 100_000,

    ep_max_iter: int = 100,

    ep_tol: float = 1e-5,

    ep_damping: float = 0.8,

    ep_seed: int | None = None,

    is_seed: int | None = None,

):

    mu_ep, Sigma_ep, X_tilde = apply_ep_to_model_no_intercept(

        X,

        y,

        max_iter=ep_max_iter,

        tol=ep_tol,

        damping=ep_damping,

        ep_seed=ep_seed,

    )


    logZ, ess = importance_sampling_evidence(

        X_tilde,

        y,

        mu_ep,

        Sigma_ep,

        S=S,

        is_seed=is_seed,

    )


    return {

        "logZ": float(logZ),

        "ess": float(ess),

        "ess_frac": float(ess / S),

        "mu_ep": mu_ep,

        "Sigma_ep": Sigma_ep,

    }



def compute_reference_logz_for_scenario(

    n: int = 600,

    p: int = 12,

    data_seed: int = 415,

    sigma_beta: float = 1.0,

    S: int = 100_000,

    ep_seed: int = 123,

    is_seed: int = 456,

):

    """

    Recreates YOUR exact dissertation data setup, then computes EP reference logZ.

    """

    sim = simulate_logistic_data(

        n=n,

        p=p,

        use_correlated_X=False,

        rho=1.0,

        sigma_beta=sigma_beta,

        sparsity=0.0,

        include_intercept=False,

        seed=data_seed,

    )


    out = compute_reference_logz_from_xy(

        sim.X,

        sim.y,

        S=S,

        ep_max_iter=100,

        ep_tol=1e-5,

        ep_damping=0.8,

        ep_seed=ep_seed,

        is_seed=is_seed,

    )


    out["X"] = sim.X

    out["y"] = sim.y

    out["sim"] = sim

    return out



if __name__ == "__main__":

    scenarios = [

        (12, 415),

        (36, 36),

        (72, 72),

    ]


    for p, data_seed in scenarios:

        out = compute_reference_logz_for_scenario(

            n=600,

            p=p,

            data_seed=data_seed,

            sigma_beta=1.0,

            S=100000,

            ep_seed=1000 + p,

            is_seed=2000 + p,

        )


        print(f"\nScenario: n=600, p={p}, data_seed={data_seed}")

        print(f"Reference logZ: {out['logZ']:.6f}")

        print(f"ESS fraction:   {out['ess_frac']:.2%}")
