import numpy as np
from scipy.stats import invgamma, beta, norm

def ar1_bayesian_sampler(y,
                         n_iter=10000,
                         phi_init=0.0,
                         sigma2_init=1.0,
                         nu0=2.0,
                         S0=1.0,
                         a=20.0,
                         b=1.5,
                         proposal_sd=0.05,
                         burn_in=1000):
    """
    Gibbs–Metropolis sampler for AR(1) model:
        y[t] = phi * y[t-1] + eps[t],    eps[t] ~ N(0, sigma2)

    Priors:
      sigma2 ~ Inverse-Gamma(nu0/2, S0/2)
      u = (phi + 1)/2 ~ Beta(a, b), so phi ∈ (−1, 1)

    Parameters
    ----------
    y : array_like, shape (T+1,)
        Observations with y[0] as initial value.
    n_iter : int
        Total number of MCMC iterations (including burn-in).
    phi_init : float
        Starting value for phi.
    sigma2_init : float
        Starting value for sigma².
    nu0, S0 : floats
        Hyperparameters for IG prior on sigma².
    a, b : floats
        Hyperparameters for Beta prior on u=(phi+1)/2.
    proposal_sd : float
        Std. dev. of Gaussian proposal for phi in MH step.
    burn_in : int
        Number of initial samples to discard.

    Returns
    -------
    dict with keys 'phi' and 'sigma2', each an array of posterior draws
    after burn-in.
    """
    # Number of usable observations (we lose y[0])
    T = len(y) - 1
    y_prev = y[:-1]   # y[0] through y[T-1]
    y_curr = y[1:]    # y[1] through y[T]

    # Allocate storage for samples
    phi_samples = np.empty(n_iter)
    sigma2_samples = np.empty(n_iter)

    # Initialize
    phi = phi_init
    sigma2 = sigma2_init

    for i in range(n_iter):
        # -------------------------------
        # 1) Sample sigma² | phi, y  (Gibbs step)
        #    Posterior: IG( (nu0 + T)/2, (S0 + SSR)/2 )
        # -------------------------------
        residuals = y_curr - phi * y_prev
        SSR = np.sum(residuals**2)
        shape = 0.5 * (nu0 + T)               # alpha parameter
        scale = 0.5 * (S0 + SSR)              # beta parameter
        sigma2 = invgamma.rvs(a=shape, scale=scale)

        # -------------------------------
        # 2) Sample phi | sigma², y  (Metropolis–Hastings step)
        #    Prior on u = (phi+1)/2 ~ Beta(a,b)
        # -------------------------------
        # Propose new phi
        phi_star = phi + norm.rvs(scale=proposal_sd)

        # Log-likelihood for current and proposed
        SSR_star = np.sum((y_curr - phi_star * y_prev)**2)
        loglike_star = -0.5 * (T * np.log(2 * np.pi * sigma2) + SSR_star / sigma2)
        loglike_cur  = -0.5 * (T * np.log(2 * np.pi * sigma2) + SSR   / sigma2)

        # Log-prior via Beta on u = (phi+1)/2
        u_star = 0.5 * (phi_star + 1)
        u_cur  = 0.5 * (phi     + 1)
        # beta.logpdf returns -inf if u_star ∉ [0,1], so proposals outside (−1,1) get auto-rejected
        logprior_star = beta.logpdf(u_star, a, b)
        logprior_cur  = beta.logpdf(u_cur,  a, b)

        # MH acceptance ratio (in log scale)
        log_r = (loglike_star + logprior_star) - (loglike_cur + logprior_cur)
        if np.log(np.random.rand()) < log_r:
            phi = phi_star

        # Store draws
        phi_samples[i]    = phi
        sigma2_samples[i] = sigma2

    # Discard burn-in and return
    return {
        'phi': phi_samples[burn_in:],
        'sigma2': sigma2_samples[burn_in:]
    }


if __name__ == "__main__":
    # ─── Example usage ─────────────────────────────────────────────
    np.random.seed(42)

    # Simulate AR(1) data
    T = 200
    true_phi = 0.7
    true_sigma2 = 0.5
    y = np.zeros(T + 1)
    for t in range(1, T + 1):
        y[t] = true_phi * y[t - 1] + np.sqrt(true_sigma2) * np.random.randn()

    # Run the sampler
    posterior = ar1_bayesian_sampler(
        y,
        n_iter=12000,
        phi_init=0.0,
        sigma2_init=1.0,
        nu0=2.0, S0=1.0,
        a=20.0, b=1.5,
        proposal_sd=0.02,
        burn_in=2000
    )

    # Summaries
    print(f"Posterior mean phi:    {np.mean(posterior['phi']):.3f}")
    print(f"Posterior mean sigma²: {np.mean(posterior['sigma2']):.3f}")
