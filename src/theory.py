# First part: Mean-field solution for T->inf
##################
import numpy as np

def mean_field_activity(lam, mu, h, dt=1):
    return mu*(1-np.exp(-h*dt))/(1-lam*(1-mu)-lam*mu*np.exp(-h*dt))

# second part: Mean-field solution for T->0
##################

def fp_solution(support, f, g):
    """
    Numerically compute the solution to the Fokker-Planck equation for the given f and g

    The integral is computed as sum over the support for two reasons: 
    1) to ensure a self-consistent formulation
    2) to speed up the evaluation

    Parameters
    ----------
    support : array-like
        Support of the pmf
    f : function
        drift term of the Fokker-Planck equation
    g : function
        diffusion term of the Fokker-Planck equation

    Returns
    -------
    pmf : array-like
        Probability mass function of the Fokker-Planck equation
    """
    # exponent = lambda x: 2 * scipy.integrate.quad(lambda x_: f(x_)/g(x_), 0, x)[0]
    # exponent = np.vectorize(exponent)
    # log_pmf = exponent(support) - np.log(g(support))
    
    # Replace continuous integral with a Riemann sum of bin size dx=1, where the value of the right boundary is chosen for each bin. This allows to evaluate all upper bounds of the integral in a cumulative sum. Since the integral from 0 to x=0 is zero, we start the cumulative sum at x=0 but subtract the first term f(0)/g(0).
    # This assumes that f and g do not change much with x, which is a reasonable assumption for f and g that depend on x/N for large N.
    fraction = f(support)/g(support)
    exponent = 2*(np.cumsum(fraction)-f(0)/g(0))     
    log_pmf = exponent - np.log(g(support))

    pmf = np.exp(log_pmf - np.max(log_pmf))
    # normalize the pmf
    pmf = pmf / np.sum(pmf)
    return pmf

def pmf_from_coupled_fokker_planck(params, h, lam, return_only_pmf=False):
    """ 
    Solution to the Mean-field coupled Fokker-Planck equations. 
    1.1) compute solution of FP equation of the part that receives input assuming a mean-field coupling to the recurrently coupled rest
    $$ p_{rec}(x_{in}) = \lambda \frac{x_in + x_{rest}} {N} = \lambda\frac{x_{in}/N}{\left(1-(1-\mu\lambda)\right)}$$
    from mean-field assumption
    $$ x_{rest} = \frac{(1-\mu)\lambda x_{in}}{1-(1-\mu)\lambda}$$ 

    1.2) compute solution of FP equation for the part that does not receive input assuming a mean-field coupling to the input part
    $$ p_{rec}(x_{rest}) = \lambda \frac{x_{in} + x_{rest}} {N} = \lambda\frac{x_{rest}/N + \mu p_\mathrm{ext}}{\left(1-\mu\lambda(1-p_\mathrm{ext})\right)}$$
    with 
    $$ x_{in} = \mu\frac{N p_\mathrm{ext} + \lambda (1-p_\mathrm{ext}) x_\mathrm{rest}}{1-\mu\lambda(1-p_\mathrm{ext})} $$
    2) convolution of the two solutions to obtain the full pmf
    """
    # probability of external activation
    dt=1
    p_ext = 1 - np.exp(-h * dt)

    # pmf of input part
    x_max = params["N"] * params["mu"]
    x_in = np.arange(0, x_max + 1)

    p_rec = lambda x: lam * x / params["N"] / (1 - (1 - params["mu"]) * lam)

    # NEW spikes generated from inactive neurons: (N^{input} - x^{input})
    # with probability of not being not activated: (1 - (1-p_rec)*(1-p_ext) )
    rate_birth = lambda x: (x_max - x) * (1 - (1 - p_rec(x)) * (1 - p_ext))
    # REMOVE spikes only if active neurons: x^{input}
    # are not activated with probability (1 - p_rec) * (1 - p_ext)
    rate_death = lambda x: x * (1 - p_rec(x)) * (1 - p_ext)

    f = lambda x: rate_birth(x) - rate_death(x)
    g = lambda x: rate_birth(x) + rate_death(x)

    # catch case of super strong drive that results in g==0 at x_max implying that the pmf is a delta distribution
    if g(x_max) == 0:
        pmf_in = np.zeros_like(x_in)
        pmf_in[-1] = 1
    else:
        pmf_in = fp_solution(x_in, f, g)

    # pmf of rest part
    x_max = params["N"] * (1 - params["mu"])
    x_rest = np.arange(0, x_max + 1, dtype=int)

    if lam == 0:
        # in this case the rest cannot receive ANY input and hence the pdf is a delta distribution
        pmf_rest = np.zeros_like(x_rest)
        pmf_rest[0] = 1
    else:
        p_rec = (
            lambda x: lam
            * (x / params["N"] + params["mu"] * p_ext)
            / (1 - params["mu"] * lam * (1 - p_ext))
        )

        # spikes generates with rate (N^{rest} - x^{rest]) * p_rec(x^{rest})
        rate_birth = lambda x: (x_max - x) * p_rec(x)
        # spikes not maintined with rate x^{rest} * (1 - p_rec(x^{rest}))
        rate_death = lambda x: x * (1 - p_rec(x))

        f = lambda x: rate_birth(x) - rate_death(x)
        g = lambda x: rate_birth(x) + rate_death(x)

        pmf_rest = fp_solution(x_rest, f, g)
    
    #return convolution of both
    pmf = np.convolve(pmf_in, pmf_rest, mode="full")
    if return_only_pmf:
        return pmf
    else:
        # This should always be [0:N]
        x = np.arange(0, len(pmf)) 
        return x, pmf
