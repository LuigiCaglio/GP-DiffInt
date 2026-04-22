import numpy as np
from scipy.optimize import minimize
import KalmanFilter_functions as KF
from Matern_52_state_space import Matern52


# Algorithm 1
def nll_matern52(params, measurements, R, dt,
                 observed_derivative = 0,
                 verbose=False, iteration_counter=None,
                 return_full_state_estimation = False):
    """Evaluate the negative log-likelihood for a Matérn 5/2 (p=2) GP-SS model.

    Parameters
    ----------
    params : array-like of shape (2,)
        [length_scale, sigma_f]
    measurements : ndarray, shape (N,)
        Measurements.
    R : ndarray, shape (1, 1)
        Measurement noise covariance.
    dt : float
        Sampling interval.
    observed_derivative : int
        Which derivative of the process is observed.
        0 --> position (process itself)
        1 --> first derivative (velocity)
        2 --> second derivative (acceleration)
    verbose : bool
        Print iteration info.
    iteration_counter : list of one int, optional
        Mutable counter so the caller can track iterations across calls, e.g. [0].
    return_full_state_estimation : bool
        Return full RTSS state estimation.
    """
    
    
    
    l_f, sig_f = params[0], params[1]
    A, Qd, Pinf, D0, D1, D2 = Matern52(l_f,sig_f,dt)  
    
    
    #set output matrix depending
    if observed_derivative == 0: # if differentiating
        H = D0
    elif observed_derivative == 1: # if computing first integral
        H = D1
    elif observed_derivative == 2: # if computing second integral
        H = D2
    elif observed_derivative > 2:
        raise ValueError("Matérn 5/2 state has derivatives 0, 1, 2 only.")


    #Run Kalman filter
    zf, Pf, nll = KF.kf_full_estimation(
        A, H, Qd, R,
        measurements.reshape(1, -1),
        return_nll=True,
        z0=np.zeros(len(A)),
        P0=Pinf
    )



                            
                            
    if verbose:
        if iteration_counter is not None:
            iteration_counter[0] += 1
            it = iteration_counter[0]
        else:
            it = "?"
        print(f"Iteration {it}  negLogLike: {np.round(nll)}  params: {params}")


    if not return_full_state_estimation:
        return nll
    else:
        #Run RTS smoother for constraint (Example in Section 4 of the paper)
        zs, Ps = KF.rtss_full_estimation(A,zf,Pf,Qd)
        return nll, zs 
     

def optimize_hyperparams_Matern52_and_measurement_noise(
    measurements,  dt,
    initial_params,
    bounds_log,
    observed_derivative,
    R=None,                  # if None, R is optimized
    maxiter=100,
    ftol=1e-4,
    verbose=True,
):
    """Optimize Matérn GP hyperparameters via NLL minimization in log space.

    Parameters
    ----------
    measurements : ndarray, shape (N,)
    dt : float
    initial_params : array-like of shape (2,), optional
        [l_f, sig_f]. Defaults to [1.0, 1.0].
    p : int
        Matérn smoothness.
    bounds_log : list of (lo, hi) tuples in log space, optional
        Bounds for [log(l_f), log(sig_f)]. Defaults to [(-4.6, 4.6), (-4.6, 4.6)].
    observed_derivative : int
        Which derivative of the process is observed.
        0 --> position (process itself)
        1 --> first derivative (velocity)
        2 --> second derivative (acceleration)
    R : ndarray of shape (1, 1), optional
        Measurement noise covariance. If provided, R is held fixed during
        optimization. If None, R is treated as a free parameter and optimized
        jointly with the GP hyperparameters (requires a third entry in
        initial_params and bounds_log).
    maxiter : int
    ftol : float
    verbose : bool

    Returns
    -------
    params_opt : ndarray, shape (2,) or shape (3,)
        Optimized [l_f, sig_f] or [l_f, sig_f, sigma2_n] in original space.
    result : OptimizeResult
    """


    counter = [0]

    def compute_negative_log_likelihood(params_log): # returns negative log-likelihood
    
        if R is None: #if measurement noise variance is optimized
            sigma2_n = np.exp(params_log[2])
            R_opt = sigma2_n * np.eye(1)
        else:       #if it is assumed to be known
            R_opt = R
            
        return nll_matern52(
            np.exp(params_log), measurements, R_opt, dt,
            observed_derivative = observed_derivative,
             verbose=verbose, iteration_counter=counter
        )

    result = minimize(
        compute_negative_log_likelihood, # objective function
        x0=np.log(initial_params),
        method='L-BFGS-B',
        bounds=bounds_log,
        options={'disp': True, 'maxiter': maxiter, 'ftol': ftol}
    )

    params_opt = np.exp(result.x)
    if verbose:
        print("Optimized parameters:", params_opt)
        print("Final neg-log-likelihood:", result.fun)

    return params_opt, result




def optimize_hyperparams_Matern52_constrained(
    measurements,
    dt,
    R,
    initial_params,
    bounds_log,
    observed_derivative=2, 
    maxiter=100,
    ftol=1e-6,
    verbose=True,
):
    """Optimize Matérn 5/2 GP hyperparameters via NLL minimization in log space
    subject to two equality constraints:

        1. Variance constraint  : σ_f² · λ⁴ = Var(measurements)
           (matches the marginal variance of the p-th derivative to the
           observed signal variance)

        2. Std constraint       : std(z_ks[0, :]) = σ_f
           (the RTS-smoothed position process has standard deviation σ_f)

    Parameters
    ----------
    measurements : ndarray, shape (N,)
    dt : float
    R : ndarray, shape (1, 1)
        Measurement noise covariance (held fixed).
    initial_params : array-like, shape (2,)
        Initial [l_f, σ_f] in original (positive) space.
    bounds_log : list of 2 (lo, hi) tuples
        Bounds for [log(l_f), log(σ_f)].
    observed_derivative : int
        Which derivative is observed (0 = position, 1 = velocity, 2 = acceleration).
    maxiter : int
    ftol : float
    verbose : bool

    Returns
    -------
    params_opt : ndarray, shape (2,)
        Optimized [l_f, σ_f] in original space.
    result : OptimizeResult
    """
    target_variance = np.var(measurements)
    counter = [0]

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def objective(theta):
        params = np.exp(theta)
        nll = nll_matern52(
            params, measurements, R, dt,
            observed_derivative=observed_derivative,
            verbose=verbose,
            iteration_counter=counter,
        )
        return nll

    # ------------------------------------------------------------------
    # Constraint 1 – variance of the observed derivative matches signal
    # ------------------------------------------------------------------
    def constraint_output_variance(theta):
        """σ_f² · λ⁴ = Var(measurements)  (equality → == 0)"""
        l_f, sig_f = np.exp(theta)
        p=2 # only valid for Matern 5/2
        ni = p + 0.5
        lambda_ = np.sqrt(2 * ni) / l_f
        return sig_f**2 * lambda_**4 - target_variance

    # ------------------------------------------------------------------
    # Constraint 2 – std of smoothed displacement  process equals σ_f
    # ------------------------------------------------------------------
    def constraint_position_std(theta):
        """std(z_ks[0, :]) - σ_f == 0"""
        params = np.exp(theta)
        sig_f = params[1]
        _, z_ks = nll_matern52(
            params, measurements, R, dt,
            observed_derivative=observed_derivative,
            verbose=False,
            return_full_state_estimation=True,
        )
        return np.std(z_ks[0]) - sig_f

    constraints = [
        {'type': 'eq', 'fun': constraint_output_variance},
        {'type': 'eq', 'fun': constraint_position_std},
    ]

    # ------------------------------------------------------------------
    # Optimize
    # ------------------------------------------------------------------
    result = minimize(
        objective,
        x0=np.log(initial_params),
        method='SLSQP',
        constraints=constraints,
        bounds=bounds_log,
        options={'disp': verbose, 'maxiter': maxiter, 'ftol': ftol},
    )

    params_opt = np.exp(result.x)
    if verbose:
        print("Optimized parameters:", params_opt)
        print("Final neg-log-likelihood:", result.fun)

    return params_opt, result




def extract_smoothed_displacement_vel_accel(zs, Ps, D0, D1, D2):
    """
    Extract displacement, velocity, and acceleration estimates and their
    marginal variances from the smoothed GP state sequence.
    """
    n_steps = zs.shape[1]
    displacement          = np.zeros(n_steps)
    displacement_variance = np.zeros(n_steps)
    velocity              = np.zeros(n_steps)
    velocity_variance     = np.zeros(n_steps)
    acceleration          = np.zeros(n_steps)
    acceleration_variance = np.zeros(n_steps)

    for k in range(n_steps):
        displacement[[k]]          = D0 @ zs[:, k]
        displacement_variance[[k]] = D0 @ Ps[:, :, k] @ D0.T

        velocity[[k]]              = D1 @ zs[:, k]
        velocity_variance[[k]]     = D1 @ Ps[:, :, k] @ D1.T

        acceleration[[k]]          = D2 @ zs[:, k]
        acceleration_variance[[k]] = D2 @ Ps[:, :, k] @ D2.T

    return (displacement, displacement_variance,
            velocity,     velocity_variance,
            acceleration, acceleration_variance)































