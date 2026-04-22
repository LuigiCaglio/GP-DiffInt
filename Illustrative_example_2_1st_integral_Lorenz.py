"""
Example 2: GP regression on Lorenz attractor
using Matérn 5/2 state-space model with joint hyperparameter optimization.

Reproduces Figure 4 from: Differentiation and Integration of Time Series via Gaussian Process
                          Regression for Structural Health Monitoring Applications
"""
import numpy as np
import matplotlib.pyplot as plt


from data_generation import generate_data_Lorenz_attractor,add_measurement_noise
from gp_optimization import  optimize_hyperparams_Matern52_and_measurement_noise

import KalmanFilter_functions as KF
from Matern_52_state_space import Matern52

from plotting_functions import plot_states_with_zoom

  
 


t, states, derivatives = generate_data_Lorenz_attractor()


#subtract the mean, which cannot be retrieved from integration    
states = states - np.mean(states,axis=0,keepdims=True)


#Generate data from Lorenz system
dt = t[1]-t[0]
n_steps = len(t)

noise_level = 0.05 # e.g., noise_level = 0.05 = 5% noise

meas_clean = derivatives[:, 0]
meas_noisy, noise_std, R = add_measurement_noise(meas_clean, noise_fraction=noise_level, seed=456)



fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
labels = ["State 1 (displacement)", "Derivatve state 1 (velocity)"]
data = [states[:, 0], derivatives[:, 0], derivatives[:, 1]]
colors = ["steelblue", "darkorange", "seagreen"]
for ax, d, label, color in zip(axs, data, labels, colors):
    ax.plot(t, d, color=color, lw=0.8, label=label)
    ax.legend()
    ax.set_ylabel(label, )
    ax.grid(True, alpha=0.3)
# ax.plot(t, meas_noisy, "o", lw=0.8, label="noisy meas")
axs[-1].set_xlabel("Time (s)")
fig.suptitle("Lorenz attractor - first state")
plt.tight_layout()
plt.show()

 

#%% optimize negative log-likelihood ()


# optimization bounds
bounds_log = [
    (np.log(0.01), np.log(100)),              #length scale l_f
    (np.log(0.01), np.log(100)),              # GP output variance sig_f
    (np.log(R[0,0]/100), np.log(R[0,0]*100))  # measurement noise variance
]
#initial guess parameters -
initial_params = np.array([1.,1., R[0,0]*10])



observed_derivative = 1 # observe first derivative to integrate

# =============================================================================
# #perform optimization neg-log-likelihood 
# =============================================================================
params_opt, result = optimize_hyperparams_Matern52_and_measurement_noise(meas_noisy, dt,
                                                     initial_params, bounds_log,
                                                     observed_derivative,)
 
#%% use optimal parameters to perform the estimation (i.e., GP regression)


l_f =   params_opt[0]
sig_f = params_opt[1]    ##variance of Matern process
sigma2_n = params_opt[2] ##measurement noise variance
R_opt = sigma2_n * np.eye(1) 



# Matern 5/2 state space matrices with optimal hyperparameters
A, Qd, Pinf, D0, D1, D2 = Matern52(l_f,sig_f,dt)  
H = [D0, D1, D2][observed_derivative] # select correct output matrix 

# =============================================================================
# #perform GP regression (Algorithm 2)
# =============================================================================

# =============================================================================
# #Run Kalman filter
# =============================================================================
zf, Pf = KF.kf_full_estimation(A,H,Qd,R_opt,meas_noisy.reshape(1,-1),z0 = np.zeros(3),P0=Pinf,
                        return_nll=False)
# =============================================================================
# #Run Rauch-Tung-Striebel smoother
# =============================================================================
zs, Ps = KF.rtss_full_estimation(A,zf,Pf,Qd)



# =============================================================================
# # compute final estimates from RTS smoother output
# =============================================================================
displacement = np.zeros(n_steps)
displacement_variance = np.zeros(n_steps)

#first derivative
velocity = np.zeros(n_steps)
velocity_variance = np.zeros(n_steps)



for k in range(n_steps):
    displacement[[k]] = D0@zs[:,k]
    displacement_variance[[k]] = D0@Ps[:,:,k]@D0.T
    
    velocity[[k]] = D1@zs[:,k]
    velocity_variance[[k]] = D1@Ps[:,:,k]@D1.T
    
 
 

fig, axs, axs_zoom = plot_states_with_zoom(
    t,
    states,
    derivatives,      
    displacement,
    velocity,
    np.zeros(n_steps),            
    displacement_variance, 
    velocity_variance,     
    np.zeros(n_steps), 
    t_full_min=0.0,
    t_full_max=t[-1],
    t_zoom_min=50,
    t_zoom_max=70,
    fontsize_axes=16,
    fontsize_legends=12,
    save_path=None,
    figsize=(12, 10),
    n_panels = 2
)
fig.set_constrained_layout(True) # to add the title
fig.suptitle(f"Lorenz system - noise {noise_level*100}%", fontsize=20)
plt.show() 


#%% comparison with (i) frequency-domain integration (HP+FFT),
#   (ii) Tikhonov-regularized integration,
#   (iii) Savitzky–Golay smoothing + trapezoidal integration

compare_with_other_methods = False
if compare_with_other_methods: 

     
    from scipy.signal import butter, sosfiltfilt, savgol_filter
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from numpy.fft import fft, ifft, fftfreq
    
    # ---------------------------
    # 1) Frequency-domain integration with high-pass
    # ---------------------------
    def integrate_freq_domain(vel, dt, f_hp=0.05, order=2):
        """
        Integrate velocity to displacement via FFT:
        1) zero-phase high-pass to remove DC/bias,
        2) divide by j*2πf in frequency domain,
        3) inverse FFT to time domain.
        """
        N = len(vel)
        fs = 1.0 / dt
        nyq = fs / 2.0
        # High-pass (remove DC / bias to avoid drift explosion at f≈0)
        sos = butter(order, f_hp / nyq, btype='high', output='sos')
        vel_hp = sosfiltfilt(sos, vel)
    
        freqs = fftfreq(N, d=dt)
        V = fft(vel_hp)
        X = np.zeros_like(V, dtype=complex)
        denom = 1j * 2 * np.pi * freqs
        # avoid division at f=0
        nz = np.abs(freqs) > 0
        X[nz] = V[nz] / denom[nz]
        x = ifft(X).real
    
        # Optional: anchor initial displacement to zero
        x -= x[0]
        return x
    
    # ---------------------------
    # 2) Tikhonov-regularized integration (solve Dx ≈ v)
    # ---------------------------
    def tikhonov_integration(vel, dt, lam=1e-2, x0=0.0, anchor_strength=1e6):
        """
        Solve: min_x ||D x - v||^2 + lam ||x||^2
        where D is first-difference / dt operator.
        Enforce x[0] = x0 via a strong diagonal anchor.
        """
        N = len(vel)
        D = sparse.diags([1, -1], [0, -1], shape=(N, N)) / dt  # (x_k - x_{k-1})/dt
        I = sparse.eye(N)
    
        A = D.T @ D + lam * I
        b = D.T @ vel
    
        # Anchor x[0] = x0 strongly
        A = A.tolil()
        A[0, 0] += anchor_strength
        b = b.copy()
        b[0] += anchor_strength * x0
        A = A.tocsc()
    
        x = spsolve(A, b)
        return x
    
    # ---------------------------
    # 3) Savitzky–Golay + trapezoidal integration
    # ---------------------------
    def savgol_integration(vel, dt, window=51, poly=3, detrend=True):
        """
        Smooth velocity with Savitzky–Golay, then integrate with cumulative trapezoid.
        Optionally remove the mean of velocity to reduce drift.
        """
        if window % 2 == 0:
            window += 1  # must be odd
        v = vel - np.mean(vel) if detrend else vel
        v_s = savgol_filter(v, window_length=window, polyorder=poly)
        # cumulative trapezoid (fast)
        x = np.cumsum((v_s[:-1] + v_s[1:]) * 0.5 * dt)
        x = np.concatenate([[0.0], x])  # set initial displacement 0
        return x
    
    # ---------------------------
    # Plot: displacement comparison (GP + FD + Tik + SG)
    # ---------------------------
    def plot_displacement_comparison(
        t,
        x_true,          # ground truth displacement, shape (N,)
        x_gp,            # GP displacement estimate, shape (N,)
        x_fd,            # frequency-domain integration result
        x_tik,           # Tikhonov integration result
        x_sg,            # Savitzky–Golay integration result
        t_min=None, t_max=None,
        title="Displacement Comparison (integration methods)",
        figsize=(12, 5)
    ):
        fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True)
        ax.plot(t, x_true, 'k', lw=1.6, label="True")
        ax.plot(t, x_gp,   color="tab:blue",   ls="--", lw=1.2, label="GP")
        ax.plot(t, x_fd,   color="tab:red",    ls=":",  lw=1.2, label="FD (HP+FFT)")
        ax.plot(t, x_tik,  color="gray",       ls="-.", lw=1.2, label="Tikhonov")
        ax.plot(t, x_sg,   color="purple",     ls="-.", lw=1.2, label="SG + trapz")
    
        if t_min is not None and t_max is not None:
            ax.set_xlim(t_min, t_max)
            # auto y-limits in window
            m = (t >= t_min) & (t <= t_max)
            vals = np.concatenate([x_true[m], x_gp[m], x_fd[m], x_tik[m], x_sg[m]])
            pad = 0.1 * (vals.max() - vals.min() + 1e-12)
            ax.set_ylim(vals.min() - pad, vals.max() + pad)
    
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("x(t)")
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(loc="best")
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        return fig, ax
    
    # ---------------------------
    # Plot: displacement error
    # ---------------------------
    def plot_displacement_errors(
        t,
        x_true,
        x_gp,
        x_fd,
        x_tik,
        x_sg,
        t_min=None, t_max=None,
        title="Displacement Errors (estimate - true)",
        figsize=(12, 5)
    ):
        e_gp  = x_gp  - x_true
        e_fd  = x_fd  - x_true
        e_tik = x_tik - x_true
        e_sg  = x_sg  - x_true
    
        fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True)
        ax.plot(t, e_gp,  color="tab:blue",  lw=1.2, label="GP")
        ax.plot(t, e_fd,  color="tab:red",   lw=1.2, label="FD (HP+FFT)")
        ax.plot(t, e_tik, color="gray",      lw=1.2, label="Tikhonov")
        ax.plot(t, e_sg,  color="purple",    lw=1.2, label="SG + trapz")
    
        if t_min is not None and t_max is not None:
            ax.set_xlim(t_min, t_max)
            m = (t >= t_min) & (t <= t_max)
            vals = np.concatenate([e_gp[m], e_fd[m], e_tik[m], e_sg[m]])
            pad = 0.1 * (vals.max() - vals.min() + 1e-12)
            ax.set_ylim(vals.min() - pad, vals.max() + pad)
    
        ax.axhline(0, color="k", lw=0.8, alpha=0.6)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Error")
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(loc="best")
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        return fig, ax



    # 1) Pick your measured velocity signal:
    vel_meas = derivatives[:, 0]  # or your own noisy velocity array

    # 2) GP-based displacement (from your RTS result)
    x_gp = displacement  # GP smoothed displacement

    # 3) Frequency-domain integration (remove DC, then divide by jω)
    x_fd = integrate_freq_domain(vel_meas, dt, f_hp=0.05, order=2)

    # 4) Tikhonov-regularized integration (stable inverse with anchor x[0]=0)
    x_tik = tikhonov_integration(vel_meas, dt, lam=1e-1, x0=0.0)

    # 5) Savitzky–Golay smoothing of velocity + trapezoidal integration
    x_sg = savgol_integration(vel_meas, dt, window=51, poly=3, detrend=True)

    # Optional: align each estimate to the true initial displacement for fair visual comparison
    if 'states' in globals():
        x_true = states[:, 0]
        for arr in (x_fd, x_tik, x_sg, x_gp):
            arr += (x_true[0] - arr[0])
    else:
        x_true = None

    # Visualize (comparison + error)
    if x_true is not None:
        plot_displacement_comparison(
            t, x_true, x_gp, x_fd, x_tik, x_sg,
            t_min=50, t_max=70,
            title="Displacement Comparison (Integration Methods)"
        )
        plot_displacement_errors(
            t, x_true, x_gp, x_fd, x_tik, x_sg,
            t_min=50, t_max=70,
            title="Displacement Errors (estimate - true)"
        )
    else:
        # If you don't have ground truth, plot only the estimates
        plot_displacement_comparison(
            t, x_gp, x_gp, x_fd, x_tik, x_sg,  # pass x_gp also as 'x_true' placeholder
            t_min=50, t_max=70,
            title="Displacement Comparison (no ground truth)"
        )