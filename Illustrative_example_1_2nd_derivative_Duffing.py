"""
Example 1: GP regression on Duffing oscillator response
using Matérn 5/2 state-space model with joint hyperparameter optimization.

Reproduces Figure 3 from: Differentiation and Integration of Time Series via Gaussian Process
                          Regression for Structural Health Monitoring Applications
"""
import numpy as np
import matplotlib.pyplot as plt


from data_generation import generate_data_Duffing_oscillator,add_measurement_noise
from gp_optimization import  optimize_hyperparams_Matern52_and_measurement_noise

import KalmanFilter_functions as KF
from Matern_52_state_space import Matern52

from plotting_functions import plot_states_with_zoom




#Generate data from Duffing oscillator
t, states, derivatives = generate_data_Duffing_oscillator()
dt = t[1]-t[0]
n_steps = len(t)

noise_level = 0.10 # e.g., noise_level = 0.05 = 5% noise

meas_clean = states[:, 0]
meas_noisy, noise_std, R = add_measurement_noise(meas_clean, noise_fraction=noise_level, seed=456)



fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
labels = ["Displacement", "Velocity", "Acceleration"]
data = [states[:, 0], derivatives[:, 0], derivatives[:, 1]]
colors = ["steelblue", "darkorange", "seagreen"]
for ax, d, label, color in zip(axs, data, labels, colors):
    ax.plot(t, d, color=color, lw=0.8, label=label)
    ax.legend()
    ax.set_ylabel(label, )
    ax.grid(True, alpha=0.3)
axs[-1].set_xlabel("Time (s)")
fig.suptitle("Duffing oscillator")
plt.tight_layout()
plt.show()

 

#%% optimize negative log-likelihood ()


# optimization bounds
bounds_log = [
    (np.log(0.01), np.log(100)),              #length scale l_f
    (np.log(0.01), np.log(100)),              # GP output variance sig_f
    (np.log(R[0,0]/100), np.log(R[0,0]*100))  # measurement noise variance
]
#initial guess parameters
initial_params = np.array([1.,1., R[0,0]*10])



observed_derivative = 0 # observe first state to differentiate

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

#second derivative
acceleration = np.zeros(n_steps)
acceleration_variance = np.zeros(n_steps)


for k in range(n_steps):
    displacement[[k]] = D0@zs[:,k]
    displacement_variance[[k]] = D0@Ps[:,:,k]@D0.T
    
    velocity[[k]] = D1@zs[:,k]
    velocity_variance[[k]] = D1@Ps[:,:,k]@D1.T
    
    acceleration[[k]] = D2@zs[:,k]
    acceleration_variance[[k]] = D2@Ps[:,:,k]@D2.T
 
 
 
 
fig, axs, axs_zoom = plot_states_with_zoom(
    t,
    states,             
    derivatives,      
    displacement,
    velocity,
    acceleration,            
    displacement_variance, 
    velocity_variance,     
    acceleration_variance, 
    t_full_min=0.0,
    t_full_max=200,
    t_zoom_min=50,
    t_zoom_max=80,
    fontsize_axes=16,
    fontsize_legends=12,
    save_path=None,
    figsize=(12, 10)
)
fig.set_constrained_layout(True) # to add the title
fig.suptitle(f"Duffing oscillator - noise {noise_level*100}%", fontsize=20)
plt.show()



#%% comparison with (i) frequency domain differentiation, (ii) Tikhonov-regularized differentiation, 
# (iii) Savitzky–Golay smoothing + differentiation


compare_with_other_methods =  False


if compare_with_other_methods:
    from plotting_functions import plot_derivative_comparison,plot_derivative_errors 
    from numpy.fft import fft, ifft, fftfreq
    from scipy.signal import butter, sosfiltfilt
     
    from scipy.signal import savgol_filter
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    
    def savgol_differentiation(signal, dt, window=51, poly=3):
        """
        Savitzky–Golay smoothing + differentiation.
        window: odd integer
        poly: polynomial degree
        """
        x_sg   = savgol_filter(signal, window_length=window, polyorder=poly)
        xd_sg  = savgol_filter(signal, window_length=window, polyorder=poly,
                               deriv=1, delta=dt)
        xdd_sg = savgol_filter(signal, window_length=window, polyorder=poly,
                               deriv=2, delta=dt)
        return x_sg, xd_sg, xdd_sg
    
    x_sg, xd_sg, xdd_sg = savgol_differentiation(meas_noisy, dt)
    
    
    
    def tikhonov_diff(y, dt, lam=1e-2):
        """
        Tikhonov-regularized differentiation.
        lam: regularization parameter
        """
        N = len(y)
        # Finite-difference integration operator
        D = sparse.diags([1, -1], [0, -1], shape=(N, N)) / dt
        I = sparse.eye(N)
    
        xprime = spsolve(D.T @ D + lam * I, D.T @ y)
        return xprime
    
    #regularization parameter can be tuned to better performance
    xd_tik  = tikhonov_diff(meas_noisy, dt, lam=0.1)
    xdd_tik = tikhonov_diff(xd_tik, dt, lam= 0.1)
    
    
    fd_sg   = [x_sg, xd_sg, xdd_sg]
    fd_tik  = [meas_noisy, xd_tik, xdd_tik] # same here
    
    def butterworth_lowpass(signal, f_cutoff, fs, order=4):
        """Zero-phase Butterworth low-pass filter.
        
        Parameters
        ----------
        signal : ndarray
        f_cutoff : float
            Cutoff frequency in Hz.
        fs : float
            Sampling frequency in Hz.
        order : int
            Filter order. Higher order = sharper roll-off.
        """
        nyq = fs / 2
        sos = butter(order, f_cutoff / nyq, btype='low', output='sos')
        return sosfiltfilt(sos, signal)
    
    # FFT of noisy measurement
    N = len(t)
    freqs = fftfreq(N, d=dt)
    Z = fft(meas_noisy)
    
    # Low-pass filter - cutoff frequency (tune this)
    fs = 1 / dt
    f_cutoff = 1.5 
    
    x_fd = butterworth_lowpass(meas_noisy, f_cutoff, fs, order=4)
    
    Z_filtered = fft(x_fd)
    omega_fft  = 2j * np.pi * freqs
    xd_fd  = ifft(Z_filtered * omega_fft).real
    xdd_fd = ifft(Z_filtered * omega_fft**2).real
    
    
    plot_derivative_comparison(
        t, derivatives, np.vstack((displacement,velocity,acceleration)),
        [x_fd, xd_fd, xdd_fd],
        x_sg, xd_sg, xdd_sg,
        xd_tik, xdd_tik
    )
    
    
    plot_derivative_errors(
        t, derivatives, np.vstack((displacement,velocity,acceleration)), 
        [x_fd, xd_fd, xdd_fd],
        x_sg, xd_sg, xdd_sg, 
        xd_tik, xdd_tik
    )
