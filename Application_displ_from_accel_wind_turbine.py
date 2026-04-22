"""
Example 1: GP regression on Duffing oscillator response
using Matérn 5/2 state-space model with joint hyperparameter optimization.

Reproduces Figures 6 and 7 (Section 5) from: Differentiation and Integration of Time Series via Gaussian Process
                          Regression for Structural Health Monitoring Applications
"""

import numpy as np
import matplotlib.pyplot as plt
from data_generation import add_measurement_noise, load_wind_turbine_response_data

from gp_optimization import optimize_hyperparams_Matern52_constrained,extract_smoothed_displacement_vel_accel


import KalmanFilter_functions as KF
from Matern_52_state_space import Matern52

from plotting_functions import plot_states_with_zoom



compare_with_other_methods =  False

# load measurements from simulation
t, true_displacement, true_velocity, true_acceleration = load_wind_turbine_response_data()

dt = t[1]-t[0]
n_steps = len(t) 


#  %% add measurement noise
meas_clean = true_acceleration # accelerations
meas_noisy, noise_std, R = add_measurement_noise(meas_clean, noise_fraction=0.05, seed=456)
R_GP = R*1/100 ## measurement noise variance assumed known

#%% optimize negative log-likelihood ()


# optimization bounds
bounds_log = [
    (-1                                ,2),              #length scale l_f
    (np.log(np.std(true_displacement)/100), np.log(np.std(true_displacement)*100)),      # GP output variance sig_f
]
#initial guess parameters
initial_params = np.array([10,10*np.std(true_displacement)]) 

observed_derivative = 2 # observe second derivative (acceleration) to integrate

#Run constrained optimization
params_opt, result = optimize_hyperparams_Matern52_constrained(
    meas_noisy.reshape(1,-1),
    dt,
    R_GP, ##measurement noise variance not optimized
    initial_params,
    bounds_log,
    observed_derivative=observed_derivative, 
    maxiter=100,
    ftol=1e-6,
    verbose=True,
) 

#%% use optimal parameters to perform the estimation (i.e., GP regression)

l_f =   params_opt[0]
sig_f = params_opt[1]    ##variance of Matern process
R_opt = R_GP ## measurement noise variance assumed known



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

(displacement, displacement_variance,
        velocity,     velocity_variance,
        acceleration, acceleration_variance) = extract_smoothed_displacement_vel_accel(zs, Ps, D0, D1, D2) 


 
 
plot_states_with_zoom(
    t,
    true_displacement.reshape(-1,1),             
    np.vstack((true_velocity,true_acceleration)).T,      
    displacement,
    velocity,
    acceleration,            
    displacement_variance, 
    velocity_variance,     
    acceleration_variance, 
    t_full_min=0.0,
    t_full_max=300,
    t_zoom_min=100,
    t_zoom_max=160,
    fontsize_axes=16,
    fontsize_legends=12,
    save_path=None,
    figsize=(12, 10)
)
plt.show()
     

    
#%% Comparison with existing methods

# =============================================================================
# Comparison with existing methods :
    # - Frequency domain integration + high-pass filter
    # - Double integration + polynomial detrending
# =============================================================================



if compare_with_other_methods:
        
    from plotting_functions import plot_integration_method_comparison
    from scipy.integrate import cumulative_trapezoid
    from scipy import signal 
    
    
    # double integration + polynomial detrend 
    def acc2disp_detrend(t,acceleration,dt,polynomial_degree):
        velocity = cumulative_trapezoid(acceleration, dx=dt, initial=0)
        
        # Second integration: velocity -> displacement
        displacement = cumulative_trapezoid(velocity, dx=dt, initial=0)
        
        # Detrend the displacement with a 5th order polynomial
        coefficients = np.polyfit(t, displacement, polynomial_degree)
        trend = np.polyval(coefficients, t)
        displacement_detrended = displacement - trend
        
        return displacement_detrended
    
    displacement_detrended = acc2disp_detrend(t,meas_noisy.flatten(),dt,polynomial_degree=30) 
     
      
    
    # double integrate using windowing + FFT + integration frequency domain + iFFT
      
        
    def highpass_filter(data, cutoff_freq, fs, order=4):
        """Apply highpass filter to remove low-frequency drift"""
        nyq = 0.5 * fs
        normal_cutoff = cutoff_freq / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return signal.filtfilt(b, a, data)
    
    def long_dft_integrate(input_signal, dt, highpass_cutoff=None):
        """
        Integrate using long DFT method from Brandt & Brincker (2014), Section 2.1
        This is the recommended method from the paper.
        
        Steps from the paper:
        1. Remove mean
        2. Compute FFT with zero padding
        3. Compute integration FRF H(k) = 1/(jω)
        4. Multiply: Y(k) = X(k) * H(k)
        5. Make Y(k) double-sided
        6. Inverse FFT
        7. Detrend output
        """
        fs = 1.0 / dt
        L = len(input_signal)
        
        # Apply highpass filter if specified
        if highpass_cutoff is not None:
            sig = highpass_filter(input_signal, highpass_cutoff, fs)
        else:
            sig = input_signal.copy()
        
        # Step 1: Remove mean
        sig = sig - np.mean(sig)
        
        # Step 2: Compute FFT with zero padding to 2L
        # rfft returns L+1 frequency bins (0 to fs/2)
        X = np.fft.rfft(sig, n=2*L)
        
        # Step 3: Compute integration FRF
        # Frequencies for rfft: f = k * fs / (2L) for k = 0, 1, ..., L
        freq = np.fft.rfftfreq(2*L, dt)
        H = np.zeros(L+1, dtype=complex)
        H[0] = 0  # DC component = 0 (can't divide by zero)
        omega = 2 * np.pi * freq[1:]
        H[1:] = 1.0 / (1j * omega)
        
        # Step 4: Multiply in frequency domain
        Y = X * H
        
        # Step 5-6: Inverse FFT (irfft handles double-sided automatically)
        y = np.fft.irfft(Y, n=2*L)
        
        # Take first L samples (second half is zeros from zero padding)
        y = y[:L]
        
        # Step 7: Detrend output (remove mean and linear trend)
        y = signal.detrend(y, type='linear')
        
        return y
    
    
    
    def apply_edge_window(signal, m):
        """
        Apply Hanning window only to first m and last m samples
        
        Parameters:
        -----------
        signal : array
            Input signal
        m : int
            Number of samples to taper at each end
        """
        windowed = signal.copy()
        n = len(signal)
        
        if m <= 0 or m > n // 2:
            return windowed
        
        # Create half Hanning window for the edges
        # Full Hanning: 0.5 * (1 - cos(2*pi*n/(N-1)))
        # We want just the rising edge (0 to 1) and falling edge (1 to 0)
        
        hann_rise = np.hanning(2*m)[:m]  # First half: 0 -> 1
        hann_fall = np.hanning(2*m)[m:]  # Second half: 1 -> 0
        
        # Apply to first m samples
        windowed[:m] *= hann_rise
        
        # Apply to last m samples
        windowed[-m:] *= hann_fall
        
        return windowed
    
    
    def acc_to_disp_windowing(acceleration, dt, highpass_cutoff=0.5, 
                               window_steps=None):
        """
        Double integrate acceleration to displacement with optional edge windowing
        
        Parameters:
        -----------
        acceleration : array
            Acceleration signal
        dt : float
            Sampling period
        highpass_cutoff : float or None
            Highpass filter cutoff in Hz
        detrend_order : int
            Polynomial order for final detrending
        window_steps : int or None
            Number of time steps to taper at beginning and end
            If None, no windowing is applied
        """
        # Apply edge windowing to acceleration if specified
        acc_windowed = apply_edge_window(acceleration, window_steps)
        
        # First integration: acceleration -> velocity
        velocity = long_dft_integrate(acc_windowed, dt, highpass_cutoff)
        
        # Second integration: velocity -> displacement
        displacement = long_dft_integrate(velocity, dt, highpass_cutoff)
        
        
        return displacement, velocity
    
    
    displacement_integration_Fourier_domain = acc_to_disp_windowing(meas_noisy.flatten(), dt,   
                                         highpass_cutoff=0.0222,
                                           window_steps=int(20/dt))[0]
        
    
     
    
    
    
    fig, (ax_full, ax_zoom, ax_error) = plot_integration_method_comparison(
        t=t,
        u=true_displacement,
        displacement_GP=displacement,
        displacement_detrended=displacement_detrended,
        displacement_fourier=displacement_integration_Fourier_domain,
        u_mean=true_displacement.mean(),
        t_full_min=0,
        t_full_max=600,
        t_zoom_min=240,
        t_zoom_max=310,
        y_zoom_lim=(-0.2, 0.2),
        save_path=None,
    )
    plt.show()
     