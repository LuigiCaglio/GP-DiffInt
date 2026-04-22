import numpy as np 
from scipy.integrate import odeint,solve_ivp

def add_measurement_noise(signal, noise_fraction=0.05, seed=None):
    """Add Gaussian noise to a clean signal.
    
    Returns the noisy signal, noise std, and measurement noise covariance R.
    """
    if seed is not None:
        np.random.seed(seed)
    noise_std = noise_fraction * np.std(signal)
    z_meas = signal + noise_std * np.random.randn(len(signal))
    R = noise_std**2 * np.eye(1)
    return z_meas, noise_std, R









def generate_data_Duffing_oscillator():
    
    
    def duffing(state, t, delta, alpha, beta, gamma, omega):
        """
        Duffing oscillator differential equations.
        x'' + delta*x' + alpha*x + beta*x^3 = gamma*cos(omega*t)
        
        Converting to first order system:
        x' = v
        v' = -delta*v - alpha*x - beta*x^3 + gamma*cos(omega*t)
        """
        x, v = state
        dxdt = v
        dvdt = -delta*v - alpha*x - beta*x**3 + gamma*(np.cos(omega*t)+\
                        np.cos(omega*2*t+1)+np.cos(omega*1.5*t+1)+\
                            +np.cos(omega*0.4*t+2)+np.cos(omega*0.2*t+1)+\
                                +np.cos(omega*12*t+1)+np.cos(omega*17*t+1))
        return [dxdt, dvdt]
    
    # Set parameters
    delta = 0.2    # damping
    alpha = -1.0   # linear stiffness
    beta = 1.0     # nonlinear stiffness
    gamma = 0.3    # forcing amplitude
    omega = 1.0    # forcing frequency
    
    # Set initial conditions
    state0 = [1.0, 0.0]  # [position, velocity]
    
    t_end = 300
    dt = 0.02
    nsteps = int(t_end/dt)
    # Create time points
    t = np.linspace(0, t_end, nsteps)
    
    # Solve the ODE
    states = odeint(duffing, state0, t, args=(delta, alpha, beta, gamma, omega))
    
    # Calculate derivatives at each point
    derivatives = np.array([duffing(state, t_, delta, alpha, beta, gamma, omega) 
                           for state, t_ in zip(states, t)])
    
    return t, states, derivatives


def generate_data_Lorenz_attractor():


    def lorenz(t, state, sigma, rho, beta):
        """Lorenz system of differential equations."""
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]

    # Set parameters
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    # Set initial conditions
    state0 = [1.0, 1.0, 1.0]

    # Create time points
    t = np.linspace(0, 100, 10001)

    # Solve the ODE
    sol = solve_ivp(lorenz, [t[0], t[-1]], state0, method='RK45', 
                    t_eval=t, args=(sigma, rho, beta))
    states = sol.y.T  # Transpose to match odeint shape

    # Calculate derivatives at each point
    derivatives = np.array([lorenz(t_, state, sigma, rho, beta) 
                           for state, t_ in zip(states, t)])
                           
                           
    return t, states, derivatives
                           
                           
                           
                           
                           
           
def load_wind_turbine_response_data():                
                           
                    
    
    # load measurements from simulation
    datafile="windturbine2D.txt"  
    data = np.loadtxt(datafile)
    
    
    steps_to_exclude = int(10/0.01) # exclude the first 10 seconds of the numerical solution associated with transient non physical behavior
    displacement = data[0,steps_to_exclude:] # displacement
    velocity = data[1,steps_to_exclude:] # velocity
    acceleration = data[2,steps_to_exclude:] # acceleration
    
    displacement = displacement-np.mean(displacement) ###mean value cannot be retrieved from integration
    
    
    fs = 100 # [Hz] - sampling frequency of numerical solution 
    dt = 1/fs
    n_steps = len(displacement)
    t=np.arange(n_steps)*dt
    
    
    
    
    #downsample measurements - high frequencies are not relevant in this case and might be associated with numerical noise
    downsampleFact = 5
    displacement = displacement[::downsampleFact]
    velocity = velocity[::downsampleFact]
    acceleration = acceleration[::downsampleFact]
    dt*=downsampleFact
    fs/=downsampleFact
    
    
    n_steps = len(displacement)
    t=np.arange(n_steps)*dt           
    
    
    return t, displacement, velocity, acceleration
    
                               
                               