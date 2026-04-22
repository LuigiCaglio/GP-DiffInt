import numpy as np

def Matern52(l_f,sig_f,dt):
    
    """
    input: l_f,sig_f,dt
    output: A, Qd,  Pinf, D0, D1, D2
    """

    p = 2
    dimension = p+1
    ni = p+0.5
    lambda_ = (2*ni)**0.5/l_f
    
    q =16/3 * sig_f**2*lambda_**5
    
    
    Fcf = np.zeros([3,3])
    Fcf[0:dimension-1,1:dimension] = np.eye(dimension-1)
    
    Fcf = np.array([ [0,            1,             0        ],
                     [0,            0,             1        ],
                     [-lambda_**3, -3*lambda_**2, -3*lambda_]])
    
    
    Lcf = np.array([[0],
                    [0],
                    [1]])
    
    Qc = np.array([[q]])
    
    
    D0 = np.array([[1, 0, 0]])
    D1 = np.array([[0, 1, 0]])
    D2 = np.array([[0, 0, 1]])
    
    
    
    # A = expm(Fcf*dt)3
    
    #Analytical solution
    e   = np.exp(-lambda_ * dt)
    ldt = lambda_ * dt
    
    A = e * np.array([
        [1 + ldt + ldt**2/2,       dt*(1 + ldt),           dt**2/2              ],
        [-lambda_**3*dt**2/2,      1 + ldt - ldt**2,        dt*(1 - ldt/2)       ],
        [lambda_**4*dt**2/2 - lambda_**3*dt,   lambda_**3*dt**2 - 3*lambda_**2*dt,   1 - 2*ldt + ldt**2/2]
    ])
    
    
    Pinf = sig_f**2 * np.array([
                    [1,              0,       -lambda_**2/3],
                    [0,     lambda_**2/3,              0  ],
                    [-lambda_**2/3,  0,        lambda_**4 ]])
 
    Qd = Pinf - A @ Pinf @ A.T
    
 
    
    
    return A, Qd,  Pinf, D0, D1, D2
 


