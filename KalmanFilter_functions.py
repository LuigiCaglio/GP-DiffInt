import numpy as np


def kf_1step(z_prev, P_prev,A,H,Q,R,d): 
    """
    Kalman filter - 1 step
    input: z_prev, P_prev,A,H,Q,R,d
    output: z_est, P_est
    """
    
    #d is the measurement vector at the current time step
    #prev stands from previous
    
    
    #prediction
    z_pred = A@z_prev
    P_pred = A@P_prev@A.T + Q
    
    
    #Kalman Gain
    # Kgain = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    S = H @ P_pred @ H.T + R
    B = P_pred @ H.T
    Kgain = np.linalg.solve(S.T, B.T).T
    
    
    # update
    z_est = z_pred + Kgain @ (d - H@ z_pred)
    P_est = P_pred - Kgain @ H @ P_pred
    
    
    return z_est, P_est
 

def kf_1step_nll(z_prev, P_prev,A,H,Q,R,d):
    """
    returns the negloglikelihood
    input: z_prev, P_prev,A,H,Q,R,d
    output: z_est, P_est ,nll_plus
    """
    ### 
    #d is the measurement vector at the current time step
    #prev stands from previous
    
    
    #prediction
    z_pred = A@z_prev
    P_pred = A@P_prev@A.T + Q
    
    
    #Kalman Gain
    Sk = H @ P_pred @ H.T + R # equation B.5 in "A gaussian Process latent force model..."
    B = P_pred @ H.T
    Kgain = np.linalg.solve(Sk.T, B.T).T
    
    
    # update
    z_est = z_pred + Kgain @ (d - H@ z_pred)
    P_est = P_pred - Kgain @ H @ P_pred
    
    
    #computation of neg-log-likelihood
    e_k = d - H@ z_pred   
    intermediate_result = np.linalg.solve(Sk, e_k)    
    sign, logdet_Sk = np.linalg.slogdet(Sk)
    nll_plus = 0.5 * (len(d) * np.log(2*np.pi) + logdet_Sk + e_k.T @ intermediate_result)

    return z_est, P_est ,nll_plus


def ks_1step(A,zp,zf,Pp,Pf,zs,Ps):
    """
    Rauch-Tung-Striebel smoother, a.k.a. Kalman Smoother
    input:A,zp,zf,Pp,Pf,zs,Ps
    output:zs_prev,Ps_prev
    """
    # Ks = Pf @ A.T @ np.linalg.inv(Pp) 
    B = Pf @ A.T
    Ks = np.linalg.solve(Pp.T, B .T).T
    zs_prev = zf + Ks @ (zs - zp)
    Ps_prev = Pf + Ks @ (Ps - Pp) @ Ks.T
    
    return zs_prev,Ps_prev



def kf_full_estimation(A,H,Qd,R,measurements,z0 = [],P0=[],
                        return_nll=False):
    """
    input: A,H,Qd,R,z_meas,z0,P0,return_nll=False
    output: zf, Pf
    or output: zf, Pf, nll
    """
    n_states = len(A)
    n_meas,nsteps = measurements.shape
    
    
    # initialize state vector and covariance
    zf = np.zeros([n_states,nsteps])
    Pf = np.zeros([n_states,n_states,nsteps])
    
    #first step (only update)
    Sk = H @ P0 @ H.T + R 
    B = P0 @ H.T
    Kgain = np.linalg.solve(Sk.T, B.T).T
    zf[:,0] = z0 + Kgain @ (measurements[:,0] - H@ z0)
    Pf[:,:,0] = P0 - Kgain @ H @ P0
        
    #main loop
    if return_nll:
        nll = 0
        for k in range(1,nsteps):
            zf[:,k],Pf[:,:,k],nll_plus = kf_1step_nll(zf[:,k-1], Pf[:,:,k-1], 
                                                      A, H, Qd, R, measurements[:,k])
            nll += nll_plus
        return zf, Pf, nll

    else:
        for k in range(1,nsteps):
            zf[:,k],Pf[:,:,k] = kf_1step(zf[:,k-1], Pf[:,:,k-1], 
                                            A, H, Qd, R, measurements[:,k])
        return zf, Pf
        

def rtss_full_estimation(A,zf,Pf,Qd):    
    """
    input: A,zf,Pf,use_tqdm=True
    output: zs, Ps
    """

    zs = np.zeros_like(zf)
    Ps = np.zeros_like(Pf)
    
    
    zs[:,-1] = zf[:,-1]  
    Ps[:,:,-1] = Pf[:,:,-1]
    
        
    nsteps = zf.shape[1] 
    
    for k in reversed(range(1,nsteps)):
        
        #predicted state mean and covariance (same as KF)
        zp_k = A@zf[:,k-1]
        Pp_k = A@Pf[:,:,k-1]@A.T + Qd
        
        #RTS smoother step
        zs[:,k-1],Ps[:,:,k-1] = ks_1step(A,zp_k,zf[:,k-1],Pp_k,Pf[:,:,k-1],
                                         zs[:,k],Ps[:,:,k]) 
    return zs, Ps
 

