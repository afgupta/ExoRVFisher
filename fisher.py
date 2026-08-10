import numpy as np
from scipy import linalg

from utils import K_from_mass

from jax import config
config.update('jax_enable_x64', True)

import jax.numpy as jnp
import jax
from exoplanet_core.jax import ops

import warnings


@jax.jit
def rv_model_circ(params, t):
    
    K, period, phi = params
    
    return K*jnp.sin(2*np.pi*t/period+phi)
jac_circ = jax.jit(jax.jacobian(rv_model_circ))

@jax.jit
def rv_model_ecc(params, t):
    
    K, period, phi, ecc, omega = params

    mean_anom = 2. * np.pi * t / period + phi

    cosf, sinf = ops.kepler(mean_anom, ecc + jnp.zeros_like(mean_anom))

    return K * (jnp.cos(omega) * (cosf + ecc) - jnp.sin(omega) * sinf)
jac_ecc = jax.jit(jax.jacobian(rv_model_ecc))


def fisher_keplerian_gamma(jac, Sigma_inv, obstime, K, period, phi, gamma_mask,
                           ecc=0, omega=0, eccentric=False):
    if eccentric:
        params=jnp.array([K, period, phi, ecc, omega], dtype=float)
    else:
        params=jnp.array([K, period, phi])

    J_kep = jac(params, obstime)
        
    J = np.concatenate((J_kep, gamma_mask.T), axis=1)
    
    
    B=J.T @ Sigma_inv @ J


    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Ill-conditioned matrix")
            C=linalg.solve(B, np.identity(len(params)+len(gamma_mask)))
    except linalg.LinAlgError:
        # print('Singular matrix')
        C=np.full(np.shape(B),np.nan)
        
    return C

def fisher_two_keplerian_gamma(jac_0, jac, Sigma_inv, obstime, params_0, K, period, phi, gamma_mask,
                           ecc=0, omega=0, eccentric=False):
    if eccentric:
        params=jnp.array([K, period, phi, ecc, omega], dtype=float)
    else:
        params=jnp.array([K, period, phi])


    J_kep_0 = jac_0(params_0, obstime)
    J_kep = jac(params, obstime)
        
    J = np.concatenate((J_kep, J_kep_0, gamma_mask.T), axis=1)
    
    
    B=J.T @ Sigma_inv @ J


    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Ill-conditioned matrix")
            C=linalg.solve(B, np.identity(len(params)+len(params_0)+len(gamma_mask)))
    except linalg.LinAlgError:
        # print('Singular matrix')
        C=np.full(np.shape(B),np.nan)
        
    return C

def fisher_single_planet(Sigma_inv, obstime, gamma_mask, mstar,
                         mass, period, phi_basis,
                         ecc=0, omega_basis=None, eccentric=False):
    if eccentric:
        Ck=np.zeros((len(phi_basis),len(omega_basis)))
        Cp=np.zeros((len(phi_basis),len(omega_basis)))
        Ch=np.zeros((len(phi_basis),len(omega_basis)))
        Ce=np.zeros((len(phi_basis),len(omega_basis)))
        Cw=np.zeros((len(phi_basis),len(omega_basis)))
    
    else:
        Ck=np.zeros((len(phi_basis)))
        Cp=np.zeros((len(phi_basis)))
        Ch=np.zeros((len(phi_basis)))
            
    if eccentric:
        
        K=K_from_mass(mass, period, mstar=mstar, ecc=ecc)
        
        for h, phi in enumerate(phi_basis):
            for w, omega in enumerate(omega_basis):
                C = fisher_keplerian_gamma(jac_ecc, Sigma_inv, obstime, K, period, phi, gamma_mask,
                                           ecc=ecc, omega=omega, eccentric=True)
        
                Ck[h][w]=C[0][0]
                Cp[h][w]=C[1][1]
                Ch[h][w]=C[2][2]
                Ce[h][w]=C[3][3]
                Cw[h][w]=C[4][4]
                

    else:
        
        K=K_from_mass(mass, period, mstar=mstar, ecc=0)

        for h, phi in enumerate(phi_basis):
            C = fisher_keplerian_gamma(jac_circ, Sigma_inv, obstime, K, period, phi, gamma_mask)

            Ck[h]=C[0][0]
            Cp[h]=C[1][1]
            Ch[h]=C[2][2]
                
    if eccentric:
        return Ck, Cp, Ch, Ce, Cw
    else:
        return Ck, Cp, Ch

    
def fisher_grid_series(Sigma_inv, obstime, gamma_mask, mstar,
                       m_basis, p_basis, phi_basis, ecc_basis=None, omega_basis=None,
                       eccentric=False):
    
    if eccentric:
        
        Ck=np.zeros((len(m_basis),len(p_basis),len(phi_basis),len(ecc_basis),len(omega_basis)))
        Cp=np.zeros((len(m_basis),len(p_basis),len(phi_basis),len(ecc_basis),len(omega_basis)))
        Ch=np.zeros((len(m_basis),len(p_basis),len(phi_basis),len(ecc_basis),len(omega_basis)))
        Ce=np.zeros((len(m_basis),len(p_basis),len(phi_basis),len(ecc_basis),len(omega_basis)))
        Cw=np.zeros((len(m_basis),len(p_basis),len(phi_basis),len(ecc_basis),len(omega_basis)))
        
        for k, mass in enumerate(m_basis):
            for p, period in enumerate(p_basis):
                K = K_from_mass(mass, period, mstar=mstar)
                for h, phi in enumerate(phi_basis):
                    for e, ecc in enumerate(ecc_basis):
                        for w, omega in enumerate(omega_basis):
                            C = fisher_keplerian_gamma(jac_ecc, Sigma_inv, obstime, K, period, phi, gamma_mask,
                                                       ecc=ecc, omega=omega, eccentric=True)
                
                            Ck[k][p][h][e][w]=C[0][0]
                            Cp[k][p][h][e][w]=C[1][1]
                            Ch[k][p][h][e][w]=C[2][2]
                            Ce[k][p][h][e][w]=C[3][3]
                            Cw[k][p][h][e][w]=C[4][4]

        return Ck, Cp, Ch, Ce, Cw


    else:
    
        Ck=np.zeros((len(m_basis),len(p_basis),len(phi_basis)))
        Cp=np.zeros((len(m_basis),len(p_basis),len(phi_basis)))
        Ch=np.zeros((len(m_basis),len(p_basis),len(phi_basis)))
        
        for k, mass in enumerate(m_basis):
            for p, period in enumerate(p_basis):
                K = K_from_mass(mass, period, mstar=mstar)
                for h, phi in enumerate(phi_basis):
                    C = fisher_keplerian_gamma(jac_circ, Sigma_inv, obstime, K, period, phi, gamma_mask)
        
                    Ck[k][p][h]=C[0][0]
                    Cp[k][p][h]=C[1][1]
                    Ch[k][p][h]=C[2][2]

        return Ck, Cp, Ch
    
def fisher_grid_series_two(Sigma_inv, obstime, gamma_mask, mstar,
                       m_basis, p_basis, phi_basis, params_0,
                       ecc_basis=None, omega_basis=None, eccentric=False):

    if eccentric:
        
        Ck=np.zeros((len(m_basis),len(p_basis),len(phi_basis),len(ecc_basis),len(omega_basis)))
        Cp=np.zeros((len(m_basis),len(p_basis),len(phi_basis),len(ecc_basis),len(omega_basis)))
        Ch=np.zeros((len(m_basis),len(p_basis),len(phi_basis),len(ecc_basis),len(omega_basis)))
        Ce=np.zeros((len(m_basis),len(p_basis),len(phi_basis),len(ecc_basis),len(omega_basis)))
        Cw=np.zeros((len(m_basis),len(p_basis),len(phi_basis),len(ecc_basis),len(omega_basis)))
        
        for k, mass in enumerate(m_basis):
            for p, period in enumerate(p_basis):
                K = K_from_mass(mass, period, mstar=mstar)
                for h, phi in enumerate(phi_basis):
                    for e, ecc in enumerate(ecc_basis):
                        for w, omega in enumerate(omega_basis):
                            C = fisher_two_keplerian_gamma(jac_ecc, jac_ecc, Sigma_inv, obstime, params_0, K, period, phi, gamma_mask,
                                                       ecc=ecc, omega=omega, eccentric=True)
                
                            Ck[k][p][h][e][w]=C[0][0]
                            Cp[k][p][h][e][w]=C[1][1]
                            Ch[k][p][h][e][w]=C[2][2]
                            Ce[k][p][h][e][w]=C[3][3]
                            Cw[k][p][h][e][w]=C[4][4]

        return Ck, Cp, Ch, Ce, Cw


    else:
    
        Ck=np.zeros((len(m_basis),len(p_basis),len(phi_basis)))
        Cp=np.zeros((len(m_basis),len(p_basis),len(phi_basis)))
        Ch=np.zeros((len(m_basis),len(p_basis),len(phi_basis)))
        
        for k, mass in enumerate(m_basis):
            for p, period in enumerate(p_basis):
                K = K_from_mass(mass, period, mstar=mstar)
                for h, phi in enumerate(phi_basis):
                    C = fisher_two_keplerian_gamma(jac_ecc, jac_circ, Sigma_inv, obstime, params_0, K, period, phi, gamma_mask)
        
                    Ck[k][p][h]=C[0][0]
                    Cp[k][p][h]=C[1][1]
                    Ch[k][p][h]=C[2][2]

        return Ck, Cp, Ch        


def fisher_grid_parallel(args, mp):
    
    Sigma_inv = args['isigma']
    obstime = args['obstime']
    gamma_mask = args['gamma_mask']
    mstar = args['mstar']
    eccentric=args['eccentric']
    # verbose = args['verbose']
    
    
    phi_basis = args['phi_basis']
    p_basis = args['p_basis']
    
    ecc_basis = args['ecc_basis']
    omega_basis = args['omega_basis']
    
    
    if eccentric:

        Ck=np.zeros((len(p_basis),len(phi_basis),len(ecc_basis),len(omega_basis)))
        Cp=np.zeros((len(p_basis),len(phi_basis),len(ecc_basis),len(omega_basis)))
        Ch=np.zeros((len(p_basis),len(phi_basis),len(ecc_basis),len(omega_basis)))
        Ce=np.zeros((len(p_basis),len(phi_basis),len(ecc_basis),len(omega_basis)))
        Cw=np.zeros((len(p_basis),len(phi_basis),len(ecc_basis),len(omega_basis)))
        
        for p, period in enumerate(p_basis):
            for h, phi in enumerate(phi_basis):
                for e, ecc in enumerate(ecc_basis):
                    K = K_from_mass(mp, period, mstar=mstar, ecc=ecc)
                    for w, omega in enumerate(omega_basis):
                        C = fisher_keplerian_gamma(jac_ecc, Sigma_inv, obstime, K, period, phi, gamma_mask,
                                                   ecc=ecc, omega=omega, eccentric=True)
                        
                        Ck[p][h][e][w]=C[0][0]
                        Cp[p][h][e][w]=C[1][1]
                        Ch[p][h][e][w]=C[2][2]
                        Ce[p][h][e][w]=C[3][3]
                        Cw[p][h][e][w]=C[4][4]
                        
        return Ck, Cp, Ch, Ce, Cw
                
    else:
        
        K = K_from_mass(mp, p_basis, mstar=mstar)
        
        Ck=np.zeros((len(p_basis),len(phi_basis)))
        Cp=np.zeros((len(p_basis),len(phi_basis)))
        Ch=np.zeros((len(p_basis),len(phi_basis)))
        
        for p, period in enumerate(p_basis):
            for h, phi in enumerate(phi_basis):
                C = fisher_keplerian_gamma(jac_circ, Sigma_inv, obstime, K[p], period, phi, gamma_mask)
    
                Ck[p][h]=C[0][0]
                Cp[p][h]=C[1][1]
                Ch[p][h]=C[2][2]        
                
        return Ck, Cp, Ch


def fisher_progression(n_obs_to_use, Sigma, obstime, instruments, mstar,
                       mass, period, phi_basis,
                       ecc=0, omega_basis=None, eccentric=False):
    
    if eccentric:
        Ck=np.zeros((len(n_obs_to_use),len(phi_basis),len(omega_basis)))
        Cp=np.zeros((len(n_obs_to_use),len(phi_basis),len(omega_basis)))
        Ch=np.zeros((len(n_obs_to_use),len(phi_basis),len(omega_basis)))
        Ce=np.zeros((len(n_obs_to_use),len(phi_basis),len(omega_basis)))
        Cw=np.zeros((len(n_obs_to_use),len(phi_basis),len(omega_basis)))
    
    else:
        Ck=np.zeros((len(n_obs_to_use),len(phi_basis)))
        Cp=np.zeros((len(n_obs_to_use),len(phi_basis)))
        Ch=np.zeros((len(n_obs_to_use),len(phi_basis)))

    for n, n_obs in enumerate(n_obs_to_use):
        
        n_inst = len(np.unique(instruments[:n_obs]))
        
        # create a binary matrix that serves as the derivative w.r.t. the
        # systemic velocity measured by each instrument
        # (derivative is always either 0 or 1)
        gamma_mask = np.zeros((n_inst, np.min([n_obs, len(obstime)])))#+1e-10
        for i, inst in enumerate(np.unique(instruments[:n_obs])):
            gamma_mask[i][instruments[:n_obs]==inst]=1        
        
        Identity=np.identity(np.min([n_obs, len(obstime)]))
        
        Sigma_inv = linalg.solve(Sigma[:n_obs,:n_obs], Identity)
            
        if eccentric:
            
            K=K_from_mass(mass, period, mstar=mstar, ecc=ecc)
            
            for h, phi in enumerate(phi_basis):
                for w, omega in enumerate(omega_basis):
                    C = fisher_keplerian_gamma(jac_ecc, Sigma_inv, obstime[:n_obs], K, period, phi, gamma_mask[:,:n_obs],
                                               ecc=ecc, omega=omega, eccentric=True)
            
                    Ck[n][h][w]=C[0][0]
                    Cp[n][h][w]=C[1][1]
                    Ch[n][h][w]=C[2][2]
                    Ce[n][h][w]=C[3][3]
                    Cw[n][h][w]=C[4][4]
                    
    
        else:
            
            K=K_from_mass(mass, period, mstar=mstar, ecc=0)
    
            for h, phi in enumerate(phi_basis):
                C = fisher_keplerian_gamma(jac_circ, Sigma_inv, obstime[:n_obs], K, period, phi, gamma_mask[:,:n_obs])

                Ck[n][h]=C[0][0]
                Cp[n][h]=C[1][1]
                Ch[n][h]=C[2][2]
                
    if eccentric:
        return Ck, Cp, Ch, Ce, Cw
    else:
        return Ck, Cp, Ch


def construct_parameter_grid(config, eccentric=False, verbose=False):
    """
    Contruct parameter arrays based on values given in configuration file
    """
    if config['mass_scale']=='log':        
        m_basis = 10**np.arange(float(config['mass_low']),float(config['mass_high'])+1e-5,float(config['mass_step']))
    elif config['mass_scale']=='linear':
        m_basis = np.arange(float(config['mass_low']),float(config['mass_high'])+1e-5,float(config['mass_step']))
    else:
        print('Mass scale not recognized. Must be either "log" or "linear".')
        return 0
    
    if config['period_scale']=='log':        
        p_basis = 10**np.arange(float(config['period_low']),float(config['period_high'])+1e-5,float(config['period_step']))
    elif config['period_scale']=='linear':
        p_basis = np.arange(float(config['period_low']),float(config['period_high'])+1e-5,float(config['period_step']))
    else:
        print('Period scale not recognized. Must be either "log" or "linear".')
        return 0

    if config['ecc_scale']=='linear':
        ecc_basis = np.arange(float(config['ecc_low']),float(config['ecc_high'])+1e-5,float(config['ecc_step']))
    else:
        print('Eccentricity scale not recognized. Currently only "linear" is recognized.')
        return 0

    phi_basis = np.arange(float(config['phi_low']),float(config['phi_high']),float(config['phi_step']))
    omega_basis = np.arange(float(config['omega_low']),float(config['omega_high']),float(config['omega_step']))

    if eccentric and verbose:
        print("Parameter dimensions are (%d,%d,%d,%d,%d). Grid contains %d elements." %
              (len(m_basis),len(p_basis),len(phi_basis),len(ecc_basis),len(omega_basis),
               int(len(m_basis)*len(p_basis)*len(phi_basis)*len(ecc_basis)*len(omega_basis))))
    elif verbose:
        print("Parameter dimensions are (%d,%d,%d). Grid contains %d elements." %
              (len(m_basis),len(p_basis),len(phi_basis),
               int(len(m_basis)*len(p_basis)*len(phi_basis))))        

    return m_basis, p_basis, phi_basis, ecc_basis, omega_basis