import numpy as np
from scipy.interpolate import interp1d

def get_bolo_corr(teff):
    teff_to_interp=[4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000]
    corr_to_interp=[1.4270115478150247,1.3077233084291158,1.2087260623207006,1.1249672368175945,1.0531232535874526,
    0.9908608226873058,0.9364574118651322,0.8885935199046348,0.8462310783980914,0.8085374670815352,0.7748351505122221]
    
    bolo_itp = interp1d(teff_to_interp, corr_to_interp, kind='linear')
    
    bolocorr_atTeff_kepler=bolo_itp(teff)
    
    return bolocorr_atTeff_kepler

def calc_nu_max(teff, logg, nu_max_sun=3100, teff_sun=5777, logg_sun=4.43):
    return nu_max_sun*10**(logg-logg_sun)/np.sqrt(teff/teff_sun)

def calc_Pg(teff, logg, nu_max_sun=3100, teff_sun=5777, logg_sun=4.43, amax_sun_rv=0.19, cenv_sun=331, deltanu_sun=134.9):
    
    nu_max = calc_nu_max(teff, logg)
    
    cenv = 0.174*nu_max**0.88
    
    agran = 3382*nu_max**-0.609
    
    Pg_phot = np.sqrt(1./(2*np.pi*cenv**2))*(agran/4.57)**(2./0.855)
    
    alpha=0.9
    
    beta = get_bolo_corr(teff)
    
    r_osc = 20*beta*(teff/teff_sun)**(-1.*alpha)
    
    return Pg_phot/r_osc**2

def calc_agran_rv(teff, logg, teff_sun=5777, logg_sun=4.43):
    agran=3382*calc_nu_max(teff, logg)**-0.609
    rgran=100.*(teff/teff_sun)**(-32./9.)*(10**logg/10**logg_sun)**(2./9.)
    
    return agran/rgran

def I1(yi,yj,a,eta,w0):
    upper=np.exp(-1.*a*yj)*(1-a**2)/(eta*w0*(1+a**2))*(np.cos(yj)+a*np.sin(yj))
    lower=np.exp(-1.*a*yi)*(1-a**2)/(eta*w0*(1+a**2))*(np.cos(yi)+a*np.sin(yi))
    return upper-lower
def I2(yi,yj,a,eta,w0):
    upper=-2*a*np.exp(-1.*a*yj)/(eta*w0*(1+a**2))*(np.sin(yj)-a*np.cos(yj))
    lower=-2*a*np.exp(-1.*a*yi)/(eta*w0*(1+a**2))*(np.sin(yi)-a*np.cos(yi))
    return upper-lower
def I_gran(yi,yj):
    return np.exp(-1.*yi)*np.sin(yi)-np.exp(-1.*yj)*np.sin(yj)

def osc_covariance(S0,w0,Q,eta,delta,d1,d2,overlapping=False):
    
    a = 1 / (2*Q*eta)
    
    if overlapping:
        y1=eta*w0*d1
        y2=0
        return 2*S0*Q/(d1**2*eta*(1+a**2))*(I1(y1,y2,a,eta,w0)-I2(y1,y2,a,eta,w0)+2*a*d1)
    else:

        y1=eta*w0*((d1+d2)/2.+delta)
        y2=eta*w0*((d2-d1)/2.+delta)
        y3=eta*w0*((d1-d2)/2.+delta)
        y4=eta*w0*(-1.*(d1+d2)/2.+delta)        
        return S0*Q/(d1*d2*eta*(1+a**2))*(I1(y1,y2,a,eta,w0)-I2(y1,y2,a,eta,w0)-I1(y3,y4,a,eta,w0)+I2(y3,y4,a,eta,w0))
def gran_covariance(S1,w1,S2,w2,delta,d1,d2,overlapping=False):
    if overlapping:
        y11=w1/np.sqrt(2.)*d1-np.pi/4.
        y12=-np.pi/4.
        y21=w2/np.sqrt(2.)*d1-np.pi/4.
        y22=-np.pi/4.
        return 2*S1/(w1*d1**2.)*np.exp(-np.pi/4.)*I_gran(y12,y11)+2*S1/d1+2*S2/(w2*d1**2)*np.exp(-np.pi/4.)*I_gran(y22,y21)+2*S2/d1
    else:

        y11=w1/np.sqrt(2.)*((d1+d2)/2.+delta)-np.pi/4.
        y12=w1/np.sqrt(2.)*((d2-d1)/2.+delta)-np.pi/4.
        y13=w1/np.sqrt(2.)*((d1-d2)/2.+delta)-np.pi/4.
        y14=w1/np.sqrt(2.)*(-1.*(d1+d2)/2.+delta)-np.pi/4.
        y21=w2/np.sqrt(2.)*((d1+d2)/2.+delta)-np.pi/4.
        y22=w2/np.sqrt(2.)*((d2-d1)/2.+delta)-np.pi/4.
        y23=w2/np.sqrt(2.)*((d1-d2)/2.+delta)-np.pi/4.
        y24=w2/np.sqrt(2.)*(-1.*(d1+d2)/2.+delta)-np.pi/4.
        return S1/(w1*d1*d2)*np.exp(-np.pi/4.)*(I_gran(y13,y14)-I_gran(y11,y12))+S2/(w2*d1*d2)*np.exp(-np.pi/4.)*(I_gran(y23,y24)-I_gran(y21,y22))

def rot_act_covariance_matern52(a0,a1,lamb,delta):
    kM52 = np.exp(-np.sqrt(5)*delta/lamb)*(1+np.sqrt(5)*delta/lamb+5*delta**2/(3*lamb**2))
    dkM52=-5/(3*lamb**2)*np.exp(-np.sqrt(5)*delta/lamb)*(1+np.sqrt(5)*delta/lamb-5*delta**2/lamb**2)
    return a0**2*kM52-a1**2*dkM52

def rot_act_covariance_qp(eta_qp,tau_qp,prot_qp,delta):
    return np.exp(-0.5*delta**2/tau_qp**2-0.5*np.sin(np.pi*delta/prot_qp)**2/eta_qp**2)  

def sqexp(delta, alpha=0.3, lamb=7):
    return alpha**2*np.exp(-0.5*delta**2/lamb**2)

def Covariance(star,observations,
               photon=True,inst=False,osc=False,gran=False, rot_act=False,
               teff=5777, logg=4.43,
               osc_hyperparameters='solar',
               gran_hyperparameters='solar',
               rot_act_hyperparameters='solar',
               inst_hyper_dict={},
               osc_hyper_dict={'S0':2.36,'w0':0.0195,'Q':7.63},
               gran_hyper_dict={'S1':18.64,'w1':5.22,'S2':0.00485,'w2':0.0173},
               rot_act_qp_hyper_dict={'wn_qp':1.44,'tau_qp':23.6,'prot_qp':28.1,'eta_qp':0.58},
               rot_act_matern52_hyper_dict={'wn_matern52':0.6,'a0':-0.106884,'a1':-1.154,'lambda':2.52}
               ):
    """
    Calculate the covariance matrix for a set of radial velocity observations of a star
   
    ----------
    Parameters:
    ----------
    star : string
         name of the target (must match the name given in 'star' column of observations)
    observations : data frame
        data frame with the following columns:
            "star" (target name)
            "start" (observation start time [days])
            "exposure_time" (observation duration [seconds])
            "photon_noise" (photon noise uncertainty [m/s])
            "instrument" (instrument name)
            "inst_jitter" (estimated white noise jitter [m/s])
    photon /inst / osc / gran : bool, optional
        if True, include the specified contribution to the covariance
    rot_act : bool or string, [False, 'matern52', 'qp']
        if 'matern52' or 'qp', include the contribution of rotationally
            modulated activity using the specified kernel
    teff : float, optional
        stellar effective temperature, default is the Solar value
    logg: float, optional
        stellar surface gravity, default is the Solar value
    osc_hyperparameters : string, ['solar','scaled','custom']
        if 'solar' (default), adopt solar hyperparameters for oscillation 
            kernel as given in Luhn et al. 2023
        if 'scaled', oscillation hyperparameters are scaled based on stellar
            properties using scaling relations given in Luhn et al. 2023
        if 'custom', specify hyperparameters with the osc_hyper_dict argument
    gran_hyperparameters : string, ['solar','scaled','custom']
        if 'solar' (default), adopt solar hyperparameters for granulation 
            kernel as given in Luhn et al. 2023
        if 'scaled', granulation hyperparameters are scaled based on stellar
            properties using scaling relations given in Luhn et al. 2023
        if 'custom', specify hyperparameters with the gran_hyper_dict argument
    gran_hyperparameters : string, ['solar','scaled','custom']
        if 'solar' (default), adopt solar hyperparameters for granulation 
            kernel as given in Luhn et al. 2023
        if 'scaled', granulation hyperparameters are scaled based on stellar
            properties using scaling relations given in Luhn et al. 2023
        if 'custom', specify hyperparameters with the gran_hyper_dict argument        
    rot_act_hyperparameters : string, ['solar','custom']
        if 'solar' (default), adopt solar hyperparameters for rotation kernel 
            in Luhn et al. 2023 ('matern52') or Langellier et al. 2021 ('qp')
        if 'custom', specify hyperparameters with the rot_act_matern52_hyper_dict
        or rot_act_qp_hyper_dict argument as appropriate
           
    -------
    Returns
    -------
    Sigma : array
        N X N covariance matrix for input observations, where N is the number
        of rows of observations for which observations['star']==star
    """
    if osc_hyperparameters=='solar':
        
        S0=2.36
        w0=0.0195
        Q=7.63
        
    elif osc_hyperparameters=='scaled':
        
        nu_max=calc_nu_max(teff, logg)
        
        Q=10**(0.4862+0.1135*np.log10(nu_max))
        w0=2*np.pi*nu_max/1e6
        Pg=calc_Pg(teff,logg)
        S0=Pg/(4*Q**2)*2.44e6
        
    elif osc_hyperparameters=='custom':
        
        S0, w0, Q = osc_hyper_dict['S0'],osc_hyper_dict['w0'],osc_hyper_dict['Q']
        
    if gran_hyperparameters=='solar':
        
        S1=18.64
        w1=5.22
        S2=0.00485
        w2=0.0173
        
    elif gran_hyperparameters=='scaled':

        nu_max=calc_nu_max(teff, logg)
        
        agran_rv=calc_agran_rv(teff, logg)
        
        w1=2*np.pi*0.317*nu_max**(0.97)/1e6
        S1=(agran_rv)**2*np.sqrt(2)/w1
        w2=2*np.pi*0.948*nu_max**(0.992)/1e6
        S2=(agran_rv)**2*np.sqrt(2)/w2
        
    elif gran_hyperparameters=='custom':
        
        w1, S1, w2, S2 = gran_hyper_dict['w1'],gran_hyper_dict['S1'], gran_hyper_dict['w2'],gran_hyper_dict['S2']
    
    if rot_act_hyperparameters=='solar':

        wn_matern52=0.6
        a0=-0.106884
        a1=-1.154
        lamb=2.52
        
        wn_qp=1.44
        tau_qp=23.6
        prot_qp=28.1
        eta_qp=0.58
        
    elif rot_act_hyperparameters=='custom':
        
        wn_matern52,a0,a1,lamb= rot_act_matern52_hyper_dict['wn_matern52'], rot_act_matern52_hyper_dict['a0'], rot_act_matern52_hyper_dict['a1'], rot_act_matern52_hyper_dict['lambda']
        
        wn_qp,tau_qp,prot_qp,eta_qp= rot_act_qp_hyper_dict['wn_qp'],rot_act_qp_hyper_dict['tau_qp'],rot_act_qp_hyper_dict['prot_qp'],rot_act_qp_hyper_dict['eta_qp']
        
    
    obstime=np.array(observations['start'][observations['star']==star])
    exptime=np.array(observations['exposure_time'][observations['star']==star])
    obstime=obstime-obstime[0]
    
    k_inst = np.zeros((len(obstime),len(obstime)))
    k_photon = np.zeros((len(obstime),len(obstime)))
    k_osc = np.zeros((len(obstime),len(obstime)))            
    k_gran = np.zeros((len(obstime),len(obstime)))
    k_rot = np.zeros((len(obstime),len(obstime)))
    
    if photon:
        photon_noise=np.array(observations['photon_noise'][observations['star']==star])
    if inst:
        wn_inst=np.array(observations['inst_jitter'][observations['star']==star])
    if osc:
        eta=np.sqrt(np.abs(1-1./(4*Q**2)))
    
    for i in range(len(obstime)):
        for j in range(i+1):
            if i==j:
                if inst:
                    k_inst[i,j]=wn_inst[i]**2
                if photon:
                    k_photon[i,j]=photon_noise[i]**2
                if rot_act=='matern52':
                    k_rot[i][j]=wn_matern52**2
                elif rot_act=='qp':
                    k_rot[i][j]=wn_qp**2
                if osc:
                    k_osc[i][j]=osc_covariance(S0,w0,Q,eta,0,exptime[i],exptime[j],overlapping=True)
                if gran:
                    k_gran[i][j]=gran_covariance(S1,w1,S2,w2,0,exptime[i],exptime[j],overlapping=True)
            else:
                delta=np.abs(obstime[i]-obstime[j])
             
                if osc=='Luhn2023':
                    cvr_osc = osc_covariance(S0,w0,Q,eta,delta*86400.,exptime[i],exptime[j],overlapping=False)
                    k_osc[i][j]=cvr_osc  
                    k_osc[j][i]=cvr_osc                
                if gran=='Luhn2023':
                    cvr_gran = gran_covariance(S1,w1,S2,w2,delta*86400.,exptime[i],exptime[j],overlapping=False)
                    k_gran[i][j]=cvr_gran
                    k_gran[j][i]=cvr_gran
                if rot_act=='matern52':
                    cvr_rot = rot_act_covariance_matern52(a0,a1,lamb,delta)
                    k_rot[i][j]=cvr_rot
                    k_rot[j][i]=cvr_rot
                elif rot_act=='qp':
                    cvr_rot=rot_act_covariance_qp(eta_qp,tau_qp,prot_qp,delta)
                    k_rot[i][j]=cvr_rot*wn_qp**2
                    k_rot[j][i]=cvr_rot*wn_qp**2
    Sigma = k_inst+k_photon+k_gran+k_osc+k_rot

    return Sigma
