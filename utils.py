import numpy as np
import astropy.time


def jd2datetime(times):
    return np.array([astropy.time.Time(time,format="jd",scale="utc").datetime for time in times])

def mass_from_orbit(P, K, mstar=1.0, ecc=0.0, units='g'):
    msun=1.9885e33
    mearth=5.97236e27
    G=6.67408e-8
    
    mp=K*1e2*(mstar*msun)**(2./3.)*(P*86400)**(1./3.)*(1-ecc**2)**0.5/(2*np.pi*G)**(1./3.)

    if units=='g':
        return mp
    elif units=='mearth':
        return mp/mearth
    elif units=='msun':
        return mp/msun
    else:
        print('Allowed units are "g", "mearth", "msun". Returning mass in grams.')
        return mp

def K_from_mass(mp, P, mstar=1.0, ecc=0.0, units='ms'):
    msun=1.9885e33
    mearth=5.97236e27
    G=6.67408e-8   
    
    if units=='ms':
        K=0.01*(2*np.pi*G)**(1./3.)*mp*mearth*(mstar*msun+mp*mearth)**(-2./3.)*(P*86400)**(-1./3.)/np.sqrt(1-ecc**2)
    elif units=='cms':
        K=(2*np.pi*G)**(1./3.)*mp*mearth*(mstar*msun+mp*mearth)**(-2./3.)*(P*86400)**(-1./3.)/np.sqrt(1-ecc**2)
    
    return K