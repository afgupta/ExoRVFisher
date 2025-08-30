import numpy as np

from scipy import linalg

import covariance

from plot_fisher import plot_progression

from fisher import fisher_progression, construct_parameter_grid


def recast_bool(string):
    """
    Convert config values from string to bool as appropriate
    """
    if string == 'True':
        return True
    elif string == 'False':
        return False
    elif string == 'None':
        return None
    else:
        return string

def extract_configuration_file(configfile):
    configLoader = configparser.ConfigParser()
    configLoader.optionxform = str
    with open(configfile) as cfgFile:
        configLoader.read_file(cfgFile)

    config = {}
    for section in configLoader.sections():
        for key, value in configLoader.items(section):
            config[key] = recast_bool(value)

    return config

if __name__ == "__main__":
    """
    Module to calculate Fisher information as a function of # of observations
    for a specified set of Keplerian orbital parameter given a set of times, RV
    measurement uncertainties, and GP kernel hyperparameters to describe the
    correlated noise contributions from stellar variability.
    
    Required command line arguments:
        survey (string): name of survey or (simulated) data set
        planet (string): name of target planet (must match entry in planetfile)
        run_id (string): customizable tag for each run
        
    Optional command line arguments:
        verbose (-v; bool): if True, prints some useful information
        inputfile (-i; string): input csv file with required columns
                                "star" (target name)
                                "start" (observation start time [days])
                                "exposure_time" (observation duration [seconds])
                                "photon_noise" (photon noise uncertainty [m/s])
                                "instrument" (instrument name)
                                and optional column
                                "inst_jitter" (estimated white noise jitter [m/s])
        starfile (-f; string): csv file with stellar parameter information. See
                               example stellar_params_hypers.csv for columns. The
                               teff and logg columns can remain empty unless 
                               'osc_hyperparameters' or 'gran_hyperparameters'
                               are set to 'scaled' in config. The hyperparameter
                               columns can remain empty unless the corresponding
                               source is set to 'custom' in config.
        planetfile (-t; string): csv file with planet parameter information,
                                 required columns are:
                                "mass" (planet mass [Earth masses])
                                "period" (orbital period [days])
                                "ecc" (orbital eccentricity)
        configfile (-g; string): file specifying a number of configurable
                                 parameters for the Fisher information
                                 calculation - see example file for format and
                                 available parameters
        eccentric (-e, bool): if True, Fisher information is computed using a
                              full Keplerian model instead of a simple sinusoid
        readcovar (-r, bool): if True, reads covariance matrix from existing file
        save (-s, bool): if True, saves the covariance matrix and its inverse
                         to file along with Fisher results and any output plots
        plot (-p, bool): if True, plots the Fisher information as a function of
                         number of observations for each parameter specified in
                         the configuration file
    """
    import os
    import time

    import configparser
    import argparse

    import pandas as pd

    parser = argparse.ArgumentParser(
                    prog = 'RV Fisher Grid',
                    description = 'Calculate Fisher information for radial velocity measurements of Keplerian orbits')
    
    parser.add_argument('survey', type=str)
    parser.add_argument('planet', type=str)
    parser.add_argument('run_id', type=str)
    parser.add_argument('-v', '--verbose', action='store_true', default=True)
    parser.add_argument('-i', '--inputfile', default='example_observations.csv')
    parser.add_argument('-f', '--starfile', default='stellar_params_hypers.csv')
    parser.add_argument('-t', '--planetfile', default='planet_params.csv')
    parser.add_argument('-g', '--configfile', default='fisher_base.config')
    parser.add_argument('-e', '--eccentric', action='store_true', default=False)
    # parser.add_argument('-l', '--parallel', action='store_true', default=False)
    # parser.add_argument('-t', '--nthreads', default=4)
    parser.add_argument('-r', '--readcovar', action='store_true', default=False)
    parser.add_argument('-s', '--save', action='store_true', default=False)
    parser.add_argument('-p', '--plot', action='store_true', default=False)
    args = parser.parse_args()

    config = extract_configuration_file(os.path.join('config',args.configfile))

    survey = args.survey
    planet = args.planet
    run_id = args.run_id
    verbose = args.verbose
    eccentric = args.eccentric
    # parallel = args.parallel
    # nthreads = args.nthreads
    inputfile = args.inputfile
    starfile = args.starfile
    planetfile = args.planetfile
    read_covar = args.readcovar
    save_results = args.save
    plot_results = args.plot

    outdir = survey+'_results'
    if os.path.isdir(outdir)==False:
        os.mkdir(outdir)
        if verbose:
            print('Creating output directory %s.' % outdir)
    elif verbose:
        print("%s output directory already exists." % outdir)

    if save_results:
        with open(os.path.join(outdir, survey+'_'+planet+'_'+run_id+'_prog_log.txt'), 'w') as f:
            print(args, file=f)
            print(config, file=f)

    observations = pd.read_csv(os.path.join('survey_files', inputfile))
    observations = observations.sort_values('start')

    stellar_params = pd.read_csv(os.path.join('params', starfile))
    planet_params = pd.read_csv(os.path.join('params', planetfile))

    try:
        ixp = np.where(planet_params.planet==planet)[0][0]
    except IndexError:
        print('No information provided in %s for planet %s.' % (planetfile, planet))

    mass = np.array(planet_params.mass)[ixp]
    period=np.array(planet_params.period)[ixp]
    ecc=np.array(planet_params.ecc)[ixp] 

    uncertainties={}
    n_obs_arr={}

    for star in np.unique(observations.star):
        if verbose:
            print('---------\n'+star+'\n---------')
        try:
            ixs = np.where(stellar_params.star==star)[0][0]
        except IndexError:
            print('No information provided in %s for star %s.' % (starfile, star))
        
        mstar = np.array(stellar_params.mstar)[ixs]
        teff = np.array(stellar_params.teff)[ixs]
        logg = np.array(stellar_params.logg)[ixs]
        
        S0, w0, Q = stellar_params.S0[ixs], stellar_params.w0[ixs], stellar_params.Q[ixs]
        w1, S1, w2, S2 = stellar_params.w1[ixs], stellar_params.S1[ixs], stellar_params.w2[ixs], stellar_params.S2[ixs]
        wn_qp, tau_qp, prot_qp, eta_qp = stellar_params.wn_qp[ixs], stellar_params.tau_qp[ixs], stellar_params.prot_qp[ixs], stellar_params.eta_qp[ixs]
        wn_matern52, a0, a1, lamb = stellar_params.wn_matern52[ixs], stellar_params.a0[ixs], stellar_params.a1[ixs], stellar_params.lamb[ixs]
        
     

        observations_star = observations[observations.star==star].reset_index(drop=True)
        
        obstime = np.array(observations_star.start)
        instruments = np.array(observations_star.instrument)
        
        # If instrument jitter not provided in input file, read from config
        if 'inst_jitter' not in observations_star.columns:
            inst_jitter=np.zeros(len(observations_star))
            for i, inst in enumerate(np.unique(observations_star.instrument)):
                try:
                    inst_jitter[observations_star.instrument==inst]=float(config[inst+'_jitter'])
                except KeyError:
                    print("Jitter value for %s not provided in config file. Defaulting to 0" % inst)
            observations_star['inst_jitter']=inst_jitter
    
        
        n_obs = len(observations_star)
        
        nobs_resolution=int(config['nobs_resolution'])
        min_obs=int(config['min_obs'])
        
        n_obs_to_use=np.arange(min_obs,n_obs+nobs_resolution,nobs_resolution, dtype=int)
        
        if read_covar==True and os.path.isfile(os.path.join(outdir, survey+'_'+star+'_covar_'+run_id+'.npy')):
            Sigma = np.load(os.path.join(outdir,survey+'_'+star+'_covar_'+run_id+'.npy'))
            
            if verbose:
                print('Covariance read from file.')
            
        else:
            
            covar_start = time.time()
            
            Sigma = covariance.Covariance(star, observations_star,
                                          photon = config['phot'], inst = config['inst'],
                                          osc = config['osc'], gran = config['gran'], rot_act = config['rot_act'],
                                          teff=teff, logg=logg,
                                          osc_hyperparameters=config['osc_hyperparameters'],
                                          gran_hyperparameters=config['gran_hyperparameters'],
                                          rot_act_hyperparameters=config['rot_act_hyperparameters'],
                                          osc_hyper_dict={'S0':S0,'w0':w0,'Q':Q},
                                          gran_hyper_dict={'S1':S1,'w1':w1,'S2':S2,'w2':w2},
                                          rot_act_qp_hyper_dict={'wn_qp':wn_qp,'tau_qp':tau_qp,'prot_qp':prot_qp,'eta_qp':eta_qp},
                                          rot_act_matern52_hyper_dict={'wn_matern52':wn_matern52,'a0':a0,'a1':a1,'lambda':lamb}
                                          )
            
            if save_results:
                np.save(os.path.join(outdir,survey+'_'+star+'_covar_'+run_id+'.npy'),Sigma)
        
            if verbose:
                print('Covariance matrix computed in %.2f s.' % float(time.time()-covar_start))        
        
        if not config['phot']:
            # Without this padding (equivalent to 1 mm/s), the matrix inversion
            # may throw errors due to being ill-conditioned
            di = np.diag_indices_from(Sigma)
            Sigma[di] += 1e-6
        
        m_basis, p_basis, phi_basis, ecc_basis, omega_basis = construct_parameter_grid(config, eccentric=eccentric, verbose=verbose)

        
        fisher_start = time.time()
        
        uncertainties[star] = fisher_progression(n_obs_to_use, Sigma, obstime, instruments, mstar,
                               mass, period, phi_basis,
                               ecc=ecc, omega_basis=omega_basis, eccentric=eccentric)
        n_obs_arr[star] = n_obs_to_use
    
        if verbose:
            print('Fisher information computed in %.2f s.' % float(time.time()-fisher_start))
    
        Ck=np.array(uncertainties[star][0], dtype=float)
        Cp=np.array(uncertainties[star][1], dtype=float)
        Ch=np.array(uncertainties[star][2], dtype=float)
        if eccentric:
            Ce=np.array(uncertainties[star][3], dtype=float)
            Cw=np.array(uncertainties[star][4], dtype=float)
    
    
        if save_results:
            np.save(os.path.join(outdir,survey+'_'+star+'_k_uncertainties_'+planet+'_'+run_id+'.npy'), np.sqrt(Ck))
            np.save(os.path.join(outdir,survey+'_'+star+'_p_uncertainties_'+planet+'_'+run_id+'.npy'), np.sqrt(Cp))    
            np.save(os.path.join(outdir,survey+'_'+star+'_phi_uncertainties_'+planet+'_'+run_id+'.npy'), np.sqrt(Ch))
            if eccentric:
                np.save(os.path.join(outdir,survey+'_'+star+'_ecc_uncertainties_'+planet+'_'+run_id+'.npy'), np.sqrt(Ce))
                np.save(os.path.join(outdir,survey+'_'+star+'_omega_uncertainties_'+planet+'_'+run_id+'.npy'), np.sqrt(Cw))
        
    plot_K=config['plot_K']
    plot_period=config['plot_period']
    plot_ecc=config['plot_ecc']
    
    
    if plot_results:
        
        plot_progression(outdir, survey, np.unique(observations.star), planet,run_id,
                         uncertainties, n_obs_arr, {'mass':mass,'period':period,'ecc':ecc},
                         plot_K = plot_K, plot_period=plot_period, plot_ecc=plot_ecc,
                         eccentric=eccentric, verbose=verbose, save=save_results)
        
