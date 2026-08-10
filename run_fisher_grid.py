import numpy as np

from scipy import linalg

import covariance

from plot_fisher import plot_sensitivity

from fisher import fisher_grid_parallel, fisher_grid_series, fisher_grid_series_two, construct_parameter_grid


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
    Module to calculate Fisher information over a grid of Keplerian orbital
    parameter given a set of times, RV measurement uncertainties, and GP kernel
    hyperparameters to describe the correlated noise contributions from stellar
    variability.
    
    Required command line arguments:
        survey (string): name of survey or (simulated) data set
        star (string): name of the target star (must match entry in starfile)
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
        configfile (-g; string): file specifying a number of configurable
                                 parameters for the Fisher information
                                 calculation - see example file for format and
                                 available parameters
        eccentric (-e, bool): if True, Fisher information is computed using a
                              full Keplerian model instead of a simple sinusoid
        parallel (-l, bool): if True, enables multithreading to calculate Fisher
                             information for multiple parameter combinations in
                             parallel
        nthreads (-t, int): number of threads to use for parallel computation
        readcovar (-r, bool): if True, reads covariance matrix from existing file
        save (-s, bool): if True, saves the covariance matrix and its inverse
                         to file along with Fisher results and any output plots
        plot (-p, bool): if True, plots Fisher information results for the
                         input parameter grid in mass-period space - plot options
                         are specified in the configuration file
    """
    import os
    import time

    import configparser
    import argparse

    import pandas as pd
    
    from multiprocessing import Pool as ThreadPool
    from functools import partial
            
    parser = argparse.ArgumentParser(
                    prog = 'RV Fisher Grid',
                    description = 'Calculate Fisher information for radial velocity measurements of Keplerian orbits')
    
    parser.add_argument('survey', type=str)
    parser.add_argument('star', type=str)
    parser.add_argument('run_id', type=str)
    parser.add_argument('-v', '--verbose', action='store_true', default=True)
    parser.add_argument('-i', '--inputfile', default='example_observations.csv')
    parser.add_argument('-f', '--starfile', default='stellar_params_hypers.csv')
    parser.add_argument('-g', '--configfile', default='fisher_base.config')
    parser.add_argument('-e', '--eccentric', action='store_true', default=False)
    parser.add_argument('-l', '--parallel', action='store_true', default=False)
    parser.add_argument('-k', '--nknown', default=0)
    parser.add_argument('-t', '--nthreads', default=4)
    parser.add_argument('-r', '--readcovar', action='store_true', default=False)
    parser.add_argument('-s', '--save', action='store_true', default=False)
    parser.add_argument('-p', '--plot', action='store_true', default=False)
    args = parser.parse_args()

    config = extract_configuration_file(os.path.join('config',args.configfile))

    survey = args.survey
    star = args.star
    run_id = args.run_id
    verbose = args.verbose
    parallel = args.parallel
    eccentric = args.eccentric
    nknown = args.nknown
    nthreads = args.nthreads
    inputfile = args.inputfile
    starfile = args.starfile
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
        with open(os.path.join(outdir, survey+'_'+star+'_'+run_id+'_grid_log.txt'), 'w') as f:
            print(args, file=f)
            print(config, file=f)

    observations = pd.read_csv(os.path.join('survey_files', inputfile))
    observations = observations.sort_values('start')
    
    stellar_params = pd.read_csv(os.path.join('params', starfile))

    try:
        ixs = np.where(stellar_params.star==star)[0][0]
    except IndexError:
        print('No information provided in %s for star %s.' % (starfile, star))
        exit()
    
    mstar = np.array(stellar_params.mstar)[ixs]
    teff = np.array(stellar_params.teff)[ixs]
    logg = np.array(stellar_params.logg)[ixs]
    
    S0, w0, Q = stellar_params.S0[ixs], stellar_params.w0[ixs], stellar_params.Q[ixs]
    w1, S1, w2, S2 = stellar_params.w1[ixs], stellar_params.S1[ixs], stellar_params.w2[ixs], stellar_params.S2[ixs]
    wn_qp, tau_qp, prot_qp, eta_qp = stellar_params.wn_qp[ixs], stellar_params.tau_qp[ixs], stellar_params.prot_qp[ixs], stellar_params.eta_qp[ixs]
    wn_matern52, a0, a1, lamb = stellar_params.wn_matern52[ixs], stellar_params.a0[ixs], stellar_params.a1[ixs], stellar_params.lamb[ixs]
    
    
    observations_star = observations[observations.star==star].reset_index(drop=True)
    if len(observations_star)==0:
        print("The observations file %s contains no entries for star %s" % (inputfile, star))
        exit()
    
    obstime = np.array(observations_star.start)
    
    # If instrument jitter not provided in input file, read from config
    if 'inst_jitter' not in observations_star.columns:
        inst_jitter=np.zeros(len(observations_star))
        for i, inst in enumerate(np.unique(observations_star.instrument)):
            try:
                inst_jitter[observations_star.instrument==inst]=float(config[inst+'_jitter'])
            except KeyError:
                print("Jitter value for %s not provided in config file. Defaulting to 0" % inst)
        observations_star['inst_jitter']=inst_jitter
    
    n_inst = len(np.unique(observations_star.instrument))
    n_obs = len(observations_star)
    
    # create a binary matrix that serves as the derivative w.r.t. the
    # systemic velocity measured by each instrument
    # (derivative is always either 0 or 1)
    gamma_mask = np.zeros((n_inst, n_obs))
    for i, inst in enumerate(np.unique(observations_star.instrument)):
        gamma_mask[i][np.array(observations_star.instrument)==inst]=1
            
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
    
    if read_covar==True and os.path.isfile(os.path.join(outdir, survey+'_'+star+'_invcovar_'+run_id+'.npy')):
        Sigma_inv = np.load(os.path.join(outdir,survey+'_'+star+'_invcovar_'+run_id+'.npy'))
    else:
        
        invcovar_start = time.time()
        
        Identity=np.identity(n_obs)
        Sigma_inv = linalg.solve(Sigma, Identity)
        
        if save_results:
            np.save(os.path.join(outdir,survey+'_'+star+'_invcovar_'+run_id+'.npy'),Sigma_inv)

        if verbose:
            print('Covariance matrix inverted in %.2f s.' % float(time.time()-invcovar_start))
    
    
    m_basis, p_basis, phi_basis, ecc_basis, omega_basis = construct_parameter_grid(config, eccentric=eccentric, verbose=verbose)

    
    fisher_start = time.time()
    
    if parallel:
        p_args={'obstime':obstime, 'gamma_mask':gamma_mask,
              'isigma':Sigma_inv, 'mstar':mstar, 'eccentric':eccentric,
              'p_basis':p_basis,'phi_basis':phi_basis,
              'ecc_basis':ecc_basis,'omega_basis':omega_basis,
              'verbose':verbose} 
        func = partial(fisher_grid_parallel, p_args)

        pool=ThreadPool(nthreads)
        
        uncertainties = np.array(list(pool.map(func, m_basis)), dtype=object)
        
        if eccentric:
            uncertainties = np.transpose(uncertainties, (1,0,2,3,4,5))
        else:
            uncertainties = np.transpose(uncertainties, (1,0,2,3))
            
    else:
        
        if int(nknown)==0:
            uncertainties = fisher_grid_series(Sigma_inv, obstime, gamma_mask, mstar,
                                   m_basis, p_basis, phi_basis, ecc_basis=ecc_basis, omega_basis=omega_basis,
                                   eccentric=eccentric)
        elif int(nknown)==1:
            
            params_0 = np.array([float(config['K_0']),float(config['period_0']),float(config['phi_0']),float(config['ecc_0']),float(config['omega_0'])])   
            
            uncertainties = fisher_grid_series_two(Sigma_inv, obstime, gamma_mask, mstar,
                                   m_basis, p_basis, phi_basis, params_0, ecc_basis=ecc_basis, omega_basis=omega_basis,
                                   eccentric=eccentric)
        else:
            print('Not yet implemented for >=1 known planet.')

    Ck=np.array(uncertainties[0], dtype=float)
    Cp=np.array(uncertainties[1], dtype=float)
    Ch=np.array(uncertainties[2], dtype=float)
    if eccentric:
        Ce=np.array(uncertainties[3], dtype=float)
        Cw=np.array(uncertainties[4], dtype=float)
    
    if verbose:
        print('Fisher information computed in %.2f s.' % float(time.time()-fisher_start))

    if save_results:
        np.save(os.path.join(outdir,survey+'_'+star+'_k_uncertainties_grid_'+run_id+'.npy'), np.sqrt(Ck))
        np.save(os.path.join(outdir,survey+'_'+star+'_p_uncertainties_grid_'+run_id+'.npy'), np.sqrt(Cp))    
        np.save(os.path.join(outdir,survey+'_'+star+'_phi_uncertainties_grid_'+run_id+'.npy'), np.sqrt(Ch))
        if eccentric:
            np.save(os.path.join(outdir,survey+'_'+star+'_ecc_uncertainties_grid_'+run_id+'.npy'), np.sqrt(Ce))
            np.save(os.path.join(outdir,survey+'_'+star+'_omega_uncertainties_grid_'+run_id+'.npy'), np.sqrt(Cw))
        
    if plot_results:
        
        plot_contours = config['plot_contours']
        plot_colormap = config['plot_colormap']
        
        plot_K_lines = config['plot_K_lines']
        plot_known_exoplanets = config['plot_known_exoplanets']
        plot_ss_planets = config['plot_ss_planets']
        
        ecc_index=int(config['ecc_index'])
        
        contour_levels = np.array(config['contour_levels'].split(','), dtype=float)
    
        plot_sensitivity(outdir, survey, star, run_id, m_basis, p_basis, np.sqrt(Ck),
                         plot_contours, plot_colormap, contour_levels=contour_levels,
                         mstar=mstar, eccentric=eccentric, ecc_index=ecc_index,
                         plot_ss_planets=plot_ss_planets,plot_known_exoplanets=plot_known_exoplanets, plot_K_lines=plot_K_lines,
                         save=save_results, verbose=verbose)