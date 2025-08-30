import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

import utils


def plot_sensitivity(outdir, survey, star, run_id, m_basis, p_basis, k_uncertainties,
                     plot_contours, plot_colormap, contour_levels=[3,5,10],
                     mstar=1.0, eccentric=False, ecc_index=0,
                     plot_ss_planets=True,plot_known_exoplanets=False, plot_K_lines=False,
                     contour_color='xkcd:crimson', 
                     verbose=False, save=False, plotformat='png'):
    
    K_grid=np.zeros((len(m_basis), len(p_basis)))
    P_grid=np.zeros((len(m_basis), len(p_basis)))
    for k, mass in enumerate(m_basis):
        for p, period in enumerate(p_basis):
            K_grid[k][p] = utils.K_from_mass(mass, period, mstar=mstar)
            P_grid[k][p] = period
    
    if eccentric:
        if ecc_index==-1:
            sigma_K = np.mean(k_uncertainties, axis=(2,3,4))
        else:
            sigma_K = np.mean(k_uncertainties, axis=(2,4))[:,:,ecc_index]
    else:
        sigma_K = np.mean(k_uncertainties, axis=(2))
        
    sensitivity=K_grid/sigma_K

    if plot_colormap:
        fig, ax = plt.subplots(1,1,figsize=(8,5))
    else:
        fig, ax = plt.subplots(1,1,figsize=(7,5))
        
    ax.tick_params(length=8, width=1.6, labelsize=15, direction='out')
    for xx in ['top','bottom','left','right']:
        ax.spines[xx].set_linewidth(2.0)

    if plot_colormap:
        
        contour_color='w'
        
        magma = cm.get_cmap('magma', 225)
        magma2=ListedColormap(magma(np.linspace(0,1,225)**0.75)[5:])
        cax=ax.pcolor(p_basis, m_basis, 1./sensitivity, cmap=magma2, vmin=0,vmax=1., shading='auto')
        
        
        cbar = plt.colorbar(cax, ticks=[0.01,0.1,0.2,0.25,0.333,0.5,1.0])
        cbar.ax.set_yticklabels(['100','10','5','4','3','2','1'])
        cbar.ax.invert_yaxis()
        cbar.set_label(label='$K/\\sigma_K$',fontsize=18, rotation=90, color='k', labelpad=-10)
        cbar.ax.tick_params(labelsize=15, width=1.5, length=15, direction='out', color='k', colors='k')
        cbar.outline.set_linewidth(2)
        cbar.outline.set_edgecolor('k')
        

    if plot_contours:

        contours=ax.contour(p_basis, m_basis, sensitivity, np.array(contour_levels), colors=contour_color, linestyles='-', linewidths=1.2, zorder=2)
        contours.levels=np.array(contour_levels, dtype=str)
        for l in range(len(contours.levels)):
            contours.levels[l]=str(int(float(contours.levels[l])))+'$-\\sigma$'
    
    if plot_ss_planets:
        ss_color='lightskyblue'
        
        ss_planet_periods=np.array([0.24,0.615,1.0,1.88,11.86,29.45,84,164.8])
        ss_planet_masses=np.array([0.0553,0.815,1.0,0.1075,317.8,95.2,14.6,17.2])

        ax.scatter(365.*ss_planet_periods,ss_planet_masses,marker=r'$\odot$', s=100, color=ss_color, label='Solar System Planets')


    ax.set_yscale('log')
    ax.set_xscale('log')
    
    ax.set_xlim(np.min(p_basis), np.max(p_basis))
    ax.set_ylim(np.min(m_basis), np.max(m_basis))
    
    ax.set_xlabel('Period [days]', fontsize=15)
    ax.set_ylabel('$m_p\\ \\sin i$ [$M_\\oplus$]', fontsize=15)
    
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(outdir,survey+'_'+star+'_sensitivity_'+run_id+'.'+plotformat), format=plotformat, dpi=400,bbox_inches='tight')
        if verbose:
            print('Figure saved to file '+os.path.join(outdir,survey+'_'+star+'_sensitivity_'+run_id+'.'+plotformat))
    plt.show()
    
    
def plot_progression(outdir, survey, stars, planet, run_id, 
                     parameter_uncertainties, n_obs_arr, input_params,
                     plot_K=True, plot_period=False, plot_ecc=False,
                     eccentric=False,
                     verbose=False, save=False, plotformat='png'):
    
    n_panels = int(plot_K)+int(plot_period)+int(plot_ecc)
    if n_panels==0: 
        print("Must choose at least one parameter to plot.")
        return 0
    if plot_ecc and not eccentric:
        n_panels-=1
    
    if n_panels==1:
        fig, ax1 = plt.subplots(1,n_panels,figsize=(n_panels*4,5))
        axes=[ax1]
    elif n_panels==2:
        fig, (ax1,ax2) = plt.subplots(1,n_panels,figsize=(n_panels*4,5))
        axes=[ax1,ax2]
    elif n_panels==3:
        fig, (ax1,ax2,ax3) = plt.subplots(1,n_panels,figsize=(n_panels*4,5))
        axes=[ax1,ax2,ax3]

    max_visits=0
    for arr in n_obs_arr.values():
        max_visits=np.max([max_visits, np.max(arr)])
    
    for ax in axes:
        ax.tick_params(length=8, width=1.6, labelsize=15, direction='out')
        for xx in ['top','bottom','left','right']:
            ax.spines[xx].set_linewidth(2.0)
            
        ax.set_xlim(0,max_visits)
        ax.set_xlabel('Number of Visits', fontsize=15)

    
    ax1.text(0.6,0.93,'$m_p = $%.1f $M_\\oplus$' % input_params['mass'], transform=ax1.transAxes, fontsize=12)
    ax1.text(0.6,0.85,'$P = $%.1f d' % input_params['period'], transform=ax1.transAxes, fontsize=12)
    if eccentric:
        ax1.text(0.6,0.77,'$e = $%.2f' % input_params['ecc'], transform=ax1.transAxes, fontsize=12)
    else:
        ax1.text(0.6,0.77,'$e = 0$', transform=ax1.transAxes, fontsize=12)
    
    i=0 # panel index
    
    if plot_K:
        for star in stars:
            Ck=np.array(parameter_uncertainties[star][0], dtype=float)
            if eccentric:
                sigma_K = np.mean(np.sqrt(Ck), axis=(1,2))
            else:
                sigma_K = np.mean(np.sqrt(Ck), axis=(1))
            axes[i].plot(n_obs_arr[star], sigma_K)
        
        axes[i].set_yscale('log')

        axes[i].set_ylabel('$\\sigma_K$ [m/s]', fontsize=15)

        
        i+=1
        
    if plot_period:
        for star in stars:
            Cp=np.array(parameter_uncertainties[star][1], dtype=float)
            if eccentric:
                sigma_p = np.mean(np.sqrt(Cp), axis=(1,2))
            else:
                sigma_p = np.mean(np.sqrt(Cp), axis=(1))
                
                
            axes[i].plot(n_obs_arr[star], sigma_p)
        
        axes[i].set_yscale('log')
        
        axes[i].set_ylabel('$\\sigma_P$ [d]', fontsize=15)

        
        i+=1
        
    if eccentric and plot_ecc:
        for star in stars:
            Ce=np.array(parameter_uncertainties[star][3], dtype=float)
            sigma_e = np.mean(np.sqrt(Ce), axis=(1,2))

            axes[i].plot(n_obs_arr[star], sigma_e)

        axes[i].set_yscale('log')
        
        axes[i].set_ylabel('$\\sigma_e$', fontsize=15)
    
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(outdir,survey+'_fisher_progress_'+planet+'_'+run_id+'.'+plotformat), format=plotformat, dpi=400,bbox_inches='tight')
        if verbose:
            print('Figure saved to file '+os.path.join(outdir,survey+'_fisher_progress_'+planet+'_'+run_id+'.'+plotformat))
            
    plt.show()
