import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit
import fitsio
from astropy.table import Table
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn

from superbit_lensing import utils
from superbit_lensing.match import MatchedTruthCatalog

import pudb

parser = ArgumentParser()

parser.add_argument('shear_file', type=str,
                    help='Shear catalog filename')
parser.add_argument('truth_file', type=str,
                    help='Truth catalog filename')
parser.add_argument('-run_name', type=str, default=None,
                    help='Name of simulation run')
parser.add_argument('-outdir', type=str, default=None,
                    help='Output directory for plots')
parser.add_argument('--show', action='store_true', default=False,
                    help='Turn on to display plots')
parser.add_argument('--vb', action='store_true', default=False,
                    help='Turn on for verbose prints')

def get_image_shape():
    '''Returns the SB single-epoch image shape (9568, 6380)'''

    return (9568, 6380)

def cut_cat_by_radius(cat, radius, xtag='X_IMAGE', ytag='Y_IMAGE'):
    assert radius > 0
    Nx, Ny = get_image_shape()
    xcen, ycen = Nx // 2, Ny // 2
    assert xcen > 0
    assert ycen > 0

    try:
        obj_radius = np.sqrt((cat[xtag]-xcen)**2 + (cat[ytag]-ycen)**2)
    except KeyError:
        xtag = xtag.lower()
        ytag = ytag.lower()
        obj_radius = np.sqrt((cat[xtag]-xcen)**2 + (cat[ytag]-ycen)**2)

    return cat[obj_radius < radius]

def cut_cats_by_radius(cat1, cat2, radius, xtag1='x_image', ytag1='y_image',
                       xtag2='X_IMAGE', ytag2='Y_IMAGE'):
    '''
    Same as above, but for large samples the radial cut can cause differences
    in cat length due to astrometric errors
    '''

    assert radius > 0
    Nx, Ny = get_image_shape()
    xcen, ycen = Nx // 2, Ny // 2
    assert xcen > 0
    assert ycen > 0

    obj_radius_1 = np.sqrt((cat1[xtag1]-xcen)**2 + (cat1[ytag1]-ycen)**2)
    obj_radius_2 = np.sqrt((cat2[xtag2]-xcen)**2 + (cat2[ytag2]-ycen)**2)

    good = np.where((obj_radius_1 < radius) & (obj_radius_2 < radius))
    return (cat1[good], cat2[good])

def plot_true_shear_dist(match, size=(8,8), outfile=None, show=None,
                         run_name=None, selection=None, radial_cut=None,
                         fontsize=14):

    true = match.true

    if selection is not None:
        true = true[selection]

    if radial_cut is not None:
        true = cut_cat_by_radius(true, radial_cut)

    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=size)

    plt.rcParams.update({'font.size': fontsize})

    bins = np.linspace(-.2, .2, 100)

    plt.hist(true['nfw_g1'], bins=bins, histtype='step', lw=2, label='g1', alpha=0.8)
    plt.hist(true['nfw_g2'], bins=bins, histtype='step', lw=2, label='g2', alpha=0.8)
    plt.axvline(0, lw=2, ls='--', c='k')

    plt.legend()
    plt.xlabel(r'True $\gamma$')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.title(f'{run_name} True Shear')

    plt.gcf().set_size_inches(size)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

    return

def plot_responses_by_class(match, size=(8,8), outfile=None, show=False,
                         run_name=None, radial_cut=None, fontsize=14):
    '''
    Plot the distribution of shear responses for both stars
    and galaxies
    '''

    shear = match.meas

    if radial_cut is not None:
        shear = cut_cat_by_radius(shear, radial_cut)

    print('shear columns: ',shear.columns)
    is_star = np.where(shear['redshift'] == 0)

    objs = {
        'star': shear[is_star],
        'gal': shear[not is_star],
        'foreground': shear[(not is_star) and
                            (shear['nfw_g1'] == 0) and
                            (shear['nfw_g2'] == 0)],
        'background': shear[(not is_star) and
                            (shear['nfw_g1'] != 0) or
                            (shear['nfw_g2'] != 0)]
    }

    plt.rcParams.update({'font.size': fontsize})
    bins = np.linspace(-3, 3, 100)

    R = np.mean([shear['r11'], shear['r22']], axis=0)
    plt.hist(
        R, bins=bins, histtype='step', alpha=0.8, label='All', lw=2
        )

    plt.axvline(0, lw=2, c='k', ls='--')

    for obj, cat in objs.items():
        R = np.mean([cat['r11'], cat['r22']], axis=0)

        plt.hist(
            R, bins=bins, histtype='step', alpha=0.8, label=obj, lw=2
            )

    if run_name is None:
        p = ''
    else:
        p = f'{run_name} '

    plt.legend()
    plt.xlabel('Response')
    plt.ylabel('Counts')
    plt.title(f'{p}Object Responses')

    plt.gcf().set_size_inches(size)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

    return

def plot_shear_responses(match, size=(8,8), outfile=None, show=False,
                         run_name=None, radial_cut=None, fontsize=14,
                         selection=None, label=None, close=True):
    '''
    Plot the distribution of shear responses for both stars
    and galaxies
    '''

    shear = match.meas

    if selection is not None:
        shear = shear[selection]

    if radial_cut is not None:
        shear = cut_cat_by_radius(shear, radial_cut)

    R = {'r11':None, 'r12': None, 'r21':None, 'r22': None}

    plt.rcParams.update({'font.size': fontsize})

    bins = np.linspace(-3, 3, 100)

    for key in R.keys():
        R[key] = shear[key]
        plt.hist(
            R[key], bins=bins, histtype='step', alpha=0.8, label=key, lw=2
            )

    plt.axvline(0, lw=2, c='k', ls='--')

    if run_name is None:
        p = ''
    else:
        p = f'{run_name} '

    plt.legend()
    plt.xlabel('Response')
    plt.ylabel('Counts')
    plt.title(f'{p}Object Responses')

    plt.gcf().set_size_inches(size)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()

    if close is True:
        plt.close()

    return

def density_scatter(x, y, ax=None, fig=None, sort=True, bins=20, s=2,
                    **kwargs):
    '''
    Scatter plot colored by 2d histogram
    '''

    # sometimes x or y can be a masked column
    try:
        x = x.filled()
    except:
        pass
    try:
        y = y.filled()
    except:
        pass

    if ax is None:
        fig , ax = plt.subplots()
    else:
        if fig is None:
            raise Exception('Must pass both fig and ax if you pass one!')
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True )
    z = interpn(
        (0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1])), data, np.vstack([x,y]).T,
        method='splinef2d', bounds_error=False
        )

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, s=s, **kwargs)

    norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
    cbar.ax.set_ylabel('Density')

    return ax

def plot_shear_calibration(match, size=(16,6), outfile=None, show=False,
                           run_name=None, gridsize=50, fontsize=14,
                           radial_cut=None, limits=None, selection=None,
                           prefix=None, **kwargs):
    '''
    Plot the shear calibration (multiplicative & additive)
    for both stars and galxies
    '''

    truth = match.true
    shear = match.meas

    if selection is not None:
        truth = truth[selection]
        shear = shear[selection]

    if radial_cut is not None:
        # truth = cut_cat_by_radius(truth, radial_cut)
        # shear = cut_cat_by_radius(shear, radial_cut)

        truth, shear = cut_cats_by_radius(truth, shear, radial_cut)

        if len(truth) != len(shear):
            raise Exception(f'len(truth) = {len(truth)} but ' +\
                            f'len(shear) = {len(shear)}')

    gtrue = {}
    gtrue['g1'] = truth['nfw_g1']
    gtrue['g2'] = truth['nfw_g2']

    gmeas= {}
    gmeas['g1'] = shear['g1_Rinv']
    gmeas['g2'] = shear['g2_Rinv']

    plt.rcParams.update({'font.size': fontsize})

    k = 1
    plt.close()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    for i in range(2):
        # plt.hexbin(g1_true, g1_meas, gridsize=gridsize)
        # plt.axhline(0, lw=2, c='k', ls='--')
        # ax = plt.gca()
        ax = axes[i]
        x, y = gtrue[f'g{k}'], gmeas[f'g{k}']
        density_scatter(x, y, ax=ax, fig=fig, **kwargs)

        # Get linear fit to scatter points
        # m, b = np.polyfit(x, y, 1)
        # b, m = Polynomial.fit(x, y, 1)
        m, m_err, c, c_err = compute_shear_bias(x, y, None, f'g{k}')

        # Compute m, b directly
        mean_gtrue = np.mean(gtrue[f'g{k}'])
        mean_gmeas = np.mean(gmeas[f'g{k}'])
        # mean_Rinv = 1. / (np.mean(shear['r11']) + np.mean(shear['r22']))

        # TODO: update for additive bias
        m_est = (mean_gmeas / mean_gtrue) - 1.

        if limits is None:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
        else:
            lims = [
                np.min(limits),
                np.max(limits)
            ]
            xlim = limits[0]
            ylim = limits[1]

        # ax.plot([np.min(x), np.max(x)], [np.min(y), np.max(y)], 'k-')
        ax.plot(lims, np.poly1d((m,c))(lims), lw=2, ls='--', c='r', label=f'm={m:.3f}; c={c:.3f}')
        ax.plot(lims, lims, 'k-', label='x=y')
        ax.legend(fontsize=12)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(rf'True $\gamma_{k}$')
        ax.set_ylabel(rf'Meas $\gamma_{k}$' + ' ($\gamma^i / <R^i>$)')
        # ax.set_title(f'<g{k}_meas> ~ (1 + {m_est:.4f}) * <g{k}_true>')
        k += 1

    plt.gcf().set_size_inches(size)

    if run_name is None:
        p = ''
    else:
        p = f'{run_name} '

    if prefix is not None:
        p += (prefix + ' ')

    plt.suptitle(f'{p}Shear Bias')

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

    return

def compute_binned_bias():
    pass

def plot_binned_shear_calibration(match, bins=None, size=(16,6), outfile=None,
                           run_name=None, gridsize=50, fontsize=14,
                           radial_cut=None, limits=None, selection=None,
                           prefix=None, show=False, **kwargs):
    '''
    Plot the shear calibration binned in true shear
    '''

    truth = match.true
    shear = match.meas

    if selection is not None:
        truth = truth[selection]
        shear = shear[selection]

    if radial_cut is not None:

        truth, shear = cut_cats_by_radius(truth, shear, radial_cut)

        if len(truth) != len(shear):
            raise Exception(f'len(truth) = {len(truth)} but ' +\
                            f'len(shear) = {len(shear)}')

    gtrue = {}
    gtrue['g1'] = truth['nfw_g1']
    gtrue['g2'] = truth['nfw_g2']

    gmeas= {}
    gmeas['g1'] = shear['g1_Rinv']
    gmeas['g2'] = shear['g2_Rinv']

    # bin data to prep for fitting
    if bins is None:
        gmin, gmax = -0.1, 0.1
        Nbins = 11
        bins = np.linspace(gmin, gmax, Nbins)

    bindx = {}
    bindx['g1'] = np.digitize(gtrue['g1'], bins)
    bindx['g2'] = np.digitize(gtrue['g2'], bins)

    plt.rcParams.update({'font.size': fontsize})
    fig, axes = plt.subplots(nrows=1, ncols=2)

    for k, g in enumerate(['g1', 'g2']):
        ax = axes[k]

        gmean = []
        gstd  = []
        for i in range(1, len(bins)):
            x = gtrue[g][bindx[g]==i]
            y = gmeas[g][bindx[g]==i]

            gmean.append(np.mean(y))
            gstd.append(np.std(y))

        # Now compute shear bias from binned results
        bin_mean = np.mean([bins[0:-1], bins[1:]], axis=0)
        m, m_err, c, c_err = compute_shear_bias(bin_mean, gmean, gstd, g)
        f = lambda g: (1.+m)*g + c
        bias_label = f'(1+{m:.5f}){g} + {c:.5f}'

        ax.errorbar(bin_mean, gmean, yerr=gstd, label='Binned g_meas')
        ax.plot(bin_mean, f(bin_mean), lw=2, ls='--', c='r', label=bias_label)
        ax.axhline(0, lw=2, ls='--', c='k')
        ax.plot(bin_mean, bin_mean, ls='-', lw=2, c='k', label='unity')
        ax.set_xlabel(f'True {g}')
        ax.set_ylabel(f'Meas {g}')
        ax.legend()

    fig.set_size_inches(size)

    if run_name is None:
        p = ''
    else:
        p = f'{run_name} '

    if prefix is not None:
        p += (prefix + ' ')

    plt.suptitle(f'{p}Shear Calibration')

    if outfile is not None:
        goutfile = outfile.replace('.png', '_{g}.png')
        plt.savefig(goutfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

    return

def compute_shear_bias(true_g, meas_g, meas_g_err, component):
    '''
    Compute multiplicative & added shear bias for sample
    and component 'g1' or 'g2'

    returns: (m, m_err, c, c_err)
    '''

    assert (component == 'g1') or (component == 'g2')
    g = component

    # x = true[f'nfw_{g}']
    # y = meas[f'{g}_Rinv']
    x = true_g
    y = meas_g

    # intercept, slope = Polynomial.fit(x, y, 1)

    def f(x, a, b):
        return a*x + b
    res, cov = curve_fit(f, x, y, sigma=meas_g_err)
    slope, intercept = res[0], res[1]

    m = slope - 1. # bias defined as y=(1+m)*x+c
    c = intercept

    # TODO: actually compute errors
    m_err, c_err = cov[0,0], cov[1,1]

    return (m, m_err, c, c_err)

def plot_bias_by_col(match, col, bins, outfile=None, show=False, run_name=None,
                     selection=None, prefix=None, size=(9,12), title=None):

    shear = match.meas

    N = len(bins) - 1
    g1_m = np.zeros(N)
    g1_c = np.zeros(N)
    g1_m_err = np.zeros(N)
    g1_c_err = np.zeros(N)
    g2_m = np.zeros(N)
    g2_c = np.zeros(N)
    g2_m_err = np.zeros(N)
    g2_c_err = np.zeros(N)

    k = 0
    for b1, b2 in zip(bins, bins[1:]):
        sample = (shear[col] >= b1) & (shear[col] < b2)

        # apply additional selection if desired
        if selection is not None:
            sample = selection & sample


        sample_true = match.true[sample]
        sample_meas = match.meas[sample]

        assert len(sample_true) == len(sample_meas)
        N = len(sample_true)

        if N > 0:
            g1_bias = compute_shear_bias(sample_true, sample_meas, 'g1')
            g1_m[k], g1_m_err[k] = g1_bias[0], g1_bias[1]
            g1_c[k], g1_c_err[k] = g1_bias[2], g1_bias[3]

            g2_bias = compute_shear_bias(sample_true, sample_meas, 'g2')
            g2_m[k], g2_m_err[k] = g2_bias[0], g2_bias[1]
            g2_c[k], g2_c_err[k] = g2_bias[2], g2_bias[3]
        else:
            g1_m[k], g1_m_err[k] = np.NaN, np.NaN
            g1_c[k], g1_c_err[k] = np.NaN, np.NaN
            g2_m[k], g2_m_err[k] = np.NaN, np.NaN
            g2_c[k], g2_c_err[k] = np.NaN, np.NaN

        k += 1

    xx = np.mean([bins[0:-1], bins[1:]], axis=0)

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

    axes[0].errorbar(xx, 10*g1_m, 10*g1_m_err, label='g1')
    axes[0].errorbar(xx, 10*g2_m, 10*g2_m_err, label='g2')
    axes[0].axhline(0, lw=2, ls='--', c='k')
    axes[0].set_ylabel(r'm [$10^{-1}$]')
    axes[0].legend()

    axes[1].errorbar(xx, 100*g1_c, 100*g1_c_err, label='g1')
    axes[1].errorbar(xx, 100*g2_c, 100*g2_c_err, label='g2')
    axes[1].axhline(0, lw=2, ls='--', c='k')
    axes[1].set_xlabel(col)
    axes[1].set_ylabel('c [$10^{-2}$]')
    axes[1].legend()

    fig.set_size_inches(size)
    fig.subplots_adjust(hspace=0.1)

    if run_name is None:
        p = ''
    else:
        p = f'{run_name} '

    if prefix is not None:
        p += (prefix + ' ')

    if title is None:
        plt.suptitle(f'{p}Shear Bias', y=0.9)
    else:
        plt.suptitle(f'{p}{title}')

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

    return

def match_cats(truth_file, shear_file, **kwargs):
    return MatchedTruthCatalog(truth_file, shear_file, **kwargs)

def plot_binned_bias_by_col(match, col, bins, gbins=None, outfile=None, show=False,
                            run_name=None, selection=None, prefix=None, size=(14,9),
                            radial_cut=None, title=None):
    '''
    Plot shear bias (binned in gbins) by col binned in bins
    '''

    truth = match.true
    shear = match.meas

    if selection is not None:
        truth = truth[selection]
        shear = shear[selection]

    if radial_cut is not None:
        truth, shear = cut_cats_by_radius(truth, shear, radial_cut)

    if len(truth) != len(shear):
        raise Exception(f'len(truth) = {len(truth)} but ' +\
                        f'len(shear) = {len(shear)}')

    Ntotal = len(shear)

    #-----------------------------------
    # bin data to prep for fitting
    if gbins is None:
        gmin, gmax = -0.1, 0.1
        Nbins = 11
        gbins = np.linspace(gmin, gmax, Nbins)

    N = len(bins) - 1
    g1_m = np.zeros(N)
    g1_c = np.zeros(N)
    g1_m_err = np.zeros(N)
    g1_c_err = np.zeros(N)
    g2_m = np.zeros(N)
    g2_c = np.zeros(N)
    g2_m_err = np.zeros(N)
    g2_c_err = np.zeros(N)

    k = 0
    for b1, b2 in zip(bins, bins[1:]):
        sample = (shear[col] >= b1) & (shear[col] < b2)

        sample_true = truth[sample]
        sample_meas = shear[sample]

        assert len(sample_true) == len(sample_meas)
        N = len(sample_true)

        if N > 0:
            for g in ['g1', 'g2']:
                gtrue = {}
                gtrue['g1'] = sample_true['nfw_g1']
                gtrue['g2'] = sample_true['nfw_g2']

                gmeas= {}
                gmeas['g1'] = sample_meas['g1_Rinv']
                gmeas['g2'] = sample_meas['g2_Rinv']

                gbindx = {}
                gbindx['g1'] = np.digitize(gtrue['g1'], gbins)
                gbindx['g2'] = np.digitize(gtrue['g2'], gbins)

                gmean = []
                gstd  = []
                bin_mean = []
                for i in range(1, len(gbins)):
                    x = gtrue[g][gbindx[g]==i]
                    y = gmeas[g][gbindx[g]==i]

                    assert len(x) == len(y)
                    if len(x) == 0:
                        continue

                    gmean.append(np.mean(y))
                    gstd.append(np.std(y))
                    bin_mean.append(np.mean([gbins[i-1], gbins[i]]))

                # Now compute shear bias from binned results
                m, m_err, c, c_err = compute_shear_bias(bin_mean, gmean, gstd, g)

                # plt.errorbar(bin_mean, gmean, gstd, label=g)
                # plt.plot(bin_mean, bin_mean, lw=2, ls='-', c='k', label='unity')
                # f = lambda g: (1.+m)*np.array(g) + c
                # plt.plot(bin_mean, f(bin_mean), lw=2, ls='--', c='r', label='calibration')
                # plt.legend()
                # plt.title(f'N={N}; m={m:.4f}; c={c:.4f}')
                # plt.show()

                if g == 'g1':
                    g1_m[k], g1_m_err[k] = m, m_err
                    g1_c[k], g1_c_err[k] = c, c_err
                else:
                    g2_m[k], g2_m_err[k] = m, m_err
                    g2_c[k], g2_c_err[k] = c, c_err

        else:
            g1_m[k], g1_m_err[k] = np.NaN, np.NaN
            g1_c[k], g1_c_err[k] = np.NaN, np.NaN
            g2_m[k], g2_m_err[k] = np.NaN, np.NaN
            g2_c[k], g2_c_err[k] = np.NaN, np.NaN

        k += 1

    xx = np.mean([bins[0:-1], bins[1:]], axis=0)

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

    axes[0].errorbar(xx, g1_m, g1_m_err, label='g1')
    axes[0].errorbar(xx, g2_m, g2_m_err, label='g2')
    axes[0].axhline(0, lw=2, ls='--', c='k')
    # axes[0].set_ylabel(r'm [$10^{-1}$]')
    axes[0].set_ylabel(r'm')
    axes[0].legend()

    axes[1].errorbar(xx, 100*g1_c, 100*g1_c_err, label='g1')
    axes[1].errorbar(xx, 100*g2_c, 100*g2_c_err, label='g2')
    axes[1].axhline(0, lw=2, ls='--', c='k')
    axes[1].set_xlabel(col)
    axes[1].set_ylabel('c [$10^{-2}$]')
    axes[1].legend()

    fig.set_size_inches(size)
    fig.subplots_adjust(hspace=0.1)

    if run_name is None:
        p = ''
    else:
        p = f'{run_name} '

    if prefix is not None:
        p += (prefix + ' ')

    if title is None:
        plt.suptitle(f'{p}Shear Bias (Ntotal={Ntotal})', y=1.1)
    else:
        plt.suptitle(f'{p}{title}')

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

    return


def main(args):

    shear_file = args.shear_file
    truth_file = args.truth_file
    run_name = args.run_name
    outdir = args.outdir
    show = args.show
    vb = args.vb

    if run_name is not None:
        p = f'{run_name}-'
    else:
        p = ''

    if outdir is not None:
        utils.make_dir(outdir)
    else:
        outdir = ''

    match = match_cats(truth_file, shear_file, match_radius=1./3600.)
    print(f'Match cat has {match.Nobjs} objs')

    #-----------------------------------------------------------------
    # Top-level shear calibration, based on fit to binned g's
    gbins = np.linspace(-0.1, 0.1, 11)
    # gbins = np.linspace(-0.05, 0.05, 21)

    outfile = os.path.join(outdir, f'{p}binned-bias.png')
    plot_binned_shear_calibration(
        match, bins=gbins, outfile=outfile, show=show, run_name=run_name
        )

    outfile = os.path.join(outdir, f'{p}binned-bias-by-s2n.png')
    s2n_bins = np.linspace(5, 30, 6)
    plot_binned_bias_by_col(
        match, 's2n_r_noshear', s2n_bins, gbins=gbins, outfile=outfile, run_name=run_name,
        show=show
        )

    outfile = os.path.join(outdir, f'{p}binned-bias-by-T.png')
    # T_bins = np.logspace(-4, 2, 6)
    T_bins = np.linspace(-1, 4, 6)
    plot_binned_bias_by_col(
        match, 'T_noshear', T_bins, gbins=gbins, outfile=outfile, run_name=run_name,
        show=show
        )

    # TODO: might want to remove in the future
    return 0

    #-----------------------------------------------------------------
    # Extra plots
    outfile = os.path.join(outdir, f'{p}true-g-dist.png')
    plot_true_shear_dist(
        match, outfile=outfile, show=show, run_name=run_name
                         )

    outfile = os.path.join(outdir, f'{p}shear-responses.png')
    plot_shear_responses(
        match, outfile=outfile, show=show, run_name=run_name
                         )

    outfile = os.path.join(outdir, f'{p}shear-calibration.png')
    plot_shear_calibration(
        match, outfile=outfile, show=show, run_name=run_name
        )

    outfile = os.path.join(outdir, f'{p}shear-calibration-zoom.png')
    plot_shear_calibration(
        match, outfile=outfile, show=show, run_name=run_name,
        limits=[[-.1, .1], [-.5, .5]], s=3
        )

    #-----------------------------------------------------------------
    # Now repeat w/ |g| cut

    gmax = 0.2
    selection = (np.abs(match.meas['g1_Rinv']) < gmax) &\
                (np.abs(match.meas['g2_Rinv']) < gmax)

    outfile = os.path.join(outdir, f'{p}true-g-dist-gcut.png')
    plot_true_shear_dist(
        match, outfile=outfile, show=show, run_name=run_name,
        selection=selection
                         )

    outfile = os.path.join(outdir, f'{p}shear-responses-gcut.png')
    plot_shear_responses(
        match, outfile=outfile, show=show, run_name=run_name,
        selection=selection
                         )

    outfile = os.path.join(outdir, f'{p}shear-calibration-gcut.png')
    plot_shear_calibration(
        match, outfile=outfile, show=show, run_name=run_name,
        selection=selection, prefix='gcut'
        )

    outfile = os.path.join(outdir, f'{p}shear-calibration-zoom-gcut.png')
    plot_shear_calibration(
        match, outfile=outfile, show=show, run_name=run_name,
        limits=[[-.1, .1], [-.5, .5]], s=3,
        selection=selection, prefix='gcut'
        )

    #-----------------------------------------------------------------
    # Now repeat w/ radial cut to force square in image
    # single images have dimensions (9568, 6380)
    # radial_cut = 6380 // 2 # pixels
    # outfile = os.path.join(outdir, f'{p}true-g-dist-rcut.png')
    # plot_true_shear_dist(
    #     match, outfile=outfile, show=show, run_name=run_name,
    #     radial_cut=radial_cut
    #                      )

    # outfile = os.path.join(outdir, f'{p}shear-responses-rcut.png')
    # plot_shear_responses(
    #     match, outfile=outfile, show=show, run_name=run_name,
    #     radial_cut=radial_cut
    #                      )

    # outfile = os.path.join(outdir, f'{p}shear-calibration-rcut.png')
    # plot_shear_calibration(
    #     match, outfile=outfile, show=show, run_name=run_name,
    #     radial_cut=radial_cut, prefix='rcut'
    #     )

    # outfile = os.path.join(outdir, f'{p}shear-calibration-zoom-rcut.png')
    # plot_shear_calibration(
    #     match, outfile=outfile, show=show, run_name=run_name,
    #     limits=[[-.1, .1], [-.5, .5]], s=3,
    #     radial_cut=radial_cut, prefix='rcut'
    #     )

    #-----------------------------------------------------------------
    # Now replicate some of the plots in mcal2

    # outfile = os.path.join(outdir, f'{p}obj-responses.png')
    # plot_responses_by_class(
    #     match, outfile=outfile, show=show, run_name=run_name,
    #     )

    outfile = os.path.join(outdir, f'{p}bias-by-s2n.png')
    bins = np.linspace(10, 20, 11)
    plot_bias_by_col(
        match, 's2n_r_noshear', bins, outfile=outfile, show=show, run_name=run_name,
        )

    outfile = os.path.join(outdir, f'{p}bias-by-s2n-gcut.png')
    bins = np.linspace(10, 20, 11)
    gmax = 0.2
    selection = (np.abs(match.meas['g1_Rinv']) < gmax) &\
                (np.abs(match.meas['g2_Rinv']) < gmax)
    title = f'|g_i|<{gmax}'
    plot_bias_by_col(
        match, 's2n_r_noshear', bins, outfile=outfile, show=show, run_name=run_name,
        selection=selection, title=title
        )

    outfile = os.path.join(outdir, f'{p}bias-by-T.png')
    bins = np.linspace(0, 10, 11)
    plot_bias_by_col(
        match, 'T_noshear', bins, outfile=outfile, show=show, run_name=run_name,
        )

    # outfile = os.path.join(outdir, f'{p}bias-by-s2n-Tcut.png')
    # bins = np.linspace(10, 20, 11)
    # Tmax = 0.2
    # selection = (np.abs(match.meas['g1_Rinv']) < gmax) &\
    #             (np.abs(match.meas['g2_Rinv']) < gmax)
    # title = f'|T_noshear|<{Tmax}'
    # plot_bias_by_col(
    #     match, 's2n_r_noshear', bins, outfile=outfile, show=show, run_name=run_name,
    #     selection=selection, title=title
    #     )

    return 0

if __name__ == '__main__':
    args = parser.parse_args()
    rc = main(args)

    if rc == 0:
        print('\nTests have completed without errors')
    else:
        print(f'\nTests failed with rc={rc}')

