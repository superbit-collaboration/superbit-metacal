import os
from glob import glob
from astropy.table import Table
from argparse import ArgumentParser

import superbit_lensing.utils as utils
from superbit_lensing.shear_profiles.shear_plots import ShearProfilePlotter
from superbit_lensing.shear_profiles.bias import compute_shear_bias

import ipdb

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('basedir', type=str,
                        help='Location of run files to analyze. Should be ' +\
                        'a list of cluster directories')
    parser.add_argument('-shear_cut', type=float, default=None,
                        help='Max tangential shear to define scale cuts')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Show plots')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite output mcal file')
    parser.add_argument('--vb', action='store_true', default=False,
                        help='Make verbose')

    return parser.parse_args()

class MeanShearProfilePlotter(ShearProfilePlotter):
    def get_alpha(self, shear_cut):
        '''
        Compute alpha over the mean profile bins, ignoring any past
        the shear cut
        '''

        if shear_cut is True:
            cat = self.cat[self.cat['shear_cut_flag'] == 0]
        else:
            cat = self.cat

        compute_shear_bias(cat)

        alpha = cat.meta['alpha']
        sig_alpha = cat.meta['sig_alpha']

        return alpha, sig_alpha

class AnalysisRunner(object):

    def __init__(self, basedir, shear_cut=None, outdir=None, logprint=None):
        '''
        basedir: str
            Location of run files to analyze. Should be a list of cluster
            directories
        shear_cut: float
            The max absolute <gtan> value beyond which data bins
            are removed
        outdir: str
            The output directory for analysis plots & cats
        logprint: LogPrint
            A LogPrint instance
        '''

        self.basedir = basedir
        self.shear_cut = shear_cut

        if logprint is None:
            logprint = utils.LogPrint(None, vb)
        self.logprint = logprint

        if outdir is None:
            outdir = os.path.join(self.basedir, 'analysis')
        self.outdir = outdir
        utils.make_dir(self.outdir)

        self.analysis_dir = os.path.join(utils.MODULE_DIR, 'analysis')

        return

    def go(self, overwrite=False, show=False):

        clusters = glob(os.path.join(self.basedir, 'cl*/'))

        for cluster in clusters:
            cl = os.path.basename(os.path.abspath(cluster))
            self.logprint(f'Starting cluster {cl}')
            self.run_mean_shear_profile(
                cluster, overwrite=overwrite, show=show
                )
            self.make_plots(
                cluster, overwrite=overwrite, show=show
                )

        return

    def run_mean_shear_profile(self, cluster, overwrite=False, show=False):
        '''
        Compute the mean shear profile and associated alpha, after removing
        bins past the shear cut

        cluster: str
            The cluster dir that holds the output results for each realization
        overwrite: bool
            Set to overwrite output files
        show: bool
            Set to show plots
        '''

        cl = os.path.basename(os.path.abspath(cluster))

        profile_script = os.path.join(
            self.analysis_dir, 'get_mean_shear_profile.py'
            )

        # regular expressions to grab all relevant catalogs for a given cluster
        shear_cats = os.path.join(
            cluster, 'r*/*shear_profile_cat.fits'
            )
        annular_cats = os.path.join(
            cluster, 'r*/*annular.fits'
            )
        outfile = os.path.join(
            self.outdir, cl, 'stacked_shear_profile_cats.fits'
            )

        base = f'python {profile_script} '
        opts = f'-shear_cats={shear_cats} -annular_cats={annular_cats} ' +\
               f'-outfile={outfile}'
        cmd = base + opts

        if self.shear_cut is not None:
            cmd += f' -shear_cut={self.shear_cut}'

        if overwrite is True:
            cmd += ' --overwrite'

        if show is True:
            cmd += ' --show'

        if self.logprint.vb is True:
            cmd += ' --vb'

        self.logprint('Computing mean shear profile\n')
        self.logprint(f'cmd = {cmd}')
        os.system(cmd)

        return

    def make_plots(self, cluster, overwrite=False, show=False):

        cl = os.path.basename(os.path.abspath(cluster))
        outdir = os.path.join(self.outdir, cl)

        mean_catfile = os.path.join(
            self.outdir, cl, 'mean_shear_profile_cats.fits'
            )

        outfile = os.path.join(outdir, 'mean_shear_profile.png')
        self._plot_mean_profile(
            mean_catfile, cluster, outfile=outfile, overwrite=overwrite,
            show=show
            )

        return

    def _plot_mean_profile(self, catfile, cluster, outfile=None, overwrite=None,
                           show=None, size=(12,9)):

        cl = os.path.basename(cluster)

        if self.shear_cut is not None:
            shear_cut = True
        else:
            shear_cut = False

        shear_plotter = MeanShearProfilePlotter(catfile)

        shear_plotter.plot_tan_profile(
            title='Mean profile for cluster {cl}', outfile=outfile, show=show,
            size=size, shear_cut=shear_cut
            )

        return

def main(args):

    basedir = args.basedir
    shear_cut = args.shear_cut
    overwrite = args.overwrite
    show = args.show
    vb = args.vb

    logdir = os.path.join(basedir, 'analysis')
    logfile = 'analysis.log'
    log = utils.setup_logger(logfile, logdir=logdir)
    logprint = utils.LogPrint(log, vb)

    runner = AnalysisRunner(
        basedir, shear_cut=shear_cut, logprint=logprint
        )

    logprint('Starting AnalysisRunner')
    runner.go(overwrite=overwrite, show=show)

    return 0

if __name__ == '__main__':

    args = parse_args()

    rc = main(args)

    if rc == 0:
        print('run_analysis.py has completed succesfully')
    else:
        print(f'run_analysis.py has failed w/ rc={rc}')
