import os
import sys
from glob import glob
from astropy.table import Table
import matplotlib.pyplot as plt
import pudb

class Diagnostics(object):
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.plotdir = None
        self.plot_outdir = None

        return

    def run(self, run_options, logprint):
        logprint(f'Running diagnostics for {self.name}')

        return

class GalSimDiagnostics(Diagnostics):

    def __init__(self, name, config):
        super(GalSimDiagnostics, self).__init__(name, config)

        self.outdir = config['outdir']
        plotdir = os.path.join(self.outdir, 'plots')
        plot_outdir = os.path.join(plotdir, name)

        for d in [plotdir, plot_outdir]:
            if not os.path.exists(d):
                os.mkdir(d)

        self.outdir = config['outdir']
        self.plotdir = plot_outdir

        return

    def run(self, run_options, logprint):

        super(GalSimDiagnostics, self).run(run_options, logprint)

        ## Check consistency of truth tables
        self.plot_compare_truths(run_options, logprint)

        return

    def plot_compare_truths(self, run_options, logprint):
        # Not obvious to me why there are multiple tables - this here
        # just to prove this.

        logprint('Diagnostic: Comparing truth catalogs...')

        truth_tables = glob(os.path.join(self.outdir, 'truth*.fits'))
        N = len(truth_tables)

        tables = []
        for fname in truth_tables:
            tables.append(Table.read(fname))

        cols = ['ra', 'flux', 'hlr']
        Nrows = len(cols)
        Nbins = 30
        ec = 'k'
        alpha = 0.75

        for i in range(1, Nrows+1):
            plt.subplot(Nrows, 1, i)

            col = cols[i-1]

            k = 1
            for t in tables:
                plt.hist(t[col], bins=Nbins, ec=ec, alpha=alpha, label=f'Truth_{k}')
                k += 1

            plt.xlabel(f'True {col}')
            plt.ylabel('Counts')
            plt.legend()

            if ('flux' in col) or ('hlr' in col):
                plt.yscale('log')

        plt.gcf().set_size_inches(9, 3*Nrows+2)

        outfile = os.path.join(self.plotdir, 'compare_truth_tables.pdf')
        plt.savefig(outfile, bbox_inches='tight')

        return

class MedsmakerDiagnostics(Diagnostics):
    pass

class MetacalDiagnostics(Diagnostics):
    pass

class ShearProfileDiagnostics(Diagnostics):
    pass

def get_diagnostics_types():
    return DIAGNOSTICS_TYPES

# NOTE: This is where you must register a new diagnostics type
DIAGNOSTICS_TYPES = {
    'galsim': GalSimDiagnostics,
    'medsmaker': MedsmakerDiagnostics,
    'Metacal': MetacalDiagnostics,
    'ShearProfile': ShearProfileDiagnostics,
}

def build_diagnostics(name, config):
    name = name.lower()

    if name in DIAGNOSTICS_TYPES.keys():
        # User-defined input construction
        diagnostics = DIAGNOSTICS_TYPES[name](name, config)
    else:
        # Attempt generic input construction
        diagnostics = Diagnostics(name, config)

    return diagnostics
