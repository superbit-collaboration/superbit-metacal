from abc import abstractmethod
import numpy as np
from copy import deepcopy
from astropy.table import Table

from superbit_lensing import utils

import ipdb

class Selector(object):
    '''
    Basic selection runner on a catalog given a selection
    config file
    '''

    # allowed cat names that corresond to a metacalibration catalog
    # if present, will compute the selection responsivity corrections
    _mcal_cat_names = ['metacal', 'mcal', 'metacalibration']

    def __init__(self, config, catalogs):
        '''
        config: str, dict
            Either a filename or config dictionary that defines
            a "Selection" rule, as defined below
        catalogs: dict of str's
            A dictionary of catalog_names:catalog_files that we will be making
            selections on

        For example:
          config:
            metacal:
              - type: Inequality
                col: T
                min: 0
                max: 10
              - type: Equality
                col: flags
                value: 0
            etc.

          catalogs:
            metacal: mcal.fits
            etc.
        '''

        if isinstance(config, str):
            self.config = utils.read_yaml(config)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError('Selection config must be a filename or a dict!')

        if not isinstance(catalogs, dict):
            raise TypeError('catalogs must be a dictionary!')
        self.catalog_files = catalogs

        self.catalogs = {}
        self.load_catalogs()

        self.selections = {}

        self.parse_config()

        # to be filled in a call to run()
        self.selected_cats = {}

        return

    def load_catalogs(self):
        for cat_name, filename in self.catalog_files.items():
            self.catalogs[cat_name] = Table.read(filename)

        return

    def parse_config(self):

        for cat_name, selections in self.config.items():
            self.selections[cat_name] = []

            if cat_name not in self.catalogs.keys():
                raise ValueError(f'{cat_name} not in passed catalogs dict!')

            for selection in selections:
                stype = selection.pop('type')
                if stype not in SELECTION_TYPES:
                    raise ValueError(f'{selection} not a valid selection type!')

                self.selections[cat_name].append(
                    build_selection(stype, selection)
                    )

        return

    def run(self, logprint=None):
        '''
        Make the desired selections on all ingested catalogs

        logprint: utils.LogPrint
            A LogPrint instance, if desired. Will use print otherwise
        '''

        if logprint is None:
            logprint = print

        for cat_name, cat in self.catalogs.items():
            selections = self.selections[cat_name]

            selected_cat = None
            for selection in selections:
                if selected_cat is None:
                    selected_cat = selection.run(
                        cat, logprint=logprint
                        )
                else:
                    selected_cat = selection.run(
                        selected_cat, logprint=logprint
                        )

            # if making selections on a mcal catalog, need to compute
            # the selection responsivity corrections
            if cat_name in self._mcal_cat_names:
                selected_cat = self.compute_mcal_responsivities(
                    cat_name, selected_cat, selections, logprint=logprint
                    )

            self.selected_cats[cat_name] = selected_cat

        return

    def compute_mcal_responsivities(self, cat_name, noshear_select, selections,
                                    use_inv=True, mcal_shear=0.01,
                                    logprint=None):
        '''
        If making selections on a metacalibration catalog, compute the
        corresponding selection responsivity corrections

        cat_name: str
            The config name of the mcal catalog we're selecting on
        noshear_select: astropy.Table
            The *post*-selection metacal catalog on `noshear` quantities
        selections: lsit
            A list of Selection objects that define the selections we're
            making on the full mcal catalog
        use_inv: bool
            If True, will compute full responsivities using R_inv instead
            of just the diagonal entries
        mcal_shear: float
            The value of the mcal shear
        logprint: utils.LogPrint
            A LogPrint instance, if desired. Will use print otherwise
        '''

        if logprint is None:
            logprint = print

        # no_shear has already been used to create noshear_select
        mcal_select = {
            '1p': None,
            '1m': None,
            '2p': None,
            '2m': None
        }

        logprint(f'noshear selection has {len(noshear_select)} objs')

        # as we will update the selection cols, want to keep track
        # of the originals
        orig_selections = deepcopy(selections)

        for mcal_type in mcal_select:
            logprint(f'Selecting on {mcal_type}')
            # use original mcal cat for base
            selected_cat = self.catalogs[cat_name]
            logprint(f'Catalog pre-selection has {len(selected_cat)} objs')

            # start fresh
            selections = deepcopy(orig_selections)

            for selection in selections:
                # update col name for given mcal shear type
                old_col = selection.col
                new_col = old_col.replace('noshear', mcal_type)
                logprint(f'Selecting on {new_col} for {mcal_type}')
                selection.update_col(new_col)
                selected_cat = selection.run(selected_cat)
                logprint(f'{mcal_type} cat now has {len(selected_cat)} objs')

            # selected_cat now has all selections made for the given
            # mcal type
            logprint(f'Final {mcal_type} catalog has {len(selected_cat)} objs')
            mcal_select[mcal_type] = selected_cat

        #----------------------------------------------------------------------
        # if R_gamma responsivities are not present, compute them now
        logprint('Computing R_gamma responsivities...')

        rcols = ['r11', 'r12', 'r21', 'r22']

        # which shear component (index + 1) to pick for a given response term
        gamma_comp = {
            'r11': 1,
            'r12': 2,
            'r21': 1,
            'r22': 2
        }

        # which shape index to pick for a given response term
        g_indx = {
            'r11': 0,
            'r12': 0,
            'r21': 1,
            'r22': 1
        }

        for rcol in rcols:
            outcol = rcol + '_g' # for root shear response

            if rcol not in noshear_select.columns:
                gc = gamma_comp[rcol] # gamma component
                gi = g_indx[rcol] # shape index
                r_g = (noshear_select[f'g_{gc}p'][:,gi] -
                       noshear_select[f'g_{gc}m'][:,gi]) / (2.*mcal_shear)
                noshear_select[outcol] = r_g

        # NOTE: here is an explicit example of what the above is doing. I
        # decided to risk opacity over transcription errors
        #
        # if 'r11_g' not in noshear_select.columns:
        #     r11_g = (noshear_select['g_1p'][:,0] -\
        #            noshear_select['g_1m'][:,0]) / (2.*mcal_shear)
        #     noshear_select['r11_g'] = r11_g
        # else:
        #     r11_g = noshear_select['r11_g']

        #----------------------------------------------------------------------
        # compute selection responsivities
        logprint('Computing R_select responsivities...')

        Nrows = len(noshear_select)
        for rcol in rcols:
            outcol = rcol + '_s' # for selection response
            gc = gamma_comp[rcol] # gamma component
            gi = g_indx[rcol] # shape index

            r_s = ( (np.mean(mcal_select[f'{gc}p']['g_noshear'][:,gi]) -
                     np.mean(mcal_select[f'{gc}m']['g_noshear'][:,gi])) /
                    (2.*mcal_shear) )

            # NOTE: The selection response is only well-defined for the
            # ensemble mean as the number of objects is different for each
            # sample. We will save them point-wise in the catalog for
            # consistency, but they will be identical per response component
            noshear_select[outcol] = r_s * np.ones(Nrows)

        # NOTE: here is an explicit example of what the above is doing. I
        # decided to risk opacity over transcription errors
        #
        # r11_s = (np.mean(mcal_select['1p']['g_noshear'][:,0]) -
        #          np.mean(mcal_select['1m']['g_noshear'][:,0])) /
        #         (2.*mcal_shear)
        # noshear_select['r11_s'] = Nrows * [r11_s]

        #----------------------------------------------------------------------
        # now compute & save the final total shear responsivities:
        # R_gamma + R_select
        logprint('Saving total responsivities...')

        for rcol in rcols:
            noshear_select[rcol] = ( noshear_select[rcol+'_g'] +
                                     noshear_select[rcol+'_s'] )

        #----------------------------------------------------------------------
        # now compute & save the response-weighted mcal shapes
        logprint('Computing response-weighted shapes...')
        ipdb.set_trace()
        # TODO!!

        # TODO: Implement weighting! See #102
        # https://github.com/superbit-collaboration/superbit-metacal/issues/102

        return noshear_select

    def write(self, outfiles=None, overwrite=False):
        '''
        Write out each of the selected catalogs

        outfiles: dict
            A dictionary of output filenames for each input
            catalog that a selection was made on. Defaults to
            the original filenames w/ 'selected' suffix
        overwrite: bool
            Set to True to overwrite existing files
        '''

        if outfiles is not None:
            if not isinstance(outfiles, dict):
                raise TypeError('outfiles must be a dict!')
            for cat_name, outfile in outfiles.items():
                if cat_name not in self.catalogs:
                    raise ValueError(f'{cat_name} does not match stored cats!')
                if not isinstance(outfile, str):
                    raise TypeError('Each outfile must be a str!')
        else:
            for cat_name, filename in self.catalog_files.items():
                outfile = filename.replace('.fits', '_selected.fits')
                self.selected_cats[cat_name].write(outfile, overwrite=overwrite)

        return

class Selection(object):
    '''
    A selection rule to apply to a catalog
    '''

    _req_pars = ['type', 'col']
    _opt_pars = {}
    _opt_pars_all = {
        'derived': False,
        'mcal': False
        }

    def __init__(self, config):
        self.config = utils.parse_config(
            config, self._req_pars, self._opt_pars, name='Selection'
            )

        return

    def run(self, catalog, logprint=None):
        '''
        Make the selection on a catalog

        catalog: np.recarray, astropy.Table
            The catalog to make the selection on
        logprint: utils.LogPrint
            A LogPrint instance, if desired. Will use print otherwise
        '''

        if logprint is None:
            logprint = print

        col = self.config['col']
        derived = self.config['derived']

        # if the selection column is a derived quantity and not an existing
        # col, we need to create it first
        if derived is True:
            # NOTE: a derived col must use instances of `catalog["{col}"]` that we
            # will evaluate
            # e.g. col = `catalog["a"] / catalog["b"]`
            #
            # because `catalog` is defined locally and shows up in the `col`
            # definition, it should evaluate correctly
            #
            # NOTE: Obviously a bit unsafe for untrusted users!
            derived_col = eval(col)
            derived_col.name = 'derived'

            catalog['derived'] = derived_col

            derived_str = 'derived '
        else:
            derived_str = ''

        logprint(f'Selecting on {derived_str}col {col} for type={self.type}')
        logprint(f'Selection config for {col}:\n{self.config}')

        # now run the subclass-specific selection
        selected = self._run(catalog, logprint)

        if derived is True:
            selected.remove_column('derived')

        return selected

    @property
    def type(self):
        return self.config['type']

    @property
    def col(self):
        return self.config['col']

    def update_col(self, new_col):
        '''
        It is sometimes useful to update the col that the selection is made on
        (e.g. when computing metacal selection responsivities)

        new_col: str
            The new column to make the selection on
        '''

        self.config['col'] = new_col

        return

    @abstractmethod
    def _run(self, catalog, logprint):
        '''
        Each subclass will define their specific form of selection running
        '''
        pass

class InequalitySelection(Selection):
    '''
    A selection based on either a single or double-sided inequality
    '''

    _opt_pars = {
        **Selection._opt_pars_all,
        **{
            'min': None,
            'max': None,
            'strict': False
            }
        }

    def __init__(self, config):
        super(InequalitySelection, self).__init__(config)

        if (self.config['min'] is None) and (self.config['max'] is None):
            raise KeyError('Must pass at least one of min or max for ' +\
                           'a InequalitySelection!')

        return

    def _run(self, catalog, logprint):
        '''
        Make the selection on a catalog

        catalog: np.recarray, astropy.Table
            The catalog to make the selection on
        logprint: utils.LogPrint
            A logprint instance
        '''

        if self.config['derived'] is True:
            col = 'derived'
        else:
            col = self.config['col']

        # include equality?
        strict = self.config['strict']

        # depending on config, can sometimes be read as a str
        if not isinstance(strict, bool):
            if isinstance(strict, str):
                strict = eval(strict)
            else:
                raise TypeError('strict must be a bool or a str that evals ' +\
                                'to a bool!')

        # NOTE: we already check that at least one exists during construction
        if self.config['min'] is not None:
            val = self.config['min']
            if strict is True:
                catalog = catalog[catalog[col] > val]
            else:
                catalog = catalog[catalog[col] >= val]
        if self.config['max'] is not None:
            val = self.config['max']
            if strict is True:
                catalog = catalog[catalog[col] < val]
            else:
                catalog = catalog[catalog[col] <= val]

        return catalog

class EqualitySelection(Selection):
    '''
    A selection based on equality to a single value
    '''

    _req_pars = ['type', 'col', 'value']

    def _run(self, catalog, logprint):
        '''
        Make the selection on a catalog

        catalog: np.recarray, astropy.Table
            The catalog to make the selection on
        logprint: utils.LogPrint
            A logprint instance
        '''

        if self.config['derived'] is True:
            col = 'derived'
        else:
            col = self.config['col']

        val = self.config['value']

        catalog = catalog[catalog[col] == val]

        return catalog

def build_selection(selection_type, selection_config):
    '''
    selection_type: str
        The name of the selection type
    kwargs: dict
        All args needed for the called selection constructor
    '''

    if selection_type in SELECTION_TYPES:
        if selection_type not in selection_config:
            # the Selection subclasses expect this
            selection_config['type'] = selection_type
        # User-defined selection construction
        return SELECTION_TYPES[selection_type](selection_config)
    else:
        raise ValueError(f'{selection_type} is not a valid selection type!')

# allow for a few different conventions
SELECTION_TYPES = {
    'default': Selection,
    'equality': EqualitySelection,
    'inequality': InequalitySelection,
    #...
    }
