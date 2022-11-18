from abc import abstractmethod
import numpy as np
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
    _mcal_cat_names = ['metacal', 'mcal', 'metacalibration']:

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

    def run(self):
        '''
        Make the desired selections on all ingested catalogs
        '''

        for cat_name, cat in self.catalogs.items():
            selections = self.selections[cat_name]

            selected_cat = None
            for selection in selections:
                if selected_cat is None:
                    selected_cat = selection.run(cat)
                else:
                    selected_cat = selection.run(selected_cat)

            # if making selections on a mcal catalog, need to compute
            # the selection responsivity corrections
            if cat_name in self._mcal_catnames:
                selected_cat = self.compute_mcal_responsivities(
                    cat_name, selected_cat, selections
                    )

            self.selected_cats[cat_name] = selected_cat

        # if self.config['mcal'] is False:
        #     selected = self._run(catalog)
        # else:
            # in this case, we need to do a bit more work. Mcal selections
            # are done across the base and all sheared images

        return

    def compute_mcal_responsivities(self, selected_mcal, selections):
        '''
        If making selections on a metacalibration catalog, compute the
        corresponding selection responsivity corrections

        cat_name: str
            The config name of the mcal catalog we're selecting on
        selected_mcal: astropy.Table
            The *post*-selection metacal catalog
        selections: lsit
            A list of Selection objects that define the selections we're
            making on the full mcal catalog
        '''

        # TODO: implement!

        return selected_mcal

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

    def run(self, catalog):
        '''
        Make the selection on a catalog

        catalog: np.recarray, astropy.Table
            The catalog to make the selection on
        '''

        derived = self.config['derived']

        # if the selection column is a derived quantity and not an existing
        # col, we need to create it first
        if derived is True:
            # NOTE: a derived col must use instances of `catalog["{col}"]` that we
            # will evaluate
            # e.g. col = `catalog["a"] / catalog["b"]`
            col = self.config['col']

            # because `catalog` is defined locally and shows up in the `col`
            # definition, it should evaluate correctly
            # NOTE: Obviously a bit unsafe for untrusted users!
            derived_col = eval(col)
            derived_col.name = 'derived'

            catalog['derived'] = derived_col

        # now run the subclass-specific selection
        selected = self._run(catalog)

        if derived is True:
            selected.remove_column('derived')

        return selected

    @abstractmethod
    def _run(self, catalog):
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

    def _run(self, catalog):
        '''
        Make the selection on a catalog

        catalog: np.recarray, astropy.Table
            The catalog to make the selection on
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

    def _run(self, catalog):
        '''
        Make the selection on a catalog

        catalog: np.recarray, astropy.Table
            The catalog to make the selection on
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
