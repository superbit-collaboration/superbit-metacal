'''
File with classes & functions useful for matching catalogs
'''

import numpy as np
import esutil.htm as htm
from astropy.table import Table

class MatchedCatalog(object):
    '''
    Could have a matched class not dependent on
    a truth catalog, but haven't had a use yet
    '''
    pass

class MatchedTruthCatalog(MatchedCatalog):

    def __init__(self, true_file, meas_file,
                 true_ratag='ra', true_dectag='dec',
                 meas_ratag='ra', meas_dectag='dec',
                 match_radius=1.0/3600, depth=14):
        '''
        match_radius is in deg, same as htm
        '''

        self.true_file = true_file
        self.meas_file = meas_file

        self.true_ratag  = true_ratag
        self.meas_ratag  = meas_ratag
        self.true_dectag = true_dectag
        self.meas_dectag = meas_dectag

        self.match_radius = match_radius
        self.depth = depth

        self.meas = None
        self.true = None
        self.Nobjs = 0

        self._match()

        return

    def _match(self):
        true_cat, meas_cat = self._load_cats()

        h = htm.HTM(self.depth)

        self.matcher = htm.Matcher(
            depth=self.depth,
            ra=true_cat[self.true_ratag],
            dec=true_cat[self.true_dectag]
            )

        id_m, id_t, dist = self.matcher.match(
            ra=meas_cat[self.meas_ratag],
            dec=meas_cat[self.meas_dectag],
            radius=self.match_radius
            )

        self.true = true_cat[id_t]
        self.meas = meas_cat[id_m]
        self.meas['separation'] = dist
        self.dist = dist

        assert len(self.true) == len(self.meas)

        self.Nobjs = len(self.true)

        return

    def _load_cats(self):
        true_cat = Table.read(self.true_file)
        meas_cat = Table.read(self.meas_file)

        return true_cat, meas_cat
