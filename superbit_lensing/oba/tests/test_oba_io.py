import unittest
import os
from pathlib import Path

from superbit_lensing.oba import IOManager

import ipdb

class TestIOManager(unittest.TestCase):
    def test_root_dir(self):
        print('Starting test_root_dir()')

        root_dir = 'test_dir'

        manager = IOManager(root_dir=root_dir)
        raw_dir = manager.RAW_DATA

        os.rmdir(root_dir)

        return

    def test_rel_root(self):
        print('Starting test_rel_root()')

        root_dir = '.'

        manager = IOManager(root_dir=root_dir)
        raw_dir = manager.RAW_DATA

        return

    def test_all_dirs(self):
        print('Starting test_all_dirs()')

        manager = IOManager()

        raw_dir = manager.RAW_DATA
        raw_clusters = manager.RAW_CLUSTERS
        oba_dir = manager.OBA_DIR
        oba_clusters = manager.OBA_CLUSTERS
        oba_results = manager.OBA_RESULTS

        return

    def test_pathlib_root(self):
        print('Starting test_pathlib_root()')

        root_dir = Path('./test_dir')

        manager = IOManager(root_dir=root_dir)
        raw_dir = manager.RAW_DATA

        os.rmdir(root_dir)

        return

    def test_print_dirs(self):
        print('Starting test_print_dirs()')

        root_dir = Path('.')

        print('Testing print_dirs() w/ root_dir')
        manager = IOManager(root_dir=root_dir)
        manager.print_dirs()

        print('Testing print_dirs() w/o root_dir')
        manager = IOManager()
        manager.print_dirs()

        return

if __name__ == '__main__':
    unittest.main()
