from pathlib import Path
import numpy as np
from astropy.table import Table
import astropy.units as u
import matplotlib.pyplot as plt

from sim_full_focus import setup_stars

import ipdb

def main():
    target_list = Path(__file__).parent / 'target_list_full_focus.csv'
    targets = Table.read(str(target_list))

    height_deg = 0.4 * u.deg
    width_deg = 0.4 * u.deg
    area = height_deg * width_deg

    Nstars = []
    density = []
    for i, target in enumerate(targets):
        target_name = target['target']
        print(f'Starting target {target_name}')
        ra, dec = target['ra'], target['dec']

        star_cat = setup_stars(
            ra, dec, height_deg, width_deg
            )

        Nstars.append(len(star_cat))
        density.append(len(star_cat)/ area.value) # counts per deg^2

        print(f'{target_name} has {len(star_cat)} stars')

    # ipdb.set_trace()
    bins = np.logspace(np.log10(200), np.log10(2e5), 25)
    plt.hist(Nstars, bins=bins, ec='k')
    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Nstars')
    plt.ylabel('counts')
    plt.show()

    plt.hist(density, bins=bins, ec='k')
    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Nstars / deg^2')
    plt.ylabel('counts')
    plt.show()

    return

if __name__ == '__main__':
    main()
