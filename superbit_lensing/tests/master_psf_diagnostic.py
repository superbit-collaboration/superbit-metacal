import psfex
import galsim,galsim.des
import treecorr
import piff
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from astropy.table import Table
import pdb
from argparse import ArgumentParser

from starmaker import StarMaker, StampBackground
from psfmaker import PSFMaker,make_output_table


parser = ArgumentParser()

# I/O file names
parser.add_argument('imdir',type=str,
                    help='Directory containing star catalogs & images')
parser.add_argument('star_cat',type=str,
                    help='Star catalog to use for PSF diagnostic')
parser.add_argument('--min_snr',type=float, default=None,
                    help='Optional S/N cut for star catalog [default=None]')
parser.add_argument('--outdir',type=str, default=None,
                    help='Output directory for diagnostics [default=./psf_diagnostics]')
# Select which diagnostics to run
parser.add_argument('--epsfex',action='store_true', default=False,
                    help='Run esheldon psfex diagnostic')
parser.add_argument('--gpsfex',action='store_true', default=False,
                    help='Run galsim.des_psfex diagnostic')
parser.add_argument('--piff',action='store_true', default=False,
                    help='Run PIFF diagnostic')
# Files for the different PSF diagnostics
parser.add_argument('--psfex_name',type=str, default=None,
                    help='PSFEx model filename')
parser.add_argument('--im_name',type=str, default=None,
                    help='FITS image filename used with PSFEx model')
parser.add_argument('--piff_name',type=str, default=None,
                    help='PIFF psf model filename')
parser.add_argument('--verbose','-v',action='store_true', default=False,
                    help='Verbosity')



def main():

    args = parser.parse_args()
    imdir = args.imdir
    star_cat = args.star_cat
    run_piff = args.piff
    run_gpsf = args.gpsfex
    run_pex  = args.epsfex
    vb = args.verbose

    if args.outdir is None:
        outdir = './psf_diagnostics'
    if args.min_snr is not None:
        min_snr = args.min_snr
    if args.im_name is not None:
        im_name = args.im_name
    if args.psfex_name is not None:
        psf_name = args.psfex_name
    if args.piff_name is not None:
        piff_name = args.piff_name

    if not os.path.isdir(outdir):
        cmd = 'mkdir -p %s' % outdir
        os.system(cmd)

    try:
        star_cat = Table.read(os.path.join(imdir,star_cat),hdu=2)
    except:
        star_cat = Table.read(os.path.join(imdir,star_cat),hdu=1)

    if min_snr is not None:
        print("selecting S/N > %.1f stars" % float(min_snr))
        wg = star_cat['SNR_WIN'] > min_snr
        star_cat = star_cat[wg]

    # Initialize background
    cs = StampBackground(star_cat)
    sky_bg,sky_std = cs.calc_star_bkg(vb=True)

    # Do star HSM fits
    sm = StarMaker(star_cat)
    sm.run(bg_obj=cs,vb=True)

    # Render PSFs, do HSM fits, save diagnostics to file
    makers = []; makers.append(sm)
    prefix = []; prefix.append('star')

    if run_pex==True:
        pex = psfex.PSFEx(os.path.join(imdir,psf_name))
        psf_pex = PSFMaker(psf_file=pex,psf_type='epsfex')
        psf_pex.run_all(stars=sm,vb=vb)
        makers.append(psf_pex); prefix.append('pex')

    if run_gpsf==True:
        psfex_des = galsim.des.DES_PSFEx(os.path.join(imdir,psf_name),os.path.join(imdir,im_name))
        psf_des = PSFMaker(psf_file=psfex_des,psf_type='gpsfex')
        psf_des.run_all(stars=sm,vb=False)
        makers.append(psf_des); prefix.append('gpsf')

    if run_piff==True:
        piff_psf = piff.read(os.path.join(imdir,piff_name))
        psf_piff = PSFMaker(psf_file=piff_psf,psf_type='piff')
        psf_piff.run_all(stars=sm,vb=False)
        makers.append(psf_piff); prefix.append('piff')

    # Write star & psf HSM fits to file
    outfile=os.path.join(outdir,'star+psf_HSM_fits.ldac')
    make_output_table(makers,prefix,outfile=outfile)


if __name__ == "__main__":
    import pdb, traceback, sys
    try:
        main()
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
