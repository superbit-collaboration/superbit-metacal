Requirements:

pip install meds
pip install psfex
pip install fitsio
SWARP - http://www.astromatic.net/download/swarp/swarp-2.38.0.tar.gz

This directory contains the Medsmaker.py script written by EM and JM, with a small change to allow variation of the PSF sampling within PSFEx. ``psfex.debug.config`` is what has changed. Both files now contain "debug" in the filename.

process_mocks.py is a script that invokes medsmaker, or in this case, medsmaker_debug. The result of the medsmaking are output to a directory called debug3/

