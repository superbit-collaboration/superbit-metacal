# superbit-metacal
Contains a collection of routines used to perform gmix/metacalibration on simulated SuperBIT images

The structure is somewhat byzantine, but here is a top-level overview, in the order in which each step is performed. More detailed descriptions for each stage are contained in their respective directories.

  - `GalSim`: contains scripts that generate the simulated SuperBIT observations used for further analysis.
  - `Medsmaker`: contains small modifications to the original superbit-ngmix scripts to allow for changes of the PSF sampling scale. Output MEDS files are contained in this directory. 
  - `Metacalibration`: contains scripts used to run the ngmix/metacalibration algorithms on the MEDS files produced by Medsmaker. This directory contains the output CSV files from running the ngmix scripts. 
  - `Shear-profiles`: contains scripts to compute the tangential/cross shear profiles and output to a file, as well as plots of the shear profiles.
