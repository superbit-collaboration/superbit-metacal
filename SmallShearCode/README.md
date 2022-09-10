A summary of the codes I have included:

- small_shear_annular.py: a copy of annular_jmac.py, except that in this version
  `compute_shear_bias` only includes those bins for which the average `nfw_gtan`
  input shear signal is less than 0.08, i.e., the quasi-linear regime for Metacalibration.
  Note that this is *not* a shear selection on galaxies, this is computing the value of a
  statistic on a subset of the data.

- small_shear_runner.py: scraping the parts of `make_annular_catalog.py` that are needed to 

  annular_jmac.py, or in this case, small_shear_annular.py

- job_small_shear.sh: calls small_shear_runner.py and submits an HPC job.

- shear_plots_fillbtwn.py: hacky version of shear_plots.py that includes a way to get filled-
  between error bars which look cool. Also draws dashed line at the shear cutoff for alpha calculation.

- get_avg_alpha.py: now prints to table and allows you to specify truth and shear profile catalog names, 
  rather than assume truth name is shear name but with truth at the end! Even better, saves the printout
  to a fits table with the average alpha & weighted mean alpha in the metadata!

- get_mean_shearprofile.py: returns the mean shear profile, as well as the stacked shear profile cats
  (not averaged) for shear calibration plots. Doesn't yet automatically produce filled between shear profile
  but you can do that yourself.

