#!/bin/bash

# Script to clean up directories according to 'oba' structure

# Check if at least one band argument is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <band1> [<band2> ...]"
  exit 1
fi

echo_only=false

# Check for --echo flag for a dry run
for arg in "$@"; do
  if [[ "$arg" == "--echo" ]]; then
    echo_only=true
    break
  fi
done

# Function to either echo or remove files
perform_action() {
  if [[ $echo_only == true ]]; then
    for file in "$@"; do
      if [[ -e $file ]]; then
        echo "$file"
      fi
    done
    # Add one line break between actions
    echo ""
  else
    rm -rf "$@"
  fi
}


# Loop through each band provided as an argument
for band in "$@"; do
  if [[ "$band" != "--echo" ]]; then
    # Remove specific files from the {band}/cal/ directory
    perform_action "${band}/cal/"*.sgm.fits "${band}/cal/"*.sub.fits "${band}/cal/"*.weight.fits "${band}/cal/"*.bkg_rms.fits

    # Remove all files from the {band}/cat/ and {band}/meds/ directories
    perform_action "${band}/cat/"* "${band}/meds/"*

    # Remove specific files from the {band}/coadd/ directory
    perform_action "${band}/coadd/"*.*

    # Remove contents of the det/cat folder
    perform_action "det/cat/"*

    # Remove specific files from the det/coadd/ folder
    perform_action "det/coadd/"*.sub.fits "det/coadd/"*.sgm.fits

    # Remove /out directory
    perform_action "${band}/out/"*.*

  fi
done

# End of script