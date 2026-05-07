#!/bin/bash
# Submit all HPC jobs.
# Run from the repository root: bash hpc/submit_all.sh

mkdir -p logs results_new results_spike

echo "=== Submitting Table 2 sweep (420 jobs) ==="
qsub hpc/run.pbs
echo

echo "Done. Monitor with: qstat -u $USER"
echo
echo "Note: Appendix D.2 non-additive experiments use R code in experiments/appendix_d2/"
echo "      Run A2D2E_main.Rmd in RStudio from the experiments/appendix_d2/ directory."
