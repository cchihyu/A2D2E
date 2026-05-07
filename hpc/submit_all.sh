#!/bin/bash
# Submit all HPC jobs.
# Run from the repository root: bash hpc/submit_all.sh

mkdir -p logs results_new results_nonadditive results_spike

echo "=== Submitting Table 2 sweep (420 jobs) ==="
qsub hpc/run.pbs
echo

echo "=== Submitting Appendix D.2 sweep (280 jobs) ==="
qsub hpc/run_nonadditive.pbs
echo

echo "Done. Monitor with: qstat -u $USER"
