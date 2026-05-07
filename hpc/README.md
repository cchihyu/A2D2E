# hpc — High-Performance Computing Scripts

PBS job scripts for running the full Table 2 sweep on Imperial College London's CX3 cluster.

## Files

| File | Purpose |
|------|---------|
| `run.pbs` | PBS job array — 420 jobs (7 envs × 5 methods × 1 model × 3 noise levels × 4 dependences) |
| `submit_all.sh` | Convenience script to submit the job array and print monitoring instructions |

## Usage

From the repository root:

```bash
bash hpc/submit_all.sh
```

Monitor progress:

```bash
qstat -u $USER
```

## Job configuration

Each array job selects one `(env, method, model, noise, dependence)` combination by decomposing `PBS_ARRAY_INDEX` in base-mixed-radix order. Training-set size is set to `100 × D` (where D is the dimensionality of the selected environment).

Key PBS settings in `run.pbs`:

| Setting | Value |
|---------|-------|
| CPUs per job | 4 |
| Memory per job | 8 GB |
| Walltime | 8 hours |
| Max concurrent jobs | 100 |

Results are written to `results/` in the repository root.

## Prerequisites

- `micromamba` with a `healthkit` environment containing the Python dependencies from `requirements.txt`
- Repository cloned to a directory accessible from the compute nodes
- Submit from the repository root so that `PBS_O_WORKDIR` points to the correct location
