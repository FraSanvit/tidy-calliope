#!/bin/bash
# Post-process Calliope result files with tidy-calliope, as a Slurm job of its own:
#   sbatch postprocess.sbatch results/                     # all .nc files in a folder
#   sbatch postprocess.sbatch results/<scenario>.nc        # one scenario
# or as a dependent job right after a model run:
#   jid=$(sbatch --parsable run-model.sbatch)
#   sbatch --dependency=afterok:$jid postprocess.sbatch results/<scenario>.nc
#
# Adapt the #SBATCH lines (partition, account) to your cluster.

#SBATCH --job-name=tidy-calliope
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --output=%x_%j.out

# Memory: annual/monthly tables need a few GB; full-year hourly tables of a ~100-node model ~20-40 GB.

CONFIG=${CONFIG:-tidy.yaml}            # settings to change (see `tidy-calliope --show-config`)
OUTPUT_DIR=${OUTPUT_DIR:-tidy-results}
SIF=${SIF:-tidy-calliope.sif}          # optional Apptainer image (see README)

cd "$SLURM_SUBMIT_DIR"
config_args=()
[ -f "$CONFIG" ] && config_args=(-c "$CONFIG")

if [ -f "$SIF" ]; then
    # container: nothing else to install
    srun apptainer run --bind "$PWD" "$SIF" "$@" "${config_args[@]}" -o "$OUTPUT_DIR"
else
    # any environment where tidy-calliope is installed (pip install tidy-calliope)
    srun tidy-calliope "$@" "${config_args[@]}" -o "$OUTPUT_DIR"
fi
