#!/usr/local_rwth/bin/zsh

# ask for less than memory
#SBATCH --mem-per-cpu=1G  #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)

# name the job
#SBATCH --job-name=rr_pp_h

#SBATCH --array=0-3

# thesis accout
#SBATCH --account=thes1045

# restrict time (format: d-hh:mm:ss)
#SBATCH --time=0-24:00:00

#SBATCH --cpus-per-task=8

# declare the merged STDOUT/STDERR file
#SBATCH --output=/home/jo378964/Dokumente/ma/bp-gbp/sim/output/rr_pp/rr_pp_highp/output.%x_%A_%a.txt

### beginning of executable commands
cd /home/jo378964/Dokumente/ma/bp-gbp/sim
module load gcc/6
srun sim_KL ./input/input_files/rr_pp/input_rr_pp_${SLURM_ARRAY_TASK_ID}.json ./output/rr_pp/rr_pp_highp
