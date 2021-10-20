#!/usr/local_rwth/bin/zsh

# ask for less than memory
#SBATCH --mem-per-cpu=1G  #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)

# name the job
#SBATCH --job-name=sf_e_l

#SBATCH --array=3,5,7,9

# thesis accout
#SBATCH --account=thes1045

#SBATCH --cpus-per-task=8

# restrict time (format: d-hh:mm:ss)
#SBATCH --time=0-01:00:00

# declare the merged STDOUT/STDERR file
#SBATCH --output=/home/jo378964/Dokumente/ma/bp-gbp/sim/output/surface_qec_lowp/output.%x_%A_%a.txt

### beginning of executable commands
cd /home/jo378964/Dokumente/ma/bp-gbp/sim
source /home/jo378964/.zshrc
srun sim_KL ./input/input_files/surface_qec/input_surface_qec_${SLURM_ARRAY_TASK_ID}.json ./output/surface_qec_lowp
