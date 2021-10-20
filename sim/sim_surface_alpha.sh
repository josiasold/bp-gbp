#!/usr/local_rwth/bin/zsh

# ask for less than memory
#SBATCH --mem-per-cpu=1G  #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)

# name the job
#SBATCH --job-name=sf_al_41

#SBATCH --array=3,5,7,9

# thesis accout
#SBATCH --account=thes1045

# restrict time (format: d-hh:mm:ss)
#SBATCH --time=0-06:00:00

# declare the merged STDOUT/STDERR file
#SBATCH --output=/home/jo378964/Dokumente/ma/bp-gbp/sim/output/bp4_surface_alpha_41/output.%x_%A_%a.txt

### beginning of executable commands
cd /home/jo378964/Dokumente/ma/bp-gbp/sim
source /home/jo378964/.zshrc
./sim_alpha ./input/input_files/surface_alpha/input_surface_alpha_${SLURM_ARRAY_TASK_ID}.json ./output/bp4_surface_alpha_41