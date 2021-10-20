#!/usr/local_rwth/bin/zsh

# ask for less than memory
#SBATCH --mem-per-cpu=1G  #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)

# name the job
#SBATCH --job-name=rr_alpha_urw

#SBATCH --array=0-3

# thesis accout
#SBATCH --account=thes1045

# restrict time (format: d-hh:mm:ss)
#SBATCH --time=0-20:00:00

# declare the merged STDOUT/STDERR file
#SBATCH --output=/home/jo378964/Dokumente/ma/bp-gbp/sim/output/bp4_rr_alphaURW/output.%x_%A_%a.txt

### beginning of executable commands
cd /home/jo378964/Dokumente/ma/bp-gbp/sim
source /home/jo378964/.zshrc
./sim_alpha ./input/input_files/rr_alpha/input_rr_alpha_${SLURM_ARRAY_TASK_ID}.json ./output/bp4_rr_alphaURW