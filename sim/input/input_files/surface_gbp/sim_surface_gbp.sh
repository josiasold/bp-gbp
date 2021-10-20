#!/usr/local_rwth/bin/zsh

# ask for less than memory
#SBATCH --mem-per-cpu=1G  #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)

# name the job
#SBATCH --job-name=s_g_d_ns

#SBATCH --array=3,5,7,9,11

# thesis accout
#SBATCH --account=thes1045

#SBATCH --cpus-per-task=16

# restrict time (format: d-hh:mm:ss)
#SBATCH --time=0-24:00:00

# declare the merged STDOUT/STDERR file
#SBATCH --output=/home/jo378964/Dokumente/ma/bp-gbp/sim/output/surface_gbp_nosplit/output.%x_%A_%a.txt

### beginning of executable commands
cd /home/jo378964/Dokumente/ma/bp-gbp/sim
# source /home/jo378964/.zshrc
module load gcc/6
srun sim_KL ./input/input_files/surface_gbp/input_surface_gbp_${SLURM_ARRAY_TASK_ID}.json ./output/surface_gbp_nosplit