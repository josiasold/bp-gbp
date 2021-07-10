#!/usr/local_rwth/bin/zsh

# ask for less than memory
#SBATCH --mem-per-cpu=2G  #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)

# name the job
#SBATCH --job-name=SERIAL_JOB

# restrict time (format: d-hh:mm:ss)
#SBATCH --time=0-20:00:00

# declare the merged STDOUT/STDERR file
#SBATCH --output=output/gbp_pp_new/output.%J.txt

### beginning of executable commands
cd $HOME/Dokumente/ma/bp-gbp/sim
source $HOME/.zshrc
./sim ./input/input_files/input_0.json ./output/gbp_pp_new