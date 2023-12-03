#!/bin/bash

#SBATCH --job-name data_pol
#SBATCH --error data_pol_error_%j.txt
#SBATCH --output data_pol_output_%j.txt
#SBATCH --mail-user pietro.girotto@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH --time 15:00:00
#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --mem 40G


# Downloading
source /home/girottopie/.bashrc

gdrive_id="15g8aCnrCOro90NildcIR1_RNh27Sk0YA"

cd /ext
gdrivedownload $gdrive_id it_20M_lines.tar.gz
tar -xzvf it_20M_lines.tar.gz

ls -l

cd /home/girottopie/Code/mr_white_game/dataset_manipulation

python3 polish_dataset.py /ext/it_20M_lines.txt

ls -l

tar -czvf it_20M_lines_polished.tar.gz /ext/it_20M_lines_polished.txt

