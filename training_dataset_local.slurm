#!/bin/bash

#SBATCH --job-name emb_train
#SBATCH --error emb_train_error_%j.txt
#SBATCH --output emb_train_output_%j.txt
#SBATCH --mail-user pietro.girotto@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH --time 15:00:00
#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --mem 40G
#SBATCH --gres=gpu:rtx


# Setting up vars
working_dir="/home/girottopie/Code/mr_white_game"
dataset_name="it_20M_lines_polished"
archive_path="${working_dir}/dataset_manipulation/${dataset_name}.tar.gz"
dataset_path="${working_dir}/dataset_manipulation/${dataset_name}.txt" # The added ext is because I have wrongly compressed the dataset

# Extracting the archive
cd $working_dir
tar -xzvf $archive_path

# Adjust output
mv $working_dir/dataset_manipulation/ext/dataset_name.txt $working_dir/dataset_manipulation
rm -r $working_dir/dataset_manipulation/ext

# Ensuring dataset pos
echo "## Trying to see dataset pos in ${dataset_path}"
readlink -f $dataset_path
echo "## end of try"

# Setting up vars for training
context_size=2
embedding_dim=300
epochs=10
batch_size=32

# Starting training
cd $working_dir
srun singularity exec --nv /nfsd/opt/sif-images/tensorflow_latest-gpu.sif python3 model_training/main.py dataset_manipulation/$dataset_name.txt $context_size $embedding_dim $epochs $batch_size
