#!/usr/bin/env bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node 4
#SBATCH --mail-type=END
#SBATCH --time=4-00:00:00
#SBATCH -o /home2/$USER/.slurm-logs/%j.out
#SBATCH -w gnode027

PROJECT_DIR=$HOME/projects/HYDRA
DATA_DIR=/share1/$USER/pepbdb_natoms200_pocket10_processed_pockets.lmdb
CONDA_ENV=HYDRA

# activate conda env
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

# copy data
mkdir -p /scratch/$USER
rsync -avzWP ada:$DATA_DIR /scratch/$USER

# change cwd
cd $PROJECT_DIR
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# run the script
python3 scripts/train.py configs/train.yml --num_gpus 4
