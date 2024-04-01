#!/usr/bin/env bash
#SBATCH --ntasks-per-node 9
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --time=4-00:00:00
#SBATCH -o /home2/$USER/.slurm-logs/%j.out
#SBATCH -w gnode087

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
python3 scripts/sample-testset.py configs/train.yml configs/sample.yml \
    --out_dir ./outputs/default-50-20240208 \
    --ndist 4 \
    --idx 3
