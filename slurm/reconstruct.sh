#!/usr/bin/env bash
#SBATCH -n 9
#SBATCH --mail-type=END
#SBATCH --time=4-00:00:00
#SBATCH -o /home2/$USER/.slurm-logs/%j.out
#SBATCH -w gnode078

PROJECT_DIR=$HOME/projects/HYDRA
DATA_DIR=/share1/$USER/pepbdb_natoms200_pocket10.tar.gz
CONDA_ENV=HYDRA

# activate conda env
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

# copy data
# mkdir -p /scratch/$USER
# rsync -avzWP ada:$DATA_DIR /scratch/$USER
# tar xvzf /scratch/$USER/pepbdb_natoms200_pocket10.tar.gz -C /scratch/$USER

# change cwd
cd $PROJECT_DIR
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# run the script
python3 scripts/reconstruct.py ./outputs/pfemp1/PF3D71200600/MEDIUM \
    configs/reconstruct.yml \
    --ndist 4 \
    --idx 3
