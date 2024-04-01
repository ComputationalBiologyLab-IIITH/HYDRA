#!/usr/bin/env bash
#SBATCH -n 9
#SBATCH --mail-type=END
#SBATCH --time=4-00:00:00
#SBATCH -o /home2/$USER/.slurm-logs/%j.out
#SBATCH -w gnode087

PROJECT_DIR=$HOME/projects/HYDRA
CONDA_ENV=docking

# activate conda env
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

# change cwd
cd $PROJECT_DIR
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# run the script
python3 scripts/eval-frodock.py \
    /scratch/$USER/RF_pepbdb \
    --ndist 4 \
    --idx 3
