import os
import subprocess
import pandas as pd


os.environ["LD_LIBRARY_PATH"] = os.path.expanduser("~/miniconda3/envs/docking/lib")

# number of openmp threads
NPROC = 8

# frodock binary
FRODOCK_BIN = os.path.expanduser("~/frodock3_linux64/run_frodock.sh")

# commands
pdb2pqr = lambda target: f"pdb2pqr --ff=CHARMM {target}.pdb {target}_clean.pdb"
frodock = lambda prot, pep: f"{FRODOCK_BIN} {prot}_clean.pdb {pep}_clean.pdb {NPROC}"


def run(prot, pep, data_dir):
    # clean PDBs
    r = subprocess.run(
        pdb2pqr(prot).split(),
        stdout=subprocess.PIPE,
        cwd=data_dir,
    )
    if r.returncode != 0:
        raise Exception("pdb2pqr failed on protein")

    r = subprocess.run(
        pdb2pqr(pep).split(),
        stdout=subprocess.PIPE,
        cwd=data_dir,
    )
    if r.returncode != 0:
        raise Exception("pdb2pqr failed on peptide")

    try:
        # run frodock
        r = subprocess.run(
            frodock(prot, pep).split(),
            stdout=subprocess.PIPE,
            cwd=data_dir,
        )

        result_df = pd.read_csv(
            f"{data_dir}/frodock/{prot}_clean-{pep}_clean_results.tsv",
            sep=r"\s+",
            header=None,
            names=[
                "rank",
                "euler1",
                "euler2",
                "euler3",
                "posX",
                "posY",
                "posZ",
                "energy",
            ],
        )

        # return the average energy of the top 5 poses
        score = result_df[:5].energy.mean().item()

    except:
        score = 0.0

    return score
