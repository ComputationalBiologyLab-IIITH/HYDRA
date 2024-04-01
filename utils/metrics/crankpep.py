import os
import subprocess

from utils import pdb2fasta
from utils.common import sanitize

ADFR_BIN = os.path.expanduser(os.environ.get("ADFR_BIN", "~/ADFR/bin"))

# commands
reduce_target = (
    lambda target: f"{ADFR_BIN}/reduce {target} > {target.split('.')[0]}H.pdb"
)
prepare_receptor = (
    lambda target: f"{ADFR_BIN}/prepare_receptor -r {target.split('.')[0]}H.pdb -o {target}qt"
)
prepare_ligand = (
    lambda target: f"{ADFR_BIN}/prepare_ligand -l {target.split('.')[0]}H.pdb -o {target}qt -A bonds_hydrogens"
)
prepare_target = (
    lambda prot, pep: f"{ADFR_BIN}/agfr -r {prot}qt -l {pep}qt -asv 1.1 -o {sanitize(prot.split('.')[0])}-{sanitize(pep.split('.')[0])}.pdb"
)
adcp = (
    lambda data_dir, prot, pep: f"{ADFR_BIN}/adcp -t {sanitize(prot.split('.')[0])}-{sanitize(pep.split('.')[0])}.trg -s {pdb2fasta.run(f'{data_dir}/{pep}').lower()} -N 10 -n 10000 -o {sanitize(prot.split('.')[0])}-{sanitize(pep.split('.')[0])}_redocked -ref {pep.split('.')[0]}.pdb"
)


def run(prot, pep, data_dir):
    # clean PDBs
    r = subprocess.run(
        reduce_target(prot.split("/")[1]),
        stdout=subprocess.PIPE,
        cwd=f"{data_dir}/{prot.split('/')[0]}",
        shell=True,
    )
    r = subprocess.run(
        reduce_target(pep.split("/")[1]),
        stdout=subprocess.PIPE,
        cwd=f"{data_dir}/{pep.split('/')[0]}",
        shell=True,
    )

    r = subprocess.run(
        prepare_receptor(prot).split(),
        stdout=subprocess.PIPE,
        cwd=data_dir,
    )
    if r.returncode != 0:
        raise Exception("prepare_receptor failed on protein")

    r = subprocess.run(
        prepare_ligand(pep.split("/")[1]).split(),
        stdout=subprocess.PIPE,
        cwd=f"{data_dir}/{pep.split('/')[0]}",
    )

    r = subprocess.run(
        prepare_target(prot, pep).split(),
        stdout=subprocess.PIPE,
        cwd=data_dir,
    )

    try:
        # run crankpep
        out = subprocess.check_output(
            adcp(data_dir, prot, pep).split(),
            cwd=data_dir,
        )

        affinity_str = out.decode().split("\n")[9].split()[1]
        affinity = float(affinity_str)

    except:
        affinity = 0.0

    return affinity
