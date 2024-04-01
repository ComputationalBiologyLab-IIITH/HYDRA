import re
import os
import subprocess

from rdkit import Chem
from rdkit.Chem.rdMolTransforms import ComputeCentroid

from utils.common import sanitize

ADFR_BIN = os.path.expanduser(os.environ.get("ADFR_BIN", "~/ADFR/bin"))
VINA_BIN = os.path.expanduser(
    os.environ.get("VINA_BIN", "~/autodock/autodock_vina_1_1_2_linux_x86/bin")
)


# commands
prepare_receptor = (
    lambda target: f"{ADFR_BIN}/prepare_receptor -r {target} -o {target}qt"
)
prepare_ligand = (
    lambda target: f"{ADFR_BIN}/prepare_ligand -l {target} -o {target}qt -A bonds_hydrogens"
)
vina = (
    lambda prot, pep, center, size: f"{VINA_BIN}/vina --cpu 1 --receptor {prot}qt --ligand {pep}qt --size_x {size[0]} --size_y {size[1]} --size_z {size[2]} --center_x {center[0]} --center_y {center[1]} --center_z {center[2]} --out /tmp/{sanitize(prot)}-{sanitize(pep)}.pdbqt"
)


def run(prot, pep, data_dir, box_size=30, local=False):
    # clean PDBs
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

    # compute centroid
    lig = Chem.MolFromPDBFile(f"{data_dir}/{pep}", removeHs=False)
    centroid = ComputeCentroid(lig.GetConformer())

    try:
        # run vinadock
        vina_cmd = vina(prot, pep, centroid, [box_size, box_size, box_size])
        if local:
            vina_cmd += " --local_only"
        out = subprocess.check_output(
            vina_cmd.split(),
            cwd=data_dir,
        )

        affinity_str = re.search(r"Affinity: (.*) \(kcal/mol\)", out.decode()).group(1)
        affinity = float(affinity_str)

    except:
        affinity = 0.0

    return affinity
