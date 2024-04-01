import os
import copy
import shutil
import numpy as np
import pyswarms as ps
import multiprocess as mp

from uuid import uuid4
from datetime import datetime
from scipy.spatial.distance import pdist, squareform

from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem, molzip

from deap import algorithms, base, cma, creator, tools

from utils.metrics import vina

# path to PDBs of amino acids
RESIDUES_PATH = "./data/residues"

MAP_AA_TO_IDX = {
    "ALA": 0,
    "CYS": 1,
    "ASP": 2,
    "GLU": 3,
    "PHE": 4,
    "GLY": 5,
    "HIS": 6,
    "ILE": 7,
    "LYS": 8,
    "LEU": 9,
    "MET": 10,
    "ASN": 11,
    "PRO": 12,
    "GLN": 13,
    "ARG": 14,
    "SER": 15,
    "THR": 16,
    "VAL": 17,
    "TRP": 18,
    "TYR": 19,
}

MAP_IDX_TO_AA = {v: k for k, v in MAP_AA_TO_IDX.items()}


# Disjoint Set Union data structure
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1


def combine_fragments(m1, m2):
    m1 = Chem.Mol(m1)
    m2 = Chem.Mol(m2)
    for atm in m1.GetAtoms():
        if atm.HasProp("atomLabel") and atm.GetProp("atomLabel") == "_R2":
            atm.SetAtomMapNum(1)
    for atm in m2.GetAtoms():
        if atm.HasProp("atomLabel") and atm.GetProp("atomLabel") == "_R1":
            atm.SetAtomMapNum(1)
    return molzip(m1, m2)


def make_peptide(monomerlist):
    monomerlist = copy.deepcopy(monomerlist)
    for idx, monomer in enumerate(monomerlist):
        for atom in monomer.GetAtoms():
            info = atom.GetMonomerInfo()
            if info is not None:
                mi = Chem.AtomPDBResidueInfo()
                mi.SetResidueNumber(idx + 1)
                mi.SetResidueName(info.GetResidueName())
                mi.SetName(info.GetName())
                atom.SetMonomerInfo(mi)

        if Chem.MolToSmiles(monomer).count("*") == 1:
            continue
        if idx == 0:
            res = monomer
        else:
            res = combine_fragments(res, monomer)

    return res


def cap_terminal(m):
    coords = []
    for atm in m.GetAtoms():
        if atm.HasProp("atomLabel") and atm.GetProp("atomLabel") == "_R1":
            atm.SetAtomMapNum(1)
        if atm.HasProp("atomLabel") and atm.GetProp("atomLabel") == "_R2":
            coords.append(
                np.array(list(m.GetConformer().GetAtomPosition(atm.GetIdx())))
            )
            atm.SetAtomMapNum(2)

    c_term = Chem.MolFromSmiles("O[*:2]")
    c_term = shift_coms([c_term], coords)[0]

    res = molzip(m, c_term)
    res = Chem.DeleteSubstructs(res, Chem.MolFromSmiles("[*:1]"))
    return res


def compute_com(conf):
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(conf.GetNumAtoms())])
    atoms = [atom for atom in conf.GetOwningMol().GetAtoms()]
    masses = np.array([atom.GetMass() for atom in atoms]).reshape(-1, 1)
    com = np.sum(pos * masses, axis=0) / np.sum(masses)

    return com


def translate(conf, delta_pos):
    for i in range(conf.GetNumAtoms()):
        x = conf.GetAtomPosition(i).x + delta_pos[0]
        y = conf.GetAtomPosition(i).y + delta_pos[1]
        z = conf.GetAtomPosition(i).z + delta_pos[2]
        conf.SetAtomPosition(i, Point3D(x, y, z))

    return conf


def shift_coms(mols, protein_pos):
    for i, mol in enumerate(mols):
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        com = compute_com(conf)
        conf = translate(conf, protein_pos[i] - com)
        mol.AddConformer(conf)

    return mols


def clean_peptide(m):
    for atom in m.GetAtoms():
        if atom.GetPDBResidueInfo() is None:
            prev = m.GetAtomWithIdx(atom.GetIdx() - 1)
            ri = Chem.AtomPDBResidueInfo()
            ri.SetResidueName(prev.GetPDBResidueInfo().GetResidueName())
            ri.SetResidueNumber(prev.GetPDBResidueInfo().GetResidueNumber())

            idx = 1
            for atom2 in m.GetAtoms():
                if atom2.GetPDBResidueInfo() is None:
                    continue
                if (
                    atom2.GetPDBResidueInfo().GetResidueNumber()
                    == ri.GetResidueNumber()
                    and atom2.GetSymbol() == atom.GetSymbol()
                ):
                    idx += 1
            ri.SetName(f" {atom.GetSymbol()}{idx} ")

            atom.SetPDBResidueInfo(ri)
    return m


def construct_peptide(mol_list):
    peptide = make_peptide(mol_list)
    peptide = cap_terminal(peptide)
    peptide = clean_peptide(peptide)

    Chem.SanitizeMol(peptide)
    Chem.AllChem.MMFFOptimizeMolecule(
        peptide,
        mmffVariant="MMFF94s",
        maxIters=1000,
        ignoreInterfragInteractions=True,
    )

    return peptide


def degree_constrained_mst(graph, degree_constraints):
    n = len(graph)
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j, graph[i][j]))

    # sort edges by weight in ascending order
    edges.sort(key=lambda x: x[2])

    uf = UnionFind(n)
    result_edges = []

    for edge in edges:
        u, v, _ = edge

        if (
            uf.find(u) != uf.find(v)
            and degree_constraints[u] > 0
            and degree_constraints[v] > 0
        ):
            uf.union(u, v)
            result_edges.append(edge)
            degree_constraints[u] -= 1
            degree_constraints[v] -= 1

    return result_edges


def topological_sort(graph):
    num_vertices = len(graph)
    in_degrees = [0] * num_vertices  # Store in-degrees of each vertex

    # calculate in-degrees
    for i in range(num_vertices):
        for j in range(num_vertices):
            if graph[j][i] == 1:
                in_degrees[i] += 1

    # initialize a queue for vertices with in-degree 0
    zero_in_degree_queue = []
    for i in range(num_vertices):
        if in_degrees[i] == 0:
            zero_in_degree_queue.append(i)

    # perform topological sorting
    topological_order = []
    while zero_in_degree_queue:
        vertex = zero_in_degree_queue.pop(0)
        topological_order.append(vertex)

        for neighbor in range(num_vertices):
            if graph[vertex][neighbor] == 1:
                in_degrees[neighbor] -= 1
                if in_degrees[neighbor] == 0:
                    zero_in_degree_queue.append(neighbor)

    # check for cycles
    if len(topological_order) != num_vertices:
        raise Exception("Graph contains a cycle!")

    return topological_order


def ordering_msl(coords):
    # weighted adjacency matrix
    n = len(coords)
    graph = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            graph[i][j] = d
            graph[j][i] = d

    # compute minimum spanning line
    msl = degree_constrained_mst(graph, [2] * len(graph))

    # unweighted undirected adjacency matrix
    conn = np.zeros((n, n), dtype=int)
    for i, j, _ in msl:
        conn[i][j] = True

    # topological sort
    order = topological_sort(conn)

    return order


def ordering_evo(
    coords,
    aas=None,
    protein=None,
    DIST_THR=10,
    SIGMA=5.0,
    NPOP=20,
    NGEN=10,
    NPROC=2,
):
    n = len(coords)

    # compute distance matrix
    dist = squareform(pdist(coords))

    # construct mask
    mask = np.ones_like(dist)
    mask = np.where(dist > DIST_THR, 0, 1)  # ignore atoms that are too far away
    mask = np.triu(mask, k=1)  # ignore diagonal and lower triangle
    indices = np.argwhere(mask == 1)  # get indices of upper triangle
    individual_size = len(indices) + 1  # +1 for determining parity of connection

    # set temporary workdir
    wd = f"/tmp/hydra/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(wd, exist_ok=True)

    # copy protein to wd
    shutil.copy(protein, f"{wd}/protein.pdb")

    def individual_to_peptide(x):
        # extract features from individual
        edges = indices[np.where(x[:-1])]
        parity = x[-1]

        # compute residue order for given edges and parity
        graph = np.zeros((n, n))
        for i, j in edges:
            graph[i][j] = 1
        order = topological_sort(graph)
        mols_ordered = get_ordered_mols(coords, aas, order, parity)

        # construct peptide
        peptide = construct_peptide(mols_ordered)

        return peptide

    def objective(x):
        # convert x to boolean array
        x_bool = np.where(x > 0.5, 1, 0).astype(bool)

        # terminate if invalid number of edges selected
        if np.sum(x_bool[:-1]) != n - 1:
            return (0,)

        # construct peptide
        peptide = individual_to_peptide(x_bool)

        # save peptide to file
        peptide_id = uuid4().hex
        Chem.MolToPDBFile(peptide, f"{wd}/{peptide_id}.pdb", confId=0)

        # compute vina energy
        affinity = vina.run("./protein.pdb", f"./{peptide_id}.pdb", wd, local=True)

        return (affinity,)

    # set up DEAP
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", objective)

    strategy = cma.Strategy(centroid=[0] * individual_size, sigma=SIGMA, lambda_=NPOP)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    pool = mp.Pool(processes=NPROC)
    toolbox.register("map", pool.map)

    hof = tools.HallOfFame(1, similar=np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # run CMAES optimization
    algorithms.eaGenerateUpdate(toolbox, ngen=NGEN, stats=stats, halloffame=hof)

    # extract features from individual
    x = np.where(hof[0] > 0.5, 1, 0).astype(bool)
    edges = indices[np.where(x[:-1])]
    parity = x[-1]

    # compute residue order for given edges and parity
    graph = np.zeros((n, n))
    for i, j in edges:
        graph[i][j] = 1

    # topological sort
    order = topological_sort(graph)

    return order, parity


def ordering_bpso(
    coords,
    aas=None,
    protein=None,
    DIST_THR=10,
    NPROC=9,
    n_particles=100,
    iters=20,
    c1=2.5,
    c2=0.5,
    w=0.9,
    k_m=0.1,
    p=2,
):
    n = len(coords)

    # compute distance matrix
    dist = squareform(pdist(coords))

    # construct mask
    mask = np.ones_like(dist)
    mask = np.where(dist > DIST_THR, 0, 1)  # ignore atoms that are too far away
    mask = np.triu(mask, k=1)  # ignore diagonal and lower triangle
    indices = np.argwhere(mask == 1)  # get indices of upper triangle
    individual_size = len(indices) + 1  # +1 for determining parity of connection

    # set temporary workdir
    wd = f"/tmp/hydra/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(wd, exist_ok=True)

    # copy protein to wd
    shutil.copy(protein, f"{wd}/protein.pdb")

    def individual_to_peptide(x):
        # extract features from individual
        edges = indices[np.where(x[:-1])]
        parity = x[-1]

        # compute residue order for given edges and parity
        graph = np.zeros((n, n))
        for i, j in edges:
            graph[i][j] = 1
        order = topological_sort(graph)
        mols_ordered = get_ordered_mols(coords, aas, order, parity)

        # construct peptide
        peptide = construct_peptide(mols_ordered)

        return peptide

    def objective(x_bool):
        # terminate if invalid number of edges selected
        if np.sum(x_bool[:-1]) != n - 1:
            return ((n - 1) - np.sum(x_bool[:-1])) ** 2

        # construct peptide
        peptide = individual_to_peptide(x_bool)

        # save peptide to file
        peptide_id = uuid4().hex
        Chem.MolToPDBFile(peptide, f"{wd}/{peptide_id}.pdb", confId=0)

        # compute vina energy
        affinity = vina.run("./protein.pdb", f"./{peptide_id}.pdb", wd, local=True)

        return affinity

    def objective_v(x_arr):
        res = []
        with mp.Pool(NPROC) as pool:
            res = pool.map(objective, x_arr)
        return np.array(res)

    # set up BinaryPSO
    options = {"c1": c1, "c2": c2, "w": w, "k": k_m * individual_size, "p": p}
    optimizer = ps.discrete.binary.BinaryPSO(
        n_particles=n_particles,
        dimensions=individual_size,
        options=options,
    )

    # run BPSO optimization
    _, x = optimizer.optimize(objective_v, iters=iters)

    # extract features from individual
    edges = indices[np.where(x[:-1])]
    parity = x[-1]

    # compute residue order for given edges and parity
    graph = np.zeros((n, n))
    for i, j in edges:
        graph[i][j] = 1

    # topological sort
    order = topological_sort(graph)

    return order, parity


def compute_ordering(xyz, aas, protein, config):
    if config.reconstruct.algorithm == "msl":
        return ordering_msl(xyz), 0
    if config.reconstruct.algorithm == "mslr":
        return ordering_msl(xyz), 1
    if config.reconstruct.algorithm == "evo":
        assert (
            config.reconstruct.evo is not None
        ), "Algorithm 'evo' requires additional parameters"
        return ordering_evo(xyz, aas, protein, **config.reconstruct.evo)
    if config.reconstruct.algorithm == "bpso":
        assert (
            config.reconstruct.bpso is not None
        ), "Algorithm 'bpso' requires additional parameters"
        return ordering_bpso(xyz, aas, protein, **config.reconstruct.bpso)

    raise ValueError("Invalid ordering algorithm")


def load_residue(aa):
    r1 = Chem.Atom("*")
    r1.SetProp("atomLabel", "_R1")
    r2 = Chem.Atom("*")
    r2.SetProp("atomLabel", "_R2")

    mol = Chem.MolFromPDBFile(f"{RESIDUES_PATH}/{aa}.pdb", removeHs=False)

    # label alpha carbons
    peptide_bond_representation = Chem.MolFromSmiles("NCC(=O)")
    for peptide_bond in mol.GetSubstructMatches(peptide_bond_representation):
        alpha_carbon = peptide_bond[1]
        atom = mol.GetAtomWithIdx(alpha_carbon)
        info = atom.GetPDBResidueInfo()
        ri = Chem.AtomPDBResidueInfo()
        ri.SetResidueNumber(info.GetResidueNumber())
        ri.SetResidueName(info.GetResidueName())
        ri.SetName(" CA ")
        atom.SetPDBResidueInfo(ri)

    # open terminals
    mol = Chem.RWMol(mol)
    r1_idx, r2_idx = None, None
    for atom in mol.GetAtoms():
        if atom.GetPDBResidueInfo().GetName().strip() == "H1":
            r1_idx = atom.GetIdx()
        elif atom.GetPDBResidueInfo().GetName().strip() == "H2":
            r2_idx = atom.GetIdx()
    mol.ReplaceAtom(r1_idx, r1)
    mol.ReplaceAtom(r2_idx, r2)

    return mol


def get_ordered_mols(xyz, aas, order, parity=0):
    # translate indices to amino acids
    mapped_aas = [MAP_IDX_TO_AA[i] for i in aas]

    # fetch and process amino acids
    mols = []
    for i, aa in enumerate(mapped_aas):
        mols.append(load_residue(aa))

    # shift center of mass of predicted AAs
    mols = shift_coms(mols, xyz)
    mols_ordered = [mols[i] for i in order]

    if parity == 0:  # NC
        return mols_ordered
    else:  # CN
        return reversed(mols_ordered)


def reconstruct_peptide(xyz, aas, protein, config):
    # get connectivity order
    order, parity = compute_ordering(xyz, aas, protein, config)

    mols_ordered = get_ordered_mols(xyz, aas, order, parity)

    peptide = construct_peptide(mols_ordered)

    return peptide
