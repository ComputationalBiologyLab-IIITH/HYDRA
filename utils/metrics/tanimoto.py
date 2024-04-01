from rdkit import Chem, DataStructs


def tanimoto(peptide1, peptide2):
    fingerprint1 = Chem.RDKFingerprint(peptide1)
    fingerprint2 = Chem.RDKFingerprint(peptide2)
    return DataStructs.TanimotoSimilarity(fingerprint1, fingerprint2)


def run(peptide1, peptide2):
    try:
        peptide1 = Chem.MolFromPDBFile(peptide1, removeHs=False)
        peptide2 = Chem.MolFromPDBFile(peptide2, removeHs=False)
        score = tanimoto(peptide1, peptide2)

    except:
        score = 1.0

    return 1 - score
