import re
import requests

# config
HOST_URI = "https://web.expasy.org/cgi-bin/protparam/protparam"

# regexes
molecular_weight = r"<B>Molecular weight:</B> (.*)"
theoretical_pI = r"<B>Theoretical pI:</B> (.*)"
half_life = r"The estimated half-life is: (.*)"
instability_index = r"The instability index \(II\) is computed to be (.*)"
stability_bool = r"This classifies the protein as (.*)"
aliphatic_index = r"<B>Aliphatic index:</B> (.*)"
GRAVY = r"<B>Grand average of hydropathicity \(GRAVY\):</B> (.*)"


def run(sequence_file):
    with open(sequence_file, "r") as f:
        sequence = f.read().strip()

    try:
        res = requests.post(HOST_URI, data={"sequence": sequence}, timeout=10)

        results = {
            "molecular_weight": float(re.search(molecular_weight, res.text).group(1)),
            "theoretical_pI": float(re.search(theoretical_pI, res.text).group(1)),
            "half_life": re.search(half_life, res.text).group(1).split(" (")[0],
            "instability_index": float(re.search(instability_index, res.text).group(1)),
            "stability_bool": re.search(stability_bool, res.text)
            .group(1)
            .replace(".", "")
            == "stable",
            "aliphatic_index": float(re.search(aliphatic_index, res.text).group(1)),
            "GRAVY": float(re.search(GRAVY, res.text).group(1)),
        }

    except:
        print("Failed for sequence: ", sequence_file)

        results = {
            "molecular_weight": None,
            "theoretical_pI": None,
            "half_life": None,
            "instability_index": None,
            "stability_bool": None,
            "aliphatic_index": None,
            "GRAVY": None,
        }

    return results
