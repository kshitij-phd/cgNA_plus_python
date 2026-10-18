from pathlib import Path
import numpy as np
from scipy.io import savemat


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "Utilities" / "Ideal_Bases_Tsukuba.txt"
OUTPUT_FILE = PROJECT_ROOT / "Utilities" / "Ideal_Bases_Tsukuba.mat"


# ---------------------------------------------------------------------
# Required MATLAB atom ordering
# ---------------------------------------------------------------------

ATOM_ORDER = {
    "A": ["C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "T": ["C1'", "N1", "C6", "C5", "C7", "C4", "O4", "C2", "O2", "N3"],
    "G": ["C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "C": ["C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "P": ["P", "O3'", "O5'", "OP1", "OP2"],
}


GROUP_TO_BASE = {
    "Adenine": "A",
    "Thymine": "T",
    "Uracil": "T",
    "Guanine": "G",
    "Cytosine": "C",
    "Phosphate": "P",
}


# ---------------------------------------------------------------------
# Parse TXT file
# ---------------------------------------------------------------------

def parse_tsukuba_file(filename):
    groups = {}

    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    i = 0

    while i < len(lines):

        line = lines[i]

        # Skip comments and blank lines
        if not line or line.startswith("#"):
            i += 1
            continue

        header = line.split()

        group_name = header[0]
        n_atoms = int(header[1])

        if group_name not in GROUP_TO_BASE:
            raise ValueError(f"Unknown group: {group_name}")

        base = GROUP_TO_BASE[group_name]

        atoms = {}

        for j in range(n_atoms):
            i += 1

            parts = lines[i].split()

            if len(parts) != 4:
                raise ValueError(
                    f"Invalid atom line:\n{lines[i]}"
                )

            x, y, z = map(float, parts[:3])
            atom_name = parts[3]

            if atom_name in atoms:
                raise ValueError(
                    f"Duplicate atom {atom_name} in {group_name}"
                )

            atoms[atom_name] = np.array([x, y, z], dtype=float)

        groups[base] = atoms

        i += 1

    return groups


# ---------------------------------------------------------------------
# Convert to MATLAB-compatible structure
# ---------------------------------------------------------------------

def build_mat_structure(groups):

    result = {}

    for base, atom_order in ATOM_ORDER.items():

        if base not in groups:
            raise ValueError(f"Missing group: {base}")

        atoms = groups[base]

        missing = [name for name in atom_order if name not in atoms]

        if missing:
            raise ValueError(
                f"{base}: missing atoms: {missing}"
            )

        extra = [name for name in atoms if name not in atom_order]

        if extra:
            raise ValueError(
                f"{base}: unexpected atoms: {extra}"
            )

        # MATLAB coord is 3 × N
        coord = np.column_stack(
            [atoms[name] for name in atom_order]
        )

        names = np.array(
            atom_order,
            dtype=object
        )

        result[base] = {
            "coord": coord,
            "names": names,
        }

    return result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print(f"Reading: {INPUT_FILE}")

    groups = parse_tsukuba_file(INPUT_FILE)

    data = build_mat_structure(groups)

    savemat(
        OUTPUT_FILE,
        data,
        do_compression=False
    )

    print(f"Written: {OUTPUT_FILE}")

    for base in ATOM_ORDER:
        print(
            f"{base}: "
            f"coord={data[base]['coord'].shape}, "
            f"atoms={len(data[base]['names'])}"
        )


if __name__ == "__main__":
    main()
