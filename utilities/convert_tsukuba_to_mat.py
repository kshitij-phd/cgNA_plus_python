from pathlib import Path

import numpy as np
from scipy.io import savemat


SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FILE = SCRIPT_DIR / "ideal_bases_tsukuba.txt"
OUTPUT_FILE = SCRIPT_DIR / "Ideal_Bases_Tsukuba.mat"


# Tsukuba group name -> MAT variable name
GROUP_MAP = {
    "Adenine": "A",
    "Thymine": "T",
    "Guanine": "G",
    "Cytosine": "C",
    "Phosphate": "P",
}

# Explicitly excluded groups
SKIP_GROUPS = {
    "Uracil",
}


def read_tsukuba(filename):

    templates = {}

    current_group = None
    expected_atoms = 0
    names = []
    coords = []

    with open(filename, "r") as f:

        for raw_line in f:

            line = raw_line.strip()

            # Skip empty lines
            if not line:
                continue

            # ---------------------------------------------------------
            # Separator
            # ---------------------------------------------------------
            if line.startswith("#"):

                # Finalize previous group
                if current_group is not None:

                    if current_group in GROUP_MAP:

                        if len(names) != expected_atoms:
                            raise ValueError(
                                f"{current_group}: expected "
                                f"{expected_atoms} atoms, found "
                                f"{len(names)}."
                            )

                        templates[GROUP_MAP[current_group]] = {
                            "names": names.copy(),
                            "coord": np.asarray(
                                coords,
                                dtype=float
                            ).T,
                        }

                current_group = None
                expected_atoms = 0
                names = []
                coords = []

                continue

            parts = line.split()

            # ---------------------------------------------------------
            # Group header
            #
            # Example:
            #
            # Adenine 11 1.0 0.0 ... 0.000 0.000 0.000
            # ---------------------------------------------------------
            if current_group is None:

                if len(parts) < 2:
                    continue

                group_name = parts[0]

                try:
                    n_atoms = int(parts[1])
                except ValueError:
                    continue

                current_group = group_name
                expected_atoms = n_atoms
                names = []
                coords = []

                # Uracil is deliberately ignored, but we still need
                # to read through its atom records.
                continue

            # ---------------------------------------------------------
            # Atom record
            #
            # x y z atom_name
            # ---------------------------------------------------------
            if len(parts) < 4:
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
            except ValueError:
                raise ValueError(
                    f"Invalid coordinate line:\n{line}"
                )

            atom_name = " ".join(parts[3:])

            names.append(atom_name)
            coords.append([x, y, z])

    # -------------------------------------------------------------
    # Final group
    # -------------------------------------------------------------

    if current_group is not None:

        if current_group in GROUP_MAP:

            if len(names) != expected_atoms:
                raise ValueError(
                    f"{current_group}: expected "
                    f"{expected_atoms} atoms, found "
                    f"{len(names)}."
                )

            templates[GROUP_MAP[current_group]] = {
                "names": names.copy(),
                "coord": np.asarray(
                    coords,
                    dtype=float
                ).T,
            }

    return templates


def matlab_atom_names(names):
    """
    Convert atom names to the MATLAB representation used by
    Ideal_Bases.mat:

        1 x N cell array

    where each cell contains a 1 x 3 character vector.
    """
    fixed_names = [
        name.ljust(3)[:3]
        for name in names
    ]

    return np.array(
        [np.array([name], dtype="U3") for name in fixed_names],
        dtype=object,
    ).reshape(1, -1)


def reorder_phosphate(template):
    """
    Tsukuba order:

        P
        OP1
        OP2
        O5'
        O3'

    Target Ideal_Bases.mat order:

        P
        O3'
        O5'
        OP1
        OP2

    The coordinates are reordered together with the atom names.
    """

    names = template["names"]
    coord = template["coord"]

    target_order = [
        "P",
        "O3'",
        "O5'",
        "OP1",
        "OP2",
    ]

    indices = []

    for atom in target_order:

        if atom not in names:
            raise ValueError(
                f"Required phosphate atom {atom!r} "
                f"was not found."
            )

        indices.append(names.index(atom))

    return {
        "names": target_order,
        "coord": coord[:, indices],
    }


def build_mat_struct(template):

    return {
        "coord": np.asarray(
            template["coord"],
            dtype=float
        ),
        "names": matlab_atom_names(
            template["names"]
        ),
    }


def main():

    print("Reading:")
    print(INPUT_FILE)

    templates = read_tsukuba(INPUT_FILE)

    required_groups = [
        "A",
        "T",
        "G",
        "C",
        "P",
    ]

    missing = [
        group
        for group in required_groups
        if group not in templates
    ]

    if missing:
        raise RuntimeError(
            "Missing template groups: "
            + ", ".join(missing)
        )

    # Reorder phosphate to the Ideal_Bases convention.
    templates["P"] = reorder_phosphate(
        templates["P"]
    )

    matlab_data = {}

    for group in required_groups:

        matlab_data[group] = build_mat_struct(
            templates[group]
        )

    savemat(
        OUTPUT_FILE,
        matlab_data,
        format="5",
        long_field_names=False,
        do_compression=False,
    )

    print("\nCreated:")
    print(OUTPUT_FILE)

    print("\nTemplates included:")

    for group in required_groups:

        names = matlab_data[group]["names"]
        coord = matlab_data[group]["coord"]

        print(
            f"  {group}: "
            f"{len(names)} atoms, "
            f"coord shape = {coord.shape}"
        )

        print(
            "    names =",
            list(names)
        )

    print("\nUracil:")
    print("  skipped")


if __name__ == "__main__":
    main()