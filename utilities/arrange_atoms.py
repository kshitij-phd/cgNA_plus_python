import numpy as np


def _validate_atom_group(atom_group, name):
    """Validate one atom group."""
    if not isinstance(atom_group, dict):
        raise TypeError(f"{name} must be a dictionary.")

    if "coord" not in atom_group:
        raise KeyError(f"{name} is missing 'coord'.")

    if "names" not in atom_group:
        raise KeyError(f"{name} is missing 'names'.")

    coord = np.asarray(atom_group["coord"], dtype=float)
    names = list(atom_group["names"])

    if coord.ndim != 2 or coord.shape[0] != 3:
        raise ValueError(
            f"{name} coordinates must have shape (3, N), "
            f"got {coord.shape}."
        )

    if coord.shape[1] != len(names):
        raise ValueError(
            f"{name}: coordinate/name count mismatch. "
            f"Coordinates contain {coord.shape[1]} atoms, "
            f"but {len(names)} names were provided."
        )

    if not np.all(np.isfinite(coord)):
        raise ValueError(
            f"{name} contains non-finite coordinates."
        )

    return {
        "coord": coord,
        "names": names,
    }


def _select_atoms_by_name(atom_group, atom_names, group_name):
    """
    Select atoms by atom name in the requested order.

    This avoids depending on the ordering of atoms in the
    underlying ideal-atom template.
    """
    atom_group = _validate_atom_group(
        atom_group,
        group_name,
    )

    names = atom_group["names"]

    indices = []

    for atom_name in atom_names:
        if atom_name not in names:
            raise ValueError(
                f"{group_name} does not contain required "
                f"atom '{atom_name}'."
            )

        indices.append(names.index(atom_name))

    coord = atom_group["coord"][:, indices]

    selected_names = [
        names[index]
        for index in indices
    ]

    return {
        "coord": coord,
        "names": selected_names,
    }


def _arrange_base_atoms(atom_group, group_name):
    """
    Validate and preserve the atom ordering supplied by
    the ideal-atom template.
    """
    return _validate_atom_group(atom_group, group_name)


def _concatenate_atom_groups(groups):
    """Concatenate atom groups along the atom dimension."""
    if not groups:
        raise ValueError(
            "At least one atom group is required."
        )

    coords = [
        group["coord"]
        for group in groups
    ]

    names = []

    for group in groups:
        names.extend(group["names"])

    coord = np.concatenate(
        coords,
        axis=1,
    )

    return {
        "coord": coord,
        "names": names,
    }


def _validate_input_groups(atoms_base, atoms_pho, nbp):
    """Validate input atom groups and their sizes."""

    if not isinstance(nbp, (int, np.integer)):
        raise TypeError(
            "nbp must be an integer."
        )

    if nbp < 1:
        raise ValueError(
            "nbp must be positive."
        )

    if len(atoms_base["W"]) != nbp:
        raise ValueError(
            "Watson base count does not match nbp."
        )

    if len(atoms_base["C"]) != nbp:
        raise ValueError(
            "Crick base count does not match nbp."
        )

    if len(atoms_pho["W"]) != nbp - 1:
        raise ValueError(
            "Watson phosphate count must be nbp - 1."
        )

    if len(atoms_pho["C"]) != nbp - 1:
        raise ValueError(
            "Crick phosphate count must be nbp - 1."
        )


def _arrange_strand(
    base_groups,
    phosphate_groups,
    nbp,
    strand_name,
):
    """
    Arrange one DNA strand for PDB output.

    Residue topology follows the MATLAB ArrangeAtoms
    implementation:

    First residue:
        O3' + C1' + base atoms

    Internal residue:
        P + OP1 + OP2 + O5' + O3'
        + C1' + base atoms

    Last residue:
        P + OP1 + OP2 + O5'
        + C1' + base atoms
    """

    arranged = [None] * nbp

    for i in range(nbp):

        groups = []

        # -------------------------------------------------
        # First residue
        # -------------------------------------------------
        if i == 0:

            groups.append(
                _select_atoms_by_name(
                    phosphate_groups[0],
                    ["O3'"],
                    f"{strand_name} phosphate 0",
                )
            )

            groups.append(
                _arrange_base_atoms(
                    base_groups[0],
                    f"{strand_name} base 0",
                )
            )

        # -------------------------------------------------
        # Internal residues
        # -------------------------------------------------
        elif i < nbp - 1:

            # Previous phosphate:
            # P, OP1, OP2, O5'
            groups.append(
                _select_atoms_by_name(
                    phosphate_groups[i - 1],
                    [
                        "P",
                        "OP1",
                        "OP2",
                        "O5'",
                    ],
                    f"{strand_name} phosphate {i - 1}",
                )
            )

            # Current phosphate:
            # O3'
            groups.append(
                _select_atoms_by_name(
                    phosphate_groups[i],
                    ["O3'"],
                    f"{strand_name} phosphate {i}",
                )
            )

            # Base:
            # C1' first, then remaining atoms
            groups.append(
                _arrange_base_atoms(
                    base_groups[i],
                    f"{strand_name} base {i}",
                )
            )

        # -------------------------------------------------
        # Last residue
        # -------------------------------------------------
        else:

            # Previous phosphate:
            # P, OP1, OP2, O5'
            groups.append(
                _select_atoms_by_name(
                    phosphate_groups[i - 1],
                    [
                        "P",
                        "OP1",
                        "OP2",
                        "O5'",
                    ],
                    f"{strand_name} phosphate {i - 1}",
                )
            )

            # Last base:
            # C1' first, then remaining atoms
            groups.append(
                _arrange_base_atoms(
                    base_groups[i],
                    f"{strand_name} base {i}",
                )
            )

        arranged[i] = _concatenate_atom_groups(
            groups
        )

    return arranged


def arrange_atoms(
    atoms_base,
    atoms_pho,
    nbp,
):
    """
    Arrange Watson and Crick atom coordinates for PDB output.

    Parameters
    ----------
    atoms_base : dict
        Base atoms with keys:
            atoms_base["W"]
            atoms_base["C"]

    atoms_pho : dict
        Phosphate atoms with keys:
            atoms_pho["W"]
            atoms_pho["C"]

    nbp : int
        Number of base pairs.

    Returns
    -------
    atoms_coord : dict
        Dictionary containing:

            atoms_coord["W"]
                Watson strand residues.

            atoms_coord["C"]
                Crick strand residues.

        Each residue is represented by:
            {
                "coord": (3, N) ndarray,
                "names": list[str]
            }

    Notes
    -----
    The Crick strand is reversed before arrangement, matching
    the MATLAB EmbedIdealAtoms implementation:

        Chain2_base = [AtomsBase.C];
        Chain2_base = Chain2_base(nbp:-1:1);

        Chain2_pho = [AtomsPho.C];
        Chain2_pho = Chain2_pho(nbp-1:-1:1);
    """

    if not isinstance(atoms_base, dict):
        raise TypeError(
            "atoms_base must be a dictionary."
        )

    if not isinstance(atoms_pho, dict):
        raise TypeError(
            "atoms_pho must be a dictionary."
        )

    required_base_keys = {"W", "C"}
    required_pho_keys = {"W", "C"}

    if not required_base_keys.issubset(
        atoms_base.keys()
    ):
        raise KeyError(
            "atoms_base must contain 'W' and 'C'."
        )

    if not required_pho_keys.issubset(
        atoms_pho.keys()
    ):
        raise KeyError(
            "atoms_pho must contain 'W' and 'C'."
        )

    _validate_input_groups(
        atoms_base,
        atoms_pho,
        nbp,
    )

    # -----------------------------------------------------
    # Watson strand
    # -----------------------------------------------------
    watson = _arrange_strand(
        base_groups=atoms_base["W"],
        phosphate_groups=atoms_pho["W"],
        nbp=nbp,
        strand_name="Watson",
    )

    # -----------------------------------------------------
    # Crick strand
    #
    # MATLAB reverses both base and phosphate arrays before
    # arranging the second strand.
    # -----------------------------------------------------
    crick_base = list(
        reversed(atoms_base["C"])
    )

    crick_pho = list(
        reversed(atoms_pho["C"])
    )

    crick = _arrange_strand(
        base_groups=crick_base,
        phosphate_groups=crick_pho,
        nbp=nbp,
        strand_name="Crick",
    )

    return {
        "W": watson,
        "C": crick,
    }