import numpy as np

from .ideal_atoms import load_ideal_atoms


COMPLEMENT = {
    "A": "T",
    "T": "A",
    "G": "C",
    "C": "G",
}


def _validate_rotation(R, name):
    """Validate a 3x3 rotation matrix."""
    R = np.asarray(R, dtype=float)

    if R.shape != (3, 3):
        raise ValueError(
            f"{name} must have shape (3, 3), got {R.shape}"
        )

    if not np.all(np.isfinite(R)):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return R


def _validate_translation(r, name):
    """Validate a 3-component translation vector."""
    r = np.asarray(r, dtype=float).reshape(-1)

    if r.size != 3:
        raise ValueError(
            f"{name} must contain 3 values, got {r.size}"
        )

    if not np.all(np.isfinite(r)):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return r


def transform_atoms(R, r, local_coord):
    """
    Transform ideal local coordinates into global coordinates.

    MATLAB convention:

        X_global = R * X_local + r

    Parameters
    ----------
    R : ndarray, shape (3, 3)
        Rotation matrix.

    r : ndarray, shape (3,)
        Translation vector.

    local_coord : ndarray, shape (3, N)
        Local atom coordinates.

    Returns
    -------
    ndarray, shape (3, N)
        Global atom coordinates.
    """

    R = _validate_rotation(R, "R")
    r = _validate_translation(r, "r")

    local_coord = np.asarray(local_coord, dtype=float)

    if local_coord.ndim != 2 or local_coord.shape[0] != 3:
        raise ValueError(
            "local_coord must have shape (3, N)."
        )

    if not np.all(np.isfinite(local_coord)):
        raise ValueError(
            "local_coord contains non-finite values."
        )

    return R @ local_coord + r[:, None]


def basepair_to_atoms(bp_level, sequence, ideal_atoms=None):
    """
    Python equivalent of MATLAB Basepair2Atoms.m.

    Parameters
    ----------
    bp_level : tuple
        Output of cgDNAUtils.frames().

        The actual Python frames() implementation returns:

            R, r, Rc, rc, Rw, rw, Rpw, rpw, Rpc, rpc

        where each object is a dictionary indexed by base-pair
        number.

    sequence : str
        DNA sequence.

    ideal_atoms : dict, optional
        Output of load_ideal_atoms().

    Returns
    -------
    atoms_base : dict
        Dictionary containing Watson and Crick base atoms.

        atoms_base["W"][i]
        atoms_base["C"][i]

    atoms_pho : dict
        Dictionary containing Watson and Crick phosphate atoms.

        atoms_pho["W"][i]
        atoms_pho["C"][i]

    Notes
    -----
    The transformations reproduce MATLAB Basepair2Atoms.m:

        Watson:
            W = Rw * IdealAtoms.(base).coord + rw

        Crick:
            Rc_eff = Rc * diag([1 -1 -1])
            C = Rc_eff * IdealAtoms.(base_c).coord + rc

        Phosphate:
            pW = Rpw * IdealAtoms.P.coord + rpw
            pC = Rpc * IdealAtoms.P.coord + rpc
    """

    # ------------------------------------------------------------
    # Load ideal atom coordinates
    # ------------------------------------------------------------

    if ideal_atoms is None:
        ideal_atoms = load_ideal_atoms()

    # ------------------------------------------------------------
    # Validate sequence
    # ------------------------------------------------------------

    sequence = sequence.upper()
    nbp = len(sequence)

    if nbp == 0:
        raise ValueError(
            "Sequence cannot be empty."
        )

    # ------------------------------------------------------------
    # Validate required ideal atom definitions
    # ------------------------------------------------------------

    required_atoms = {"A", "T", "G", "C", "P"}

    missing_atoms = required_atoms - set(ideal_atoms.keys())

    if missing_atoms:
        raise KeyError(
            "Missing ideal atom definitions: "
            f"{sorted(missing_atoms)}"
        )

    # ------------------------------------------------------------
    # Unpack the actual output structure of frames()
    #
    # frames() returns:
    #
    # R, r, Rc, rc, Rw, rw, Rpw, rpw, Rpc, rpc
    #
    # Each is a dictionary indexed by base-pair number.
    # ------------------------------------------------------------

    if not isinstance(bp_level, (tuple, list)):
        raise TypeError(
            "bp_level must be the tuple/list returned by frames()."
        )

    if len(bp_level) != 10:
        raise ValueError(
            "Expected frames() to return 10 objects "
            f"(R, r, Rc, rc, Rw, rw, Rpw, rpw, Rpc, rpc), "
            f"but received {len(bp_level)}."
        )

    (
        R,
        r,
        Rc,
        rc,
        Rw,
        rw,
        Rpw,
        rpw,
        Rpc,
        rpc,
    ) = bp_level

    # ------------------------------------------------------------
    # Validate that all frame components are dictionaries
    # ------------------------------------------------------------

    frame_dicts = {
        "R": R,
        "r": r,
        "Rc": Rc,
        "rc": rc,
        "Rw": Rw,
        "rw": rw,
        "Rpw": Rpw,
        "rpw": rpw,
        "Rpc": Rpc,
        "rpc": rpc,
    }

    for name, frame_dict in frame_dicts.items():

        if not isinstance(frame_dict, dict):
            raise TypeError(
                f"{name} must be a dictionary, "
                f"got {type(frame_dict).__name__}."
            )

    # ------------------------------------------------------------
    # Check that base-pair frames contain all required indices
    # ------------------------------------------------------------

    expected_indices = set(range(nbp))

    for name, frame_dict in frame_dicts.items():

        actual_indices = set(frame_dict.keys())

        missing_indices = expected_indices - actual_indices

        if missing_indices:
            raise KeyError(
                f"{name} is missing base-pair indices: "
                f"{sorted(missing_indices)}"
            )

    # ------------------------------------------------------------
    # Output containers
    # ------------------------------------------------------------

    atoms_base = {
        "W": [None] * nbp,
        "C": [None] * nbp,
    }

    atoms_pho = {
        "W": [],
        "C": [],
    }

    # ------------------------------------------------------------
    # MATLAB:
    #
    # Rc = Rc * diag([1 -1 -1])
    #
    # Important:
    # The multiplication order is Rc @ reflection.
    # ------------------------------------------------------------

    reflection = np.diag(
        [1.0, -1.0, -1.0]
    )

    # ============================================================
    # Loop over base pairs
    # ============================================================

    for i, base in enumerate(sequence):

        # --------------------------------------------------------
        # Validate base
        # --------------------------------------------------------

        if base not in COMPLEMENT:
            raise ValueError(
                f"Unsupported DNA base '{base}' "
                f"at position {i + 1}."
            )

        complement = COMPLEMENT[base]

        # ========================================================
        # Watson base
        #
        # MATLAB:
        #
        # Rw = bp_level(i).Rw;
        # rw = bp_level(i).rw;
        #
        # W = Rw*IdealAtoms.(base).coord
        #     + repmat(rw,...)
        # ========================================================

        Rw_i = _validate_rotation(
            Rw[i],
            f"Rw[{i}]"
        )

        rw_i = _validate_translation(
            rw[i],
            f"rw[{i}]"
        )

        W_coord = transform_atoms(
            Rw_i,
            rw_i,
            ideal_atoms[base]["coord"]
        )

        atoms_base["W"][i] = {
            "coord": W_coord,
            "names": list(
                ideal_atoms[base]["names"]
            ),
        }

        # ========================================================
        # Crick base
        #
        # MATLAB:
        #
        # base_c = wcc(base);
        #
        # Rc = bp_level(i).Rc*diag([1 -1 -1]);
        #
        # C = Rc*IdealAtoms.(base_c).coord
        #     + repmat(rc,...)
        # ========================================================

        Rc_i = _validate_rotation(
            Rc[i],
            f"Rc[{i}]"
        )

        rc_i = _validate_translation(
            rc[i],
            f"rc[{i}]"
        )

        Rc_effective = Rc_i @ reflection

        C_coord = transform_atoms(
            Rc_effective,
            rc_i,
            ideal_atoms[complement]["coord"]
        )

        atoms_base["C"][i] = {
            "coord": C_coord,
            "names": list(
                ideal_atoms[complement]["names"]
            ),
        }

        # ========================================================
        # Watson phosphate
        #
        # MATLAB:
        #
        # Rpw = bp_level(i).Rpw;
        # rpw = bp_level(i).rpw;
        #
        # if isempty(Rpw) ~= 1 && isempty(rpw) ~= 1
        #     pW = Rpw*IdealAtoms.P.coord + rpw;
        # end
        #
        # In Python frames(), terminal entries are empty lists [].
        # ========================================================

        Rpw_i = Rpw[i]
        rpw_i = rpw[i]

        if (
            np.size(Rpw_i) != 0
            and np.size(rpw_i) != 0
        ):

            Rpw_i = _validate_rotation(
                Rpw_i,
                f"Rpw[{i}]"
            )

            rpw_i = _validate_translation(
                rpw_i,
                f"rpw[{i}]"
            )

            pW_coord = transform_atoms(
                Rpw_i,
                rpw_i,
                ideal_atoms["P"]["coord"]
            )

            atoms_pho["W"].append({
                "coord": pW_coord,
                "names": list(
                    ideal_atoms["P"]["names"]
                ),
            })

        # ========================================================
        # Crick phosphate
        # ========================================================

        Rpc_i = Rpc[i]
        rpc_i = rpc[i]

        if (
            np.size(Rpc_i) != 0
            and np.size(rpc_i) != 0
        ):

            Rpc_i = _validate_rotation(
                Rpc_i,
                f"Rpc[{i}]"
            )

            rpc_i = _validate_translation(
                rpc_i,
                f"rpc[{i}]"
            )

            pC_coord = transform_atoms(
                Rpc_i,
                rpc_i,
                ideal_atoms["P"]["coord"]
            )

            atoms_pho["C"].append({
                "coord": pC_coord,
                "names": list(
                    ideal_atoms["P"]["names"]
                ),
            })

    return atoms_base, atoms_pho


# ================================================================
# Standalone validation
# ================================================================

if __name__ == "__main__":

    from cgNA_plus.classes.cgDNAclass import cgDNA
    from cgNA_plus.modules import cgDNAUtils as tools

    sequence = "GCATCATCAGACTACTACTCAGC"

    print()
    print("=" * 60)
    print("Testing Basepair2Atoms equivalent")
    print("=" * 60)
    print()

    # ------------------------------------------------------------
    # Construct cgDNA object
    # ------------------------------------------------------------

    print("Creating cgDNA object...")

    data = cgDNA(
        sequence,
        "ps2_cgf"
    )

    print("cgDNA construction: PASSED")
    print()

    # ------------------------------------------------------------
    # Construct base-pair frames
    # ------------------------------------------------------------

    print("Constructing base-pair frames...")

    bp_level = tools.frames(
        data.ground_state
    )

    print("Frame construction: PASSED")
    print()

    # ------------------------------------------------------------
    # Construct atom coordinates
    # ------------------------------------------------------------

    print("Constructing atom coordinates...")

    atoms_base, atoms_pho = basepair_to_atoms(
        bp_level,
        sequence
    )

    print("Basepair atom construction: PASSED")
    print()

    # ------------------------------------------------------------
    # Basic counts
    # ------------------------------------------------------------

    print("Atom group counts")
    print("-" * 30)

    print(
        "Watson bases :",
        len(atoms_base["W"])
    )

    print(
        "Crick bases  :",
        len(atoms_base["C"])
    )

    print(
        "Watson P     :",
        len(atoms_pho["W"])
    )

    print(
        "Crick P      :",
        len(atoms_pho["C"])
    )

    print()

    # ------------------------------------------------------------
    # Expected counts
    # ------------------------------------------------------------

    assert len(atoms_base["W"]) == len(sequence)
    assert len(atoms_base["C"]) == len(sequence)

    assert len(atoms_pho["W"]) == len(sequence) - 1
    assert len(atoms_pho["C"]) == len(sequence) - 1

    print("Expected atom-group counts: PASSED")
    print()

    # ------------------------------------------------------------
    # Validate base coordinates
    # ------------------------------------------------------------

    for strand in ["W", "C"]:

        for i, atom_group in enumerate(
            atoms_base[strand]
        ):

            coord = atom_group["coord"]
            names = atom_group["names"]

            assert coord.ndim == 2
            assert coord.shape[0] == 3
            assert coord.shape[1] == len(names)

            assert np.all(
                np.isfinite(coord)
            )

    print(
        "Base coordinate dimensions: PASSED"
    )

    print(
        "Base coordinates finite: PASSED"
    )

    # ------------------------------------------------------------
    # Validate phosphate coordinates
    # ------------------------------------------------------------

    for strand in ["W", "C"]:

        for atom_group in atoms_pho[strand]:

            coord = atom_group["coord"]
            names = atom_group["names"]

            assert coord.shape == (
                3,
                5
            )

            assert len(names) == 5

            assert np.all(
                np.isfinite(coord)
            )

    print(
        "Phosphate coordinate dimensions: PASSED"
    )

    print(
        "Phosphate coordinates finite: PASSED"
    )

    # ------------------------------------------------------------
    # Print atom counts
    # ------------------------------------------------------------

    print()
    print("Atoms per base pair")
    print("-" * 30)

    for i, base in enumerate(sequence):

        W_count = len(
            atoms_base["W"][i]["names"]
        )

        C_count = len(
            atoms_base["C"][i]["names"]
        )

        print(
            f"BP {i + 1:2d} "
            f"{base}/{COMPLEMENT[base]} : "
            f"W={W_count:2d}, "
            f"C={C_count:2d}"
        )

    print()

    print("=" * 60)
    print("ALL BASEPAIR ATOM TESTS: PASSED")
    print("=" * 60)
    print()