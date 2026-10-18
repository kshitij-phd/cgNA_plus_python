import numpy as np

from cgNA_plus.classes.cgDNAclass import cgDNA
from cgNA_plus.modules import cgDNAUtils as tools

from utilities.basepair_atoms import (
    basepair_to_atoms,
)
from utilities.ideal_atoms import (
    load_ideal_atoms,
)


SEQUENCE = "GCATCATCAGACTACTACTCAGC"


def distance(coord, i, j):
    """Distance between atoms i and j in a (3, N) coordinate array."""
    return np.linalg.norm(
        coord[:, i] - coord[:, j]
    )


def check_rotation_matrices(bp_level):
    """
    Check that cgNA+ frame rotation matrices are
    proper rotations:

        R.T @ R = I
        det(R) = +1
    """

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

    rotation_sets = {
        "R": R,
        "Rc": Rc,
        "Rw": Rw,
    }

    max_orth_error = 0.0
    min_det = np.inf
    max_det_error = 0.0

    for name, matrices in rotation_sets.items():

        for i, matrix in matrices.items():

            matrix = np.asarray(
                matrix,
                dtype=float,
            )

            if matrix.size == 0:
                continue

            orth_error = np.linalg.norm(
                matrix.T @ matrix
                - np.eye(3),
                ord=np.inf,
            )

            determinant = np.linalg.det(
                matrix
            )

            max_orth_error = max(
                max_orth_error,
                orth_error,
            )

            min_det = min(
                min_det,
                determinant,
            )

            max_det_error = max(
                max_det_error,
                abs(determinant - 1.0),
            )

    print("Rotation-matrix validation")
    print("-" * 40)
    print(
        f"Maximum orthogonality error : "
        f"{max_orth_error:.3e}"
    )
    print(
        f"Minimum determinant         : "
        f"{min_det:.12f}"
    )
    print(
        f"Maximum determinant error   : "
        f"{max_det_error:.3e}"
    )

    assert max_orth_error < 1e-10
    assert abs(min_det - 1.0) < 1e-10

    print("Rotation matrices: PASSED")
    print()


def check_phosphate_geometry(ideal_atoms, atoms_pho):
    """
    Verify that rigid-body transformations preserve
    the internal phosphate geometry.
    """

    ideal = ideal_atoms["P"]["coord"]

    # Tsukuba atom ordering:
    #
    # 0 = P
    # 1 = OP1
    # 2 = OP2
    # 3 = O5'
    # 4 = O3'

    ideal_distances = {
        "P-OP1": distance(ideal, 0, 1),
        "P-OP2": distance(ideal, 0, 2),
        "P-O5'": distance(ideal, 0, 3),
        "P-O3'": distance(ideal, 0, 4),
    }

    print("Ideal Tsukuba phosphate geometry")
    print("-" * 40)

    for name, value in ideal_distances.items():
        print(
            f"{name:6s} : {value:.6f} Å"
        )

    print()

    max_error = 0.0

    for strand in ["W", "C"]:

        print(
            f"{strand}-strand transformed phosphate geometry"
        )
        print("-" * 40)

        for i, atom_group in enumerate(
            atoms_pho[strand]
        ):

            coord = atom_group["coord"]

            transformed_distances = {
                "P-OP1": distance(coord, 0, 1),
                "P-OP2": distance(coord, 0, 2),
                "P-O5'": distance(coord, 0, 3),
                "P-O3'": distance(coord, 0, 4),
            }

            for name in ideal_distances:

                error = abs(
                    transformed_distances[name]
                    - ideal_distances[name]
                )

                max_error = max(
                    max_error,
                    error,
                )

        print(
            f"Maximum internal-distance error: "
            f"{max_error:.3e} Å"
        )
        print()

    # A pure rotation + translation should preserve
    # distances to numerical precision.
    assert max_error < 1e-10

    print(
        "Phosphate rigid-body distance preservation: PASSED"
    )
    print()


def check_all_atom_finiteness(atoms_base, atoms_pho):
    """Check every generated atom coordinate."""

    total_atoms = 0

    for strand in ["W", "C"]:

        for atom_group in atoms_base[strand]:

            coord = atom_group["coord"]

            assert np.all(
                np.isfinite(coord)
            )

            total_atoms += coord.shape[1]

        for atom_group in atoms_pho[strand]:

            coord = atom_group["coord"]

            assert np.all(
                np.isfinite(coord)
            )

            total_atoms += coord.shape[1]

    print("Global coordinate validation")
    print("-" * 40)
    print(
        f"Total generated atoms : {total_atoms}"
    )
    print(
        "All coordinates finite : PASSED"
    )
    print()


def main():

    print()
    print("=" * 60)
    print("cgNA+ ATOM GEOMETRY VALIDATION")
    print("=" * 60)
    print()

    # ------------------------------------------------------------
    # Load ideal coordinates
    # ------------------------------------------------------------

    ideal_atoms = load_ideal_atoms()

    # ------------------------------------------------------------
    # Construct cgNA+
    # ------------------------------------------------------------

    print("Constructing cgNA+ structure...")

    data = cgDNA(
        SEQUENCE,
        "ps2_cgf",
    )

    print("cgNA+ construction: PASSED")
    print()

    # ------------------------------------------------------------
    # Construct frames
    # ------------------------------------------------------------

    print("Constructing cgNA+ frames...")

    bp_level = tools.frames(
        data.ground_state
    )

    print("Frame construction: PASSED")
    print()

    # ------------------------------------------------------------
    # Construct atom coordinates
    # ------------------------------------------------------------

    print("Generating atom coordinates...")

    atoms_base, atoms_pho = basepair_to_atoms(
        bp_level,
        SEQUENCE,
        ideal_atoms=ideal_atoms,
    )

    print("Atom construction: PASSED")
    print()

    # ------------------------------------------------------------
    # Test 1: rotation matrices
    # ------------------------------------------------------------

    check_rotation_matrices(
        bp_level
    )

    # ------------------------------------------------------------
    # Test 2: phosphate geometry
    # ------------------------------------------------------------

    check_phosphate_geometry(
        ideal_atoms,
        atoms_pho,
    )

    # ------------------------------------------------------------
    # Test 3: all coordinates finite
    # ------------------------------------------------------------

    check_all_atom_finiteness(
        atoms_base,
        atoms_pho,
    )

    # ------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------

    print("=" * 60)
    print("ALL GEOMETRY TESTS: PASSED")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
