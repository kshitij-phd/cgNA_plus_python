import numpy as np

from cgNA_plus.classes.cgDNAclass import cgDNA
from cgNA_plus.modules import cgDNAUtils as tools

from utilities.basepair_atoms import basepair_to_atoms
from utilities.arrange_atoms import arrange_atoms


SEQUENCE = "GCATCATCAGACTACTACTCAGC"

COMPLEMENT = {
    "A": "T",
    "T": "A",
    "G": "C",
    "C": "G",
}


def reverse_complement(sequence):
    return "".join(
        COMPLEMENT[b]
        for b in reversed(sequence)
    )


def main():

    print()
    print("=" * 60)
    print("ARRANGEATOMS STRAND-ORDER VALIDATION")
    print("=" * 60)
    print()

    data = cgDNA(
        SEQUENCE,
        "ps2_cgf",
    )

    bp_level = tools.frames(
        data.ground_state
    )

    atoms_base, atoms_pho = basepair_to_atoms(
        bp_level,
        SEQUENCE,
    )

    atoms_coord = arrange_atoms(
        atoms_base,
        atoms_pho,
        len(SEQUENCE),
    )

    # ------------------------------------------------------------
    # Expected Crick sequence
    # ------------------------------------------------------------

    expected_crick = reverse_complement(
        SEQUENCE
    )

    print("Watson sequence:")
    print(SEQUENCE)
    print()

    print("Expected Crick sequence:")
    print(expected_crick)
    print()

    # ------------------------------------------------------------
    # Determine base identity from C1' + base-specific atoms.
    #
    # The atom names in each residue are enough to identify
    # the Tsukuba base.
    # ------------------------------------------------------------

    expected_W = list(SEQUENCE)
    expected_C = list(expected_crick)

    def identify_base(atom_names):

        names = set(atom_names)

        if "N6" in names:
            return "A"

        if "O4" in names and "C7" in names:
            return "T"

        if "O4" in names and "C7" not in names:
            return "U"

        if "O6" in names and "N2" in names:
            return "G"

        if "N4" in names:
            return "C"

        raise ValueError(
            f"Could not identify base from atoms: "
            f"{sorted(names)}"
        )

    # ------------------------------------------------------------
    # Check Watson order
    # ------------------------------------------------------------

    print("Watson strand")
    print("-" * 40)

    for i, expected in enumerate(expected_W):

        # Base atoms occur after the phosphate atoms.
        # Identify using the complete residue.
        actual = identify_base(
            atoms_coord["W"][i]["names"]
        )

        print(
            f"Residue {i + 1:2d}: "
            f"expected={expected} "
            f"actual={actual}"
        )

        assert actual == expected

    print(
        "Watson strand order: PASSED"
    )
    print()

    # ------------------------------------------------------------
    # Check Crick order
    # ------------------------------------------------------------

    print("Crick strand")
    print("-" * 40)

    for i, expected in enumerate(expected_C):

        actual = identify_base(
            atoms_coord["C"][i]["names"]
        )

        print(
            f"Residue {i + 1:2d}: "
            f"expected={expected} "
            f"actual={actual}"
        )

        assert actual == expected

    print(
        "Crick reverse-complement order: PASSED"
    )
    print()

    # ------------------------------------------------------------
    # Check first/last residue
    # ------------------------------------------------------------

    print("Terminal residue check")
    print("-" * 40)

    W_first = atoms_coord["W"][0]["names"]
    W_last = atoms_coord["W"][-1]["names"]

    C_first = atoms_coord["C"][0]["names"]
    C_last = atoms_coord["C"][-1]["names"]

    # First residue:
    # MATLAB ArrangeAtoms uses phosphate atom 2, which is O3'.
    assert W_first[0] == "O3'"
    assert C_first[0] == "O3'"

    # The first residue must not contain a P atom because there is
    # no previous phosphate.
    assert "P" not in W_first
    assert "P" not in C_first

    # Internal residue:
    # MATLAB ordering is P, OP1, OP2, O5', O3', followed by bases.
    W_internal = atoms_coord["W"][1]["names"]
    C_internal = atoms_coord["C"][1]["names"]

    expected_phosphate = ["P", "OP1", "OP2", "O5'", "O3'"]

    assert W_internal[:5] == expected_phosphate
    assert C_internal[:5] == expected_phosphate

    # Last residue:
    # MATLAB ArrangeAtoms uses the previous phosphate atoms only:
    # P, OP1, OP2, O5'. There is no O3' from the next phosphate.
    assert W_last[:4] == ["P", "OP1", "OP2", "O5'"]
    assert C_last[:4] == ["P", "OP1", "OP2", "O5'"]

    assert "O3'" not in W_last
    assert "O3'" not in C_last

    print("First-residue phosphate ordering: PASSED")
    print("Internal-residue phosphate ordering: PASSED")
    print("Last-residue phosphate ordering : PASSED")
    print()

    print("=" * 60)
    print("ALL STRAND-ORDER TESTS: PASSED")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
