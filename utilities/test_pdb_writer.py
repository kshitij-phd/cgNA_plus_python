from pathlib import Path

import numpy as np

from cgNA_plus.classes.cgDNAclass import cgDNA
from cgNA_plus.modules import cgDNAUtils as tools

from utilities.basepair_atoms import basepair_to_atoms
from utilities.arrange_atoms import arrange_atoms
from utilities.pdb_writer import write_pdb


SEQUENCE = "GCATCATCAGACTACTACTCAGC"
OUTPUT = Path("base_atoms_test.pdb")


def main():

    print("=" * 60)
    print("PDB WRITER VALIDATION")
    print("=" * 60)

    # ------------------------------------------------------------
    # Construct cgNA+
    # ------------------------------------------------------------

    data = cgDNA(SEQUENCE, "ps2_cgf")

    print(f"\nSequence : {SEQUENCE}")
    print(f"Length   : {len(SEQUENCE)}")

    # ------------------------------------------------------------
    # Construct frames
    # ------------------------------------------------------------

    bp_level = tools.frames(data.ground_state)

    # ------------------------------------------------------------
    # Convert cgNA+ frames to ideal atoms
    # ------------------------------------------------------------

    atoms_base, atoms_pho = basepair_to_atoms(
        bp_level,
        SEQUENCE,
    )

    # ------------------------------------------------------------
    # Arrange atoms according to MATLAB ArrangeAtoms.m
    # ------------------------------------------------------------

    atoms_coord = arrange_atoms(
        atoms_base,
        atoms_pho,
        len(SEQUENCE),
    )

    # ------------------------------------------------------------
    # Write PDB
    # ------------------------------------------------------------

    output = write_pdb(
        atoms_coord,
        SEQUENCE,
        OUTPUT,
    )

    print(f"\nPDB written to: {output}")

    # ------------------------------------------------------------
    # Read generated PDB
    # ------------------------------------------------------------

    lines = output.read_text().splitlines()

    atom_lines = [
        line for line in lines
        if line.startswith("ATOM")
    ]

    ter_lines = [
        line for line in lines
        if line == "TER"
    ]

    end_lines = [
        line for line in lines
        if line == "END"
    ]

    print("\nPDB structure")
    print("-" * 40)

    print(f"Total ATOM records : {len(atom_lines)}")
    print(f"TER records        : {len(ter_lines)}")
    print(f"END records        : {len(end_lines)}")

    # ------------------------------------------------------------
    # Expected atom count
    # ------------------------------------------------------------

    expected_atoms = 0

    for residue in atoms_coord["W"]:
        expected_atoms += len(residue["names"])

    for residue in atoms_coord["C"]:
        expected_atoms += len(residue["names"])

    assert len(atom_lines) == expected_atoms

    print("ATOM count         : PASSED")

    # ------------------------------------------------------------
    # Chain validation
    # ------------------------------------------------------------

    chains = [line[21] for line in atom_lines]

    n_watson = sum(chain == "A" for chain in chains)
    n_crick = sum(chain == "B" for chain in chains)

    expected_watson = sum(
        len(residue["names"])
        for residue in atoms_coord["W"]
    )

    expected_crick = sum(
        len(residue["names"])
        for residue in atoms_coord["C"]
    )

    assert n_watson == expected_watson
    assert n_crick == expected_crick

    print("Chain A atom count: PASSED")
    print("Chain B atom count: PASSED")

    # ------------------------------------------------------------
    # TER / END
    # ------------------------------------------------------------

    assert len(ter_lines) == 1
    assert len(end_lines) == 1

    print("TER record        : PASSED")
    print("END record        : PASSED")

    # ------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------

    for line in atom_lines:

        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])

        assert np.isfinite(x)
        assert np.isfinite(y)
        assert np.isfinite(z)

    print("Coordinates finite : PASSED")

    # ------------------------------------------------------------
    # Serial numbering
    # ------------------------------------------------------------

    serials = [
        int(line[6:11])
        for line in atom_lines
    ]

    assert serials == list(range(1, len(atom_lines) + 1))

    print("Atom serials       : PASSED")

    # ------------------------------------------------------------
    # Print first and last records
    # ------------------------------------------------------------

    print("\nFirst 5 ATOM records")
    print("-" * 40)

    for line in atom_lines[:5]:
        print(line)

    print("\nLast 5 ATOM records")
    print("-" * 40)

    for line in atom_lines[-5:]:
        print(line)

    print("\n" + "=" * 60)
    print("ALL PDB WRITER TESTS: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

