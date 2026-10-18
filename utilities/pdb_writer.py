from pathlib import Path


def _terminal_residue_name(base, residue_index, nbp):
    """
    Match the MATLAB makePDB.m residue naming convention.

    First residue  -> D<base>5
    Last residue   -> D<base>3
    Internal        -> D<base>
    """
    if residue_index == 0:
        suffix = "5"
    elif residue_index == nbp - 1:
        suffix = "3"
    else:
        suffix = ""

    return f"D{base}{suffix}"


def _write_atom_line(
    fp,
    serial,
    atom_name,
    residue_name,
    chain_id,
    residue_number,
    x,
    y,
    z,
):
    """
    Write an ATOM record using the formatting of MATLAB makePDB.m.
    """

    fp.write(
        f"ATOM  {serial:5d} {atom_name:>4s} {residue_name:>3s} "
        f"{chain_id}{residue_number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}\n"
    )


def write_pdb(atoms_coord, sequence, filename):
    """
    Write cgNA+ ideal-atom coordinates to a PDB file.

    Parameters
    ----------
    atoms_coord : dict
        Output of arrange_atoms(), with:
            atoms_coord["W"]
            atoms_coord["C"]

        Each residue is:
            {
                "coord":  (3, n_atoms) ndarray,
                "names":  list[str]
            }

    sequence : str
        Original Watson-strand sequence.

    filename : str or Path
        Output PDB filename.

    Returns
    -------
    Path
        Path to the generated PDB file.
    """

    sequence = sequence.upper()
    nbp = len(sequence)

    if nbp == 0:
        raise ValueError("Sequence cannot be empty.")

    if not isinstance(atoms_coord, dict):
        raise TypeError("atoms_coord must be a dictionary.")

    if "W" not in atoms_coord or "C" not in atoms_coord:
        raise KeyError('atoms_coord must contain "W" and "C".')

    watson = atoms_coord["W"]
    crick = atoms_coord["C"]

    if len(watson) != nbp:
        raise ValueError(
            f"Watson residue count ({len(watson)}) "
            f"does not match sequence length ({nbp})."
        )

    if len(crick) != nbp:
        raise ValueError(
            f"Crick residue count ({len(crick)}) "
            f"does not match sequence length ({nbp})."
        )

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    serial = 1

    with filename.open("w", encoding="utf-8") as fp:

        # ------------------------------------------------------------
        # Watson strand: chain A
        # ------------------------------------------------------------
        for i, base in enumerate(sequence):

            residue_name = _terminal_residue_name(base, i, nbp)
            residue_number = i + 1

            residue = watson[i]
            coords = residue["coord"]
            names = residue["names"]

            if coords.shape != (3, len(names)):
                raise ValueError(
                    f"Watson residue {i + 1}: "
                    f"coordinate/name mismatch."
                )

            for j, atom_name in enumerate(names):

                x = coords[0, j]
                y = coords[1, j]
                z = coords[2, j]

                _write_atom_line(
                    fp,
                    serial,
                    atom_name,
                    residue_name,
                    "A",
                    residue_number,
                    x,
                    y,
                    z,
                )

                serial += 1

        # MATLAB makePDB.m
        fp.write("TER\n")

        # ------------------------------------------------------------
        # Crick strand: chain B
        #
        # arrange_atoms() has already reversed the Crick strand.
        # Therefore sequence[i] here must be obtained from the
        # reverse-complement sequence.
        # ------------------------------------------------------------

        complement = {
            "A": "T",
            "T": "A",
            "G": "C",
            "C": "G",
        }

        crick_sequence = "".join(
            complement[base]
            for base in reversed(sequence)
        )

        for i, base in enumerate(crick_sequence):

            residue_name = _terminal_residue_name(base, i, nbp)

            # MATLAB uses i + nbp with i starting at 1.
            residue_number = i + nbp + 1

            residue = crick[i]
            coords = residue["coord"]
            names = residue["names"]

            if coords.shape != (3, len(names)):
                raise ValueError(
                    f"Crick residue {i + 1}: "
                    f"coordinate/name mismatch."
                )

            for j, atom_name in enumerate(names):

                x = coords[0, j]
                y = coords[1, j]
                z = coords[2, j]

                _write_atom_line(
                    fp,
                    serial,
                    atom_name,
                    residue_name,
                    "B",
                    residue_number,
                    x,
                    y,
                    z,
                )

                serial += 1

        fp.write("END\n")

    return filename
