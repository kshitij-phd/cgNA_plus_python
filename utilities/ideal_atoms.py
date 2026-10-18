"""
ideal_atoms.py

Load ideal atom templates for cgNA+ atom reconstruction.

The loader reads MATLAB .mat files having the structure:

    A.coord
    A.names

    T.coord
    T.names

    G.coord
    G.names

    C.coord
    C.names

    P.coord
    P.names

The numerical values depend on the selected MATLAB file, while the
Python interface is kept identical to the previous implementation.

Default file:
    Utilities/Ideal_Bases_Tsukuba.mat

The expected atom ordering is the same as the original MATLAB
Ideal_Bases_original.mat:

A:
    C1' N9 C8 N7 C5 C6 N6 N1 C2 N3 C4

T:
    C1' N1 C6 C5 C7 C4 O4 C2 O2 N3

G:
    C1' N9 C8 N7 C5 C6 O6 N1 C2 N2 N3 C4

C:
    C1' N1 C2 O2 N3 C4 N4 C5 C6

P:
    P O3' O5' OP1 OP2
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat


# ---------------------------------------------------------------------
# Canonical atom ordering
# ---------------------------------------------------------------------
# IMPORTANT:
# This ordering follows the original MATLAB
# Ideal_Bases_original.mat / EmbedIdealAtoms / ArrangeAtoms convention.
ATOM_ORDER = {
    "A": [
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "N6",
        "N1",
        "C2",
        "N3",
        "C4",
    ],

    "T": [
        "C1'",
        "N1",
        "C6",
        "C5",
        "C7",
        "C4",
        "O4",
        "C2",
        "O2",
        "N3",
    ],

    "G": [
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "O6",
        "N1",
        "C2",
        "N2",
        "N3",
        "C4",
    ],

    "C": [
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "N4",
        "C5",
        "C6",
    ],

    "P": [
        "P",
        "O3'",
        "O5'",
        "OP1",
        "OP2",
    ],
}


# Expected number of atoms in each ideal template.
EXPECTED_NATOM = {
    "A": 11,
    "T": 10,
    "G": 12,
    "C": 9,
    "P": 5,
}


# ---------------------------------------------------------------------
# Default MAT-file location
# ---------------------------------------------------------------------
def _default_mat_file():
    """
    Return the default ideal-bases MAT file.

    Directory structure:

        cgNA_plus/
        ├── utilities/
        │   └── ideal_atoms.py
        │
        └── Utilities/
            └── Ideal_Bases_Tsukuba.mat
    """

    project_root = Path(__file__).resolve().parent.parent

    return project_root / "Utilities" / "Ideal_Bases_Tsukuba.mat"


# ---------------------------------------------------------------------
# MATLAB string conversion
# ---------------------------------------------------------------------
def _matlab_string_to_python(value):
    """
    Convert a MATLAB string/character representation returned by
    scipy.io.loadmat into a normal Python string.

    MATLAB cell arrays may appear as:
        np.ndarray(dtype=object)

    MATLAB character arrays may appear as:
        np.ndarray(dtype='<U1')

    MATLAB strings may also appear as:
        np.ndarray(dtype='<U...')

    This function handles the common cases used in the ideal atom MAT.
    """

    # Already a Python string
    if isinstance(value, str):
        return value.strip()

    # NumPy scalar string
    if isinstance(value, np.str_):
        return str(value).strip()

    # NumPy scalar object
    if isinstance(value, np.ndarray):

        # Object array containing one item
        if value.dtype == object:
            if value.size == 1:
                return _matlab_string_to_python(value.reshape(-1)[0])

            # Sometimes MATLAB stores a character string as an object
            # array. Join the flattened characters/items.
            parts = [
                _matlab_string_to_python(x)
                for x in value.reshape(-1)
            ]

            return "".join(parts).strip()

        # Character array
        if value.dtype.kind in {"U", "S"}:

            # Single element
            if value.size == 1:
                return str(value.reshape(-1)[0]).strip()

            # MATLAB char row/column
            return "".join(
                str(x)
                for x in value.reshape(-1)
            ).strip()

        # Numeric array containing character codes
        if np.issubdtype(value.dtype, np.integer):
            try:
                return "".join(
                    chr(int(x))
                    for x in value.reshape(-1)
                ).strip()
            except (ValueError, TypeError):
                pass

    # Generic fallback
    return str(value).strip()


# ---------------------------------------------------------------------
# Extract atom names
# ---------------------------------------------------------------------
def _extract_atom_names(names):
    """
    Convert the MATLAB 'names' field into:

        ['C1'', 'N9', ...]

    regardless of the common scipy.io.loadmat representation.
    """

    if names is None:
        raise ValueError("MAT-file contains no 'names' field.")

    names = np.asarray(names, dtype=object)

    extracted = []

    for item in names.reshape(-1):

        # MATLAB cell containing a string
        name = _matlab_string_to_python(item)

        name = name.strip()

        if not name:
            raise ValueError(
                "Encountered an empty atom name in the MAT-file."
            )

        extracted.append(name)

    return extracted


# ---------------------------------------------------------------------
# Extract coordinates
# ---------------------------------------------------------------------
def _extract_coordinates(coord, base):
    """
    Convert MATLAB coordinate array into a NumPy array with shape:

        (3, N_atoms)

    Coordinates are assumed to be in Angstrom.
    """

    if coord is None:
        raise ValueError(
            f"MAT-file entry '{base}' contains no 'coord' field."
        )

    coord = np.asarray(coord, dtype=float)

    expected_natom = EXPECTED_NATOM[base]

    # The expected representation is 3 x N.
    if coord.shape == (3, expected_natom):
        pass

    # Allow N x 3 as a defensive compatibility case.
    elif coord.shape == (expected_natom, 3):
        coord = coord.T

    else:
        raise ValueError(
            f"Invalid coordinate shape for base '{base}': "
            f"{coord.shape}. Expected "
            f"(3, {expected_natom}) or "
            f"({expected_natom}, 3)."
        )

    if not np.all(np.isfinite(coord)):
        raise ValueError(
            f"Non-finite coordinates detected for '{base}'."
        )

    return coord


# ---------------------------------------------------------------------
# Extract one MATLAB structure
# ---------------------------------------------------------------------
def _extract_template(mat_data, base):
    """
    Extract one ideal atom template from the MATLAB structure.
    """

    if base not in mat_data:
        raise KeyError(
            f"MAT-file does not contain required entry '{base}'. "
            f"Available entries: "
            f"{sorted(k for k in mat_data if not k.startswith('__'))}"
        )

    entry = mat_data[base]

    # scipy.io.loadmat normally returns a MATLAB struct as a
    # NumPy structured array when simplify_cells=False.
    if not isinstance(entry, np.ndarray):
        raise TypeError(
            f"Unexpected MATLAB structure type for '{base}': "
            f"{type(entry)}"
        )

    if entry.size != 1:
        raise ValueError(
            f"MAT-file entry '{base}' must contain exactly one structure."
        )

    entry = entry.reshape(-1)[0]

    # Structured-array representation
    if entry.dtype.names is not None:

        fields = entry.dtype.names

        if "coord" not in fields:
            raise KeyError(
                f"Entry '{base}' does not contain field 'coord'. "
                f"Fields found: {fields}"
            )

        if "names" not in fields:
            raise KeyError(
                f"Entry '{base}' does not contain field 'names'. "
                f"Fields found: {fields}"
            )

        coord = entry["coord"]
        names = entry["names"]

    else:
        raise TypeError(
            f"Entry '{base}' is not a MATLAB structure with named "
            f"fields. dtype={entry.dtype}"
        )

    coord = _extract_coordinates(coord, base)
    names = _extract_atom_names(names)

    expected_names = ATOM_ORDER[base]

    if len(names) != len(expected_names):
        raise ValueError(
            f"Atom count mismatch for '{base}': "
            f"MAT-file contains {len(names)} names, "
            f"but {len(expected_names)} are expected."
        )

    # -------------------------------------------------------------
    # Verify atom identity and order
    # -------------------------------------------------------------
    if names != expected_names:

        # Check whether it is merely a different ordering.
        if set(names) == set(expected_names):
            raise ValueError(
                f"Atom ordering mismatch for '{base}'.\n"
                f"Expected:\n  {expected_names}\n"
                f"Found:\n  {names}\n\n"
                "The MAT file must use the canonical MATLAB atom order."
            )

        missing = [
            x for x in expected_names
            if x not in names
        ]

        extra = [
            x for x in names
            if x not in expected_names
        ]

        raise ValueError(
            f"Atom identity mismatch for '{base}'.\n"
            f"Expected:\n  {expected_names}\n"
            f"Found:\n  {names}\n"
            f"Missing:\n  {missing}\n"
            f"Extra:\n  {extra}"
        )

    # -------------------------------------------------------------
    # Check duplicate names
    # -------------------------------------------------------------
    if len(set(names)) != len(names):
        raise ValueError(
            f"Duplicate atom names detected for '{base}': {names}"
        )

    # -------------------------------------------------------------
    # Check coordinate dimensions
    # -------------------------------------------------------------
    if coord.shape != (3, len(names)):
        raise ValueError(
            f"Coordinate/name mismatch for '{base}': "
            f"coord={coord.shape}, names={len(names)}"
        )

    return {
        "coord": coord,
        "names": names,
    }


# ---------------------------------------------------------------------
# Load ideal atoms
# ---------------------------------------------------------------------
def load_ideal_atoms(filename=None):
    """
    Load ideal atom templates from a MATLAB .mat file.

    Parameters
    ----------
    filename : str or pathlib.Path, optional
        MATLAB ideal-bases file.

        If None, the default is:

            Utilities/Ideal_Bases_Tsukuba.mat

    Returns
    -------
    dict
        Dictionary containing:

            atoms["A"]["coord"]
            atoms["A"]["names"]

            atoms["T"]["coord"]
            atoms["T"]["names"]

            atoms["G"]["coord"]
            atoms["G"]["names"]

            atoms["C"]["coord"]
            atoms["C"]["names"]

            atoms["P"]["coord"]
            atoms["P"]["names"]

    Notes
    -----
    Coordinates are in Angstrom.
    """

    # -------------------------------------------------------------
    # Resolve filename
    # -------------------------------------------------------------
    if filename is None:
        filename = _default_mat_file()

    filename = Path(filename).expanduser().resolve()

    if not filename.exists():
        raise FileNotFoundError(
            "\n"
            "Ideal atom MAT file was not found.\n\n"
            f"Requested file:\n"
            f"    {filename}\n\n"
            "Expected default location:\n"
            f"    {_default_mat_file()}\n"
        )

    if filename.suffix.lower() != ".mat":
        raise ValueError(
            f"Expected a MATLAB .mat file, got:\n"
            f"    {filename}"
        )

    # -------------------------------------------------------------
    # Read MATLAB file
    # -------------------------------------------------------------
    try:
        mat_data = loadmat(
            filename,
            struct_as_record=True,
            squeeze_me=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not read MATLAB MAT file:\n"
            f"    {filename}\n\n"
            f"Original error:\n"
            f"    {exc}"
        ) from exc

    # -------------------------------------------------------------
    # Check top-level variables
    # -------------------------------------------------------------
    required = ["A", "T", "G", "C", "P"]

    available = sorted(
        key
        for key in mat_data.keys()
        if not key.startswith("__")
    )

    missing = [
        key
        for key in required
        if key not in mat_data
    ]

    if missing:
        raise KeyError(
            "The MAT file is missing required top-level entries:\n"
            f"    {missing}\n\n"
            "Required entries:\n"
            f"    {required}\n\n"
            "Available entries:\n"
            f"    {available}"
        )

    # -------------------------------------------------------------
    # Extract all templates
    # -------------------------------------------------------------
    atoms = {}

    for base in required:
        atoms[base] = _extract_template(
            mat_data,
            base,
        )

    # -------------------------------------------------------------
    # Final global validation
    # -------------------------------------------------------------
    for base in required:

        coord = atoms[base]["coord"]
        names = atoms[base]["names"]

        expected_shape = (
            3,
            EXPECTED_NATOM[base],
        )

        if coord.shape != expected_shape:
            raise RuntimeError(
                f"Internal validation failed for '{base}': "
                f"{coord.shape} != {expected_shape}"
            )

        if len(names) != EXPECTED_NATOM[base]:
            raise RuntimeError(
                f"Internal validation failed for '{base}': "
                f"{len(names)} != {EXPECTED_NATOM[base]}"
            )

        if not np.all(np.isfinite(coord)):
            raise RuntimeError(
                f"Internal validation failed for '{base}': "
                "non-finite coordinates."
            )

    return atoms


# ---------------------------------------------------------------------
# Pretty validation/report function
# ---------------------------------------------------------------------
def validate_ideal_atoms(atoms):
    """
    Perform an independent validation of a loaded ideal-atom dictionary.

    Raises
    ------
    ValueError
        If any structure is inconsistent.
    """

    required = ["A", "T", "G", "C", "P"]

    if not isinstance(atoms, dict):
        raise TypeError(
            "Ideal atoms must be returned as a dictionary."
        )

    if set(atoms.keys()) != set(required):
        raise ValueError(
            "Ideal atom dictionary contains unexpected keys.\n"
            f"Expected: {required}\n"
            f"Found:    {list(atoms.keys())}"
        )

    for base in required:

        if not isinstance(atoms[base], dict):
            raise TypeError(
                f"atoms['{base}'] must be a dictionary."
            )

        if "coord" not in atoms[base]:
            raise KeyError(
                f"atoms['{base}'] has no 'coord'."
            )

        if "names" not in atoms[base]:
            raise KeyError(
                f"atoms['{base}'] has no 'names'."
            )

        coord = np.asarray(
            atoms[base]["coord"],
            dtype=float,
        )

        names = list(atoms[base]["names"])

        expected_names = ATOM_ORDER[base]
        expected_shape = (
            3,
            EXPECTED_NATOM[base],
        )

        if coord.shape != expected_shape:
            raise ValueError(
                f"{base}: coordinate shape {coord.shape} "
                f"!= {expected_shape}"
            )

        if names != expected_names:
            raise ValueError(
                f"{base}: atom ordering is incorrect.\n"
                f"Expected: {expected_names}\n"
                f"Found:    {names}"
            )

        if not np.all(np.isfinite(coord)):
            raise ValueError(
                f"{base}: coordinates contain NaN or Inf."
            )

    return True


# ---------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------
if __name__ == "__main__":

    print()
    print("=" * 70)
    print("cgNA+ IDEAL ATOM LOADER TEST")
    print("=" * 70)

    mat_file = _default_mat_file()

    print()
    print("MAT file:")
    print(f"    {mat_file}")

    print()
    print("Loading ideal atoms...")

    atoms = load_ideal_atoms(mat_file)

    print("Loading successful.")

    print()
    print("Running independent validation...")

    validate_ideal_atoms(atoms)

    print("Validation successful.")

    print()
    print("Loaded structures:")
    print("-" * 70)

    for base in ["A", "T", "G", "C", "P"]:

        coord = atoms[base]["coord"]
        names = atoms[base]["names"]

        print(f"{base}")
        print(f"    coordinate shape : {coord.shape}")
        print(f"    number of atoms  : {len(names)}")
        print(f"    atom names       : {' '.join(names)}")
        print(
            f"    coordinate range : "
            f"{coord.min(): .6f} to {coord.max(): .6f} Å"
        )

    print()
    print("=" * 70)
    print("ALL IDEAL-ATOM TESTS PASSED")
    print("=" * 70)
    print()