import numbers

from ..classes.dsDNAclass import dsDNA


def cgDNAp3dplot(data, sequence, Reframe=True, Axis=False):
    """
    MATLAB-compatible 3D cgDNA ground-state plot.

    Parameters
    ----------
    data : cgDNA object
        cgDNA data structure containing the ground-state shape
        and sequence.

    Reframe : bool
        Re-express the complete DNA structure in the coordinate
        frame of the first Watson base pair.

    Axis : bool
        Show Cartesian axes if True.

    Returns
    -------
    fig, ax
        Matplotlib figure and 3D axes.
    """

    # ------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------
    if not hasattr(data, "ground_state"):
        raise TypeError(
            "data must contain a 'ground_state' attribute."
        )

    if not isinstance(sequence, str):
        raise TypeError("sequence must be a string.")

    sequence = sequence.upper()

    if len(sequence) == 0:
        raise ValueError("DNA sequence cannot be empty.")

    if any(base not in "ATCG" for base in sequence):
        raise ValueError(
            "DNA sequence contains characters outside A/T/C/G."
        )

    if not isinstance(Reframe, bool):
        raise TypeError("Reframe must be True or False.")

    if not isinstance(Axis, bool):
        raise TypeError("Axis must be True or False.")

    # ------------------------------------------------------------
    # Construct dsDNA representation
    # ------------------------------------------------------------
    dna = dsDNA()

    dna.ConstructFromShape(
        data.ground_state,
        sequence
    )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = dna.plot3D(
        Reframe=Reframe,
        Axis=Axis
    )

    return fig, ax
