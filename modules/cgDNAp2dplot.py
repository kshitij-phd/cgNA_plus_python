import numpy as np
import matplotlib.pyplot as plt
import mplcursors

from . import cgDNAUtils as tools


def cgDNAp2dplot(
    data,
    RotDeg=True,
    FontSize=12,
    LineWidth=1.0,
    Marker="o",
    MarkerSize=4,
):
    """
    MATLAB cgDNAp2dplot-equivalent visualization for a single cgDNA object.

    Two figures are generated:

    Figure 1:
        12 cgDNA coordinates
        - Tilt, Roll, Twist
        - Shift, Slide, Rise
        - Buckle, Propeller, Opening
        - Shear, Stretch, Stagger

    Figure 2:
        12 phosphate coordinates
        - W Rot 1, W Rot 2, W Rot 3
        - W Tra 1, W Tra 2, W Tra 3
        - C Rot 1, C Rot 2, C Rot 3
        - C Tra 1, C Tra 2, C Tra 3

    Interaction:
        Click a data point to display only:

            x = ...
            y = ...

        No additional annotation text is added.

    Parameters
    ----------
    data : cgDNA object
        cgDNA object containing ground_state, sequence and nbp.

    RotDeg : bool
        If True, rotational coordinates are displayed in degrees.
        If False, rotational coordinates remain in rad/5.

    FontSize : float
        Font size for axes labels, titles and tick labels.

    LineWidth : float
        Line width.

    Marker : str
        Matplotlib marker style.

    MarkerSize : float
        Marker size.

    Returns
    -------
    fig_cgDNA : matplotlib.figure.Figure
        Figure containing the 12 cgDNA coordinates.

    fig_pho : matplotlib.figure.Figure
        Figure containing the 12 phosphate coordinates.
    """

    # ============================================================
    # 1. INPUT VALIDATION
    # ============================================================

    if not hasattr(data, "ground_state"):
        raise TypeError(
            "Input object must contain 'ground_state'."
        )

    if not hasattr(data, "nbp"):
        raise TypeError(
            "Input object must contain 'nbp'."
        )

    if not hasattr(data, "seq"):
        raise TypeError(
            "Input object must contain 'seq'."
        )

    if FontSize <= 0:
        raise ValueError(
            "FontSize must be positive."
        )

    if LineWidth <= 0:
        raise ValueError(
            "LineWidth must be positive."
        )

    if MarkerSize <= 0:
        raise ValueError(
            "MarkerSize must be positive."
        )

    if not isinstance(
        RotDeg,
        (bool, np.bool_, int, np.integer)
    ):
        raise TypeError(
            "RotDeg must be 0/1 or False/True."
        )

    if RotDeg not in (0, 1, False, True):
        raise ValueError(
            "RotDeg must be 0 or 1."
        )

    # ============================================================
    # 2. DECOMPOSE cgNA+ GROUND STATE
    # ============================================================

    (
        intra_r,
        intra_t,
        pho_C_r,
        pho_C_t,
        inter_r,
        inter_t,
        pho_W_r,
        pho_W_t,
    ) = tools.DecomposeCoord(data.ground_state)

    N = data.nbp

    # ------------------------------------------------------------
    # Important:
    #
    # DecomposeCoord returns:
    #
    # intra_r : (3, N)
    # intra_t : (3, N)
    #
    # inter_r : (3, N-1)
    # inter_t : (3, N-1)
    #
    # pho_W_r : (3, N-1)
    # pho_W_t : (3, N-1)
    # pho_C_r : (3, N-1)
    # pho_C_t : (3, N-1)
    #
    # We KEEP this orientation.
    # ============================================================

    expected_shapes = {
        "intra_r": (3, N),
        "intra_t": (3, N),
        "inter_r": (3, N - 1),
        "inter_t": (3, N - 1),
        "pho_W_r": (3, N - 1),
        "pho_W_t": (3, N - 1),
        "pho_C_r": (3, N - 1),
        "pho_C_t": (3, N - 1),
    }

    actual_shapes = {
        "intra_r": intra_r.shape,
        "intra_t": intra_t.shape,
        "inter_r": inter_r.shape,
        "inter_t": inter_t.shape,
        "pho_W_r": pho_W_r.shape,
        "pho_W_t": pho_W_t.shape,
        "pho_C_r": pho_C_r.shape,
        "pho_C_t": pho_C_t.shape,
    }

    for name, expected in expected_shapes.items():

        actual = actual_shapes[name]

        if actual != expected:
            raise ValueError(
                f"Unexpected {name} shape: {actual}. "
                f"Expected {expected}."
            )

    # ============================================================
    # 3. ROTATIONAL COORDINATE CONVERSION
    # ============================================================

    if RotDeg:

        # cgNA+ rotational coordinates are represented
        # in rad/5.
        #
        # rad/5 -> degrees:
        #
        # (rad/5) * 5 * 180/pi
        # = (rad/5) * 900/pi
        #
        # In the cgNA+ Python implementation this is
        # conventionally written as:
        #
        # 36/pi
        #
        # for the stored nondimensional rotational variable.

        factor = 36.0 / np.pi

        intra_r = factor * intra_r
        inter_r = factor * inter_r
        pho_W_r = factor * pho_W_r
        pho_C_r = factor * pho_C_r

        rot_unit = "degrees"

    else:

        rot_unit = "rad/5"

    # ============================================================
    # 4. TITLES
    # ============================================================

    cgDNA_titles = [
        "Tilt",
        "Roll",
        "Twist",
        "Shift",
        "Slide",
        "Rise",
        "Buckle",
        "Propeller",
        "Opening",
        "Shear",
        "Stretch",
        "Stagger",
    ]

    cgDNA_ylabel = [
        f"({rot_unit})",
        f"({rot_unit})",
        f"({rot_unit})",
        "(Å)",
        "(Å)",
        "(Å)",
        f"({rot_unit})",
        f"({rot_unit})",
        f"({rot_unit})",
        "(Å)",
        "(Å)",
        "(Å)",
    ]

    phosphate_titles = [
        "W Rot 1",
        "W Rot 2",
        "W Rot 3",
        "W Tra 1",
        "W Tra 2",
        "W Tra 3",
        "C Rot 1",
        "C Rot 2",
        "C Rot 3",
        "C Tra 1",
        "C Tra 2",
        "C Tra 3",
    ]

    phosphate_ylabel = [
        f"({rot_unit})",
        f"({rot_unit})",
        f"({rot_unit})",
        "(Å)",
        "(Å)",
        "(Å)",
        f"({rot_unit})",
        f"({rot_unit})",
        f"({rot_unit})",
        "(Å)",
        "(Å)",
        "(Å)",
    ]

    # ============================================================
    # 5. SEQUENCE
    # ============================================================

    sequence = str(data.seq).upper()

    if len(sequence) != N:
        raise ValueError(
            f"Sequence length ({len(sequence)}) does not match "
            f"data.nbp ({N})."
        )

    # ============================================================
    # 6. X-COORDINATES
    # ============================================================

    # Base-pair coordinates
    x_bp = np.arange(1, N + 1)

    # Base-pair-step coordinates
    #
    # 1.5, 2.5, ..., N-0.5
    #
    # corresponding to MATLAB:
    #
    # x + 0.5
    #
    x_step = np.arange(1, N) + 0.5

    # Phosphate coordinates
    #
    # Watson: 2 ... N
    # Crick : 1 ... N-1
    #
    x_pho_W = np.arange(2, N + 1)
    x_pho_C = np.arange(1, N)

    xlim = (0, N + 1)

    # ============================================================
    # 7. FIGURE 1 — cgDNA COORDINATES
    # ============================================================

    fig_cgDNA, axes = plt.subplots(
        4,
        3,
        figsize=(16, 14),
        constrained_layout=True,
    )

    cgDNA_curves = [

        # Inter-base-pair rotational
        (x_step, inter_r[0, :]),   # Tilt
        (x_step, inter_r[1, :]),   # Roll
        (x_step, inter_r[2, :]),   # Twist

        # Inter-base-pair translational
        (x_step, inter_t[0, :]),   # Shift
        (x_step, inter_t[1, :]),   # Slide
        (x_step, inter_t[2, :]),   # Rise

        # Intra-base-pair rotational
        (x_bp, intra_r[0, :]),     # Buckle
        (x_bp, intra_r[1, :]),     # Propeller
        (x_bp, intra_r[2, :]),     # Opening

        # Intra-base-pair translational
        (x_bp, intra_t[0, :]),     # Shear
        (x_bp, intra_t[1, :]),     # Stretch
        (x_bp, intra_t[2, :]),     # Stagger
    ]

    # ============================================================
    # 8. PLOT cgDNA COORDINATES
    # ============================================================

    for j, ax in enumerate(axes.flat):

        x, y = cgDNA_curves[j]

        line, = ax.plot(
            x,
            y,
            linewidth=LineWidth,
            marker=Marker,
            markersize=MarkerSize,
            picker=True,
        )

        # --------------------------------------------------------
        # Axis limits
        # --------------------------------------------------------

        ax.set_xlim(*xlim)

        # --------------------------------------------------------
        # Base-pair labels
        # --------------------------------------------------------

        ax.set_xticks(x_bp)

        ax.set_xticklabels(
            list(sequence),
            fontsize=FontSize,
        )

        # --------------------------------------------------------
        # Labels
        # --------------------------------------------------------

        ax.set_ylabel(
            cgDNA_ylabel[j],
            fontsize=FontSize,
            labelpad=4,
        )

        if j < 6:
            xlabel = "basepair step"
        else:
            xlabel = "basepair"

        ax.set_xlabel(
            xlabel,
            fontsize=FontSize,
            labelpad=3,
        )

        ax.set_title(
            cgDNA_titles[j],
            fontsize=FontSize,
            pad=6,
        )

        # --------------------------------------------------------
        # Grid
        # --------------------------------------------------------

        ax.grid(
            True,
            alpha=0.35,
        )

        # --------------------------------------------------------
        # Tick size
        # --------------------------------------------------------

        ax.tick_params(
            axis="both",
            labelsize=FontSize - 1,
            pad=2,
        )

        # --------------------------------------------------------
        # CLICK-TO-DISPLAY x,y
        #
        # IMPORTANT:
        #
        # hover=False means annotation appears only after
        # clicking/selecting a data point.
        #
        # The annotation contains ONLY:
        #
        # x = ...
        # y = ...
        # --------------------------------------------------------

        cursor = mplcursors.cursor(
            line,
            hover=False,
        )

        @cursor.connect("add")
        def on_add(
            sel,
            x=x,
            y=y,
        ):

            index = int(sel.index)

            x_value = float(x[index])
            y_value = float(y[index])

            sel.annotation.set_text(
                f"x = {x_value:.6g}\n"
                f"y = {y_value:.6g}"
            )

    # ============================================================
    # FIGURE TITLE
    # ============================================================

    fig_cgDNA.suptitle(
        "cgDNA Ground-State Coordinates",
        fontsize=FontSize + 2,
    )

    # ============================================================
    # 9. FIGURE 2 — PHOSPHATE COORDINATES
    # ============================================================

    phosphate_coordinates = np.vstack(
        (
            pho_W_r,
            pho_W_t,
            pho_C_r,
            pho_C_t,
        )
    )

    fig_pho, axes = plt.subplots(
        4,
        3,
        figsize=(16, 14),
        constrained_layout=True,
    )

    # ============================================================
    # 10. PLOT PHOSPHATE COORDINATES
    # ============================================================

    for j, ax in enumerate(axes.flat):

        # --------------------------------------------------------
        # Watson coordinates
        # --------------------------------------------------------

        if j < 6:
            x = x_pho_W

        # --------------------------------------------------------
        # Crick coordinates
        # --------------------------------------------------------

        else:
            x = x_pho_C

        y = phosphate_coordinates[j, :]

        line, = ax.plot(
            x,
            y,
            linewidth=LineWidth,
            marker=Marker,
            markersize=MarkerSize,
            picker=True,
        )

        # --------------------------------------------------------
        # Axis limits
        # --------------------------------------------------------

        ax.set_xlim(*xlim)

        # --------------------------------------------------------
        # Base-pair labels
        # --------------------------------------------------------

        ax.set_xticks(x_bp)

        ax.set_xticklabels(
            list(sequence),
            fontsize=FontSize,
        )

        # --------------------------------------------------------
        # Axis labels
        # --------------------------------------------------------

        ax.set_ylabel(
            phosphate_ylabel[j],
            fontsize=FontSize,
            labelpad=4,
        )

        ax.set_xlabel(
            "basepair",
            fontsize=FontSize,
            labelpad=3,
        )

        ax.set_title(
            phosphate_titles[j],
            fontsize=FontSize,
            pad=6,
        )

        # --------------------------------------------------------
        # Grid
        # --------------------------------------------------------

        ax.grid(
            True,
            alpha=0.35,
        )

        # --------------------------------------------------------
        # Tick size
        # --------------------------------------------------------

        ax.tick_params(
            axis="both",
            labelsize=FontSize - 1,
            pad=2,
        )

        # --------------------------------------------------------
        # CLICK-TO-DISPLAY x,y
        # --------------------------------------------------------

        cursor = mplcursors.cursor(
            line,
            hover=False,
        )

        @cursor.connect("add")
        def on_add(
            sel,
            x=x,
            y=y,
        ):

            index = int(sel.index)

            x_value = float(x[index])
            y_value = float(y[index])

            sel.annotation.set_text(
                f"x = {x_value:.6g}\n"
                f"y = {y_value:.6g}"
            )

    # ============================================================
    # FIGURE TITLE
    # ============================================================

    fig_pho.suptitle(
        "cgNA+ Phosphate Coordinates",
        fontsize=FontSize + 2,
    )

    # ============================================================
    # 11. RETURN FIGURES
    # ============================================================

    return fig_cgDNA, fig_pho