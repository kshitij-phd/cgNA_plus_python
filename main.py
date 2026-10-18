import numpy as np
import matplotlib
matplotlib.use("qtagg")
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.sparse.linalg import eigsh
from cgNA_plus.classes.cgDNAclass import cgDNA
from cgNA_plus.modules.cgDNAp2dplot import cgDNAp2dplot
from cgNA_plus.modules.cgDNAp3dplot import cgDNAp3dplot


# User settings
SEQUENCE = "GGCCGGGTCGTAGCAAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCTACCGCGTTTTAACCGCCAATAGGATTACTTACTAGTCTCTACG"
PARAMETER_SET = "ps2_cgf"
RUN_PLOTS = True
RUN_PDB = True


def main():
    print("=" * 60)
    print("cgNA+ DNA calculation")
    print("=" * 60)
    print(f"Sequence       : {SEQUENCE}")
    print(f"Number of bp   : {len(SEQUENCE)}")
    print(f"Parameter set  : {PARAMETER_SET}\n")

    # Construct cgNA+ model
    data = cgDNA(SEQUENCE, PARAMETER_SET)

    print("cgDNA construction: OK")
    print(f"Parameter file : {data.paramset}")
    print(f"Number of DOFs : {data.dofs}")
    print(f"Ground state   : {data.ground_state.shape}")
    print(f"Stiffness      : {data.stiff.shape}")
    print(f"Nonzero K      : {data.stiff.nnz}\n")

    # Save cgNA+ data in MATLAB-compatible format
    savemat(
        "cgNAplus_python.mat",
        {
            "sequence": SEQUENCE,
            "parameter_set": PARAMETER_SET,
            "nbp": data.nbp,
            "dofs": data.dofs,
            "ground_state": data.ground_state,
            "stiffness": data.stiff.toarray(),
        },
    )

    print("Saved: cgNAplus_python.mat\n")

    # Visualization
    if RUN_PLOTS:
        print("Generating interactive 2D plots...")

        fig_cgDNA, fig_pho = cgDNAp2dplot(
            data, RotDeg=True, FontSize=12, LineWidth=1.0,
            Marker="o", MarkerSize=4
        )

        fig_cgDNA.savefig("cgNAplus_2D_cgDNA.png", dpi=300, bbox_inches="tight")
        fig_pho.savefig("cgNAplus_2D_phosphate.png", dpi=300, bbox_inches="tight")

        print("Saved: cgNAplus_2D_cgDNA.png")
        print("Saved: cgNAplus_2D_phosphate.png")

        print("Generating 3D structure...")

        fig_3d, ax_3d = cgDNAp3dplot(
            data, SEQUENCE, Reframe=True, Axis=False
        )

        fig_3d.savefig("cgNAplus_3D.png", dpi=300, bbox_inches="tight")
        print("Saved: cgNAplus_3D.png")

    # PDB generation
    if RUN_PDB:
        print("Generating PDB...")
        data.makePDB()
        print("PDB generation: completed\n")

    print("=" * 60)
    print("cgNA+ calculation completed successfully")
    print("=" * 60)

    if RUN_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()