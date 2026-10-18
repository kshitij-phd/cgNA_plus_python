import sys
#sys.path.append('./modules')
#import cgDNAUtils as tools
from ..modules import cgDNAUtils as tools
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np 

class dsDNA:
    
    def __init__(self):
        self.empty = True
        self.R,self.r = {},{}
        self.Rw,self.rw = {},{}
        self.Rc,self.rc = {},{}
        self.Rpw,self.rpw = {},{}
        self.Rpc,self.rpc = {},{}
        self.seq = {}
        self.shape={}
        self.nbp = {}
        
    def ConstructFromShape(self,s,sequence):
        self.empty = False
        self.seq = sequence
        self.shape = s 
        self.nbp = int(len(sequence))
        (R, r, Rc, rc, Rw, rw, Rpw , rpw , Rpc, rpc)=tools.frames(s)
        
        for i in range(len(sequence)):
            self.R[i],self.r[i] = R[i],r[i]
            self.Rw[i],self.rw[i] = Rw[i],rw[i]
            self.Rc[i],self.rc[i] = Rc[i],rc[i]
            self.Rpw[i],self.rpw[i] = Rpw[i],rpw[i]
            self.Rpc[i],self.rpc[i] = Rpc[i],rpc[i]
            
    def ConstructFromFrames(self,fraFile,pfraFile):
        return 0 



    def getBaseConfig(self,bp_nbr,CompStrand):

        if CompStrand:		
            dsign = -1
            R,r = self.Rc[bp_nbr],self.rc[bp_nbr]
            base = tools.comp(self.seq[bp_nbr])
        else:
            dsign = 1
            R,r = self.Rw[bp_nbr],self.rw[bp_nbr]
            base = self.seq[bp_nbr]            

        if base == 'A':
            color = "r"
            dy = dsign*6
        elif base == 'T':
            color = "b"
            dy = dsign*4.2
        elif base == 'C':
            color = "y"
            dy = dsign*4.2
        elif base == 'G':
            dy = dsign*6
            color = "g"
            
        dz, dx, Y = 0.36, 4.20, -1.2*dsign     


        verteces_RB =[ [-0.5*dx, -0.5*Y      , -0.5*dz],
                       [ 0.5*dx, -0.5*Y      , -0.5*dz],
                       [ 0.5*dx,  dy - 0.5*Y , -0.5*dz],
                       [-0.5*dx,  dy - 0.5*Y , -0.5*dz],
                       [-0.5*dx, -0.5*Y      ,  0.5*dz],
                       [ 0.5*dx, -0.5*Y      ,  0.5*dz],
                       [ 0.5*dx,  dy - 0.5*Y ,  0.5*dz],
                       [-0.5*dx,  dy - 0.5*Y ,  0.5*dz]]
     
        RB_conf = np.add(np.transpose(np.matmul(R, np.transpose(verteces_RB))), np.tile(np.transpose(r), [8, 1]))

        return color, RB_conf

    def getPhosphateConfig(self, R, r, size=1.0):
        """
        Construct a solid tetrahedral representation of one phosphate
        rigid body.

        R : (3,3) ndarray
            Phosphate orientation matrix.

        r : (3,) ndarray
            Phosphate position in Angstrom.

        size : float
            Characteristic tetrahedron size in Angstrom.
        """

        R = np.asarray(R, dtype=float)
        r = np.asarray(r, dtype=float).reshape(3)

        if R.shape != (3, 3):
            raise ValueError(
                f"Invalid phosphate rotation matrix shape: {R.shape}"
            )

        if not np.all(np.isfinite(R)):
            raise ValueError(
                "Phosphate rotation matrix contains non-finite values."
            )

        if not np.all(np.isfinite(r)):
            raise ValueError(
                "Phosphate position contains non-finite values."
            )

        if size <= 0:
            raise ValueError("Phosphate size must be positive.")

        # ------------------------------------------------------------
        # Regular tetrahedron in the local phosphate frame
        # ------------------------------------------------------------
        #
        # One vertex points approximately along +z and the remaining
        # three vertices form the triangular base.
        #
        h = size

        vertices_local = np.array([
            [0.0, 0.0, h],
            [ h, 0.0, 0.0],
            [-0.5 * h,  np.sqrt(3.0) * 0.5 * h, 0.0],
            [-0.5 * h, -np.sqrt(3.0) * 0.5 * h, 0.0],
        ])

        # Center tetrahedron at the phosphate reference position.
        vertices_local -= np.mean(vertices_local, axis=0)

        # Transform into the global cgNA+ frame.
        vertices_global = (
            R @ vertices_local.T
        ).T + r

        return vertices_global

    def getRigidBodies(self):   
        
        color, colorC, RB, RBC = {}, {}, {}, {} 

        for i in range(self.nbp):
            color[i], RB[i] = self.getBaseConfig(i, False) 
            colorC[i], RBC[i] = self.getBaseConfig(i, True)
        return RBC, colorC, RB, color

    def plot3D(self, Reframe=True, Axis=False):
        """
        Plot the reconstructed cgDNA structure as Watson/Crick rigid bodies.

        Parameters
        ----------
        Reframe : bool
            If True, express the complete structure in the coordinate frame
            of the first Watson base pair. This removes arbitrary global
            translation and rotation without changing internal geometry.

        Axis : bool
            If True, show x/y/z axes.
            If False, hide the axes.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the 3D DNA structure.

        ax : matplotlib.axes._subplots.Axes3DSubplot
            3D axes.
        """

        if self.empty:
            raise RuntimeError("Cannot plot an empty dsDNA object.")

        # ------------------------------------------------------------
        # Obtain rigid-body geometry
        # ------------------------------------------------------------
        RBC, colorC, RB, color = self.getRigidBodies()

        # ------------------------------------------------------------
        # Optional global reframe
        #
        # We use the first Watson base-pair frame:
        #
        #     r' = R0^T (r - r0)
        #
        #     R' = R0^T R
        #
        # This is a rigid-body transformation and therefore preserves
        # all relative distances and orientations.
        # ------------------------------------------------------------
        if Reframe:

            R0 = np.asarray(self.Rw[0], dtype=float)
            r0 = np.asarray(self.rw[0], dtype=float).reshape(3)

            # Check that R0 is a proper rotation matrix.
            rotation_error = np.linalg.norm(R0.T @ R0 - np.eye(3))
            determinant = np.linalg.det(R0)

            if rotation_error > 1e-8:
                raise ValueError(
                    f"Invalid first Watson rotation matrix: "
                    f"orthogonality error = {rotation_error:.3e}"
                )

            if determinant <= 0:
                raise ValueError(
                    f"Invalid first Watson rotation matrix: "
                    f"det(R0) = {determinant:.6f}"
                )

            Q = R0.T

            # Transform all vertices.
            for i in range(self.nbp):

                RB[i] = (Q @ (RB[i] - r0).T).T
                RBC[i] = (Q @ (RBC[i] - r0).T).T

        # ------------------------------------------------------------
        # Create figure
        # ------------------------------------------------------------
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection="3d")

        # ------------------------------------------------------------
        # Draw Watson and Crick bases
        # ------------------------------------------------------------
        all_vertices = []

        for bp in range(self.nbp):

            # -------------------------
            # Watson strand
            # -------------------------
            k = RB[bp]
            c = color[bp]

            lower = [
                tuple(k[j]) for j in [0, 1, 2, 3]
            ]

            upper = [
                tuple(k[j]) for j in [4, 5, 6, 7]
            ]

            poly = Poly3DCollection(
                [lower, upper],
                alpha=1,
                facecolors=c,
                edgecolors="black",
                linewidths=0.7,
            )

            ax.add_collection3d(poly)

            # -------------------------
            # Crick strand
            # -------------------------
            k = RBC[bp]
            c = colorC[bp]

            lower = [
                tuple(k[j]) for j in [0, 1, 2, 3]
            ]

            upper = [
                tuple(k[j]) for j in [4, 5, 6, 7]
            ]

            poly = Poly3DCollection(
                [lower, upper],
                alpha=1,
                facecolors=c,
                edgecolors="black",
                linewidths=0.7,
            )

            ax.add_collection3d(poly)

            all_vertices.extend(RB[bp])
            all_vertices.extend(RBC[bp])

            # ------------------------------------------------------------
            # Draw phosphate groups
            #
            # Watson: indices 1 ... N-1
            # Crick : indices 0 ... N-2
            #
            # The terminal entries are intentionally empty in frames().
            # ------------------------------------------------------------

            phosphate_vertices = []

            phosphate_W_color = "orange"
            phosphate_C_color = "purple"

            # -------------------------
            # Watson phosphates
            # -------------------------
            for i in range(1, self.nbp):

                vertices = self.getPhosphateConfig(
                    self.Rpw[i],
                    self.rpw[i],
                    size=1.0,
                )

                if Reframe:
                    vertices = (Q @ (vertices - r0).T).T

                faces = [
                    [vertices[0], vertices[1], vertices[2]],
                    [vertices[0], vertices[2], vertices[3]],
                    [vertices[0], vertices[3], vertices[1]],
                    [vertices[1], vertices[3], vertices[2]],
                ]

                poly = Poly3DCollection(
                    faces,
                    alpha=1.0,
                    facecolors=phosphate_W_color,
                    edgecolors="black",
                    linewidths=0.7,
                )

                ax.add_collection3d(poly)
                phosphate_vertices.extend(vertices)

            # -------------------------
            # Crick phosphates
            # -------------------------
            for i in range(0, self.nbp - 1):

                vertices = self.getPhosphateConfig(
                    self.Rpc[i],
                    self.rpc[i],
                    size=1.0,
                )

                if Reframe:
                    vertices = (Q @ (vertices - r0).T).T

                faces = [
                    [vertices[0], vertices[1], vertices[2]],
                    [vertices[0], vertices[2], vertices[3]],
                    [vertices[0], vertices[3], vertices[1]],
                    [vertices[1], vertices[3], vertices[2]],
                ]

                poly = Poly3DCollection(
                    faces,
                    alpha=1.0,
                    facecolors=phosphate_C_color,
                    edgecolors="black",
                    linewidths=0.7,
                )

                ax.add_collection3d(poly)
                phosphate_vertices.extend(vertices)

            all_vertices.extend(phosphate_vertices)   

        # ------------------------------------------------------------
        # Determine limits from actual geometry
        # ------------------------------------------------------------
        all_vertices = np.asarray(all_vertices, dtype=float)

        xyz_min = np.min(all_vertices, axis=0)
        xyz_max = np.max(all_vertices, axis=0)

        center = 0.5 * (xyz_min + xyz_max)
        extent = np.max(xyz_max - xyz_min)

        if extent <= 0:
            raise ValueError("Degenerate 3D geometry: zero spatial extent.")

        # Small margin around the structure.
        margin = 0.08 * extent
        half = 0.5 * extent + margin

        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)

        # Equal aspect ratio is important for DNA geometry.
        ax.set_box_aspect((1, 1, 1))

        # ------------------------------------------------------------
        # Axis option — MATLAB compatible meaning
        # Axis=False -> hide axes
        # Axis=True  -> show axes
        # ------------------------------------------------------------
        if Axis:
            ax.set_xlabel("x (Å)")
            ax.set_ylabel("y (Å)")
            ax.set_zlabel("z (Å)")
        else:
            ax.set_axis_off()

        ax.set_title(
            f"cgNA+ 3D Ground-State Structure ({self.nbp} bp)"
        )

        return fig, ax          
