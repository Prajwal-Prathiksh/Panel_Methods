import os
import math as math
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import ntpath
from numba import njit
from scipy import interpolate
np.seterr('raise')


@njit
def is_complex(x):
    """
    Check if a value is complex

    Parameters
    ----------
    x : float
        Value to check

    Returns
    -------
    bool
        True if complex, False otherwise
    """
    if np.imag(x) == 0:
        return False
    else:
        return True


def compute_circulation(a, b, x0, y0, numT, Vx, Vy, X, Y):
    """
    Compute the circulation around the defined ellipse

    Parameters
    ----------
    a : float
        Horizontal axis half-length
    b : float
        Vertical axis half-length
    x0 : float
        Ellipse center X coordinate
    y0 : float
        Ellipse center Y coordinate
    numT : int
        Number of points for integral
    Vx : ndarray
        Array of X-grid values
    Vy : ndarray
        Array of Y-grid values

    Returns
    -------
    Gamma : float
        Circulation [length^2/time]
    xC : ndarray
        X-values of integral curve [numT x 1]
    yC : ndarray
        Y-values of integral curve [numT x 1]
    VxC : ndarray
        Velocity X-component on integral curve [numT x 1]
    VyC : ndarray
        Velocity Y-component on integral curve [numT x 1]
    """

    # Discretized ellipse into angles [rad]
    t = np.linspace(0, 2 * np.pi, numT)

    # X coordinates of ellipse
    xC = a * np.cos(t) + x0

    # Y coordinates of ellipse
    yC = b * np.sin(t) + y0

    # Interpolate X velocities from grid to ellipse points
    fx = interpolate.RectBivariateSpline(Y, X, Vx)

    # Interpolate Y velocities from grid to ellipse points
    fy = interpolate.RectBivariateSpline(Y, X, Vy)

    # X velocity component on ellipse
    VxC = fx.ev(yC, xC)

    # Y velocity component on ellipse
    VyC = fy.ev(yC, xC)

    # Compute integral using trapezoid rule
    Gamma = -(np.trapz(VxC, xC) + np.trapz(VyC, yC))

    return Gamma, xC, yC, VxC, VyC


@njit
def compute_kl_vpm(XC, YC, XB, YB, phi, S):
    """
    Compute the integral expression for constant strength vortex panels.
    Vortex panel strengths are constant, but can change from panel to panel.
    Geometric integral for panel-normal    : K(ij).
    Geometric integral for panel-tangential: L(ij).

    Parameters
    ----------
    XC  : ndarray
        X-coordinate of control points
    YC  : ndarray
        Y-coordinate of control points
    XB  : ndarray
        X-coordinate of boundary points
    YB  : ndarray
        Y-coordinate of boundary points
    phi : float
        Angle between positive X-axis and interior of panel
    S   : float
        Length of panel

    Returns
    -------
    K   : float
        Value of panel-normal integral (Ref [1])
    L   : float
        Value of panel-tangential integral (Ref [2])
    """

    # Number of panels
    # Number of panels
    numPan = len(XC)

    # Initialize arrays
    # Initialize K integral matrix
    K = np.zeros(numPan * numPan).reshape(numPan, numPan)
    # Initialize L integral matrix
    L = np.zeros(numPan * numPan).reshape(numPan, numPan)

    # Compute integral
    # Loop over i panels
    for i in range(numPan):
        # Loop over j panels
        for j in range(numPan):

            # If panel j is not the same as panel i
            if (j != i):
                # Compute intermediate values

                # A term
                A = -(XC[i] - XB[j]) * np.cos(phi[j]) - \
                    (YC[i] - YB[j]) * np.sin(phi[j])

                # B term
                B = (XC[i] - XB[j])**2 + (YC[i] - YB[j])**2

                # C term (normal)
                Cn = -np.cos(phi[i] - phi[j])

                # D term (normal)
                Dn = (XC[i] - XB[j]) * np.cos(phi[i]) + \
                    (YC[i] - YB[j]) * np.sin(phi[i])

                # C term (tangential)
                Ct = np.sin(phi[j] - phi[i])

                # D term (tangential)
                Dt = (XC[i] - XB[j]) * np.sin(phi[i]) - \
                    (YC[i] - YB[j]) * np.cos(phi[i])

                # E term
                E = np.sqrt(B - A**2)

                # If E term is 0 or complex or a NAN or an INF
                if (E == 0 or is_complex(E) or np.isnan(E) or np.isinf(E)):
                    # Set K value equal to zero
                    K[i, j] = 0
                    # Set L value equal to zero
                    L[i, j] = 0

                else:
                    # Compute K

                    # First term in K equation
                    term1 = 0.5 * Cn * np.log((S[j]**2 + 2 * A * S[j] + B) / B)

                    # Second term in K equation
                    term2 = ((Dn - A * Cn) / E) * \
                        (math.atan2((S[j] + A), E) - math.atan2(A, E))

                    # Compute K integral
                    K[i, j] = term1 + term2

                    # Compute L

                    # First term in L equation
                    term1 = 0.5 * Ct * np.log((S[j]**2 + 2 * A * S[j] + B) / B)

                    # Second term in L equation
                    term2 = ((Dt - A * Ct) / E) * \
                        (math.atan2((S[j] + A), E) - math.atan2(A, E))

                    # Compute L integral
                    L[i, j] = term1 + term2

            # Zero out any problem values
            # If K term is complex or a NAN or an INF
            if (is_complex(K[i, j]) or np.isnan(K[i, j]) or np.isinf(K[i, j])):
                # Set K value equal to zero
                K[i, j] = 0

            # If L term is complex or a NAN or an INF
            if (is_complex(L[i, j]) or np.isnan(L[i, j]) or np.isinf(L[i, j])):
                # Set L value equal to zero
                L[i, j] = 0

    # Return both K and L matrices
    return K, L


@njit
def streamline_vpn(XP, YP, XB, YB, phi, S):
    """
    Compute the integral expression for constant strength vortex panels.
    Vortex panel strengths are constant, but can change from panel to panel.
    Geometric integral for X-direction: Nx(pj).
    Geometric integral for Y-direction: Ny(pj).

    Parameters
    ----------
    XP  : float
        X-coordinate of computation point, P
    YP  : float
        Y-coordinate of computation point, P
    XB  : ndarray
        X-coordinate of boundary points
    YB  : ndarray
        Y-coordinate of boundary points
    phi : float
        Angle between positive X-axis and interior of panel
    S   : float
        Length of panel

    Returns
    -------
    Nx  : float
        Value of X-direction geometric integral
    Ny  : float
        Value of Y-direction geometric integral
    """

    # Number of panels
    # Number of panels (control points)
    numPan = len(XB) - 1

    # Initialize arrays
    # Initialize Nx integral array
    Nx = np.zeros(numPan)
    # Initialize Ny integral array
    Ny = np.zeros(numPan)

    # Compute Nx and Ny
    # Loop over all panels
    for j in range(numPan):
        # Compute intermediate values

        # A term
        A = -(XP - XB[j]) * np.cos(phi[j]) - (YP - YB[j]) * np.sin(phi[j])

        # B term
        B = (XP - XB[j])**2 + (YP - YB[j])**2

        # Cx term (X-direction)

        Cx = np.sin(phi[j])

        # Dx term (X-direction)
        Dx = -(YP - YB[j])

        # Cy term (Y-direction)
        Cy = -np.cos(phi[j])

        # Dy term (Y-direction)
        Dy = XP - XB[j]

        # E term
        E = math.sqrt(B - A**2)

        # If E term is 0 or complex or a NAN or an INF
        if (E == 0 or is_complex(E) or np.isnan(E) or np.isinf(E)):
            # Set Nx value equal to zero
            Nx[j] = 0

            # Set Ny value equal to zero
            Ny[j] = 0
        else:
            # Compute Nx, Ref [1]
            # First term in Nx equation
            term1 = 0.5 * Cx * np.log((S[j]**2 + 2 * A * S[j] + B) / B)

            # Second term in Nx equation
            term2 = ((Dx - A * Cx) / E) * \
                (math.atan2((S[j] + A), E) - math.atan2(A, E))

            # Compute Nx integral
            Nx[j] = term1 + term2

            # Compute Ny, Ref [1]
            # First term in Ny equation
            term1 = 0.5 * Cy * np.log((S[j]**2 + 2 * A * S[j] + B) / B)

            # Second term in Ny equation
            term2 = ((Dy - A * Cy) / E) * \
                (math.atan2((S[j] + A), E) - math.atan2(A, E))

            # Compute Ny integral
            Ny[j] = term1 + term2

        # Zero out any problem values
        # If Nx term is complex or a NAN or an INF
        if (is_complex(Nx[j]) or np.isnan(Nx[j]) or np.isinf(Nx[j])):
            # Set Nx value equal to zero
            Nx[j] = 0

        # If Ny term is complex or a NAN or an INF
        # if (np.iscomplex(Ny[j]) or np.isnan(Ny[j]) or np.isinf(Ny[j])):
        if (np.isnan(Ny[j]) or np.isinf(Ny[j])):
            # Set Ny value equal to zero
            Ny[j] = 0

    return Nx, Ny


class XFOIL_CLASS:
    """
    Class for interfacing with XFOIL.

    Parameters
    ----------
    NACA : str
        NACA airfoil name
    PPAR : float
        PPAR menu options
    AoA : float
        Angle of attack (in degrees)
    load_type : str
        To load the airfoil or create a new one (default: 'create')
    """

    def __init__(self, NACA, PPAR, AoA, load_type='create'):
        self.NACA = NACA
        self.PPAR = PPAR
        self.AoA = AoA
        self.load_type = load_type

        if self.load_type == "create":
            self.airfoil_name = self.NACA
            self.xFoilResults = dict(airfoil_name=self.airfoil_name)

        elif self.load_type == "load":
            # Create GUI for open file dialog box
            root = Tk()
            # File types allowed to be loaded
            ftypes = [('dat file', "*.dat")]
            # Title of the dialog box GUI
            ttl = "Select Airfoil File"
            # Initial directory of the dialog box GUI
            dir1 = '/Airfoil_DAT_Selig/'
            # Needed for closing the Tk window later
            root.withdraw()
            # Needed for closing the Tk window later
            root.update()

            # User input of airfoil file to load
            root.fileName = askopenfilename(filetypes=ftypes,
                                            initialdir=dir1,
                                            title=ttl)
            # Destroy the Tk window
            root.destroy()

            # https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
            head, tail = ntpath.split(root.fileName)
            # Retain only airfoil name, not extension
            self.airfoil_name = tail[0:len(tail) - 4]

            self.xFoilResults = dict(airfoil_name=self.airfoil_name)


def create_elliptical_panels(numB, a=1., b=1.):
    """
    Creates an ellipse with a specified number of panels, such that
    a^2 + b^2 = 1.
    Note: If a and b are equal, a circle of radius 1 is created.

    Parameters
    ----------
    numB : int
        Number of boundary points
    a : float, default = 1.
        Semi-major axis of ellipse
    b : float, default = 1.
        Semi-minor axis of ellipse

    Returns
    -------
    XB : ndarray
        X-coordinate of boundary points
    YB : ndarray
        Y-coordinate of boundary points
    numPan : int
        Number of panels
    """
    # Boundary point angle offset [deg]
    tO = (360 / (numB - 1)) / 2

    # Normalise axes' length
    if a == b:
        a = b = 1.
    else:
        norm = math.sqrt(a**2 + b**2)
        a, b = a / norm, b / norm

    # Angles used to compute boundary points
    # Create angles for computing boundary point locations [deg]
    theta = np.linspace(0, 360, numB)

    # Add panel angle offset [deg]
    theta = theta + tO

    # Convert from degrees to radians [rad]
    theta = theta * (np.pi / 180)

    # Boundary points
    # Compute boundary point X-coordinate [radius of 1]
    XB = a * np.cos(theta)

    # Compute boundary point Y-coordinate [radius of 1]
    YB = b * np.sin(theta)

    # Number of panels
    # Number of panels (control points)
    numPan = len(XB) - 1

    return XB, YB, numPan


def correct_panels_orientation(numPan, XB, YB):
    # Check for direction of points
    edge = np.zeros(numPan)

    # Loop over all panels
    for i in range(numPan):
        # Compute edge values
        edge[i] = (XB[i + 1] - XB[i]) * (YB[i + 1] + YB[i])

    # Sum of all edge values
    sumEdge = np.sum(edge)

    # If panels are CCW, flip them (don't if CW)
    if (sumEdge < 0):
        # Flip the X-data array
        XB = np.flipud(XB)

        # Flip the Y-data array
        YB = np.flipud(YB)

    return XB, YB


def compute_panel_geometries(numPan, XB, YB, AoA):

    # Convert AoA to radians [rad]
    AoAR = AoA * (np.pi / 180)

    # Initialize control point X-coordinate
    XC = np.zeros(numPan)

    # Initialize control point Y-coordinate
    YC = np.zeros(numPan)

    # Initialize panel length array
    S = np.zeros(numPan)

    # Initialize panel orientation angle array
    phi = np.zeros(numPan)

    # Find geometric quantities of the airfoil
    # Loop over all panels
    for i in range(numPan):
        # X-value of control point
        XC[i] = 0.5 * (XB[i] + XB[i + 1])

        # Y-value of control point
        YC[i] = 0.5 * (YB[i] + YB[i + 1])

        # Change in X between boundary points
        dx = XB[i + 1] - XB[i]

        # Change in Y between boundary points
        dy = YB[i + 1] - YB[i]

        # Length of the panel
        S[i] = (dx**2 + dy**2)**0.5

        # Angle of panel (positive X-axis to inside face)
        phi[i] = math.atan2(dy, dx)

        # Make all panel angles positive [rad]
        if (phi[i] < 0):
            phi[i] = phi[i] + 2 * np.pi

    # Compute angle of panel normal w.r.t. horizontal and include AoA
    # Angle of panel normal [rad]
    delta = phi + (np.pi / 2)

    # Angle of panel normal and AoA [rad]
    beta = delta - AoAR

    # Make all panel angles between 0 and 2pi [rad]
    beta[beta > 2 * np.pi] = beta[beta > 2 * np.pi] - 2 * np.pi

    return XC, YC, S, beta, delta, phi


def populate_matrices_vpm(numPan, K, beta, Vinf):
    # Populate A matrix
    # Initialize the A matrix
    A = np.zeros([numPan, numPan])
    # Loop over all i panels
    for i in range(numPan):
        # Loop over all j panels
        for j in range(numPan):
            # If the panels are the same
            if (i == j):
                # Set A equal to zero
                A[i, j] = 0

            # If panels are not the same
            else:
                # Set A equal to negative geometric integral
                A[i, j] = -K[i, j]

    # Populate b array
    # Initialize the b array
    b = np.zeros(numPan)
    # Loop over all panels
    for i in range(numPan):
        # Compute RHS array
        b[i] = -Vinf * 2 * np.pi * np.cos(beta[i])

    return A, b


def satisfy_kutta_condition_vpm(numPan, A, b, pct=100):
    # Satisfy the Kutta condition
    # Replace this panel with Kutta condition equation
    panRep = int((pct / 100) * numPan) - 1
    # If we specify the last panel
    if (panRep >= numPan):
        # Set appropriate replacement panel index
        panRep = numPan - 1

    # Set all colums of the replaced panel equation to zero
    A[panRep, :] = 0
    # Set first column of replaced panel equal to 1
    A[panRep, 0] = 1
    # Set last column of replaced panel equal to 1
    A[panRep, numPan - 1] = 1
    # Set replaced panel value in b array equal to zero
    b[panRep] = 0

    return A, b


def compute_panel_velocities(numPan, gamma, beta, L, Vinf):
    # Compute velocities
    # Initialize tangential velocity array
    Vt = np.zeros(numPan)
    # Initialize pressure coefficient array
    Cp = np.zeros(numPan)
    # Loop over all i panels
    for i in range(numPan):
        # Reset summation value to zero
        addVal = 0
        # Loop over all j panels
        for j in range(numPan):
            # Sum all tangential vortex panel terms
            addVal = addVal - (gamma[j] / (2 * np.pi)) * L[i, j]

        # Compute tangential velocity by adding uniform flow and i=j terms
        Vt[i] = Vinf * np.sin(beta[i]) + addVal + gamma[i] / 2
        # Compute pressure coefficient
        Cp[i] = 1 - (Vt[i] / Vinf)**2

    return Vt, Cp


def compute_force_coefficients(XC, phi, beta, AoAR, Cp, S):
    # Compute normal and axial force coefficients
    # Normal force coefficient []
    CN = -Cp * S * np.sin(beta)
    # Axial force coefficient []
    CA = -Cp * S * np.cos(beta)

    # Compute lift and drag coefficients
    # Decompose axial and normal to lift coefficient []
    CL = sum(CN * np.cos(AoAR)) - sum(CA * np.sin(AoAR))
    # Decompose axial and normal to drag coefficient []
    CD = sum(CN * np.sin(AoAR)) + sum(CA * np.cos(AoAR))
    CM = sum(Cp * (XC - 0.25) * S * np.cos(phi))
    return CN, CA, CL, CD, CM
