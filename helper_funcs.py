import math as math
import numpy as np
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
            term2 = ((Dx - A * Cx) / E) * (math.atan2((S[j] + A), E) - math.atan2(A, E))

            # Compute Nx integral
            Nx[j] = term1 + term2

            # Compute Ny, Ref [1]
            # First term in Ny equation
            term1 = 0.5 * Cy * np.log((S[j]**2 + 2 * A * S[j] + B) / B)

            # Second term in Ny equation
            term2 = ((Dy - A * Cy) / E) * (math.atan2((S[j] + A), E) - math.atan2(A, E))

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
