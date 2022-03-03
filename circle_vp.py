###########################################################################
# Imports
###########################################################################
# Standard library imports
import argparse
import time as time
import numpy as np
import math as math
import matplotlib.pyplot as plt
from matplotlib import path


# Local imports
from helper_funcs import *

###########################################################################
# Code
###########################################################################


def cli_parser():
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-n', '--numb', action='store', dest='numB', type=int, required=True,
        help='Number of boundary points (including endpoint).'
    )
    parser.add_argument(
        '-v', '--vinf', action='store', dest='Vinf', type=float, default=1.,
        help='Free stream velocity.'
    )
    parser.add_argument(
        '-A', '--aoa', action='store', dest='AoA', type=float, default=0.,
        help='Angle of attack.'
    )
    parser.add_argument(
        '-a', '--ellipse-a', action='store', dest='ellipse_a', type=float,
        default=1., help='Semi-major axis of ellipse.'
    )
    parser.add_argument(
        '-b', '--ellipse-b', action='store', dest='ellipse_b', type=float,
        default=1., help='Semi-minor axis of ellipse.'
    )
    parser.add_argument(
        '--pct', action='store', dest='replacement_pct', type=float,
        default=100., help='Panel replacement percentage.'
    )
    parser.add_argument(
        '--dpi', action='store', dest='dpi', type=int, default=300.,
        help='DPI of output image.'
    )
    args = parser.parse_args()
    return args


# KNOWNS
args = cli_parser()

Vinf = args.Vinf
AoA = args.AoA
numB = args.numB

# Convert AoA to radians [rad]
AoAR = AoA * (np.pi / 180)

# Plotting flags
flagPlot = [1,      # Shape polygon with panel normal vectors
            1,      # Geometry boundary pts, control pts, first panel
            1,      # Analytical and SPM pressure coefficient plot
            1,      # Streamline plot
            1]      # Pressure coefficient contour plot

# Grid parameters
# X & Y grid for streamlines and contours
nGridX = nGridY = 150

# X-grid extents [min, max]
xVals = [-2, 2]

# Y-grid extents [min, max]
yVals = [-2, 2]

# %% FUNCTIONS
XB, YB, numPan = create_elliptical_panels(
    numB=numB, a=args.ellipse_a, b=args.ellipse_b
)
XB, YB = correct_panels_orientation(numPan, XB, YB)
XC, YC, S, beta, delta, phi = compute_panel_geometries(numPan, XB, YB, AoA)
K, L = compute_kl_vpm(XC, YC, XB, YB, phi, S)
A, b = populate_matrices_vpm(numPan, K, beta, Vinf)
A, b = satisfy_kutta_condition_vpm(numPan, A, b, pct=args.replacement_pct)

gamma = np.linalg.solve(A, b)
print("Sum of gamma: ", sum(gamma * S))

Vt, Cp = compute_panel_velocities(numPan, gamma, beta, L, Vinf)

# Analytical angles and pressure coefficients
# Analytical theta angles [rad]
analyticTheta = np.linspace(0, 2 * np.pi, 200)
# Analytical pressure coefficient []
analyticCP = 1 - 4 * np.sin(analyticTheta)**2

CN, CA, CL, CD, CM = compute_force_coefficients(XC, phi, beta, AoAR, Cp, S)

# Print the results to the Console
print("\n======= RESULTS =======")
print("Lift Coefficient (CL)")
# From Kutta-Joukowski lift equation
print(f"  K-J  : {2*sum(gamma*S)}")

# From this VPM code
print(f"  VPM  : {CL}")
print("\nMoment Coefficient (CM)")
print(f"  VPM  : {CM}")

# %% COMPUTE STREAMLINES - REF [4]

if (flagPlot[3] == 1 or flagPlot[4] == 1):
    # Streamline parameters
    # Percentage of streamlines of the grid
    slPct = 25
    # Create array of Y streamline starting points
    Ysl = np.linspace(yVals[0], yVals[1], int((slPct / 100) * nGridY))
    # Create array of X streamline starting points
    Xsl = xVals[0] * np.ones(len(Ysl))
    # Concatenate X and Y streamline starting points
    XYsl = np.vstack((Xsl.T, Ysl.T)).T

    # Generate the grid points
    # X-values in evenly spaced grid
    Xgrid = np.linspace(xVals[0], xVals[1], nGridX)
    # Y-values in evenly spaced grid
    Ygrid = np.linspace(yVals[0], yVals[1], nGridY)
    # Create meshgrid from X and Y grid arrays
    XX, YY = np.meshgrid(Xgrid, Ygrid)

    # Initialize velocities
    # Initialize X velocity matrix
    Vx = np.zeros([nGridX, nGridY])
    # Initialize Y velocity matrix
    Vy = np.zeros([nGridX, nGridY])

    # Path to figure out if grid point is inside polygon or not
    # Concatenate XB and YB geometry points
    AF = np.vstack((XB.T, YB.T)).T
    # Create a path for the geometry
    afPath = path.Path(AF)

    # Solve for grid point X and Y velocities
    tic = time.perf_counter()
    # Loop over X-grid points
    for m in range(nGridX):
        # Loop over Y-grid points
        for n in range(nGridY):
            # Current iteration's X grid point
            XP = XX[m, n]
            # Current iteration's Y grid point
            YP = YY[m, n]
            # Compute Nx and Ny geometric integrals
            Nx, Ny = streamline_vpn(XP, YP, XB, YB, phi, S)
            # Check if grid points are in object
            # - If they are, assign a velocity of zero
            # If (XP,YP) is in the body
            if afPath.contains_points([(XP, YP)]):
                # Set X-velocity equal to zero
                Vx[m, n] = 0
                # Set Y-velocity equal to zero
                Vy[m, n] = 0
            else:
                # Compute X-velocity
                Vx[m, n] = Vinf * np.cos(AoAR) + sum(-gamma * Nx / (2 * np.pi))
                # Compute Y-velocity
                Vy[m, n] = Vinf * np.sin(AoAR) + sum(-gamma * Ny / (2 * np.pi))
    toc = time.perf_counter()
    print("\n\nSTREAMLINE_VPM: %.2f seconds" % (toc - tic))

    # Compute grid point velocity magnitude and pressure coefficient
    # Compute magnitude of velocity vector []
    Vxy = np.sqrt(Vx**2 + Vy**2)
    # Pressure coefficient []
    CpXY = 1 - (Vxy / Vinf)**2

# %% PLOTTING

# FIGURE: Shape polygon with panel normal vectors
if (flagPlot[0] == 1):
    # Angles for "perfect" circle
    angCirc = np.linspace(0, 2 * np.pi, 1000)
    # "Perfect" circle X values
    xCirc = np.cos(angCirc)
    # "Perfect" circle Y values
    yCirc = np.sin(angCirc)
    # Create figure
    fig = plt.figure(1)
    # Clear the axes
    plt.cla()
    # Plot the circle that polygon is approximating
    plt.plot(xCirc, yCirc, 'k--')
    # Plot the paneled circle
    plt.fill(XB, YB, 'k')
    # Initialize 'X'
    X = np.zeros(2)
    # Initialize 'Y'
    Y = np.zeros(2)

    # Loop over all panels
    for i in range(numPan):
        # Set X start of panel orientation vector
        X[0] = XC[i]
        # Set X end of panel orientation vector
        X[1] = XC[i] + S[i] * np.cos(delta[i])
        # Set Y start of panel orientation vector
        Y[0] = YC[i]
        # Set Y end of panel orientation vector
        Y[1] = YC[i] + S[i] * np.sin(delta[i])
        # If it's the first panel index
        if (i == 0):
            # Plot the first panel
            plt.plot(X, Y, 'b-', label='First Panel')
        # If it's the second panel index
        elif (i == 1):
            # Plot the second panel
            plt.plot(X, Y, 'g-', label='Second Panel')
        # If it's neither the first nor second panel index
        else:
            # Plot the rest of the panels
            plt.plot(X, Y, 'r-')
    # Set X-label
    plt.xlabel('X-Axis')
    # Set Y-label
    plt.ylabel('Y-Axis')
    # Set title
    plt.title('Panel Geometry')
    # Set axes equal
    plt.axis('equal')
    # Show legend
    plt.legend()
    fname = os.path.join('figs', 'ellipses', 'panel_geometry.png')
    plt.savefig(fname, dpi=args.dpi, bbox_inches='tight')

# FIGURE: Geometry with the following indicated:
# - Boundary points, control points, first panel, second panel
if (flagPlot[1] == 1):
    # Create figure
    fig = plt.figure(2)
    # Get ready for plotting
    plt.cla()
    # Plot polygon
    plt.plot(XB, YB, 'k-', label='Panels')
    plt.plot([XB[0], XB[1]], [YB[0], YB[1]], 'b-',
             label='First Panel')            # Plot first panel
    plt.plot([XB[1], XB[2]], [YB[1], YB[2]], 'g-',
             label='Second Panel')           # Plot second panel

    # Plot boundary points
    plt.plot(
        XB, YB, 'ko', markerfacecolor='k', label='Boundary Points'
    )

    # Plot control points
    plt.plot(
        XC, YC, 'ko', markerfacecolor='r', label='Control Points'
    )
    # Set X-label
    plt.xlabel('X-Axis')
    # Set Y-label
    plt.ylabel('Y-Axis')
    # Set title
    plt.title('Panel Geometry 2')
    # Set axes equal
    plt.axis('equal')
    # Show legend
    plt.legend()
    fname = os.path.join('figs', 'ellipses', 'panel_geometry2.png')
    plt.savefig(fname, dpi=args.dpi, bbox_inches='tight')

# FIGURE: Analytical and SPM pressure coefficient
if (flagPlot[2] == 1):
    # Create figure
    fig = plt.figure(3)
    # Get ready for plotting
    plt.cla()

    # Plot analytical pressure coefficient
    plt.plot(
        analyticTheta * (180 / np.pi), analyticCP, 'b-', label='Analytical'
    )

    # Plot panel method pressure coefficient
    plt.plot(
        beta * (180 / np.pi), Cp, 'ks', markerfacecolor='r', label='VPM'
    )
    # Set X-label
    plt.xlabel('Angle [deg]')
    # Set Y-label
    plt.ylabel('Pressure Coefficient')
    # Set title
    plt.title('Pressure Coefficient Comparison')
    # Set X-limits
    plt.xlim(0, 360)
    # Set Y-limits
    plt.ylim(-3.5, 1.5)
    # Show legend
    plt.legend()
    fname = os.path.join('figs', 'ellipses', 'pressure_coefficient_comparison.png')
    plt.savefig(fname, dpi=args.dpi, bbox_inches='tight')

# FIGURE: Streamlines
if (flagPlot[3] == 1):
    # Create figure
    fig = plt.figure(5)
    # Get ready for plotting
    plt.cla()
    # Ignore underflow error message
    np.seterr(under="ignore")

    # Plot streamlines
    plt.streamplot(
        XX, YY, Vx, Vy, linewidth=0.5, density=40, color='r', arrowstyle='-',
        start_points=XYsl
    )
    plt.clim(vmin=0, vmax=2)
    # Plot airfoil as black polygon
    plt.fill(XB, YB, 'k')
    # Set X-label
    plt.xlabel('X Units')
    # Set Y-label
    plt.ylabel('Y Units')
    # Set axes equal
    plt.gca().set_aspect('equal')
    # Set X-limits
    plt.xlim(xVals)
    # Set Y-limits
    plt.ylim(yVals)
    fname = os.path.join('figs', 'ellipses', 'streamlines.png')
    plt.savefig(fname, dpi=args.dpi, bbox_inches='tight')

# FIGURE: Pressure coefficient contours
if (flagPlot[4] == 1):
    # Create figure
    fig = plt.figure(6)
    # Get ready for plotting
    plt.cla()
    # Plot contour
    plt.contourf(XX, YY, CpXY, 500, cmap='jet')
    # Plot airfoil as black polygon
    plt.fill(XB, YB, 'k')
    # Set X-label
    plt.xlabel('X Units')
    # Set Y-label
    plt.ylabel('Y Units')
    # Set axes equal
    plt.gca().set_aspect('equal')
    # Set X-limits
    plt.xlim(xVals)
    # Set Y-limits
    plt.ylim(yVals)
    plt.colorbar()
    fname = os.path.join('figs', 'ellipses', 'pressure_coefficient_contours.png')
    plt.savefig(fname, dpi=args.dpi, bbox_inches='tight')
