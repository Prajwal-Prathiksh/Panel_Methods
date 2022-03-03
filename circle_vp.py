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
    '-a', '--ellipse-a', action='store', dest='ellipse_a', type=float, default=1.,
    help='Semi-major axis of ellipse.'
    )    
    parser.add_argument(
    '-b', '--ellipse-b', action='store', dest='ellipse_b', type=float, default=1.,
    help='Semi-minor axis of ellipse.'
    )    
    parser.add_argument(
    '--pct', action='store', dest='replacement_pct', type=float,
    default=100., help='Panel replacement percentage.'
    )
    parser.add_argument(
    '--dpi', action='store', dest='dpi', type=int,
    default=300., help='DPI of output image.'
    )
    args = parser.parse_args()
    return args


# KNOWNS
args = cli_parser()

Vinf = args.Vinf
AoA  = args.AoA
numB = args.numB

# Convert AoA to radians [rad]
AoAR = AoA*(np.pi/180)           

# Plotting flags
close_plots = False
flagPlot = [1,      # Shape polygon with panel normal vectors
            1,      # Geometry boundary pts, control pts, first panel, second panel
            1,      # Analytical and SPM pressure coefficient plot
            1,      # Streamline plot
            1]      # Pressure coefficient contour plot

# Grid parameters
# X & Y grid for streamlines and contours
nGridX = nGridY = 150

# X-grid extents [min, max]
xVals    = [-2, 2]

# Y-grid extents [min, max]
yVals    = [-2, 2]

# %% FUNCTIONS
XB, YB, numPan = create_elliptical_panels(numB=numB, a=args.ellipse_a, b = args.ellipse_b)                                                        
XB, YB = correct_panels_orientation(numPan, XB, YB)
XC, YC, S, beta, delta, phi = compute_panel_geometries(numPan, XB, YB, AoA)
K, L = compute_kl_vpm(XC, YC, XB, YB, phi, S)
A, b = populate_matrices_vpm(numPan, K, beta, Vinf)
A, b = satisfy_kutta_condition_vpm(numPan, A, b, pct=args.replacement_pct)

gamma = np.linalg.solve(A,b)
print("Sum of gamma: ",sum(gamma*S))

Vt, Cp = compute_panel_velocities(numPan, gamma, beta, L, Vinf)

# Analytical angles and pressure coefficients
analyticTheta = np.linspace(0,2*np.pi,200)                                      # Analytical theta angles [rad]
analyticCP    = 1 - 4*np.sin(analyticTheta)**2                                  # Analytical pressure coefficient []

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
    slPct  = 25                                                                 # Percentage of streamlines of the grid
    Ysl    = np.linspace(yVals[0],yVals[1],int((slPct/100)*nGridY))             # Create array of Y streamline starting points
    Xsl    = xVals[0]*np.ones(len(Ysl))                                         # Create array of X streamline starting points
    XYsl   = np.vstack((Xsl.T,Ysl.T)).T                                         # Concatenate X and Y streamline starting points
    
    # Generate the grid points
    Xgrid  = np.linspace(xVals[0],xVals[1],nGridX)                              # X-values in evenly spaced grid
    Ygrid  = np.linspace(yVals[0],yVals[1],nGridY)                              # Y-values in evenly spaced grid
    XX, YY = np.meshgrid(Xgrid,Ygrid)                                           # Create meshgrid from X and Y grid arrays
    
    # Initialize velocities
    Vx     = np.zeros([nGridX,nGridY])                                          # Initialize X velocity matrix
    Vy     = np.zeros([nGridX,nGridY])                                          # Initialize Y velocity matrix
    
    # Path to figure out if grid point is inside polygon or not
    AF     = np.vstack((XB.T,YB.T)).T                                           # Concatenate XB and YB geometry points
    afPath = path.Path(AF)                                                      # Create a path for the geometry
    
    # Solve for grid point X and Y velocities
    tic = time.perf_counter()
    for m in range(nGridX):                                                     # Loop over X-grid points
        for n in range(nGridY):                                                 # Loop over Y-grid points
            XP     = XX[m,n]                                                    # Current iteration's X grid point
            YP     = YY[m,n]                                                    # Current iteration's Y grid point
            Nx, Ny = streamline_vpn(XP,YP,XB,YB,phi,S)                          # Compute Nx and Ny geometric integrals
            # Check if grid points are in object
            # - If they are, assign a velocity of zero
            if afPath.contains_points([(XP,YP)]):                               # If (XP,YP) is in the body
                Vx[m,n] = 0                                                     # Set X-velocity equal to zero
                Vy[m,n] = 0                                                     # Set Y-velocity equal to zero
            else:
                Vx[m,n] = Vinf*np.cos(AoAR) + sum(-gamma*Nx/(2*np.pi))          # Compute X-velocity
                Vy[m,n] = Vinf*np.sin(AoAR) + sum(-gamma*Ny/(2*np.pi))          # Compute Y-velocity
    toc = time.perf_counter()
    print("\n\nSTREAMLINE_VPM: %.2f seconds" % (toc-tic))

    # Compute grid point velocity magnitude and pressure coefficient
    Vxy  = np.sqrt(Vx**2 + Vy**2)                                               # Compute magnitude of velocity vector []
    CpXY = 1 - (Vxy/Vinf)**2                                                    # Pressure coefficient []

# %% PLOTTING

# FIGURE: Shape polygon with panel normal vectors
if (flagPlot[0] == 1):
    angCirc = np.linspace(0,2*np.pi,1000)                                       # Angles for "perfect" circle
    xCirc = np.cos(angCirc)                                                     # "Perfect" circle X values
    yCirc = np.sin(angCirc)                                                     # "Perfect" circle Y values
    fig = plt.figure(1)                                                         # Create figure
    plt.cla()                                                                   # Clear the axes
    plt.plot(xCirc,yCirc,'k--')                                                 # Plot the circle that polygon is approximating
    plt.fill(XB,YB,'k')                                                         # Plot the paneled circle
    X = np.zeros(2)                                                             # Initialize 'X'
    Y = np.zeros(2)                                                             # Initialize 'Y'
    for i in range(numPan):                                                     # Loop over all panels
        X[0] = XC[i]                                                            # Set X start of panel orientation vector
        X[1] = XC[i] + S[i]*np.cos(delta[i])                                    # Set X end of panel orientation vector
        Y[0] = YC[i]                                                            # Set Y start of panel orientation vector
        Y[1] = YC[i] + S[i]*np.sin(delta[i])                                    # Set Y end of panel orientation vector
        if (i == 0):                                                            # If it's the first panel index
            plt.plot(X,Y,'b-',label='First Panel')                              # Plot the first panel
        elif (i == 1):                                                          # If it's the second panel index
            plt.plot(X,Y,'g-',label='Second Panel')                             # Plot the second panel
        else:                                                                   # If it's neither the first nor second panel index
            plt.plot(X,Y,'r-')                                                  # Plot the rest of the panels
    plt.xlabel('X-Axis')                                                        # Set X-label
    plt.ylabel('Y-Axis')                                                        # Set Y-label
    plt.title('Panel Geometry')                                                 # Set title
    plt.axis('equal')                                                           # Set axes equal
    plt.legend()                                                                # Show legend
    fname = os.path.join('figs','panel_geometry.png')                          
    plt.savefig(fname, dpi=args.dpi, bbox_inches='tight')                                                  

# FIGURE: Geometry with the following indicated:
# - Boundary points, control points, first panel, second panel
if (flagPlot[1] == 1):
    fig = plt.figure(2)                                                         # Create figure
    plt.cla()                                                                   # Get ready for plotting
    plt.plot(XB,YB,'k-',label='Panels')                                         # Plot polygon
    plt.plot([XB[0], XB[1]],[YB[0], YB[1]],'b-',label='First Panel')            # Plot first panel
    plt.plot([XB[1], XB[2]],[YB[1], YB[2]],'g-',label='Second Panel')           # Plot second panel
    plt.plot(XB,YB,'ko',markerfacecolor='k',label='Boundary Points')            # Plot boundary points
    plt.plot(XC,YC,'ko',markerfacecolor='r',label='Control Points')             # Plot control points
    plt.xlabel('X-Axis')                                                        # Set X-label
    plt.ylabel('Y-Axis')                                                        # Set Y-label
    plt.title('Panel Geometry 2')                                               # Set title
    plt.axis('equal')                                                           # Set axes equal
    plt.legend()                                                                # Show legend
    fname = os.path.join('figs','panel_geometry2.png')                          
    plt.savefig(fname, dpi=args.dpi, bbox_inches='tight')                                                  

# FIGURE: Analytical and SPM pressure coefficient
if (flagPlot[2] == 1):
    fig = plt.figure(3)                                                         # Create figure
    plt.cla()                                                                   # Get ready for plotting
    plt.plot(analyticTheta*(180/np.pi),analyticCP,'b-',label='Analytical')      # Plot analytical pressure coefficient
    plt.plot(beta*(180/np.pi),Cp,'ks',markerfacecolor='r',label='VPM')          # Plot panel method pressure coefficient
    plt.xlabel('Angle [deg]')                                                   # Set X-label
    plt.ylabel('Pressure Coefficient')                                          # Set Y-label
    plt.title('Pressure Coefficient Comparison')                                # Set title
    plt.xlim(0, 360)                                                            # Set X-limits
    plt.ylim(-3.5, 1.5)                                                         # Set Y-limits
    plt.legend()                                                                # Show legend
    fname = os.path.join('figs','pressure_coefficient_comparison.png')                          
    plt.savefig(fname, dpi=args.dpi, bbox_inches='tight')                                          

# FIGURE: Streamlines
if (flagPlot[3] == 1):
    fig = plt.figure(5)                                                         # Create figure
    plt.cla()                                                                   # Get ready for plotting
    np.seterr(under="ignore")                                                   # Ignore underflow error message
    plt.streamplot(XX,YY,Vx,Vy, linewidth=0.5, density=40, color='r',           # Plot streamlines
                   arrowstyle='-', start_points=XYsl)
    plt.clim(vmin=0, vmax=2)
    plt.fill(XB,YB,'k')                                                         # Plot airfoil as black polygon
    plt.xlabel('X Units')                                                       # Set X-label
    plt.ylabel('Y Units')                                                       # Set Y-label
    plt.gca().set_aspect('equal')                                               # Set axes equal
    plt.xlim(xVals)                                                             # Set X-limits
    plt.ylim(yVals)                                                             # Set Y-limits
    fname = os.path.join('figs','streamlines.png')                         
    plt.savefig(fname, dpi=args.dpi, bbox_inches='tight')                                               

# FIGURE: Pressure coefficient contours
if (flagPlot[4] == 1):
    fig = plt.figure(6)                                                         # Create figure
    plt.cla()                                                                   # Get ready for plotting
    plt.contourf(XX,YY,CpXY,500,cmap='jet')                                     # Plot contour
    plt.fill(XB,YB,'k')                                                         # Plot airfoil as black polygon
    plt.xlabel('X Units')                                                       # Set X-label
    plt.ylabel('Y Units')                                                       # Set Y-label
    plt.gca().set_aspect('equal')                                               # Set axes equal
    plt.xlim(xVals)                                                             # Set X-limits
    plt.ylim(yVals)                                                             # Set Y-limits
    plt.colorbar()
    fname = os.path.join('figs','pressure_coefficient_contours.png')                         
    plt.savefig(fname, dpi=args.dpi, bbox_inches='tight')                                                  
