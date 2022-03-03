###########################################################################
# Imports
###########################################################################
# Standard library imports
import argparse
import time as time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path


# Local imports
from XFOIL import XFOIL
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
        '-n', '--n-panels', action='store', dest='numPan', type=int, default=170,
        help='Number of panel nodes.'
    )
    parser.add_argument(
        '-v', '--vinf', action='store', dest='Vinf', type=float, default=1.,
        help='Free stream velocity.'
    )
    parser.add_argument(
        '--naca', action='store', dest='NACA', type=str, default="0012",
        help='NACA airfoil to be used.'
    )
    parser.add_argument(
        '-A', '--aoa', action='store', dest='AoA', type=float, default=0.,
        help='Angle of attack.'
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
NACA = args.NACA

# Convert AoA to radians [rad]
AoAR = AoA * (np.pi / 180)

# Flag to specify creating or loading airfoil
flagAirfoil = [1,                                                               # Create specified NACA airfoil in XFOIL
               0]                                                               # Load Selig-format airfoil from directory

# Plotting flags
flagPlot = [1,      # Airfoil with panel normal vectors
            1,      # Geometry boundary pts, control pts, first panel, second panel
            1,      # Cp vectors at airfoil surface panels
            1,      # Pressure coefficient comparison (XFOIL vs. VPM)
            1,      # Airfoil streamlines
            1]      # Pressure coefficient contour

# PPAR menu options
PPAR = [str(args.numPan + 1),                                                                  # "Number of panel nodes"
        '4',                                                                    # "Panel bunching paramter"
        '1.5',                                                                  # "TE/LE panel density ratios"
        '1',                                                                    # "Refined area/LE panel density ratio"
        '1 1',                                                                  # "Top side refined area x/c limits"
        '1 1']                                                                  # "Bottom side refined area x/c limits"

# Grid parameters
nGridX   = 150                                                                   # X-grid for streamlines and contours
nGridY   = 150                                                                   # Y-grid for streamlines and contours
xVals    = [-1, 1.5]                                                          # X-grid extents [min, max]
yVals    = [-1.5, 1.5]                                                          # Y-grid extents [min, max]

# %% XFOIL - CREATE/LOAD AIRFOIL

# Call XFOIL function to obtain the following:
# - Airfoil coordinates
# - Pressure coefficient along airfoil surface
# - Lift, drag, and moment coefficients
xFoilResults = XFOIL(NACA, PPAR, AoA, flagAirfoil)

# Separate out results from XFOIL function results
afName  = xFoilResults[0]                                                       # Airfoil name
xFoilX  = xFoilResults[1]                                                       # X-coordinate for Cp result
xFoilY  = xFoilResults[2]                                                       # Y-coordinate for Cp result
xFoilCP = xFoilResults[3]                                                       # Pressure coefficient
XB      = xFoilResults[4]                                                       # Boundary point X-coordinate
YB      = xFoilResults[5]                                                       # Boundary point Y-coordinate
xFoilCL = xFoilResults[6]                                                       # Lift coefficient
xFoilCD = xFoilResults[7]                                                       # Drag coefficient
xFoilCM = xFoilResults[8]                                                       # Moment coefficient

# Number of boundary points and panels
numPts = len(XB)                                                                # Number of boundary points
numPan = numPts - 1                                                             # Number of panels (control points)

# %% FUNCTIONS
XB, YB = correct_panels_orientation(numPan, XB, YB)
XC, YC, S, beta, delta, phi = compute_panel_geometries(numPan, XB, YB, AoA)
K, L = compute_kl_vpm(XC, YC, XB, YB, phi, S)
A, b = populate_matrices_vpm(numPan, K, beta, Vinf)
A, b = satisfy_kutta_condition_vpm(numPan, A, b, pct=args.replacement_pct)

gamma = np.linalg.solve(A, b)
print("\nSum of gamma: ", sum(gamma * S))

Vt, Cp = compute_panel_velocities(numPan, gamma, beta, L, Vinf)

CN, CA, CL, CD, CM = compute_force_coefficients(XC, phi, beta, AoAR, Cp, S)

# Print the results to the Console
print("\n======= RESULTS =======")
print("Lift Coefficient (CL)")
# From Kutta-Joukowski lift equation
print(f"  K-J  : {2*sum(gamma*S)}")

# From this VPM code
print(f"  VPM  : {CL}")
print(f"XFOIL  : {xFoilCL}")
print("\nMoment Coefficient (CM)")
print(f"  VPM  : {CM}")
print(f"XFOIL  : {xFoilCM}")

# %% COMPUTE STREAMLINES - REF [4]

if (flagPlot[4] == 1 or flagPlot[5] == 1):                                      # If we are plotting streamlines or pressure coefficient contours
    
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

# %% CIRCULATION AND VORTEX STRENGTH CHECK

if (flagPlot[4] == 1 or flagPlot[5] == 1):                                      # If we are plotting streamlines or Cp contours
    # Compute circulation
    aa   = 0.75                                                                 # Ellipse horizontal half-length
    bb   = 0.25                                                                 # Ellipse vertical half-length
    x0   = 0.5                                                                  # Ellipse center X-coordinate
    y0   = 0                                                                    # Ellipse center Y-coordinate
    numT = 5000                                                                 # Number of points on ellipse
    Circulation, xC, yC, VxC, VyC = compute_circulation(aa,bb,x0,y0,            # Compute circulation around ellipse
                                                       numT,Vx,Vy,Xgrid,Ygrid)
    
    # Print values to Console
    print("\n\n======= CIRCULATION RESULTS =======")
    print("Sum of L   : %2.8f" % sum(gamma*S))                                  # Print sum of vortex strengths
    print("Circulation: %2.8f" % Circulation)                                   # Print circulation
    print("Lift Coef  : %2.8f" % (2.0*Circulation))                             # Lift coefficient from K-J equation
    
# %% PLOTTING

# FIGURE: Airfoil with panel normal vectors
if (flagPlot[0] == 1):
    fig = plt.figure(1)                                                         # Create the figure
    plt.cla()                                                                   # Clear the axes
    plt.fill(XB,YB,'k')                                                         # Plot the airfoil
    X = np.zeros(2)                                                             # Initialize 'X'
    Y = np.zeros(2)                                                             # Initialize 'Y'
    for i in range(numPan):                                                     # Loop over all panels
        X[0] = XC[i]                                                            # Set X start of panel orientation vector
        X[1] = XC[i] + S[i]*np.cos(delta[i])                                    # Set X end of panel orientation vector
        Y[0] = YC[i]                                                            # Set Y start of panel orientation vector
        Y[1] = YC[i] + S[i]*np.sin(delta[i])                                    # Set Y end of panel orientation vector
        if (i == 0):                                                            # If it's the first panel index
            plt.plot(X,Y,'b-',label='First Panel')                              # Plot normal vector for first panel
        elif (i == 1):                                                          # If it's the second panel index
            plt.plot(X,Y,'g-',label='Second Panel')                             # Plot normal vector for second panel
        else:                                                                   # If it's neither the first nor second panel index
            plt.plot(X,Y,'r-')                                                  # Plot normal vector for all other panels
    plt.xlabel('X Units')                                                       # Set X-label
    plt.ylabel('Y Units')                                                       # Set Y-label
    plt.title('Panel Geometry')                                                 # Set title
    plt.axis('equal')                                                           # Set axes equal
    plt.legend()                                                                # Display legend
    fname = os.path.join('figs', 'airfoil', 'airfoil_geometry.png')
    plt.savefig(fname, dpi=args.dpi, bbox_inches='tight')

# FIGURE: Geometry with the following indicated:
# - Boundary points, control points, first panel, second panel
if (flagPlot[1] == 1):
    fig = plt.figure(2)                                                         # Create figure
    plt.cla()                                                                   # Get ready for plotting
    plt.plot(XB,YB,'k-')                                                        # Plot airfoil panels
    plt.plot([XB[0], XB[1]],[YB[0], YB[1]],'b-',label='First Panel')            # Plot first panel
    plt.plot([XB[1], XB[2]],[YB[1], YB[2]],'g-',label='Second Panel')           # Plot second panel
    plt.plot(XB,YB,'ko',markerfacecolor='k',label='Boundary Pts')               # Plot boundary points (black circles)
    plt.plot(XC,YC,'ko',markerfacecolor='r',label='Control Pts')                # Plot control points (red circles)
    plt.xlabel('X Units')                                                       # Set X-label
    plt.ylabel('Y Units')                                                       # Set Y-label
    plt.axis('equal')                                                           # Set axes equal
    plt.legend()                                                                # Display legend
    fname = os.path.join('figs', 'airfoil', 'airfoil_geometry2.png')
    plt.savefig(fname, dpi=args.dpi, bbox_inches='tight')

# FIGURE: Cp vectors at airfoil control points
if (flagPlot[2] == 1):
    fig = plt.figure(3)                                                         # Create figure
    plt.cla()                                                                   # Get ready for plotting
    
    # Scale and make positive all Cp values
    Cps = np.absolute(Cp*0.15)
    X = np.zeros(2)
    Y = np.zeros(2)   

    posOnce = negOnce = True
    for i in range(len(Cps)):                                                   # Loop over all panels
        X[0] = XC[i]                                                            # Control point X-coordinate
        X[1] = XC[i] + Cps[i]*np.cos(delta[i])                                  # Ending X-value based on Cp magnitude
        Y[0] = YC[i]                                                            # Control point Y-coordinate
        Y[1] = YC[i] + Cps[i]*np.sin(delta[i])                                  # Ending Y-value based on Cp magnitude
        
        if (Cp[i] < 0):
            if posOnce:
                plt.plot(X,Y,'r-', label=r'$C_p < 0$')
                posOnce = False
            else:
                plt.plot(X,Y,'r-')
        elif (Cp[i] >= 0):
            if negOnce:
                plt.plot(X,Y,'b-', label=r'$C_p \geq 0$')
                negOnce = False
            else:
                plt.plot(X,Y,'b-')

    # Plot the airfoil as black polygon
    plt.fill(XB,YB,'k')

    plt.xlabel('X Units')
    plt.ylabel('Y Units')
    plt.gca().set_aspect('equal')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -1), ncol = 2)
    fig.subplots_adjust(bottom=0.25)
    
    fname = os.path.join('figs', 'airfoil', 'airfoil_cp.png')
    plt.savefig(fname, dpi=args.dpi, bbox_inches='tight')

# FIGURE: Pressure coefficient comparison (XFOIL vs. VPM)
if (flagPlot[3] == 1):
    fig = plt.figure(4)                                                         # Create figure
    plt.cla()                                                                   # Get ready for plotting
    midIndX = int(np.floor(len(xFoilCP)/2))                                     # Airfoil middle index for XFOIL data
    midIndS = int(np.floor(len(Cp)/2))                                          # Airfoil middle index for VPM data
    plt.plot(xFoilX[0:midIndX],xFoilCP[0:midIndX],                              # Plot Cp for upper surface of airfoil from XFoil
             'b-',label='XFOIL Upper')
    plt.plot(xFoilX[midIndX+1:len(xFoilX)],xFoilCP[midIndX+1:len(xFoilX)],      # Plot Cp for lower surface of airfoil from XFoil
             'r-',label='XFOIL Lower')
    plt.plot(XC[midIndS+1:len(XC)],Cp[midIndS+1:len(XC)],                       # Plot Cp for upper surface of airfoil from panel method
             'ks',markerfacecolor='b',label='VPM Upper')
    plt.plot(XC[0:midIndS],Cp[0:midIndS],                                       # Plot Cp for lower surface of airfoil from panel method
             'ks',markerfacecolor='r',label='VPM Lower')
    plt.xlim(0,1)                                                               # Set X-limits
    plt.xlabel('X Coordinate')                                                  # Set X-label
    plt.ylabel('Cp')                                                            # Set Y-label
    plt.title('Pressure Coefficient')                                           # Set title
    plt.legend()                                                                # Display legend
    plt.gca().invert_yaxis()                                                    # Invert Cp (Y) axis
    fname = os.path.join('figs', 'airfoil', 'airfoil_cp_comparison.png')
    plt.savefig(fname, dpi=args.dpi, bbox_inches='tight')
    
# FIGURE: Airfoil streamlines
if (flagPlot[4] == 1):
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
    fname = os.path.join('figs', 'airfoil', 'airfoil_streamlines.png')
    plt.savefig(fname, dpi=args.dpi, bbox_inches='tight')

# FIGURE: Pressure coefficient contour
if (flagPlot[5] == 1):
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
    fname = os.path.join(os.getcwd(),'CpContour.png')
    fname = os.path.join('figs', 'airfoil', 'airfoil_cp_contour.png')
    plt.savefig(fname, dpi=args.dpi, bbox_inches='tight')
