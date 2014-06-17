###
# ******************************************
# implicit time stepping implementation of 2D diffusion problem
# Ben Cumming, CSCS
#
# Adapted from Fortran by Lucas Benedicic, CSCS
# *****************************************
#
# A small benchmark app that solves the 2D fisher equation using second-order
# finite differences.
#
# Syntax:
#   ./main nx ny nt t
###
import sys
import time
import numpy as np

from data import Options, Solution, Boundary
from linalg import LinAlg
from operators import diffusion



def readcmdline (args):
    """
    Returns a dict containing the command line arguments and their values.-
    """
    ret_value = dict ( )

    if len (args) < 5 or len (args) > 6:
        sys.stderr.write ("Usage: %s nx ny nt t verbose\n" % args[0])
        sys.stderr.write ("\tnx\tnumber of gridpoints in x-direction\n")
        sys.stderr.write ("\tny\tnumber of gridpoints in y-direction\n")
        sys.stderr.write ("\tnt\tnumber of timesteops\n")
        sys.stderr.write ("\tt\ttotal time\n")
        sys.stderr.write ("\tverbose\t(option) if set verbose output is enabled")
        sys.exit (1)

    # read nx
    try:
        ret_value['nx'] = int (args[1])
    except ValueError:
        sys.stderr.write ('nx must be positive integer')

    # read ny
    try:
        ret_value['ny'] = int (args[2])
    except ValueError:
        sys.stderr.write ('ny must be positive integer')

    # read nt
    try:
        ret_value['nt'] = int (args[3])
    except ValueError:
        sys.stderr.write ('nt must be positive integer')

    # read total time
    try:
        tot_time = float (args[4])
    except ValueError:
        sys.stderr.write ('t must be positive real value')

    # set verbose output if a fith argument has been passed
    ret_value['verbose_output'] = bool (len (args) == 5)

    # total number of grid points
    ret_value['N'] = ret_value['nx'] * ret_value['ny']

    # compute timestep size
    ret_value['dt'] = tot_time / ret_value['nt']

    # compute the distance between grid points
    # assume that x dimension has length 1.0
    ret_value['dx'] = 1.0 / (ret_value['nx'] - 1.0)

    # set alpha, assume diffusion coefficient D is 1
    ret_value['alpha'] = (ret_value['dx'] ** 2) / (1.0 * ret_value['dt'])

    return ret_value


def main (argv):
    """
    Starting point.-
    """
    # ****************** read command line arguments ******************

    # read command line arguments
    options = readcmdline (sys.argv)
    options = Options (**options)
    nx = options.nx
    ny = options.ny
    N  = options.N
    nt = options.nt

    # ****************** setup compute domain ******************

    print ('========================================================================')
    print ('                      Welcome to mini-stencil!')
    print ('mesh :: %d * %d \t    dx = %.5f' % (nx, ny, options.dx))
    print ('time :: %d \t time steps from 0 .. %.5f' % (nt, nt * options.dt))
    print ('========================================================================')

    # ****************** constants ******************
    one = 1.
    zero = 0.

    # ****************** allocate memory ******************

    # allocate solution fields
    sol = Solution (x_new = np.zeros ((nx, ny), dtype=np.float64),
                    x_old = np.zeros ((nx, ny), dtype=np.float64))

    # allocate local variables
    b = np.zeros ((N), dtype=np.float64)
    deltax = np.zeros_like (b)

    # allocate boundary vectors
    # and set dirichlet boundary conditions to 0 all around
    boundary = Boundary (bndN = np.zeros ((nx), dtype=np.float64),
                         bndS = np.zeros ((nx), dtype=np.float64),
                         bndE = np.zeros ((ny), dtype=np.float64),
                         bndW = np.zeros ((ny), dtype=np.float64))

    # initialize the linear algebra solver
    la_solver = LinAlg (nx * ny)

    # set the initial condition
    # a circle of concentration 0.1 centred at (xdim/4, ydim/4) with radius
    # no larger than 1/8 of both xdim and ydim
    xc = 1.0/4.0
    yc = float (ny - 1) * options.dx / 4
    radius = min (xc, yc) / 2.0
    for j in range (ny):
        y = float (j - 1) * options.dx
        for i in range (nx):
            x = float (i - 1) * options.dx
            if ( (x-xc)**2 + (y-yc)**2 < radius**2):
                sol.x_new[i, j] = 0.1

    # ****************** serial reference version ******************
    time_in_bcs  = 0.0
    time_in_diff = 0.0
    flops_bc     = 0
    flops_diff   = 0
    flops_blas1  = 0
    iters_cg     = 0
    iters_newton = 0

    # start timer
    timespent = -time.time ( )

    # main timeloop
    alpha = options.alpha
    tolerance = 1.e-6
    for timestep in range (nt):
        # set x_new and x_old to be the solution
        sol._replace (x_old = np.copy (sol.x_new))

        converged = False
        for it in range (50):
            # compute residual : requires both x_new and x_old
            diffusion (sol.x_new, 
                       b, 
                       options, 
                       sol, 
                       boundary)
            residual = la_solver.ss_norm2 (b)

            # check for convergence
            if (residual < tolerance):
                converged = True
                break

            # solve linear system to get -deltax
            #call ss_cg (deltax, b, 200, tolerance, cg_converged)
            cg_converged = la_solver.ss_cg (deltax, b, 200, tolerance)

            # check that the CG solver converged
            if info != 0:
                sys.stderr.write ("The CG solver did not converge.\n")
                sys.exit (1)

            # update solution
            #call ss_axpy(x_new, -one, deltax, N)
            x_new += (deltax * -1)

        # update the iteration counter
        iters_newton += it

        # output some statistics
        if (converged and options.verbose_output):
            print ('step %d required %d iterations for residual %.5f' % (timestep, it, residual))

        if not converged:
            print ('step %d')
            sys.stderr.write (' ERROR : nonlinear iterations failed to converge\n')
            sys.exit (1)

    # get times
    timespent += time.time ( )
    flops_total = flops_diff + flops_blas1

    # ***** write final solution to BOV file for visualization *****
    # binary data
    with open ('output.bin', 'w') as f:
        sol.x_new.tofile (f)
    
    # metadata
    with open ('output.bov', 'w') as f:
        f.writelines (('TIME: 0.0\n',
                       'DATA_FILE: output.bin\n',
                       'DATA_SIZE: %d %d 1\n' % (nx, ny),
                       'DATA_FORMAT: DOUBLE\n',
                       'VARIABLE: phi\n',
                       'DATA_ENDIAN: LITTLE\n',
                       'CENTERING: nodal\n',
                       'BYTE_OFFSET: 4\n',
                       'BRICK_SIZE: 1.0 %.2f 1.0\n' % (float (ny-1)*options.dx)
                     ))

    # print table sumarizing results
    print ('--------------------------------------------------------------------------------')
    print ('simulation took %.5f seconds' % timespent)
    print ('                %d conjugate gradient iterations, i.e., %.5f iterations per second' % (iters_cg, iters_cg / timespent))
    print ('                %d nonlinear newton iterations' % iters_newton)
    print ('-------------------------------------------------------------------------------')

    # ****************** cleanup ******************
    
    # deallocate global fields
    #deallocate(x_new, x_old)
    #deallocate(bndN, bndS, bndE, bndW)

    print ('Goodbye!')



if __name__ == '__main__':
    main (sys.argv)

