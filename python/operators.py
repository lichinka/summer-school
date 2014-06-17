# ******************************************
# operators.f90
# based on min-app code written by Oliver Fuhrer, MeteoSwiss
# modified by Ben Cumming, CSCS
# adapted from Fortran by Lucas Benedicic, CSCS
# *****************************************
#
# Description: Contains simple operators which can be used on 3d-meshes
#
import numpy as np



def diffusion (u, s, opt, sol, bnd):
    """
    The diffusion stencil:

        u   the input array;
        opt the `Options' named tuple;
        sol the `Solution' named tuple;
        bnd the `Boundary' named tuple.-
    """
    dxs   = 1000.0 * (opt.dx ** 2)
    alpha = opt.alpha
    iend  = opt.nx - 1
    jend  = opt.ny - 1

    # the interior grid points
    for j in range (1, jend):
        for i in range (1, iend):
            s[(j*opt.nx) + i] = ( -(4.0 + alpha) * u[i, j]    # central point
                                  + u[i-1, j] + u[i+1, j]     # east and west
                                  + u[i, j-1] + u[i, j+1]     # north and south
                                  + alpha * sol.x_old[i,j]
                                  + dxs * u[i,j] * (1.0 - u[i, j]) )
    # the east boundary
    i = opt.nx - 1
    for j in range (1, jend):
        s[(j*opt.nx) + i] = ( -(4.0 + alpha) * u[i, j]
                              + u[i-1, j] + u[i, j-1] + u[i, j+1]
                              + alpha * sol.x_old[i, j] + bnd.bndE[j] 
                              + dxs * u[i, j] * (1.0 - u[i, j]) )

    # the west boundary
    i = 1
    for j in range (2, jend):
        s[(j*opt.nx) + i] = ( -(4.0 + alpha) * u[i, j]
                              + u[i+1, j] + u[i, j-1] + u[i, j+1]
                              + alpha * sol.x_old[i, j] + bnd.bndW[j]
                              + dxs * u[i, j] * (1.0 - u[i, j]) )

    # the north boundary (plus NE and NW corners)
    j = opt.ny - 1
    i = 0   # NW corner
    s[(j*opt.nx) + i] = ( -(4.0 + alpha) * u[i,j]
                          + u[i+1, j] + u[i, j-1]
                          + alpha * sol.x_old[i,j]
                          + bnd.bndW[j] + bnd.bndN[i]
                          + dxs * u[i, j] * (1.0 - u[i,j]) )

    # north boundary
    for i in range (1, iend):
        s[(j*opt.nx) + i] = ( -(4.0 + alpha) * u[i,j]
                              + u[i-1, j] + u[i+1, j] + u[i, j-1] 
                              + alpha * sol.x_old[i, j] + bnd.bndN[i] 
                              + dxs * u[i, j] * (1.0 - u[i, j]) )

    i = opt.nx - 1  # NE corner
    s[(j*opt.nx) + i] = ( -(4.0 + alpha) * u[i, j]
                          + u[i-1, j] + u[i, j-1] 
                          + alpha * sol.x_old[i, j]
                          + bnd.bndE[j] + bnd.bndN[i]
                          + dxs * u[i, j] * (1.0 - u[i,j]) )

    # the south boundary
    j = 0
    i = 0   # SW corner
    s[(j*opt.nx) + i] = ( -(4.0 + alpha) * u[i, j]
                          + u[i-1, j] + u[i, j-1] 
                          + alpha * sol.x_old[i, j]
                          + bnd.bndW[j] + bnd.bndS[i]
                          + dxs * u[i, j] * (1.0 - u[i,j]) )

    # south boundary
    for i in range (1, iend):
        s[(j*opt.nx) + i] = ( -(4.0 + alpha) * u[i, j]
                              + u[i-1, j] + u[i, j-1] 
                              + alpha * sol.x_old[i, j]
                              + bnd.bndS[i]
                              + dxs * u[i, j] * (1.0 - u[i,j]) )

    i = opt.nx - 1  # SE corner
    s[(j*opt.nx) + i] = ( -(4.0 + alpha) * u[i, j]
                          + u[i-1, j] + u[i, j-1] 
                          + alpha * sol.x_old[i, j]
                          + bnd.bndE[j] + bnd.bndS[i]
                          + dxs * u[i, j] * (1.0 - u[i,j]) )

    """
    ! accumulate the flop counts
    ! 8 ops total per point
    flops_diff =  flops_diff                    &
                    + 12 * (options%nx-2) * (options%ny-2) & ! interior points
                    + 11 * (options%nx-2 + options%ny-2) &   ! NESW boundary points
                    + 11 * 4                                 ! corner points
    """

