// ******************************************
// implicit time stepping implementation of 2D diffusion problem
// Ben Cumming, CSCS
// *****************************************

// A small benchmark app that solves the 2D fisher equation using second-order
// finite differences.

// Syntax: ./main nx ny nt t

#include <omp.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include <algorithm>

#include "data.h"
#include "linalg.h"
#include "operators.h"
#include "stats.h"

using namespace data;
using namespace linalg;
using namespace operators;
using namespace stats;

// ==============================================================================

// read command line arguments
static void readcmdline(Discretization& options, int argc, char* argv[])
{
    if (argc != 5)
    {
        printf("Usage: main nx ny nt t\n");
        printf("  nx  number of gridpoints in x-direction\n");
        printf("  ny  number of gridpoints in y-direction\n");
        printf("  nt  number of timesteps\n");
        printf("  t   total time\n");
        exit(1);
    }

    // read nx
    options.nx = atoi(argv[1]);
    if (options.nx < 1)
    {
        fprintf(stderr, "nx must be positive integer\n");
        exit(-1);
    }

    // read ny
    options.ny = atoi(argv[2]);
    if (options.ny < 1)
    {
        fprintf(stderr, "ny must be positive integer\n");
        exit(-1);
    }

    // read nt
    options.nt = atoi(argv[3]);
    if (options.nt < 1)
    {
        fprintf(stderr, "nt must be positive integer\n");
        exit(-1);
    }
    
    // read total time
    double t = atof(argv[4]);
    if (t < 0)
    {
        fprintf(stderr, "t must be positive real value\n");
        exit(-1);
    }

    // store the parameters
    options.N = options.nx * options.ny;

    // compute timestep size
    options.dt = t / options.nt;
    
    // compute the distance between grid points
    // assume that x dimension has length 1.0
    options.dx = 1. / (options.nx - 1);
    
    // set alpha, assume diffusion coefficient D is 1
    options.alpha = (options.dx * options.dx) / (1. * options.dt);
}

// ==============================================================================

int main(int argc, char* argv[])
{
    // read command line arguments
    readcmdline(options, argc, argv);
    int nx = options.nx;
    int ny = options.ny;
    int N  = options.N;
    int nt = options.nt;

    printf("========================================================================\n");
    printf("                      Welcome to mini-stencil!\n");
    printf("mesh :: %d * %d, dx = %f\n", nx, ny, options.dx);
    printf("time :: %d, time steps from 0 .. %f\n", nt, options.nt * options.dt);
    printf("========================================================================\n");

    // allocate global fields
    x_new = new double[nx * ny];
    x_old = new double[nx * ny];
    bndN = new double[nx];
    bndS = new double[nx];
    bndE = new double[ny];
    bndW = new double[ny];

    double* b = new double[N];
    double* deltax = new double[N];

    // set dirichlet boundary conditions to 0 all around
    std::fill(bndN, bndN+nx, 0.);
    std::fill(bndS, bndS+nx, 0.);
    std::fill(bndE, bndE+ny, 0.);
    std::fill(bndW, bndW+ny, 0.);

    // set the initial condition
    // a circle of concentration 0.1 centred at (xdim/4, ydim/4) with radius
    // no larger than 1/8 of both xdim and ydim
    std::fill(x_new, x_new+nx*ny, 0.);
    double xc = 1.0 / 4.0;
    double yc = (ny - 1) * options.dx / 4;
    double radius = fmin(xc, yc) / 2.0;
    for (int j = 0; j < ny; j++)
    {
        double y = (j - 1) * options.dx;
        for (int i = 0; i < nx; i++)
        {
            double x = (i - 1) * options.dx;
            if ((x - xc) * (x - xc) + (y - yc) * (y - yc) < radius * radius)
                x_new[i+nx*j] = 0.1;
        }
    }

    double time_in_bcs = 0.0;
    double time_in_diff = 0.0;
    flops_bc = 0;
    flops_diff = 0;
    flops_blas1 = 0;
    verbose_output = false;
    iters_cg = 0;
    iters_newton = 0;

    // start timer
    double timespent = -omp_get_wtime();

    // main timeloop
    double alpha = options.alpha;
    double tolerance = 1.e-6;
    for (int timestep = 1; timestep <= nt; timestep++)
    {
        // set x_new and x_old to be the solution
        ss_copy(x_old, x_new, N);

        double residual;
        bool converged = false;
        int it;
        for (it=0; it<50; it++)
        {
            // compute residual : requires both x_new and x_old
            diffusion(x_new, b);
            residual = ss_norm2(b, N);

            // check for convergence
            if (residual < tolerance)
            {
                converged = true;
                break;
            }

            // solve linear system to get -deltax
            bool cg_converged = false;
            ss_cg(deltax, b, 200, tolerance, cg_converged);

            // check that the CG solver converged
            if (!cg_converged) break;

            // update solution
            ss_axpy(x_new, -1.0, deltax, N);
        }
        iters_newton += it+1;

        // output some statistics
        //if (converged && verbose_output)
        if (converged && verbose_output)
            printf("step %d required %d iterations for residual %E\n", timestep, it, residual);
        if (!converged)
        {
            fprintf(stderr, "step %d ERROR : nonlinear iterations failed to converge\n", timestep);
            break;
        }
    }

    // get times
    timespent += omp_get_wtime();
    unsigned long long flops_total = flops_diff + flops_blas1;

    ////////////////////////////////////////////////////////////////////
    // write final solution to BOV file for visualization
    ////////////////////////////////////////////////////////////////////

    // binary data
    {
        FILE* output = fopen("output.bin", "w");
        fwrite(x_new, sizeof(double), nx * ny, output);
        fclose(output);
    }

    // metadata
    {
        FILE* output = fopen("output.bov", "wb");
        fprintf(output, "TIME: 0.0\n");
        fprintf(output, "DATA_FILE: output.bin\n");
        fprintf(output, "DATA_SIZE: %d, %d, 1\n", nx, ny);
        fprintf(output, "DATA_FORMAT: DOUBLE\n");
        fprintf(output, "VARIABLE: phi\n");
        fprintf(output, "DATA_ENDIAN: LITTLE\n");
        fprintf(output, "CENTERING: nodal\n");
        fprintf(output, "BRICK_SIZE: 1.0 %f 1.0\n", (ny - 1) * options.dx);
        fclose(output);
    }

    // print table sumarizing results
    printf("--------------------------------------------------------------------------------\n");
    printf("simulation took %f seconds\n", timespent);
    printf("%d conjugate gradient iterations, at rate of %8.1f iters/second\n",
            int(iters_cg),
            float(iters_newton/timespent));
    printf("%d newton iterations\n", int(iters_newton));
    printf("--------------------------------------------------------------------------------\n");

    // deallocate global fields
    delete[] x_new;
    delete[] x_old;
    delete[] bndN;
    delete[] bndS;
    delete[] bndE;
    delete[] bndW;

    printf("Goodbye!\n");

    return 0;
}

