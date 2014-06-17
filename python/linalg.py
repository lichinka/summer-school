###
# linear algebra subroutines
# Ben Cumming @ CSCS
# adapted from Fortran by Lucas Benedicic, CSCS
###
import numpy as np

from operators import diffusion



class LinAlg (object):
    """
    This class provides linear algebra functions.-
    """
    def _check_dim (self, lst, expected_dim=1):
        """
        Checks that the dimension of matrices in lst is as expected:

            lst: a tuple of numpy arrays;
            expected_dim: the expected number of dimensions.-
        """
        for x in lst:
            assert len (np.shape (x)) == expected_dim, \
                   'Dimension should be %d, but got %d' % (expected_dim,
                                                           len (np.shape (x)))


    def __init__ (self, N):
        """
        Initializes temporary storage fields used by the cg solver:

            N:  the size of the 1D arrays that will be used later.

        I do this here so that the fields are persistent between calls
        to the CG solver. This is useful if we want to avoid malloc/free calls
        on the device for the OpenACC implementation (feel free to suggest a 
        better method for doing this).-
        """
        self._Ap    = np.zeros ((N), dtype=np.float64)
        self._r     = np.zeros ((N), dtype=np.float64)
        self._p     = np.zeros ((N), dtype=np.float64)
        self._Fx    = np.zeros ((N), dtype=np.float64)
        self._Fxold = np.zeros ((N), dtype=np.float64)
        self._v     = np.zeros ((N), dtype=np.float64)
        self._xold  = np.zeros ((N), dtype=np.float64)


    def ss_dot (self, x, y):
        """
        Computes the inner product of x and y:

            x: a 1D numpy array;
            y: a 1D numpy array.-
        """
        self._check_dim ([x, y])
        ret_value = 0.0

        for i in range (len (x)):
            ret_value += x[i] * y[i]

        # TODO record the number of floating point oporations
        #flops_blas1 = flops_blas1 + 2*N
        return ret_value


    def ss_norm2 (self, x):
        """
        Computes the 2-norm of x:

            x: a 1D numpy array.-
        """
        self._check_dim ([x])
        ret_value = 0.0

        for i in range (len (x)):
            ret_value += x[i] * x[i]

        # TODO record the number of floating point oporations
        #flops_blas1 = flops_blas1 + 2*N
        return ret_value


    def ss_axpy (self, y, alpha, x):
        """
        Computes y = alpha*x + y, with x and y vectors:
        
            x:     a 1D numpy array;
            y:     a 1D numpy array;
            alpha: a scalar.-
        """
        self._check_dim ([x, y])

        for i in range (len (x)):
            y[i] += alpha * x[i]

        # TODO record the number of floating point oporations
        #flops_blas1 = flops_blas1 + 2*N


    def ss_add_scaled_diff (self, y, x, alpha, l, r):
        """
        Computes y = x + alpha*(l-r):

            x:     a 1D numpy array;
            y:     a 1D numpy array;
            l:     a 1D numpy array;
            r:     a 1D numpy array;
            alpha: a scalar.-
        """
        self._check_dim ([x, y, l, r])

        for i in range (len (x)):
            y[i] = x[i] + alpha * (l[i] * r[i])

        # update the flops counter
        #flops_blas1 = flops_blas1 + 3*N


    def ss_scaled_diff (self, y, alpha, l, r):
        """
        Computes y = alpha*(l-r):

            y:     a 1D numpy array;
            alpha: a scalar;
            l:     a 1D numpy array;
            r:     a 1D numpy array.-
        """
        self._check_dim ([y, l, r])

        for i in range (len (y)):
            y[i] = alpha * (l[i] - r[i])

        # update the flops counter
        #flops_blas1 = flops_blas1 + 2*N


    def ss_scale (self, y, alpha, x):
        """
        Computes y := alpha*x

            y:     a 1D numpy array;
            alpha: a scalar;
            x:     a 1D numpy array.-
        """
        self._check_dim ([y, x])

        for i in range (len (y)):
            y[i] = alpha * x[i]

        # TODO update the flops counter
        #flops_blas1 = flops_blas1 + N


    def ss_lcomb (self, y, alpha, x, beta, z):
        """
        Computes linear combination of two vectors y = alpha*x + beta*z:

            y:     a 1D numpy array;
            alpha: a scalar;
            x:     a 1D numpy array;
            beta:  a scalar;
            z:     a 1D numpy array.-
        """
        self._check_dim ([y, x, z])

        for i in range (len (y)):
            y[i] = alpha * x[i] + beta * z[i]

        # TODO update the flops counter
        #flops_blas1 = flops_blas1 + 3*N


    def ss_cg (self, x, b, maxiters, tol):
        """
        Conjugate gradient solver: solve the linear system A*x = b for x
        The matrix A is implicit in the objective function for the diffusion 
        equation. Returns 'True' if the system converged within the given
        tolerance.
        The value in x constitutes the "first guess" at the solution ON ENTRY, 
        ON EXIT it contains the solution:

                x:          a 1D numpy array;
                b:          a 1D numpy array;
                maxiters:   the maximum number of iterations to run;
                tol:        the tolerance of the approximation.-
        """
        self._check_dim ([x, b])
        ret_value = False

        # useful constants
        zero = 0.0
        one  = 1.0

        # epslion value use for matrix-vector approximation
        eps     = 1.e-8
        eps_inv = 1./eps

        # copy the input array before changing it
        self._xold = np.copy (x)

        # matrix vector multiplication is approximated with
        #   A*v = 1/epsilon * ( F(x+epsilon*v) - F(x) )
        #       = 1/epsilon * ( F(x+epsilon*v) - Fxold )
        # we compute Fxold at startup
        # we have to keep x so that we can compute the F(x+exps*v)
        diffusion (x, self._Fxold)

        # v = x + epsilon*x
        self.ss_scale (self._v, one+eps, x)

        # Fx = F(v)
        diffusion (self._v, self._Fx)

        # r = b - A*x
        # where A*x = (Fx-Fxold)/eps
        self.ss_add_scaled_diff (self._r, 
                                 self._b,
                                 -eps_inv,
                                 self._Fx,
                                 self._Fxold)
        # p = r
        self._p = np.copy (self._r)

        # rold = <r,r>
        rold = self.ss_dot (r, r)

        # check for convergence
        ret_value = (np.sqrt (rold) < tol)

        for it in range (maxiter):
            # Ap = A*p
            self.ss_lcomb (self._v,
                           one,
                           self._xold,
                           eps,
                           self._p)
            diffusion (self._v, self._Fx)
            self.ss_scaled_diff (self._Ap, eps_inv, self._Fx, self._Fxold)

            # alpha = rold / p*Ap
            alpha = rold / self._ss_dot (self._p, self._Ap)

            # x += alpha*p
            self.ss_axpy (x, alpha, self._p)

            # r -= alpha*Ap
            self.ss_axpy (self._r, -alpha, self._Ap)

            # find new norm
            rnew = self.ss_dot (self._r, self._r)

            # test for convergence
            ret_value = ( np.sqrt (rnew) < tol )
            if ret_value:
                break

            # p = r + rnew.rold * p
            self.ss_lcomb (self._p, one, self._r, rnew/rold, self._p)

            rold = rnew

        # TODO iters_cg = iters_cg + iter

        if not ret_value:
            sys.stderr.write ('ERROR: CG failed to converge after %d iterations' % (iterations))

        return ret_value

