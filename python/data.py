"""
! define some helper types that can be used to pass simulation
! data around without haveing to pass individual parameters
"""
from collections import namedtuple


Discretization = namedtuple ('Discretization', [
                                'nx',   # x dimension
                                'ny',   # y dimension
                                'nt',   # number of time steps
                                'N',    # total number of grid points
                                'dt',   # time step size
                                'dx',   # distance between grid points
                                'alpha',# dx^2/(D*dt)
                            ])
Options = namedtuple ('Options', Discretization._fields + ('verbose_output',))

# store information about the domain decomposition
Subdomain = namedtuple ('Subdomain', [
                            # i and j dimensions of the global decomposition 
                            'i_dimension',  
                            'j_dimension',
                            # i and j indices of this sub-domain
                            'i_index', 
                            'j_index',
                            # boolean flags that indicate whether the sub-
                            # domain is on any of the four global boundaries
                            'on_boundary_north',
                            'on_boundary_south',
                            'on_boundary_east',
                            'on_boundary_west'
                        ])

# fields that hold the solution
Solution = namedtuple ('Solution', [
                            'x_new',
                            'x_old'
                      ])

# the boundary vectors
Boundary = namedtuple ('Boundary', [
                            'bndN', # north boundary vector
                            'bndS', # south boundary vector
                            'bndE', # east boundary vector
                            'bndW'  # west boundary vector
                      ])
