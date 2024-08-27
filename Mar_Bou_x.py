import numpy as np
from pathlib import Path
from tqdm import tqdm
from mpi4py import MPI

from dolfinx import fem, mesh, io, default_scalar_type, log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from basix.ufl import element, mixed_element
from ufl import (Measure, SpatialCoordinate, TestFunctions, TrialFunctions,
                 div, exp, inner, split, grad, Identity, dot, dx, inner, div,
                 derivative, Constant)
from petsc4py import PETSc
import pdb

log.set_log_level(log.LogLevel.INFO)

#############################  END  ################################

#################### Define Parallel Variables ####################

if __name__ == '__main__':
    # Get the global communicator
    comm = MPI.COMM_WORLD

    # Get the rank of the process
    rank = comm.Get_rank()

    # Get the size of the communicator (total number of processes)
    size = comm.Get_size()

#############################  END  ################################


##################### Physical Constants ################################

GRAVITY = -10  # Acceleration due to gravity (m/s^2)
RHO1 = 760  # Fluid density (kg/m^3)
MU1 = 4.94 * 10 ** -4  # Dynamic viscosity (Pa.s)
K1 = 0.1  # Thermal conductivity (W/m.K)
CP1 = 2090  # Heat capacity (J/kg.K)
ALPHA1 = 1.3 * 10**-3  # Thermal expansion coefficient (1/K)
GAMMA = -8 * 10 ** -5  # Surface tension temperature derivative (N/m.K)

# Temperature Constants:
T_REF = 273.15  # Reference temperature (K)
T_RIGHT = 273.15  # Temperature on the right boundary (K)
DELTA_T = 2  # Temperature difference (K)
T_LEFT = T_RIGHT + DELTA_T  # Temperature on the left boundary (K)

#############################  END  ################################

##################### Mesh Refinement Functions For Bounderies ######################

def refine_mesh_near_boundary(domain: mesh.Mesh, threshold, coords):

    # Unpack domain coordinates
    (X0, Y0), (X1, Y1) = coords

    cell_map = domain.topology.index_map(domain.topology.dim)
    num_cells_on_process = cell_map.size_local + cell_map.num_ghosts
    cells = np.arange(num_cells_on_process, dtype=np.int32)

    marked_cells = []

    mps = mesh.compute_midpoints(domain, domain.topology.dim, cells)

    # Iterate through each cell in the mesh
    for idx, mp in enumerate(mps):
        x_mid, y_mid = mp[0], mp[1]

        # Calculate the distance from the cell's midpoint to the boundary
        dist_to_left_boundary = abs(x_mid - X0)
        dist_to_right_boundary = abs(x_mid - X1)
        dist_to_bottom_boundary = abs(y_mid - Y0)
        dist_to_top_boundary = abs(y_mid - Y1)

        # Mark cells for refinement if they're within the threshold distance from any boundary
        if (min(dist_to_left_boundary, dist_to_right_boundary) < threshold or
            min(dist_to_bottom_boundary, dist_to_top_boundary) < threshold
        ):
            marked_cells.append(idx)

    domain.topology.create_connectivity(domain.topology.dim,1)
    marked_edges = mesh.compute_incident_entities(
        domain.topology, np.array(marked_cells), domain.topology.dim, 1)

    # Refine the mesh based on the marked cells
    refined_mesh = mesh.refine(domain, marked_edges)

    return refined_mesh


def refine_mesh_near_corners(domain: mesh.Mesh, threshold, coords):


    # Unpack domain coordinates
    (X0, Y0), (X1, Y1) = coords

    cell_map = domain.topology.index_map(domain.topology.dim)
    num_cells_on_process = cell_map.size_local + cell_map.num_ghosts
    cells = np.arange(num_cells_on_process, dtype=np.int32)

    marked_cells = []

    mps = mesh.compute_midpoints(domain, domain.topology.dim, cells)

    # Iterate through each cell in the mesh
    for idx, mp in enumerate(mps):
        x_mid, y_mid = mp[0], mp[1]

        # Calculate the distance from the cell's midpoint to the corners
        dist_to_bottom_left_corner = np.sqrt((x_mid - X0)**2 + (y_mid - Y0)**2)
        dist_to_bottom_right_corner = np.sqrt((x_mid - X1)**2 + (y_mid - Y0)**2)
        dist_to_top_left_corner = np.sqrt((x_mid - X0)**2 + (y_mid - Y1)**2)
        dist_to_top_right_corner = np.sqrt((x_mid - X1)**2 + (y_mid - Y1)**2)

        # Mark cells for refinement if they're within the threshold distance from any corner
        if ( dist_to_top_left_corner < threshold):
            np.append(marked_cells, idx)

    domain.topology.create_connectivity(domain.topology.dim,1)
    marked_edges = mesh.compute_incident_entities(
        domain.topology, np.array(marked_cells), domain.topology.dim, 1)

    # Refine the mesh based on the marked cells
    refined_mesh = mesh.refine(domain, marked_edges)

    return refined_mesh


#############################  END  ################################


############################## Define domain sizes and discretization parameters ################################

if __name__ == '__main__':

    # Define grid spacing in x and y directions (meters)

    grid_spacing_x = 0.1e-3  # 0.1 mm converted to meters
    grid_spacing_y = 0.1e-3  # 0.1 mm converted to meters

    # Define time step for the simulation (arbitrary units)
    dt = 1000



    # Adjust the domain length to ensure it is divisible by the grid spacing and slightly larger than the desired size

    domain_length_x = 10e-3
    domain_length_y = 5e-3


    # Calculate the number of divisions along each axis based on approximate domain size and grid spacing
    num_divisions_x = int( domain_length_x  / grid_spacing_x)
    num_divisions_y = int( domain_length_y / grid_spacing_y)

    origin = [0,0]

    # Calculate the top right corner based on the origin and adjusted domain lengths
    top_right_corner = [origin[0] + domain_length_x, origin[1] + domain_length_y]

    # Create the initial rectangular mesh using the defined corners and number of divisions
    initial_mesh = mesh.create_rectangle(MPI.COMM_WORLD, [origin, top_right_corner],
                                         [num_divisions_x, num_divisions_y])

    # Define Domain

    coords = [ ( 0.0 , 0.0 ) ,( 0.0 + domain_length_x , 0.0 + domain_length_y ) ]

#############################  END  ################################

############################ Modify Initial Mesh ######################
if __name__ == '__main__':
    mar_domain = initial_mesh

    mar_domain  = refine_mesh_near_boundary( mar_domain, 0.2e-3, coords )
    mar_domain  = refine_mesh_near_boundary( mar_domain, 0.2e-3, coords )


    mar_domain  = refine_mesh_near_corners( mar_domain, 0.3e-3, coords  )


#############################  END  ################################


######################################################################

def create_function_spaces(domain):


    # Define finite elements for velocity, pressure, and temperature
    P2 = element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))  # Velocity
    P1 = element("Lagrange", domain.basix_cell(), 1)  # Pressure
    PT = element( "Lagrange", domain.basix_cell(), 1 )#temperature

    # Define mixed elements
    mix_element = mixed_element([P2, P1, PT])

    # Create a function space
    W = fem.functionspace(domain, mix_element)

    # Define test functions
    v_test, q_test, s_test = TestFunctions(W)

    # Define current and previous solutions
    upT = fem.Function(W)  # Current solution
    upT0 = fem.Function(W)  # Previous solution

    # Split functions to access individual components
    u_answer, p_answer, T_answer = split(upT)  # Current solution
    u_prev, p_prev, T_prev = split(upT0)  # Previous solution

    return W, v_test, q_test, s_test, upT, upT0, u_answer, p_answer, T_answer, u_prev, p_prev, T_prev

# Usage example:
# W, v_test, q_test, s_test, upT, upT0, u_answer, p_answer, T_answer, u_prev, p_prev, T_prev = create_function_spaces(mesh)

#############################  END  ################################


############################ Defining Equations ###########################

# Related Functions for defining equaions
def epsilon(u):

    return 0.5 * (grad(u) + grad(u).T)

def sigma(u, p, mu1):

    return 2 * mu1 * epsilon(u) - p * Identity(len(u))

def Traction(T, n_v, gamma):

    return gamma * (grad(T) - dot(n_v, grad(T)) * n_v)


# main equaions

def F1(u_answer, q_test, dt):

    F1 = inner(div(u_answer), q_test) * dt * dx

    return F1

def F2(u_answer, u_prev, p_answer, T_answer, v_test, dt, rho1, n_v, mu1, gamma, alpha1, ds1, dx1):

    global GRAVITY, T_REF

    F2 = (
        inner((u_answer - u_prev) / dt, v_test) * dx
        + inner(dot(u_answer, grad(u_answer)), v_test) * dx
        + (1/rho1) * inner(sigma(u_answer, p_answer, mu1), epsilon(v_test)) * dx
        - (1/rho1) * inner(Traction(T_answer, n_v, gamma), v_test) * ds1(1)
        # Uncomment the following lines if buoyancy force is needed
        + inner(GRAVITY * alpha1 * (T_answer - T_REF), v_test[1]) * dx  # Bouyancy y-component

        # Remeber alpha1 ?!
    )

    return F2

def F3(T_answer, T_prev, u_answer, s_test, dt, rho1, Cp1, K1):


    F3 = ( inner((T_answer - T_prev) / dt, s_test) * dx
          + inner(grad(s_test), K1/(rho1 * Cp1) * grad(T_answer)) * dx
          +   inner(s_test, dot(u_answer, grad(T_answer))) * dx)

    return F3


def solve_navier_stokes_heat_transfer(domain, Bc, dt, upT, W, rho1, mu1, gamma, n_v, alpha1, Cp1, K1, absolute_tolerance, relative_tolerance, u_answer, u_prev, T_answer, T_prev, p_answer, v_test, q_test, s_test, ds1, dx1):
    '''
    Redefined for dolfinx
    See https://jsdokken.com/dolfinx-tutorial/chapter4/newton-solver.html
    '''

    # Define weak forms
    F1_form = F1(u_answer, q_test, dt)
    F2_form = F2(u_answer, u_prev, p_answer, T_answer, v_test, dt, rho1, n_v, mu1, gamma, alpha1, ds1, dx1)
    F3_form = F3(T_answer, T_prev, u_answer, s_test, dt, rho1, Cp1, K1)

    # Define the combined weak form
    F = F1_form + F2_form + F3_form

    # Define the Jacobian
    J = derivative(F, upT)

    # Set up the nonlinear variational problem
    problem = NonlinearProblem(F, upT, Bc, J)

    # Set up the solver
    solver = NewtonSolver(MPI.COMM_WORLD, problem)

    solver.convergence_criterion = "incremental"
    solver.rtol = relative_tolerance
    solver.atol = absolute_tolerance
    # solver.report = True

    ksp = solver.krylov_solver

    ksp.setInitialGuessNonzero(True)

    # PETSC database options, example
    # https://petsc.org/release/manualpages/KSP/KSPSetInitialGuessNonzero/#options-database-key
    # Not entirely sure what the following options do, commenting them out for now.

    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "gmres"
    # opts[f"{option_prefix}ksp_rtol"] = relative_tolerance
    opts[f"{option_prefix}pc_type"] = "hypre"
    # opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
    # opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 1
    # opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"
    ksp.setFromOptions()

    # pdb.set_trace()

    return solver


#############################  END  ########################################

############################ Boundary Condition Section #################

def define_boundary_condition(W: fem.FunctionSpace, coords, T_LEFT, T_RIGHT) :
    # Define the Domain boundaries based on the previous setup
    (X0, Y0), (X1, Y1) = coords

    domain = W.mesh

    # In order: top, bottom, right, left
    boundaries = [
        (1, lambda x: np.isclose(x[1], Y1)),
        (2, lambda x: np.isclose(x[1], Y0)),
        (3, lambda x: np.isclose(x[0], X1)),
        (4, lambda x: np.isclose(x[0], X0)),
    ]

    # Collapse the subspace, otherwise dolfinx will complain
    # https://fenicsproject.discourse.group/t/dolfinx-dirichlet-bcs-for-mixed-function-spaces/7844/2
    # https://fenicsproject.discourse.group/t/meaning-of-collapse/10641/2
    W0, _ = W.sub(0).collapse()
    W1, _ = W.sub(1).collapse()
    # Unlike in Fenics, the dofmap generated is distinct for each subspace?
    # Need to collapse them seperately
    W2, _ = W.sub(2).collapse()

    # get boundary DoFs
    # Not certain about the output format; two np arrays but we only use
    # the 1st for normal subsapces, 2 for vector subspace?
    dofs_T_Vel = fem.locate_dofs_geometrical((W.sub(0), W0), boundaries[0][1])
    dofs_B_Vel = fem.locate_dofs_geometrical((W.sub(0), W0), boundaries[1][1])
    dofs_R_Vel = fem.locate_dofs_geometrical((W.sub(0), W0), boundaries[2][1])
    dofs_L_Vel = fem.locate_dofs_geometrical((W.sub(0), W0), boundaries[3][1])

    D_C = 0

    # Since the vector field is in 2d, we need to interpolate it to a 2d array
    # Probably also needs to handle the 3d case? (add another stack)
    # https://fenicsproject.discourse.group/t/dirichletbcs-assignment-for-coupled-vector-field-problem/9050/5

    def zeros_vector_expression(x):
        return np.zeros((W.mesh.topology.dim, x.shape[1]))

    u_Dvec = fem.Function(W0)
    u_Dvec.interpolate(zeros_vector_expression)

    # Define Dirichlet boundary conditions
    # first bc fixes y velocity to 0?
    bc_v_top = fem.dirichletbc(PETSc.ScalarType(D_C), dofs_T_Vel[0], W.sub(0).sub(1))
    # Basing the format on the follwing:
    # https://fenicsproject.discourse.group/t/impose-a-dirichlet-bc-on-a-vector-sub-space/14346
    bc_v_bottom = fem.dirichletbc(u_Dvec, dofs_B_Vel, W.sub(0))
    bc_v_right = fem.dirichletbc(u_Dvec, dofs_R_Vel, W.sub(0))
    bc_v_left = fem.dirichletbc(u_Dvec, dofs_L_Vel, W.sub(0))

    dofs_R_Temp = fem.locate_dofs_geometrical((W.sub(2), W2), boundaries[2][1])
    dofs_L_Temp = fem.locate_dofs_geometrical((W.sub(2), W2), boundaries[3][1])

    bc_T_left = fem.dirichletbc(PETSc.ScalarType(T_LEFT), dofs_L_Temp[0], W.sub(2))
    bc_T_right = fem.dirichletbc(PETSc.ScalarType(T_RIGHT), dofs_R_Temp[0], W.sub(2))

    # Point for setting pressure
    zero_pressure_point = fem.locate_dofs_geometrical(
        (W.sub(1), W1),
        lambda x: np.all([np.isclose(x[1], Y1), np.isclose(x[0], X0)], axis = 0)
    )

    bc_p_zero = fem.dirichletbc(PETSc.ScalarType(0), zero_pressure_point[0], W.sub(1))

    # Combine all boundary conditions

    bc_all = [bc_v_left, bc_v_right, bc_v_bottom, bc_v_top, bc_T_left, bc_T_right, bc_p_zero]

    # ******************************************
    # Create meshtags for marking the subdomains

    fdim = domain.topology.dim - 1

    facet_indices, facet_markers = [], []
    for (marker, locator) in boundaries:
        facets = mesh.locate_entities(domain, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    sub_domains = mesh.meshtags(domain, fdim, facet_indices[sorted_facets],
                              facet_markers[sorted_facets])

    # Define measures with the subdomain marking
    ds = Measure("ds", domain=domain, subdomain_data=sub_domains)  # For boundary integration

    # Define an interior domain class to mark the interior of the domain

    def interior(x):
        return np.invert(np.any([np.isclose(x[1], Y1), np.isclose(x[1], Y0)], axis=0))

    # Mark the interior domain
    interior_facets = mesh.locate_entities(domain, fdim, interior)

    interior_domains = mesh.meshtags(domain, fdim, interior_facets,
                                     np.full_like(interior_facets, 1))

    # Define the dx measure for the interior domain
    dx = Measure("dx", domain=domain, subdomain_data=interior_domains)

    return ds, dx, bc_all

#############################  END  ################################





#################### Define Step 1 For Solving  ####################
if __name__ == '__main__':
    W, v_test, q_test, s_test, upT, upT0, u_answer, p_answer, T_answer, u_prev, p_prev, T_prev = create_function_spaces(mar_domain)

    n_v = fem.Constant(mar_domain, np.array([0.0, 1.0]))

    ds1, dx1, bc_all = define_boundary_condition(W, coords, T_LEFT, T_RIGHT)



    # solver = solve_navier_stokes_heat_transfer(
    #     domain, bc_all, dt, upT, W, RHO1, MU1, GAMMA, n_v, ALPHA1, CP1, K1, 1E-5, 1E-6 )

    solver = solve_navier_stokes_heat_transfer(
        mar_domain, bc_all, dt, upT, W, RHO1, MU1, GAMMA, n_v, ALPHA1, CP1, K1,  1E-8 , 1E-7,
        u_answer, u_prev, T_answer, T_prev, p_answer, v_test, q_test, s_test, ds1, dx1)



#############################  END  ###############################



#################### Define Initial Condition ####################

def initial_conditions():

    ics = [
        (0, lambda x: np.zeros((W.mesh.topology.dim, x.shape[1]))),
        (1, lambda x: np.zeros_like(x[0])),
        (2, lambda x: np.full_like(x[0], 273.15))
    ]

    return ics


if __name__ == '__main__':
    # initial_v  = InitialConditions( degree = 2 )
    ics = initial_conditions()
    for idx, ic in ics:
        upT.sub(idx).interpolate(ic)
        upT0.sub(idx).interpolate(ic)

#############################  END  ################################

############################ File Section #########################

if __name__ == '__main__':
    file = io.XDMFFile(mar_domain.comm, Path("Mar_Bou_x.xdmf"), 'w') # File Name To Save #


def write_simulation_data(sol_func: fem.Function, time, file : io.XDMFFile, variable_names):

    # Configure file parameters
    # Autoconfigured in dolfinx?

    # Split the combined function into its components
    functions = sol_func.split()

    # Check if the number of variable names matches the number of functions
    if variable_names and len(variable_names) != len(functions):
        raise ValueError("The number of variable names must match the number of functions.")

    # Rename and write each function to the file
    for i, func in enumerate(functions):
        name = variable_names[i] if variable_names else f"Variable_{i}"
        func.name = f"{name} solution"
        file.write_function(func, time)

    file.close()


if __name__ == '__main__':
    T = fem.Constant(mar_domain, default_scalar_type(0))

    file.write_mesh(mar_domain)

    variable_names = ["Vel", "Press", "T"]  # Adjust as needed


    write_simulation_data(upT0, T, file , variable_names=variable_names )


#############################  END  ###############################


########################### Solving Loop  #########################


if __name__ == '__main__':
    # Time-stepping loop
    for it in tqdm(range(1000)):


        # Write data to file at certain intervals
        if it % 100 == 0:
            write_simulation_data(upT, T.value, file, variable_names)


        # Solve the system
        no_of_it, converged = solver.solve(upT)

        assert converged

        # Update the previous solution
        upT0.x.array[:] = upT.x.array
        upT.x.scatter_forward()

        # Update time
        T.value += dt

        # Printing Informations Related to solutions behaviour


        if rank == 0 and it% 1000 ==0  :  # Only print for the root process

            print(" ├─ Iteration: " + str(it), flush=True)
#############################  END  ###############################