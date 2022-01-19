# ----------------------------------------------------------------------------------------
# 					SIMULATION PARAMETERS FOR THE PIC-CODE SMILEI
# ----------------------------------------------------------------------------------------
#
# Remember: never override the following names:
#           SmileiComponent, Species, Laser, Collisions, DiagProbe, DiagParticles,
#           DiagScalar, DiagPhase or ExtField
#

import math as m
import numpy as np
import random as rd

resy = 4.				# spatial resolution along y axis (i.e number of cell per de)
resx = 4.				# spatial resolution along x axis
rest = np.sqrt(2.*resx**2)/0.95		# time-resolution (I use dt=0.95*CFL)

# electron normalisation :

c       = 1.
wpewce  = 2.
mime    = 25.


#Harris initialisation
B0    = 1./wpewce           	# amplitude of the magnetic field (we use 1 for the magnetosheat magnetic field)
n0    = 1.                      # normalization of the density
v0    = 1./(np.sqrt(mime)*wpewce)# normalization of the velocity
L     = 1. *np.sqrt(mime)        # transverse width of the B-field distribution

box_size = [64*np.sqrt(mime), 32*np.sqrt(mime)]
x0    = box_size[0]*0.5 #6.4 *np.sqrt(mime)        # position of the perturbation 
y0    = box_size[1]*0.5 #12.8*np.sqrt(mime)        # position of the layer

nb    = 0.2*n0              # background density


#perturbation
psi0   = 1.5*np.sqrt(mime) / wpewce
db    = 0.5
sigma = 1.5

theta = 0.2         # Te/Ti ratio

pbc   = (B0)**2    # pressure balance constant (arbitrary).

lx = 2*L
lz = 1*L
a0=1*L/np.sqrt(mime)
ap=0.5*a0

# Magnetic field components (fluid equilibrium)

def Bx(x,y):
    return B0 * np.tanh((y-y0)/L)


#Perturbed B
def Bx_(x,y):
    xx = x - x0
    yy = y - y0 #(y + ((1/resy)/2.) -y0)
    return Bx(x,y) - ap*yy/lz**2*np.exp(-xx**2/(2.*lx**2)-yy**2/(2.*lz**2))/wpewce

def By_(x,y):
    xx = x - x0
    yy = y - y0

    return 0. +ap*xx/lx**2*np.exp(-xx**2/(2.*lx**2)-yy**2/(2.*lz**2))/wpewce


# n = ion density (fluid equilibrium) = electron density
def n(x,y):
    return nb + n0 / np.cosh((y-y0)/L)**2 


#Current components

def Jz(x,y):
    xx = x -x0 #(x + ((1/resy)/2.))-x0
    yy = y -y0 #(y + ((1/resy)/2.))-y0
    perturbation = ap*(-xx**2/lx**4 + 1/lx**2 -yy**2/lz**4 + 1/lz**2)*np.exp(-xx**2/(2.*lx**2)-yy**2/(2.*lz**2))/wpewce*np.sqrt(mime)
    return -B0/L * (1. - np.tanh((y-y0)/L)**2) + perturbation

#Ion velocity components

def vix(x,y):
    return 0.
	
def viy(x,y):
    return 0.

def viz(x,y):
    return Jz(x,y)/n(x,y)/(1+theta)

#Ion temperature

def Ti(x,y):
    return (pbc-B0**2/2.)/(1+theta)/n0 
    #return (pbc-By(x,y)**2/2.)/(1+theta)/n(x,y)

#Electron velocity components

def vex(x,y):
    return 0.

def vey(x,y):
    return 0.

def vez(x,y):
    return -theta*viz(x,y)

#Electron temperature
def Te(x,y):
    return Ti(x,y)*theta


Main(
     geometry = '2Dcartesian',
          
     interpolation_order = 2,

     timestep = 1./rest,
     
     simulation_time = 30200/rest, # 30.0*wpewce*mime,
     #
     cell_length = [1./resx, 1./resy],
     #grid_length = box_size,
     grid_length = [box_size[0], box_size[1]],

     number_of_patches = [64,32],        

     maxwell_solver = 'Yee',
     

     EM_boundary_conditions = [['periodic'],['reflective']],


     random_seed = 0,


     print_every = int(1),


     patch_arrangement = "hilbertian",
)


LoadBalancing(
              initial_balance = True,
              every = 150
)

#Vectorization(
#              mode = "adaptive",
#              reconfigure_every = 20,
#              initial_mode = "on"
#)


# DEFINE EXTERNAL FIELD

ExternalField(
        field = 'Bx',
        profile = Bx_
)


ExternalField(
        field = 'By',
        profile = By_
)

# Ions

Species(
        name = 'ion',
        mass = mime,
        charge = 1.,
        particles_per_cell = 50, # if using fewer particles - it sometimes restarts successfully
        position_initialization = 'random',
        momentum_initialization = 'maxwell-juettner',
        number_density = n,
        temperature = [Ti,Ti,Ti],
        mean_velocity = [0,0,viz],
        boundary_conditions  = [['periodic'],['reflective']],
        pusher = 'boris'
	)


# Electrons

Species(
        name = 'eon',
        mass = 1,
        charge = -1.,
        particles_per_cell = 50,
        position_initialization = 'random',
        momentum_initialization = 'maxwell-juettner',
        number_density = n,
        temperature = [Te,Te,Te],
        mean_velocity = [0,0,vez],
        boundary_conditions  = [['periodic'],['reflective']],
        pusher = 'boris'
	)	

Species(
        name = 'ionbg',
        mass = mime,
        charge = 1.,
        particles_per_cell = 50,
        position_initialization = 'random',
        momentum_initialization = 'cold',
        number_density = nb,
        mean_velocity = [0,0,0],
        boundary_conditions  = [['periodic'],['reflective']],
        pusher = 'boris'
    )


# Electrons

Species(
        name = 'eonbg',
        mass = 1,
        charge = -1.,
        particles_per_cell = 50,
        position_initialization = 'random',
        momentum_initialization = 'cold',
        number_density = nb,
        mean_velocity = [0,0,0],
        boundary_conditions  = [['periodic'],['reflective']],
        pusher = 'boris'
    )

# ---------------------
# DIAGNOSTIC PARAMETERS
# ---------------------

globalEvery = int(100)

# Scalar diagnostics (all used)
DiagScalar(
           every=globalEvery
)

# Field diagnostics (we select only the one we are interested in)
DiagFields(
           every = globalEvery,
           #time_average = 2,
           #fields = ['Ex','Ey','Ez','Bx','By','Bz','Rho_ion','Rho_eon','Jx_ion','Jx_eon','Jy_ion','Jy_eon','Jz_ion','Jz_eon']
           fields = ['Ex','Ey','Ez','Bx','By','Bz','Rho_ion','Rho_eon','Rho_ionbg','Rho_eonbg','Jx_ion','Jx_eon','Jx_ionbg','Jx_eonbg','Jy_ion','Jy_eon','Jy_ionbg','Jy_eonbg','Jz_ion','Jz_eon','Jz_ionbg','Jz_eonbg']

)


##Pressure diagnostics #Can restart when including this block
#species_all = [["ion"], ["eon"]]
#for comp in [["x","x"],["x","y"],["x","z"],["y","y"],["y","z"],["z","z"]]:
#	for species in species_all:
#      #print("weight_v" + comp[0] + " _p" + comp[1])
#	    DiagParticleBinning(      
#	        deposited_quantity = "weight_v" + comp[0] + "_p" + comp[1],
#	        every = globalEvery,
#	        species = species,
#	        axes = [
#	                ["x", 0., Main.grid_length[0], int(Main.grid_length[0]*resx)+1],
#	                ["y", 0., Main.grid_length[1], int(Main.grid_length[1]*resy)+1]
#	                ]
#	    )

# Fails to restart when including this block
#DiagParticleBinning(        
#    name = "ekin_dist_eon",
#    deposited_quantity = "weight",
#    every = globalEvery,
#    species = ["eon"],
#    axes = [ ["ekin",    0.02,    2.,   400, "logscale"] ]
#)
#
#DiagParticleBinning(
#    name = "ekin_dist_eon2",
#    deposited_quantity = "weight",
#    every = globalEvery,
#    species = ["eon"],
#    axes = [ ["ekin",    0.02,    2.,   400, "logscale"] ]
#)

# ---------------------
# STOP AND RESTART
# ---------------------

Checkpoints(
            # Directory of the input *.h5 and scalars.txt
#           restart_dir = "./",
            restart_dir = None,
#            restart_number = 0,

            dump_minutes = 2,

            # change to False for long simulations
            exit_after_dump = True,

            # number of dump we keep for later restart (or distribution studies)
            keep_n_dumps = 2
)



