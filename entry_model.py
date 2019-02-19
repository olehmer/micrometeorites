"""
This code models the entry of iron micrometeorites into an atmosphere. For help
with this code email info@lehmer.us.

Owen Lehmer - 1/14/19
"""

from multiprocessing import Pool, cpu_count
from math import sin, cos, pi, floor, sqrt
from scipy.integrate import solve_ivp
from scipy import stats
from tqdm import tqdm
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


#Define constants here
EARTH_RAD = 6.37E6 #radius of Earth [m]
EARTH_MASS = 5.97E24 #Earth mass [kg]
GRAV_CONST = 6.67E-11 #gravitational constant [N m2 kg-2]
KB = 1.381E-23 #Boltzmann constant [J K-1]
PROTON_MASS = 1.67E-27 #mass of a proton [kg]
SIGMA = 5.67E-8 #Stefan-Boltzmann constant [W m-2 K-4]
GAS_CONST = 8.314 #ideal gas constant [J mol-1 K-1]
M_FE = 0.0558 #molecular weight of Fe [kg mol-1]
M_O = 0.016 #molecular weight of O [kg mol-1]
M_FEO = M_FE + M_O #molecular weight of FeO [kg mol-1]
L_V = 6.050E6 #latent heat of vaporization for FeO [J kg-1] from Genge
C_SP = 390 #specific heat of FeO from Stolen et al. (2015) [J K-1 kg-1]
FE_MELTING_TEMP = 1809 #temperature at which Fe melts [K]
FEO_MELTING_TEMP = 1720 #melting temp of Fe) [K]

#densities from from the paragraph below equation 10 in Genge et al. (2016)
RHO_FE = 7000 #liquid Fe density [kg m-3]
RHO_FEO = 4400 #liquid FeO density [kg m-3]

GAMMA = 1.0
CO2_FAC = -1 #the CO2 concentration for this model, -1 turns it off and uses O2


class impactAngleDistribution(stats.rv_continuous):
    """
    The probability distribution of impact angles. This class will generate
    random entry angles from the distribution defined in Love and Brownlee 
    (1991).
    """

    def __init__(self):
        """
        Set the lower limit to 0 degrees and the upper limit to 90 degrees
        """
        super().__init__(a=0, b=pi/2)

    def _pdf(self, x):
        prob = 0
        if 0 < x < pi/2:
            #between 0 and 90 degrees
            prob = sin(2*x)
        return prob

    def sample(self, size=1, random_state=None):
        return self.rvs(size=size, random_state=random_state)

class initialVelocityDistribution(stats.rv_continuous):
    """
    The probability distribution of initial velocity. This class will generate
    random initial velocities from the distribution defined in Love and Brownlee 
    (1991).

    Note: the values generated from sample are in [km s-1]

    Second note: the velocity is capped at 25 km/s. This is done because faster
    entries require smaller time steps, which makes the model take unreasonably
    slow. This is likely acceptable as velocities under 25 km/s account for 
    97 percent of the incident velocities, so the final result should be 
    representative of the distribution.
    """

    def __init__(self):
        """
        Set the lower limit to 11.2 [km s-1] and the upper to 72 [km s-1]
        """
        super().__init__(a=11.2, b=25) #upper limit set to 25 km s-1


    def _pdf(self, x):
        prob = 0
        if x > 11.2:
            prob = 1.791E5*x**-5.394
        return prob

    def sample(self, size=1, random_state=None):
        return self.rvs(size=size, random_state=random_state)


class initialMassDistribution(stats.rv_continuous):
    """
    The probability distribution of initial mass. This class will generate
    random initial masses from the distribution defined in Love and Brownlee 
    (1991). The masses can easily be converted to initial radii.

    Note: the values generated from the PDF are in [g]
    """

    def __init__(self):
        """
        Set the lower limit to 3.665E-9 [g] (5 [micron] Fe radius) and the upper 
        0.02932 [g] (1000 [micron] Fe radius).
        """
        super().__init__(a=3.665E-9, b=0.02932)


    def _pdf(self, x):
        prob = 0
        if 3.665E-9 < x < 0.02932:
            prob = ((2.2E3*x**0.306+15)**-4.38 + 1.3E-9*(x + 10**11*x**2 + \
                    10**27*x**4)**-0.36)/4.50936E-13
            prob = float(prob)
        return prob

    def sample(self, size=1, random_state=None):
        return self.rvs(size=size, random_state=random_state)


def zStatistic(population_data, sample_data):
    """
    Calculate the z score and the corresponding probability that the sample
    data is from the same population as the population data. It is assumed that
    the population (calculated from the model) is approximately normally 
    distributed.

    Inputs:
        population_data - 1D array of population data
        sample_data     - 1D array of sample data

    Returns:
        p_value - the probability the sample is from the population (two sided)
    """

    pop_mean = np.mean(population_data)
    pop_std = np.std(population_data)

    sample_mean = np.mean(sample_data)

    sample_size = len(sample_data)
    ste = pop_std/(sample_size)**0.5 #standard error

    z_score = (sample_mean - pop_mean)/ste
    print(z_score)

    p_value = stats.norm.sf(abs(z_score))*2

    return p_value



def Genge_2017_Fe_Fraction():
    """
    Using the data from Figure 4 of Genge et al. (2017), this function returns
    the Fe fraction for each of the reported micrometeorites. The values in 
    the figure are giving in %-area. It is worth noting that this is a minimum
    value for the Fe bead as the preparation process is unlikely to grind the
    bead at exactly the middle, so the widest Fe bead may not be displayed.

    From the data in Figure 4 there are 34 micrometeorites with Fe cores, and
    50 micrometeorites that are fully oxidized to either wustite, magnetite, or
    some combination of the two.

    Returns:
        fe_mass_fractions - array of Fe mass fractions for each micrometeorite
    """

    #the data from figure 4, truncated to 3 decimal points
    genge_data = [
        0.921,
        0.487,
        0.485,
        0.437,
        0.431,
        0.422,
        0.413,
        0.356,
        0.333,
        0.319,
        0.300,
        0.287,
        0.276,
        0.264,
        0.212,
        0.211,
        0.181,
        0.178,
        0.171,
        0.171,
        0.169,
        0.127,
        0.127,
        0.108,
        0.155,
        0.261,
        0.052,
        0.393,
        0.493,
        0.281,
        0.390,
        0.076,
        0.191,
        0.020]


    return genge_data



def atmospheric_density_and_oxygen(altitude):
    """
    This function will return the atmospheric density and the total oxygen
    density (including atomic O and O2) for the given altitude. This is done
    using data from the MSISE-90 model. The data is in 1km steps, which this
    function will linearly interpolate between. The data arrays come from the
    function ReadAndProcessAtmosData(). The arrays were hardcoded to avoid 
    reading them in during each model run.

    Inputs:
        altitude - the altitude above the Earth's center [km]

    Returns:
        rho_a - atmospheric density [kg m-3]
        rho_o - total oxygen density [kg m-3]
    """

    alt = (altitude - EARTH_RAD)/1000 #convert to traditional altitude [km]
    rho_a = 0
    rho_o = 0

    #data from atmosphere_data.txt
    alt_data = [0.00e+00, 1.00e+00, 2.00e+00, 3.00e+00, 4.00e+00, 5.00e+00, 
                6.00e+00, 7.00e+00, 8.00e+00, 9.00e+00, 1.00e+01, 1.10e+01, 
                1.20e+01, 1.30e+01, 1.40e+01, 1.50e+01, 1.60e+01, 1.70e+01, 
                1.80e+01, 1.90e+01, 2.00e+01, 2.10e+01, 2.20e+01, 2.30e+01, 
                2.40e+01, 2.50e+01, 2.60e+01, 2.70e+01, 2.80e+01, 2.90e+01, 
                3.00e+01, 3.10e+01, 3.20e+01, 3.30e+01, 3.40e+01, 3.50e+01, 
                3.60e+01, 3.70e+01, 3.80e+01, 3.90e+01, 4.00e+01, 4.10e+01, 
                4.20e+01, 4.30e+01, 4.40e+01, 4.50e+01, 4.60e+01, 4.70e+01, 
                4.80e+01, 4.90e+01, 5.00e+01, 5.10e+01, 5.20e+01, 5.30e+01, 
                5.40e+01, 5.50e+01, 5.60e+01, 5.70e+01, 5.80e+01, 5.90e+01, 
                6.00e+01, 6.10e+01, 6.20e+01, 6.30e+01, 6.40e+01, 6.50e+01,
                6.60e+01, 6.70e+01, 6.80e+01, 6.90e+01, 7.00e+01, 7.10e+01, 
                7.20e+01, 7.30e+01, 7.40e+01, 7.50e+01, 7.60e+01, 7.70e+01, 
                7.80e+01, 7.90e+01, 8.00e+01, 8.10e+01, 8.20e+01, 8.30e+01, 
                8.40e+01, 8.50e+01, 8.60e+01, 8.70e+01, 8.80e+01, 8.90e+01, 
                9.00e+01, 9.10e+01, 9.20e+01, 9.30e+01, 9.40e+01, 9.50e+01, 
                9.60e+01, 9.70e+01, 9.80e+01, 9.90e+01, 1.00e+02, 1.01e+02, 
                1.02e+02, 1.03e+02, 1.04e+02, 1.05e+02, 1.06e+02, 1.07e+02, 
                1.08e+02, 1.09e+02, 1.10e+02, 1.11e+02, 1.12e+02, 1.13e+02, 
                1.14e+02, 1.15e+02, 1.16e+02, 1.17e+02, 1.18e+02, 1.19e+02, 
                1.20e+02, 1.21e+02, 1.22e+02, 1.23e+02, 1.24e+02, 1.25e+02,
                1.26e+02, 1.27e+02, 1.28e+02, 1.29e+02, 1.30e+02, 1.31e+02, 
                1.32e+02, 1.33e+02, 1.34e+02, 1.35e+02, 1.36e+02, 1.37e+02,
                1.38e+02, 1.39e+02, 1.40e+02, 1.41e+02, 1.42e+02, 1.43e+02,
                1.44e+02, 1.45e+02, 1.46e+02, 1.47e+02, 1.48e+02, 1.49e+02,
                1.50e+02, 1.51e+02, 1.52e+02, 1.53e+02, 1.54e+02, 1.55e+02,
                1.56e+02, 1.57e+02, 1.58e+02, 1.59e+02, 1.60e+02, 1.61e+02,
                1.62e+02, 1.63e+02, 1.64e+02, 1.65e+02, 1.66e+02, 1.67e+02, 
                1.68e+02, 1.69e+02, 1.70e+02, 1.71e+02, 1.72e+02, 1.73e+02,
                1.74e+02, 1.75e+02, 1.76e+02, 1.77e+02, 1.78e+02, 1.79e+02,
                1.80e+02, 1.81e+02, 1.82e+02, 1.83e+02, 1.84e+02, 1.85e+02,
                1.86e+02, 1.87e+02, 1.88e+02, 1.89e+02, 1.90e+02]
    
    rho_a_data = [1.21e+00, 1.09e+00, 9.84e-01, 8.90e-01, 8.05e-01, 7.28e-01, 
                  6.57e-01, 5.90e-01, 5.28e-01, 4.70e-01, 4.15e-01, 3.65e-01, 
                  3.19e-01, 2.77e-01, 2.39e-01, 2.06e-01, 1.76e-01, 1.50e-01, 
                  1.28e-01, 1.09e-01, 9.23e-02, 7.86e-02, 6.69e-02, 5.70e-02, 
                  4.86e-02, 4.15e-02, 3.55e-02, 3.03e-02, 2.59e-02, 2.22e-02,
                  1.90e-02, 1.63e-02, 1.39e-02, 1.20e-02, 1.03e-02, 8.84e-03, 
                  7.63e-03, 6.60e-03, 5.72e-03, 4.97e-03, 4.32e-03, 3.77e-03,
                  3.30e-03, 2.89e-03, 2.54e-03, 2.24e-03, 1.98e-03, 1.75e-03, 
                  1.55e-03, 1.37e-03, 1.22e-03, 1.08e-03, 9.60e-04, 8.53e-04,
                  7.58e-04, 6.74e-04, 5.98e-04, 5.31e-04, 4.71e-04, 4.17e-04,
                  3.69e-04, 3.26e-04, 2.88e-04, 2.53e-04, 2.23e-04, 1.96e-04,
                  1.71e-04, 1.50e-04, 1.31e-04, 1.14e-04, 9.88e-05, 8.57e-05,
                  7.41e-05, 6.37e-05, 5.50e-05, 4.74e-05, 4.08e-05, 3.51e-05,
                  3.00e-05, 2.55e-05, 2.16e-05, 1.82e-05, 1.53e-05, 1.27e-05,
                  1.05e-05, 8.67e-06, 7.10e-06, 5.79e-06, 4.69e-06, 3.79e-06,
                  3.05e-06, 2.45e-06, 1.96e-06, 1.57e-06, 1.26e-06, 1.01e-06,
                  8.20e-07, 6.65e-07, 5.43e-07, 4.47e-07, 3.70e-07, 3.09e-07,
                  2.60e-07, 2.21e-07, 1.88e-07, 1.61e-07, 1.38e-07, 1.19e-07,
                  1.03e-07, 8.90e-08, 7.70e-08, 6.66e-08, 5.77e-08, 4.99e-08,
                  4.33e-08, 3.76e-08, 3.27e-08, 2.86e-08, 2.50e-08, 2.19e-08,
                  1.93e-08, 1.70e-08, 1.51e-08, 1.34e-08, 1.20e-08, 1.08e-08,
                  9.76e-09, 8.87e-09, 8.09e-09, 7.41e-09, 6.81e-09, 6.28e-09,
                  5.81e-09, 5.38e-09, 5.00e-09, 4.66e-09, 4.36e-09, 4.08e-09,
                  3.82e-09, 3.59e-09, 3.38e-09, 3.18e-09, 3.00e-09, 2.84e-09,
                  2.69e-09, 2.55e-09, 2.42e-09, 2.30e-09, 2.18e-09, 2.08e-09,
                  1.98e-09, 1.89e-09, 1.81e-09, 1.72e-09, 1.65e-09, 1.58e-09,
                  1.51e-09, 1.45e-09, 1.39e-09, 1.33e-09, 1.28e-09, 1.23e-09,
                  1.18e-09, 1.14e-09, 1.10e-09, 1.06e-09, 1.02e-09, 9.82e-10,
                  9.47e-10, 9.14e-10, 8.83e-10, 8.53e-10, 8.24e-10, 7.96e-10,
                  7.70e-10, 7.45e-10, 7.21e-10, 6.98e-10, 6.76e-10, 6.55e-10,
                  6.35e-10, 6.16e-10, 5.97e-10, 5.79e-10, 5.62e-10, 5.46e-10, 
                  5.30e-10, 5.14e-10, 5.00e-10, 4.86e-10, 4.72e-10]

    rho_o_data = [2.83e-01, 2.54e-01, 2.29e-01, 2.07e-01, 1.88e-01, 1.70e-01, 
                  1.53e-01, 1.38e-01, 1.23e-01, 1.09e-01, 9.68e-02, 8.51e-02,
                  7.43e-02, 6.46e-02, 5.58e-02, 4.79e-02, 4.10e-02, 3.50e-02,
                  2.98e-02, 2.53e-02, 2.15e-02, 1.83e-02, 1.56e-02, 1.33e-02,
                  1.13e-02, 9.67e-03, 8.26e-03, 7.06e-03, 6.04e-03, 5.17e-03,
                  4.42e-03, 3.79e-03, 3.25e-03, 2.78e-03, 2.39e-03, 2.06e-03,
                  1.78e-03, 1.54e-03, 1.33e-03, 1.16e-03, 1.01e-03, 8.80e-04,
                  7.69e-04, 6.74e-04, 5.93e-04, 5.22e-04, 4.60e-04, 4.07e-04,
                  3.60e-04, 3.19e-04, 2.83e-04, 2.52e-04, 2.24e-04, 1.99e-04,
                  1.77e-04, 1.57e-04, 1.39e-04, 1.24e-04, 1.10e-04, 9.72e-05,
                  8.59e-05, 7.59e-05, 6.70e-05, 5.90e-05, 5.19e-05, 4.55e-05,
                  3.99e-05, 3.48e-05, 3.04e-05, 2.64e-05, 2.30e-05, 1.99e-05,
                  1.72e-05, 1.48e-05, 1.28e-05, 1.10e-05, 9.45e-06, 8.11e-06,
                  6.93e-06, 5.89e-06, 4.99e-06, 4.19e-06, 3.51e-06, 2.91e-06,
                  2.41e-06, 1.97e-06, 1.61e-06, 1.31e-06, 1.06e-06, 8.49e-07,
                  6.80e-07, 5.43e-07, 4.32e-07, 3.44e-07, 2.75e-07, 2.20e-07,
                  1.77e-07, 1.43e-07, 1.16e-07, 9.48e-08, 7.81e-08, 6.49e-08,
                  5.43e-08, 4.57e-08, 3.87e-08, 3.29e-08, 2.81e-08, 2.40e-08,
                  2.06e-08, 1.77e-08, 1.52e-08, 1.31e-08, 1.13e-08, 9.76e-09,
                  8.44e-09, 7.32e-09, 6.37e-09, 5.56e-09, 4.86e-09, 4.28e-09,
                  3.77e-09, 3.35e-09, 2.98e-09, 2.67e-09, 2.40e-09, 2.17e-09,
                  1.98e-09, 1.81e-09, 1.67e-09, 1.54e-09, 1.43e-09, 1.33e-09,
                  1.24e-09, 1.16e-09, 1.08e-09, 1.02e-09, 9.60e-10, 9.07e-10,
                  8.57e-10, 8.13e-10, 7.71e-10, 7.33e-10, 6.98e-10, 6.66e-10,
                  6.36e-10, 6.08e-10, 5.81e-10, 5.57e-10, 5.34e-10, 5.13e-10,
                  4.93e-10, 4.74e-10, 4.56e-10, 4.39e-10, 4.23e-10, 4.08e-10,
                  3.94e-10, 3.81e-10, 3.68e-10, 3.56e-10, 3.44e-10, 3.34e-10,
                  3.23e-10, 3.13e-10, 3.04e-10, 2.95e-10, 2.86e-10, 2.78e-10,
                  2.70e-10, 2.63e-10, 2.55e-10, 2.48e-10, 2.42e-10, 2.35e-10,
                  2.29e-10, 2.23e-10, 2.18e-10, 2.12e-10, 2.07e-10, 2.02e-10,
                  1.97e-10, 1.92e-10, 1.87e-10, 1.83e-10, 1.79e-10, 1.75e-10,
                  1.71e-10, 1.67e-10, 1.63e-10, 1.60e-10, 1.56e-10]
    
    if 0 < alt < 190:
        #linearly interpolate between the two closest points
        idx = int(floor(alt))
        frac_low = 1 - (alt-alt_data[idx])/(alt_data[idx+1] - 
                                            alt_data[idx])

        rho_a = rho_a_data[idx]*frac_low + rho_a_data[idx+1]*(1-frac_low)
        rho_o = rho_o_data[idx]*frac_low + rho_o_data[idx+1]*(1-frac_low)

    return rho_a, rho_o


def get_radius_and_density(m_fe, m_feo, not_array=True):
    """
    Calculate the radius and bulk density of the micrometeorite using the mass
    fraction of Fe to FeO.

    Inputs:
        M_Fe  - mass of Fe in the micrometeorite [kg]
        M_FeO - mass of FeO in the micrometeorite [kg]

    Returns:
        new_rad - updated micrometeorite radius [m]
        new_rho - updated micrometeorite density [kg m-3]
    """

    if not_array and m_fe + m_feo <= 0:
        return 0, 0
    volume = m_fe/RHO_FE + m_feo/RHO_FEO
    new_rad = (3*volume/(4*pi))**(1/3)

    new_rho = (m_fe + m_feo)/volume

    return new_rad, new_rho



def simulate_particle_ivp(input_mass, input_vel, input_theta, 
        max_time_step=0.0005):
    """
    Top level function to simulate a micrometeorite using the solve_ivp function
    from scipy.

    Inputs:
        input_mass    - the initial mass of the micrometeorite [kg]
        input_vel     - the initial entry velocity of the micrometeorite [m s-1]
        input_theta   - initial entry angle of the micrometeorite [radians]
        max_time_step - the maximum time step to use in the simulation [s]

    Returns:
        res - the output from solve_ivp()
    """


    def sim_func(_, y_in):
        """
        The callable function passed to solve_ivp()
        
        Inputs:
            _    - placeholder for time at which to calculate [s] : NOT USED
            y_in - inputs to the function, has the form:
                TODO: put in form

        Returns:
            dy_dt - the derivative of each y value at t
        """

        alt, vel_tan, vel_rad, mass_fe, mass_feo, temp = y_in

        #read CO2_FAC from command line if present
        if len(sys.argv) == 2:
            CO2_FAC = float(sys.argv[1])

        if mass_fe < 0:
            mass_fe = 0
        if mass_feo < 0:
            mass_feo = 0

        if mass_fe == 0 and mass_feo == 0:
            return [0, 0, 0, 0, 0, 0]

        rad, rho_m = get_radius_and_density(mass_fe, mass_feo)

        #calculate gravity
        gravity = GRAV_CONST*EARTH_MASS/alt**2

        #get the total velocity and angle
        vel = (vel_tan**2 + vel_rad**2)**0.5 #velocity magnitude
        #theta = asin(vel_tan/vel) #angle above surface

        #calculate the atmospheric density and total oxygen density
        rho_a, rho_o = atmospheric_density_and_oxygen(alt)
        if CO2_FAC != -1:
            rho_o = rho_a*CO2_FAC*(16/44) #the O is 16/44 of the mass of CO2

        #calculate the radial and tangential velocity derivatives 
        #we've assumed a flat Earth here
        dvel_tan_dt = -0.75*rho_a*vel_tan**2/(rho_m*rad)
        dvel_rad_dt = gravity - 0.75*rho_a*vel_rad**2/(rho_m*rad)

        #calculate the altitude derivative
        dalt_dt = -vel_rad 

        #the mass derivative
        #Genge equation 13, which is in [dynes cm-2], converted to [Pa] here
        p_v = 10**(10.3-20126/temp)

        #genge equation 12 
        ox_enc = 0
        if temp > FE_MELTING_TEMP or (mass_feo > 0 and temp > FEO_MELTING_TEMP):
            #the particle is molten, let oxygen be absorbed
            ox_enc = GAMMA*rho_o*pi*rad**2*vel

        #Genge equation 7, but the Langmuir formula has been adjusted for SI
        #this mass loss rate is in [kg s-1] of FeO
        dm_evap_dt = 0
        if mass_feo > 0: 
            #make sure there's some FeO, in practice this is always the case
            #once the melting point is reached
            dm_evap_dt = 4*pi*rad**2*p_v*sqrt(M_FEO/(2*pi*GAS_CONST*temp))

        dmass_feo_dt = -dm_evap_dt + (M_FEO/M_O)*ox_enc
        dmass_fe_dt = -(M_FE/M_O)*ox_enc
        
        #oxidation via CO2 is endothermic so DELTA_H_OX is negative
        DELTA_H_OX = -465000 #heat of oxidation for CO2 + Fe -> CO +FeO [J kg-1]
        if CO2_FAC == -1:
            #oxidation via oxygen is exothermic
            DELTA_H_OX = 3716000 #heat of oxidation [J kg-1] from Genge

        dq_ox_dt = DELTA_H_OX*(M_FEO/M_O)*ox_enc #Genge equation 14

        #equation 6 of Genge (2016). This has the oxidation energy considered
        #which is described by equation 14
        dtemp_dt = 1/(rad*C_SP*rho_m)*\
                   (3*rho_a*vel**3/8 - 3*L_V*dm_evap_dt/(4*pi*rad**2) - 
                    3*SIGMA*temp**4 + 3*dq_ox_dt/(4*pi*rad**2))

        return [dalt_dt, 
                dvel_tan_dt, 
                dvel_rad_dt, 
                dmass_fe_dt,
                dmass_feo_dt,
                dtemp_dt]

    #collect the initial values for solve_ivp()
    y_0 = [190000+EARTH_RAD, #initial altitude, 190 [km]
           sin(input_theta)*input_vel, #initial tangential velocity [m s-1]
           cos(input_theta)*input_vel, #initial radial velocity [m s-1]
           input_mass, #initial Fe mass [kg]
           0, #initial FeO mass [kg], always 0
           300] #initial temperature of micrometeorite [K], not important

    #the time range [s] used by solve_ivp()
    time_range = [0, 25]


    res = solve_ivp(sim_func, time_range, y_0, max_step=max_time_step)

    return res

def get_final_radius_and_fe_area_from_sim(data):
    """
    Calculate the final radius and the final Fe area fraction from the solve_ivp
    results.

    Inputs:
        data - the data from the solve_ivp() function

    Returns:
        rad     - final micrometeorite radius [m]
        fe_frac - final micrometeorite Fe fractional area
    """

    fe_mass = data[3, -1]
    feo_mass = data[4, -1]

    #replace negative values
    if fe_mass < 0:
        fe_mass = 0
    if feo_mass < 0:
        feo_mass = 0

    rad = 0
    fe_frac = 0
    if fe_mass > 0 or feo_mass > 0:
        #make sure the micrometeorite didn't completely evaporate
        rad = get_radius_and_density(fe_mass, feo_mass)[0]
        fe_rad = get_radius_and_density(fe_mass, 0)[0]
        feo_rad = get_radius_and_density(0, feo_mass)[0]

        fe_area = pi*fe_rad**2
        feo_area = pi*feo_rad**2
        fe_frac = fe_area/(fe_area + feo_area)

    return [rad, fe_frac]

def readModelDataFile(filename):
    """
    Read the data from an output file.

    Inputs:
        filename - the file to read

    Returns:
        result - the data from the file
    """
    
    file_obj = open(filename, "r")
    result = []

    for line in file_obj:
        line_split = line.split()
        if len(line_split) == 1:
            num_val = float(line_split[0])
            result.append(num_val)
        else:
            nums = []
            for num in line_split:
                num_val = float(num)
                nums.append(num_val)
            result.append(tuple(nums))

    return result



def saveModelData(data, filename):
    """
    Takes an array and saves it to a file.

    Inputs:
        data     - input array to save
        filename - the filename to use for the data
    """

    file_obj = open(filename, "w")
    for d in data:
        line = ""
        if isinstance(d, tuple) or isinstance(d, np.ndarray):
            for item in d:
                line += "%2.10e "%item
        else:
            line += "%2.10e"%d
        line += "\n"
        file_obj.write(line)
    file_obj.close()


def multithreadWrapper(args):
    """
    This function will pass the multithreaded run arguments to simulateParticle
    then return the simulation parameters.

    Input:
        args - a tuple with the form (mass, velocity, impact angle)

    Returns:
        result - a tuple with the form (radius, fe_area)
    """

    mass, velocity, theta = args
    
    tries = -1 #track the number of attempts
    MAX_TRIES = 4 #the orders of magnitude to reduce by overall, in increments
                  #of 1
    initial_max_step = 0.01 #the initial time step to use [s]
    result = (0, 0)

    while tries < MAX_TRIES: 
        tries += 1
        try:
            res = simulate_particle_ivp(mass, velocity, theta, 
                    max_time_step=initial_max_step*10**(-tries))
            data = res.y
            final_radius, fe_area = get_final_radius_and_fe_area_from_sim(data)

            result = (final_radius, fe_area)
        except ValueError:
            pass
        else:
            tries = MAX_TRIES + 1

#    print("-------------------------------------------")
#    if tries == MAX_TRIES + 1:
#        print("Passed with inputs:")
#    else:
#        rad = get_radius_and_density(mass, 0)[0]
#        result = (rad, 1)
#        print("Failed with inputs:")
#
#    print("Mass: %2.3e kg"%(mass))
#    print("Radius: %0.1f microns"%(result[0]/(1.0E-6)))
#    print("Velocity: %0.2f km s-1"%(velocity))
#    print("Theta: %0.1f"%(theta*180/pi))

    if tries != MAX_TRIES + 1:
        #this run didn't converge, return negative values
        result = ( -1, -1)


    return result


def generateRandomSampleData(num_samples=100, output_dir="rand_sim"):
    """
    Randomly sample from the input parameters (impact angle, velocity, radius)
    a given number of times.

    Inputs:
        num_samples - the number of simulations to run
        output_dir  - the directory to which the output file will be saved.
    """

    if __name__ == '__main__':

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            sys.stderr.write("The directory \""+output_dir+"\" already exists.\n")
            sys.stderr.write("Overwrite files in \""+output_dir+"\"? [y/n]: ")
            resp = input()
            if resp not in ("y", "Y"):
                return

        thetas = impactAngleDistribution().sample(size=num_samples)
        velocities = initialVelocityDistribution().sample(size=num_samples)
        velocities = velocities*1000 #convert from [km s-1] to [m s-1]
        masses = initialMassDistribution().sample(size=num_samples)
        masses = masses/1000 #convert from [g] to [kg]

        args_array = []
        for i in range(num_samples):
            args = (masses[i], velocities[i], thetas[i])
            args_array.append(args)


        with Pool(cpu_count()-1) as p:
            results = list(tqdm(p.imap(multithreadWrapper, args_array), 
                                total=num_samples))

            saveModelData(args_array, output_dir+"/args_array.dat")
            saveModelData(results, output_dir+"/results.dat")

            #delete runs with -1 in them, these failed to converge
            results = np.array(results)
            args_array = np.array(args_array)
            bad_val_inds = np.argwhere(results < 0)
            results = np.delete(results, bad_val_inds[:, 0], 0)
            args_array = np.delete(args_array, bad_val_inds[:, 0], 0)
            saveModelData(args_array, output_dir+"/clean_args_array.dat")
            saveModelData(results, output_dir+"/clean_results.dat")


def runMultithreadAcrossParams(output_dir="output"):
    """
    Run the simulation across the parameter ranges of initial radius, velocity,
    and impact angle (theta).

    Inputs:
        output_dir  - the directory to which the output file will be saved.
    """
    if __name__ == '__main__':

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            print("The directory \""+output_dir+"\" already exists.")
            resp = input("Overwrite files in \""+output_dir+"\"? [y/n]: ")
            if resp not in ("y", "Y"):
                return

        mass_count = 35
        vel_count = 2 
        the_count = 25
        #masses between 5 and 100 microns [kg]
        masses = np.linspace(3.665E-12, 2.932E-8, mass_count)
        #velocities = np.linspace(11200, 20000, vel_count)
        velocities = [12000, 18000]
        thetas = np.linspace(0*pi/180, 80*pi/180, the_count)
        #thetas = np.array([0,45,70])*pi/180 

        length = len(masses)*len(velocities)*len(thetas)

        args_array = []
        for i in range(0, len(masses)):
            for j in range(0, len(velocities)):
                for k in range(0, len(thetas)):
                    args = (masses[i], velocities[j], thetas[k])
                    args_array.append(args)

        with Pool(cpu_count()-1) as p:
            results = list(tqdm(p.imap(multithreadWrapper, args_array), 
                                total=length))

            saveModelData(masses, output_dir+"/masses.dat")
            saveModelData(velocities, output_dir+"/velocities.dat")
            saveModelData(thetas, output_dir+"/thetas.dat")
            saveModelData(args_array, output_dir+"/args_array.dat")
            saveModelData(results, output_dir+"/results.dat")



def plot_particle_parameters(input_mass, input_vel, input_theta):
    """
    Function to plot the various parameters of the simulation. This function
    was used to generate Figure 1.
    """

    res = simulate_particle_ivp(input_mass, input_vel, input_theta)
    data = res.y
    times = res.t

    alts = data[0, :]
    velocities = (data[1, :]**2 + data[2, :]**2)**0.5
    data[3, data[3, :] < 0] = 0 #remove small negative values
    data[4, data[4, :] < 0] = 0 #remove small negative values
    fe_fracs = data[3, :]/(data[3, :] + data[4, :])
    rads = get_radius_and_density(data[3, :], data[4, :], not_array=False)[0]
    temps = data[5, :]

    start_ind = -1
    end_ind = -1
    last_ind = -1
    for i in range(0, len(temps)):
        if start_ind < 0 and temps[i] > FE_MELTING_TEMP:
            start_ind = i
        if end_ind < 0 and start_ind > 0 and temps[i] < FEO_MELTING_TEMP:
            end_ind = i-1
        if temps[i] > FEO_MELTING_TEMP:
            last_ind = i

    print("Molten start: %0.1f seconds"%(times[start_ind]))
    print("Molten end: %0.1f seconds"%(times[end_ind]))


    rad, frac = get_final_radius_and_fe_area_from_sim(data)
    mass_frac = fe_fracs[-1]
    print("Final radius: %0.1f [microns]"%(rad/(1.0E-6)))
    print("Final Fe area fraction: %0.2f"%(frac))
    print("Final Fe mass fraction: %0.2f"%(mass_frac))
    ind = np.argmax(temps)
    print("Max temp: %0.0f [K]"%(temps[ind]))
    print("Altitude of max temp: %0.1f [km]"%((alts[ind]-EARTH_RAD)/1000))


    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(6, 8), sharex=True)

    ax1.plot(times, temps)
    ax1.plot(times[start_ind:end_ind], temps[start_ind:end_ind], 
             color="#ff7f0e")
    if last_ind - end_ind > 0:
        ax1.plot(times[end_ind:last_ind], temps[end_ind:last_ind], 
                 color="red")
    ax1.set_ylabel("Temp. [K]")
    ax1.text(0.025, 0.9, "A", 
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax1.transAxes)
    
    ax2.plot(times, velocities/1000)
    ax2.plot(times[start_ind:end_ind], velocities[start_ind:end_ind]/1000,
             color="#ff7f0e")
    if last_ind - end_ind > 0:
        ax2.plot(times[end_ind:last_ind], velocities[end_ind:last_ind]/1000,
                 color="red")
    ax2.set_ylabel(r"Vel. [km s$^{-1}$]")
    ax2.text(0.025, 0.9, "B", 
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax2.transAxes)

    rads = np.array(rads)*(1.0E6)
    ax3.plot(times, rads)
    ax3.plot(times[start_ind:end_ind], rads[start_ind:end_ind], 
             color="#ff7f0e")
    if last_ind - end_ind > 0:
        ax3.plot(times[end_ind:last_ind], rads[end_ind:last_ind], 
                 color="red")
    ax3.set_ylabel(r"Radius [$\mu$m]")
    ax3.text(0.025, 0.9, "C", 
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax3.transAxes)

    ax4.plot(times, fe_fracs)
    ax4.plot(times[start_ind:end_ind], fe_fracs[start_ind:end_ind],
             color="#ff7f0e")
    if last_ind - end_ind > 0:
        ax4.plot(times[end_ind:last_ind], fe_fracs[end_ind:last_ind], 
                 color="red")
    ax4.set_ylabel("Fe Frac.")
    ax4.text(0.025, 0.9, "D", 
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax4.transAxes)

    alts = (alts-EARTH_RAD)/1000
    ax5.plot(times, alts)
    ax5.plot(times[start_ind:end_ind], alts[start_ind:end_ind], 
             color="#ff7f0e")
    if last_ind - end_ind > 0:
        ax5.plot(times[end_ind:last_ind], alts[end_ind:last_ind], 
                 color="red")
    ax5.set_ylabel("Alt. [km]")
    ax5.set_xlabel("Time [s]")
    ax5.text(0.025, 0.9, "E", 
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax5.transAxes)


    plt.show()


def zStatAndPlot(directory="rand_sim"):
    """
    Calculate the z-statistic and plot fe area histogram
    """

    results = readModelDataFile(directory+"/clean_results.dat")

    fe_frac_array = []

    for i in range(len(results)):
        rad = results[i][0]
        frac = results[i][1]

        if rad > 5*1.0E-6 and 0 < frac < 1:
            fe_frac_array.append(frac)

    genge_data = Genge_2017_Fe_Fraction()

    p_value = zStatistic(fe_frac_array, genge_data)
    print("Z-test gives p-value: %0.5f"%(p_value))
    mean = np.mean(fe_frac_array)
    std = np.std(fe_frac_array)
    print("Number of samples used: %d (out of %d)"%(len(fe_frac_array), len(results)))
    print("Mean: %0.4f, Std: %0.4f"%(mean, std))
    print("Genge mean: %0.4f, Std: %0.4f"%(np.mean(genge_data), np.std(genge_data)))


    plt.hist(fe_frac_array, bins=20, normed=True, alpha=0.5, color="#1f77b4")
    plt.hist(genge_data, bins=20, normed=True, alpha=0.5, color="#ff7f0e")
    plt.errorbar([np.mean(genge_data)], [7.8], xerr=[2*np.std(genge_data)], 
                 fmt='-o', color="#ff7f0e")
    plt.errorbar([mean], [7.0], xerr=[std], fmt='-o', color="#1f77b4")
    plt.xlabel("Fe Fractional Area")
    plt.ylabel("Model Counts (blue), Genge Data (orange)")
    #plt.title("Model Mean (red), Genge Mean (green), bars=95%")
    plt.gca().set_yticks([])
    plt.show()


def plotMultithreadResultsRadiusVsTheta(directory="output"):
    """
    Plot the particle radius vs impact parameter for various velocities. The 
    displayed output is specified by param, which defaults to maximum temp.

    """

    masses = np.array(readModelDataFile(directory+"/masses.dat"))
    velocities = np.array(readModelDataFile(directory+"/velocities.dat"))
    thetas = np.array(readModelDataFile(directory+"/thetas.dat"))
    results = readModelDataFile(directory+"/results.dat")

    radii = get_radius_and_density(masses, 0, not_array=False)[0]

    #the velocities to display (well, the closest available in the dataset)
    velocities_in = np.array([12000, 18000])


    rad_theta12 = np.zeros((len(thetas), len(radii)))
    rad_theta18 = np.zeros((len(thetas), len(radii)))

    the_len = len(thetas)
    vel_len = len(velocities)

    velocity_vals = []
    for vel in velocities_in:
        index = np.abs(velocities - vel).argmin()
        velocity_vals.append(velocities[index])

    for i in range(0, len(radii)):
        for j in range(0, len(velocities)): #just 2 velocities
            for k in range(0, len(thetas)):
                if velocities[j] == velocity_vals[0]:
                    rad_theta12[k][i] = results[i*vel_len*the_len + \
                            j*the_len + k][1] 
                    
                if velocities[j] == velocity_vals[1]:
                    rad_theta18[k][i] = results[i*vel_len*the_len + \
                            j*the_len + k][1] 


#    fig, (ax0,ax1) = plt.subplots(1,2, figsize=(8,6))

    plt.xlabel("Initial Radius [micron]")
    plt.ylabel("Entry Angle [degrees]")
    plt.imshow(rad_theta18, origin="lower", cmap="cool", extent=[5, 100, 80, 0],
               interpolation="none", aspect="auto")
    plt.colorbar()
    plt.title("12 km/s")

#    levels = [0, 1]
#    CS = ax0.contour(radii/(1.0E-6), thetas*180/pi, rad_theta12, levels)

#    plt.gca().invert_yaxis()
#    plt.clabel(CS, inline=1, fontsize=10)
#    ax0.set_ylabel("Entry Angle")
#    ax0.set_title(r"%0.1f [km s$^{-1}$]"%(velocity_vals[0]/1000))
#    ax0.invert_yaxis()

#    CS1 = ax1.contour(radii/(1.0E-6), thetas*180/pi, rad_theta18, levels)
#    plt.clabel(CS1, inline=1, fontsize=10)
#    plt.xlabel("Radius [microns]")
#    plt.ylabel("Entry Angle")
#    ax1.set_title(r"%0.1f [km s$^{-1}$]"%(velocity_vals[1]/1000))
#    ax1.invert_yaxis()
#

    plt.show()



def plotRandomIronPartition(directory="rand_sim", use_all=False):
    """
    Plot the random data in the style of figure 4 from Genge et al 2017.

    Inputs:
        directory - the directory to find the data in
    """

    results = readModelDataFile(directory+"/clean_results.dat")

    particle_fractions = []

    for i in range(len(results)):
        frac = results[i][1]
        if frac < 1 or use_all:
            particle_fractions.append(frac)

    particle_fractions.sort()
    particle_fractions = particle_fractions[::-1]

    num_entries = len(particle_fractions)
    width = 1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(num_entries):
        frac = particle_fractions[i]
        #draw the rectangles representing the fraction of each Fe phase
        r0 = Rectangle((i, 0), width, frac, color="grey")
        r1 = Rectangle((i, frac), width, 1, color="lightblue")
        ax.add_patch(r0)
        ax.add_patch(r1)

    plt.xlim(0, num_entries)
    #plt.title("Fractional Area of Fe (grey) to FeO (blue)")
    plt.ylabel("Fe Fractional Area (Cross Section)")
    plt.xlabel("Model Runs")
    plt.show()


def plot_co2_data_mean(directory="co2_runs"):
    """
    Calculate the mean Fe area for varying co2 levels
    """

    num_runs = 35 
    means = np.zeros(num_runs)
    co2_percents = np.zeros(num_runs)
    std_tops = np.zeros(num_runs)
    std_bots = np.zeros(num_runs)

    not_printed = True

    for i in range(0, num_runs):
        val = i+1
        fname = "/co2_%d/results.dat"%(val)

        results = readModelDataFile(directory + fname)

        particle_fractions = []

        has_pure_ox = False

        for j in range(len(results)):
            frac = results[j][1]
            if 0 < frac < 1:
                particle_fractions.append(frac)

            if frac == 0:
                has_pure_ox = True

        means[i] = np.mean(particle_fractions)
        std = np.std(particle_fractions)
        std_tops[i] = means[i] + std
        std_bots[i] = means[i] - std
        #the CO2 percents in the model were done by mass, convert to volume
        #here. Assume Pure CO2 and N2 atmosphere
        vol_frac = 7*(val/100)/(11-4*val/100)

        print("%d wt %% gives %0.2f vol %%"%(val, vol_frac*100))
        co2_percents[i] = vol_frac*100

        if has_pure_ox and not_printed:
            print("At %0.2f%% CO2 fully oxidized exists"%(co2_percents[i]))
            not_printed = False


    tomkins_data = [0.555, 0.003] #tomkins fractional areas
    t_mean = np.mean(tomkins_data)
    t_std = np.std(tomkins_data)

    std_tops = np.clip(std_tops, 0, 1)
    std_bots = np.clip(std_bots, 0, 1)

    r0 = Rectangle((15.2, 0), 13, 1, color="lightblue", alpha=0.5, zorder=1)
    plt.gca().add_patch(r0)
    plt.plot(co2_percents, means, zorder=3)
    plt.fill_between(co2_percents, std_tops, std_bots, color="grey", alpha=0.5,
                     zorder=2)
    plt.errorbar([18.2], [t_mean], yerr=[t_std*2], fmt='-o', zorder=4)
    plt.xlim(1, floor(co2_percents[-1]))
    plt.ylim(0, 1)
    plt.xlabel(r"Atmospheric CO${_2}$ [Volume %]")
    plt.ylabel("Fe Fraction")
    plt.show()





#50 micron radius has mass 3.665E-9 kg
#Figure 1: this function runs a basic, single model run
#plot_particle_parameters(3.665E-9, 13200, 45*pi/180)

#plot_co2_data_mean(directory="co2_data")
generateRandomSampleData(output_dir="co2_data_correct_hox/co2_%0.0f"%(
                         float(sys.argv[1])*100),
                         num_samples=100)
#plotRandomIronPartition(directory="rand_sim_hires_gamma1.0", use_all=True)
#zStatAndPlot(directory="correct_hox_modern_gamma07")
#runMultithreadAcrossParams(output_dir="new_output")
#plotMultithreadResultsRadiusVsTheta(directory="new_output")

