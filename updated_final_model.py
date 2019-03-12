"""
This code models the entry of iron micrometeorites into an atmosphere. For help
with this code email info@lehmer.us.

Owen Lehmer - 1/14/19
"""

from multiprocessing import Pool, cpu_count
from math import sin, cos, pi, floor, ceil, sqrt, exp
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
M_CO2 = 0.044 #molecular weight of CO2 [kg mol-1]
L_V = 6.050E6 #latent heat of vaporization for FeO [J kg-1] from Genge
C_SP = 390 #specific heat of FeO from Stolen et al. (2015) [J K-1 kg-1]
FE_MELTING_TEMP = 1809 #temperature at which Fe melts [K]
FEO_MELTING_TEMP = 1720 #melting temp of Fe) [K]

#densities from from the paragraph below equation 10 in Genge et al. (2016)
RHO_FE = 7000 #liquid Fe density [kg m-3]
RHO_FEO = 4400 #liquid FeO density [kg m-3]

GAMMA = 0.8


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


def fe_co2_rate_constant(temp):
    """
    Calculate the rate constant for the reaction Fe + CO2 -> FeO + CO from
    Smirnov (2008). The reaction is assumed to be first order.

    Inputs:
        temp - the temperature of the micrometeorite [K]

    Returns:
        k_fe_co2 - the reaction rate constant [m3 mol-1 s-1]
    """

    k_fe_co2 = 2.9E8*exp(-15155/temp)
    return k_fe_co2

def dynamic_ode_solver(func, start_time, max_time, initial_guess, 
        param_to_monitor, max_param_delta,
        base_time_step, min_step_time, end_condition):
    """
    Solves a system of ordinary differential equations with dynamic time steps.
    The passed in function (func) should take two arguments, a time step value,
    and the current parameter values. It will have the form:
        func(time_step, ys)
    The func provided should return the derivatives of each value considered.
    This routine will start at time, start_time, and run until max_time is 
    reached, or the end condition is met. The end condition is set by the
    end_condition paramter, which should be a function like func that takes
    both the current time (NOT TIME STEP) and the system values like:
        end_condition(current_time, ys)
    and should return 0 if the integration should terminate, or 1 if not. This
    solver will attempt each time step and if the % change in the monitored 
    parameter is greater than the provided max_param_delta (as a fraction), then
    the time step will be reduced by half (repeatedly if necessary) until the
    minimum step time is reached (min_step_time). The step time will slowly
    relax back to the base time step (base_time_step) specified by the input. 
    The parameter to monitor should be an integer in the ys array. For example,
    if the initial guess has values:
        initial_guess = [position, velocity]
    and you want to make sure the position never changes by more than 1% you'd
    set param_to_monitor=0 and max_param_delta=0.01. 

    Inputs:
        func             - function that takes time step and values, returns 
                           derivatives
        start_time       - simulation start time [s]
        max_time         - maximum time to run simulation before returning an 
                           error [s]
        initial_guess    - array with initial parameter values
        param_to_monitor - index of which parameter to track
        max_param_delta  - fractional difference to tolerate in param_to_monitor. 
                           A value of 0.01 means changes must be less than 1%. 
                           A value of 0.2 would mean values must be less than 20%.
        base_time_step   - the default step size to use in integration [s]
        min_step_time    - the smallest time step to allow. If max_param_delta 
                           is exceeded at the smallest allowed time step an 
                           error status will be returned.
        end_condition    - function that takes current time and values, returns 
                           0 if the integration should end, 1 if not.

    Returns:
        times  - the array of time values [s] calculated 
        ys     - the array of parameter values at each time step
        status - the status of the solver. Values are:
                    -1 : failure because max_param_delta was exceeded at the 
                         smallest allowed time step.
                     0 : simulation ended without meeting end condition (it hit
                         max_time).
                     1 : simulation reached end condition successfully.
    """

    y_cur = np.array(initial_guess)

    times = []
    ys = []
    current_time = start_time
    cur_step_size = base_time_step
    next_relax = -1 #if >0 this is the next time to increase the time step

    end_cond_val = True #set to false if the end condition is met 

    status = 0 #status of the solver
    not_failed = True #set to false if the solver fails

    while current_time < max_time and end_cond_val and not_failed:
        if end_condition(current_time, y_cur) == 0:
            end_cond_val = False #we hit the end condition
            status = 1 #success!
        else:
            #first check if the time step should relax
            if cur_step_size < base_time_step and current_time >= next_relax:
                #the step should relax, double it
                next_relax = -1
                cur_step_size = cur_step_size*2
                if cur_step_size > base_time_step:
                    #make sure the base time step isn't exceeded
                    cur_step_size = base_time_step

            #get the current value of the param to monitor
            monitor_cur = y_cur[param_to_monitor]

            #run the function to get the new derivative values
            deltas = np.array(func(cur_step_size, y_cur))

            #calculate the new values
            y_new = y_cur + deltas*cur_step_size

            #get the new monitor parameter
            monitor_new = y_new[param_to_monitor]

            #check the percent change in the monitor
            if abs(monitor_new - monitor_cur)/monitor_new > max_param_delta:
                #the change was larger than allowed. Reduce the step size and
                #try again

                #first check if the step size is already minimal, fail if so
                if cur_step_size == min_step_time:
                    not_failed = False
                    status = -1 #time step fail
                else:
                    #not at the smallest allowed step, so halve the step size
                    cur_step_size = cur_step_size/2
                    if cur_step_size < min_step_time:
                        cur_step_size = min_step_time

                    if next_relax < 0:
                        #the time to try increasing step size isn't set, set it
                        next_relax = current_time + base_time_step
                
            else:
                #the step succeeded, add the new values and increment
                current_time += cur_step_size
                ys.append(y_new)
                times.append(current_time)
                y_cur = y_new

    return times, ys, status



def simple_ode_integrate(func, time_range, initial_guess, step_size, 
        end_condition, max_delta=50):
    """
    Perform a simple Newtonian integration of the passed in func. This function
    will follow the conventions of solve_ivp.

    Inputs:
        func - function that returns derivatives of each value. func should have
               form func(t,y).
        time_range - the time range to consider [start, end]
        initial_guess - initial y parameters for func
        step_size - the time step to use [s]
        end_condition - function that returns true if the program should end
        max_delta - the maximum tolerable change in a single timestep for temp

    Returns
        times - the time values calculated at
        ys - the simulation values at each time
        status - the result of the run:
            1 = success
            0 = failed
    """
    max_num_steps = (time_range[1]-time_range[0])/step_size

    y_cur = np.array(initial_guess)

    times = []
    ys = []
    current_time = 0
    status = 1

    index = 0
    while index < max_num_steps:
        index += 1
        if end_condition(step_size, ys) == 0:
            index = max_num_steps
        else:
            temp_old = y_cur[5]
            deltas = np.array(func(step_size, y_cur))
            y_cur = y_cur + deltas*step_size
            temp_new = y_cur[5]
            if abs(temp_old - temp_new) > max_delta:
                status = 0
                index = max_num_steps
            current_time += step_size
            ys.append(y_cur)
            times.append(current_time)


    return times, ys, status





def simulate_particle_ivp(input_mass, input_vel, input_theta, co2_percent=-1):
    """
    Top level function to simulate a micrometeorite using the solve_ivp function
    from scipy.

    Inputs:
        input_mass    - the initial mass of the micrometeorite [kg]
        input_vel     - the initial entry velocity of the micrometeorite [m s-1]
        input_theta   - initial entry angle of the micrometeorite [radians]
        co2_percent   - CO2 mass fraction. If set to -1 use O2, not CO2
        max_time_step - the maximum time step to use in the simulation [s]
        zero_break    - should the simulation stop when temp goes below 0?

    NOTE: during data generation zero_break is set to false so an error is 
    generated in a try statement and the time step is reduced.

    Returns:
        res - the output from solve_ivp()
    """


    def solidified(_, y_cur, tracker):
        """
        Stop the solver when the particle has solidified after melting. Also
        stop the calculation if the particle is smaller than our minimum mass 
        of 3.665E-12 [kg] (5 [micron] Fe radius).

        Inputs:
            _       - placeholder for time
            y_cur   - current simulation values
            tracker - object that stores whether to stop or not.

        Returns:
            0 - stop the simulation
            1 - don't stop the simulation
        """

        result = 1
        total_mass = y_cur[3] + y_cur[4]
        if tracker["solidified"] or total_mass < 3.665E-12:
            result = 0
        return result


    def sim_func(time, y_in, tracker):
        """
        The callable function passed to solve_ivp()
        
        Inputs:
            time - time at which to calculate [s]
            y_in - inputs to the function, has the form:
                TODO: put in form
            tracker - holds the previous time to find the time step

        Returns:
            dy_dt - the derivative of each y value at t
        """

        alt, vel_tan, vel_rad, mass_fe, mass_feo, temp = y_in

        time_step = time 

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
        if co2_percent != -1:
            #CO2 is the oxidant, so store CO2 density on the oxidant variable
            rho_o = rho_a*co2_percent #use rho_o to track CO2 density

        #calculate the radial and tangential velocity derivatives 
        #we've assumed a flat Earth here
        dvel_tan_dt = -0.75*rho_a*vel_tan**2/(rho_m*rad)
        dvel_rad_dt = gravity - 0.75*rho_a*vel_rad**2/(rho_m*rad)

        #calculate the altitude derivative
        dalt_dt = -vel_rad 

        #the mass derivative
        #Genge equation 13, which is in [dynes cm-2], converted to [Pa] here
        p_v = 10**(10.3-20126/temp)

        #a note about ox_enc. This variable is recording the total oxygen (in
        #kg s-1) that is being absorbed by the micrometeorite. For CO2 reactions
        #this calculation uses kinetics while reactions with O2 just follow the
        #total oxygen concentration, as described by Genge.
        ox_enc = 0
        if temp > FE_MELTING_TEMP or (mass_feo > 0 and temp > FEO_MELTING_TEMP):
            if co2_percent != -1:
                #this is oxidation via CO2, use kinetics
                k_rate = fe_co2_rate_constant(temp) #[m3 mol-1 s-1]

                #the total CO2 accumulated by the micrometeorite in this timestep
                co2_total = 0.75*vel*rho_o/rad/M_CO2 #[mol m-3 s-1]
                                               
                co2_conc = co2_total*time_step #CO2 molarity [mol m-3]
                mm_vol = 4/3*pi*rad**3
                fe_moles = mass_fe/mm_vol/M_FE #Fe molarity [mol m-3]
                rate = k_rate*fe_moles*co2_conc #reaction rate [mol m-3 s-1]

                #see if the reaction is limited by reactant availability or 
                #the speed of the reaction
                if rate < co2_total:
                    #the moles of FeO produced was less than the number of CO2
                    #moles available. So the reaction is limited by the reaction
                    #rate, not by the reactant concentration.
                    ox_enc = rate*M_O*mm_vol #total oxygen absorbed [kg s-1]
                else:
                    #the reaction would have consumed more CO2 if available, so
                    #let all the CO2 encountered be used.
                    ox_enc = co2_total*M_O*mm_vol #[kg s-1]

            else:
                #let oxygen be absorbed following Genge
                ox_enc = GAMMA*rho_o*pi*rad**2*vel

        #Genge equation 7, but the Langmuir formula has been adjusted for SI
        #this mass loss rate is in [kg s-1] of FeO
        dm_evap_fe_dt = 0 #if we need to evaporate Fe store it here
        dm_evap_dt = 4*pi*rad**2*p_v*sqrt(M_FEO/(2*pi*GAS_CONST*temp)) #FeO evap

        if dm_evap_dt*time_step > mass_feo: 
            #we're evaporating more FeO than exists, so evaporate Fe as well and
            #find what fraction of dt we evaporate FeO, then the rest of dt
            #we'll assume Fe is evaporating
            feo_evap_frac = mass_feo/dm_evap_dt/time_step #FeO evaporate frac
            fe_evap_frac = 1.0 - feo_evap_frac
            p_v_fe = 10**(11.51 - 1.963e4/temp) #Fe evap rate from Wang (1994)
            dm_evap_fe_dt = 4*pi*rad**2*p_v_fe*sqrt(M_FE/(2*pi*GAS_CONST*temp))
            dm_evap_fe_dt *= fe_evap_frac
            dm_evap_dt *= feo_evap_frac

        dmass_feo_dt = -dm_evap_dt + (M_FEO/M_O)*ox_enc
        dmass_fe_dt = -(M_FE/M_O)*ox_enc - dm_evap_fe_dt

        #combine all the evaporative loses here
        #NOTE: we've assumed the latent heat of FeO=Fe for evaporation here
        total_evap_dt = dm_evap_fe_dt + dm_evap_dt
        
        #oxidation via CO2 is endothermic so DELTA_H_OX is negative
        DELTA_H_OX = -465000 #heat of oxidation for CO2 + Fe -> CO +FeO [J kg-1]
        if co2_percent == -1:
            #oxidation via oxygen is exothermic
            DELTA_H_OX = 3716000 #heat of oxidation [J kg-1] from Genge

        dq_ox_dt = DELTA_H_OX*(M_FEO/M_O)*ox_enc #Genge equation 14

        #equation 6 of Genge (2016). This has the oxidation energy considered
        #which is described by equation 14
        dtemp_dt = 1/(rad*C_SP*rho_m)*\
                   (3*rho_a*vel**3/8 - 3*L_V*total_evap_dt/(4*pi*rad**2) - 
                    3*SIGMA*temp**4 + 3*dq_ox_dt/(4*pi*rad**2))


        #check the temperatures, stop the simulation if the temp has peaked
        #and solidified
        #first set the peak temperature if needed
        if temp > tracker["peak_temp"]:
            tracker["peak_temp"] = temp

        if temp < tracker["peak_temp"]/2 and temp < FEO_MELTING_TEMP:
            
            #one last check on the temperature. Sometimes the solver oscillates
            #for a step or two and can trigger an end, only end if the step was
            #less than a 10% change in temp. The oscillation will be caught as
            #an error and trigger a rerun with a smaller step size.
            if temp > tracker["last_temp"]*0.9:
                #setting this to True stops the simulation
                tracker["solidified"] = True
        tracker["last_temp"] = temp


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
           0, #initial FeO mass [kg], always 0 at start
           300] #initial temperature of micrometeorite [K], not important

    #the time range [s] used by solve_ivp()
    time_range = [0, 120]

    #we need delta_t and states, so track the time and melt state with this obj
    tracker = {"time": 0, "solidified": False, "peak_temp": 0, "last_temp":0} 

    end_cond = lambda t, y: solidified(t, y, tracker)
    end_cond.terminal = True


    start_time = 0
    max_time = 300
    param_to_monitor = 5 #monitor temperature
    max_param_delta = 0.001 #allow 0.1% change, max
    base_time_step = 0.01
    min_step_time = 0.000001
    res = dynamic_ode_solver(lambda t, y: sim_func(t, y, tracker), start_time, 
            max_time, y_0, param_to_monitor, max_param_delta, 
            base_time_step, min_step_time, end_cond)

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

    fe_mass = data[-1, 3]
    feo_mass = data[-1, 4]

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
        if isinstance(d, (tuple, np.ndarray)):
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
        args - a tuple with the form (mass, velocity, impact angle, CO2 wt %)

    Returns:
        result - a tuple with the form (radius, fe_area)
    """

    mass, velocity, theta, CO2_fac = args
    
    result = (0, 0)

    times, data, status = simulate_particle_ivp(mass, velocity, theta, 
            co2_percent=CO2_fac)
    final_radius, fe_area = get_final_radius_and_fe_area_from_sim(np.array(data))

    result = (final_radius, fe_area)

    if status != 1:
        #the try failed, return the error value
        #NOTE: this is because the time step was too too large for the input
        #parameters. The only time this happens is for very fast, very large 
        #particles (that are very rare), so it has negligable impact.
        result = (-1, -1)


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

        #read from command line for CO2, if present
        CO2_fac = -1
        if len(sys.argv) == 2:
            CO2_fac = float(sys.argv[1])
        else:
            sys.stderr.write("The simulation is being run with O2. Proceed? [y/n] ")
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
            args = (masses[i], velocities[i], thetas[i], CO2_fac)
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
        #vel_count = 2 
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



def plot_particle_parameters(input_mass, input_vel, input_theta, CO2_fac,
        max_step=0.005):
    """
    Function to plot the various parameters of the simulation. This function
    was used to generate Figure 1.
    """

    times, data, stat = simulate_particle_ivp(input_mass, input_vel, input_theta, 
            co2_percent=CO2_fac)

    print("status: %d"%(stat))

    times = np.array(times)
    data = np.array(data)

    alts = data[:, 0]
    velocities = (data[:, 1]**2 + data[:, 2]**2)**0.5
    #data[3, data[3, :] < 0] = 0 #remove small negative values
    #data[4, data[4, :] < 0] = 0 #remove small negative values
    fe_fracs = data[:, 3]/(data[:, 3] + data[:, 4])
    rads = get_radius_and_density(data[:, 3], data[:, 4], not_array=False)[0]
    temps = data[:, 5]

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

        if rad > 5*1.0E-6 and 0 < frac < 0.97:
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

    num_runs = 25
    means = np.zeros(num_runs)
    co2_percents = np.zeros(num_runs)
    std_tops = np.zeros(num_runs)
    std_bots = np.zeros(num_runs)

    not_printed = True
    pure_ox_val = 100

    for i in range(0, num_runs):
        val = (i+1)*3
        fname = "/co2_%d/clean_results.dat"%(val)

        results = readModelDataFile(directory + fname)

        particle_fractions = []

        has_pure_ox = False


        for j in range(len(results)):
            frac = results[j][1]
            if 0 < frac < 0.97:
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
            pure_ox_val = co2_percents[i]


    tomkins_data = [0.555, 0.003] #tomkins fractional areas
    t_mean = np.mean(tomkins_data)
    t_std = np.std(tomkins_data)
    mean_ind = np.argmin(np.abs(means - t_mean))

    #TODO make sure the t_mean is between two points
    ind_dir = -1
    if t_mean < means[mean_ind]:
        #linearly interpolate to next point
        ind_dir = 1
    gap = abs(means[mean_ind] - means[mean_ind + ind_dir])
    val = abs(t_mean - means[mean_ind])
    cur_frac = val/gap
    t_co2_val = (1-cur_frac)*co2_percents[mean_ind] + \
                cur_frac*co2_percents[mean_ind + ind_dir]

    std_tops = np.clip(std_tops, 0, 1)
    std_bots = np.clip(std_bots, 0, 1)

    #set the font size of the labels
    for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        label.set_fontsize(16)
    font_size = {'size': '18'}

    r0 = Rectangle((pure_ox_val, 0), co2_percents[-1]-pure_ox_val, 1, 
            color="lightblue", alpha=0.5, zorder=1)
    plt.gca().add_patch(r0)
    plt.plot(co2_percents, means, zorder=3)
    plt.fill_between(co2_percents, std_tops, std_bots, color="grey", alpha=0.5,
                     zorder=2)
    plt.errorbar([t_co2_val], [t_mean], yerr=[t_std*2], fmt='-o', zorder=4)
    plt.xlim(ceil(co2_percents[0]), floor(co2_percents[-1]))
    plt.ylim(0, 1)
    plt.xlabel(r"Atmospheric CO${_2}$ [Volume %]", fontdict=font_size)
    plt.ylabel("Fe Fraction", fontdict=font_size)
    plt.show()


def analyzeData(input_dir):
    results = np.array(readModelDataFile(input_dir+"/clean_results.dat"))
    inputs = np.array(readModelDataFile(input_dir+"/clean_args_array.dat"))

    result_frac = np.array(results[:, 1])
    result_rad = np.array(results[:, 0])

    input_mass = np.array(inputs[:, 0])
    input_vel = np.array(inputs[:, 1])
    input_theta = np.array(inputs[:, 2])
    input_co2 = inputs[0][3]

    f = lambda x: 0.9 if x<0.9 else x
    result_frac = np.array([f(x) for x in result_frac])

    cm = plt.cm.get_cmap("Set1")
    scplt = plt.scatter(input_mass, input_vel/1000, c=result_frac, cmap=cm)
    plt.xscale("log")
    plt.xlim(1.0E-12, 1.0E-4)
    cbar = plt.colorbar(scplt, boundaries=np.linspace(0.9,1,11))
    plt.xlabel("Input Mass [kg]")
    plt.ylabel(r"Input Velocity [km s$^{-1}$]")
    plt.title(r"CO$_{2}$ set to %0.0f wt %%"%(input_co2*100))
    cbar.set_label("Fe Fractional Area")
    plt.show()







#theta = impactAngleDistribution().sample(size=1)[0]
#velocity = initialVelocityDistribution().sample(size=1)[0]
#velocity = velocity*1000 #convert from [km s-1] to [m s-1]
#mass = initialMassDistribution().sample(size=1)[0]
#mass = mass/1000 #convert from [g] to [kg]
#theta = 45*pi/180
#velocity = 28000
#mass = 4E-5
#print("Inputs: theta=%0.1f [deg], vel=%0.1f [km s-1], mass=%2.2e [kg]"%(theta*180/pi, velocity/1000, mass))
#plot_particle_parameters(mass, velocity, theta, CO2_fac=0.3, max_step=0.0001)
        


#50 micron radius has mass 3.665E-9 kg
#Figure 1: this function runs a basic, single model run
#plot_particle_parameters(3.665E-9, 11200, 45*pi/180, CO2_fac=0.5)

#Figure - main results!
#plot_co2_data_mean(directory="co2_data_updated")

#main function to generate data, read from command line
generateRandomSampleData(output_dir="co2_data/co2_%0.0f"%(
                         float(sys.argv[1])*100),
                         num_samples=200)
#main function for data but no command line
#generateRandomSampleData(output_dir="test_run",
#        num_samples=500)

#Figure - plot that compares to modern micrometeorite collection
#zStatAndPlot(directory="correct_hox_modernO2_gamma07")
#zStatAndPlot(directory="test_run")


#runMultithreadAcrossParams(output_dir="new_output")
#plotMultithreadResultsRadiusVsTheta(directory="new_output")
#plotRandomIronPartition(directory="test_run", use_all=True)

#analyzeData("test_run")
