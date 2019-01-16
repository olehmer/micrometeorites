###############################################################################
# 8/27/18 - Owen Lehmer
#
# This code implements the equations of Genge et al. (2016), which follows the
# model of Love and Brownlee (1991). This file will calculate the properties of
# an iron micrometeorite entering Earth's atmosphere.
#
# To run the model:
#
# Contact info@lehmer.us with questions or comments on this code.
###############################################################################

from math import sin, cos, sqrt, atan, asin, pi, exp, floor
from scipy.optimize import curve_fit
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy import stats, integrate
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
import matplotlib.pyplot as plt
import os

#Define constants here
gravity_0 = 9.8 #gravity at Earth's surface [m s-2]
earth_rad = 6.37E6 #radius of Earth [m]
kb = 1.381E-23 #Boltzmann constant [J K-1]
proton_mass = 1.67E-27 #mass of a proton [kg]
sigma = 5.67E-8 #Stefan-Boltzmann constant [W m-2 K-4]
gas_const = 8.314 #ideal gas constant [J mol-1 K-1]
fudge = 2


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
        if x>0 and x<pi/2:
            #between 0 and 90 degress
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
        super().__init__(a=11.2, b=25)


    def _pdf(self, x):
        prob = 0
        if x>11.2:
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
        if x>3.665E-9 and x<0.02932:
            prob = ((2.2E3*x**0.306+15)**-4.38 + 1.3E-9*(x + 10**11*x**2 + 
                    10**27*x**4)**-0.36)/4.50936E-13
            prob = float(prob)
        return prob

    def sample(self, size=1, random_state=None):
        return self.rvs(size=size, random_state=random_state)


def genge2017ModernMicrometeoritesComp():
    """
    Return arrays with the atomic Fe and atomic O in Wustite micrometeorites.
    The values here are taken from the wustite entries in Table 2 or Genge et 
    al. (2017).

    NOTE: The percents here do not add up to 100% in all cases. The model does
    not consider things like Ni and Co and Cr so we'll only compare the ratio
    of Fe to O.

    Returns:
        Fe_percents - the atomic Fe percent
        O_percents  - the atomic O percent
    """

    Fe_percents = [73.68,
                   74.83,
                   66.56,
                   74.56,
                   74.56,
                   69.60,
                   70.94,
                   75.20,
                   75.20,
                   75.08,
                   75.44,
                   76.19,
                   74.74,
                   65.74,
                   63.52,
                   70.11,
                   74.52,
                   68.86,
                   63.53,
                   76.43,
                   69.19,
                   73.69,
                   74.39,
                   65.89]

    O_percents = [22.80,
                  22.21,
                  21.39,
                  21.68,
                  21.68,
                  21.56,
                  21.57,
                  21.65,
                  22.05,
                  21.68,
                  21.89,
                  22.04,
                  21.73,
                  21.18,
                  21.34,
                  22.26,
                  22.10,
                  21.26,
                  21.84,
                  22.01,
                  21.62,
                  21.93,
                  21.70,
                  22.10]

#    for i in range(len(O_percents)):
#        total = O_percents[i] + Fe_percents[i]
#        O_percents[i] = O_percents[i]/total
#        Fe_percents[i] = Fe_percents[i]/total

    O_percents = np.array(O_percents)
    Fe_percents = np.array(Fe_percents)

    O_mean = np.mean(O_percents)
    O_std = np.std(O_percents)

    Fe_mean = np.mean(Fe_percents)
    Fe_std = np.std(Fe_percents)

    print("O %% mean: %2.2f, std: %2.2f"%(O_mean, O_std))
    print("Fe %% mean: %2.2f, std: %2.2f"%(Fe_mean, Fe_std))

    ax1 = plt.subplot(121)
    (num, bins, p) = ax1.hist(Fe_percents, bins=20, normed=True)
    ax1.set_title("Fe Atomic Percent")
    ax1.set_xlabel("Fe Atomic Percent")
    ax1.set_ylim(0, np.max(num)*1.2)

    ax2 = plt.subplot(122)
    (num, bins, p) = ax2.hist(O_percents, bins=20, normed=True)
    ax2.set_title("O Atomic Percent")
    ax2.set_xlabel("O Atomic Percent")
    ax2.set_ylim(0, np.max(num)*1.2)

    plt.show()


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



def USStandardAtmosFit(alt, a=1.273, b=2.934, c=0.901, d=0.3766):
    """
    The best fit to the Us Standard atmosphere. The fitting was done with
    curve_fit and generated values set as the defaults.

    Inputs:
        alt     - the altitude above Earth's center [m]
        a,b,c,d - constants fit by curve_fit
    """
    scale_height = 8400 #atmospheric scale height [m]
    isothermal_temp = 288 #temperature of the atmosphere [K]
    p_sur = 1.0E5 #surface pressure [Pa]
    m_bar = 29*proton_mass #mean molecular weight of the atmosphere [kg m-3]

    #make alt an array if scalar
    if not hasattr(alt, "__len__"):
        alt = [alt]

    rho_a = np.zeros_like(alt)
    for i in range(len(rho_a)):
        height = alt[i] - earth_rad
        
        pressure =  p_sur*exp(-height/scale_height*a)*b
        
        if height/1000 > 130: #over 130 km
            pressure = pressure + c**(height**d)

        rho_a[i] = m_bar*pressure/(kb*isothermal_temp)
    return rho_a

def ReadAndProcessAtmosData():
    """
    Reads the atmospheric model data from the file atmosphere_data.txt. This
    data comes from the MSISE-90 model. It is used in the
    AtmosphericDensityAndOxygen() function.
    """

    data = np.loadtxt("atmosphere_data.txt", skiprows=32)
    alt_data = data[:,0]
    o_density = data[:,1] #number density [cm-3]
    o2_density = data[:,2] #number density [cm-3]
    rho_a_data = data[:,3] #total atmospheric density [g cm-3]

    #convert the number densities for O and O2 to total oxygen density. The
    #1.0E6 factor converts from cm-3 to m-3
    o_atom_mass = 16*proton_mass
    rho_o_data = (o_density + 2*o2_density)*o_atom_mass*(1.0E6)

    #convert the density to [kg m-3]
    rho_a_data = rho_a_data*1000

    #print them nicely for copy and paste
    array_to_print = rho_o_data
    result = "["
    for i in range(len(array_to_print)):
        result += "%2.2e, "%(array_to_print[i])
    result += "]"


    print(result)


def AtmosphericDensityAndOxygen(altitude):
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

    alt = (altitude - earth_rad)/1000 #convert to traditional altitude [km]
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
    
    if alt < 190 and alt > 0:
        #linearly interpolate between the two closest points
        idx = int(floor(alt))
        frac_low = 1 - (alt-alt_data[idx])/(alt_data[idx+1] - 
                alt_data[idx])

        rho_a = rho_a_data[idx]*frac_low + rho_a_data[idx+1]*(1-frac_low)
        rho_o = rho_o_data[idx]*frac_low + rho_o_data[idx+1]*(1-frac_low)

    return rho_a, rho_o



def US1976StandardAtmosphere(altitude):
    """
    Gives the total density and the oxygen density for a given altitude from
    the 1976 US Standard Atmosphere. Assume no atmosphere above 190km, which 
    follows Love and Brownlee (1991). 

    Inputs:
        altitude - the micrometeorite altitude above the Earth's center [m]
    
    Returns:
        rho_a - total atmospheric density [kg m-3]
        rho_o - total oxygen density (both O2 and atomic O) [kg m-3]
    """

    alt = altitude - earth_rad 
    rho_a = 0
    rho_o = 0

    alt_data = [ #altitude points [m]
        #70000, #point from hydrostatic
        40000,
        45000,
        50000,
        55000,
        60000,
        65000, 
        70000,
        75000,
        80000,
        86000,
        90000,
        95000,
        100000,
        110000,
        120000,
        130000,
        140000,
        150000,
        160000,
        170000,
        180000,
        190000]

    rho_a_data = [ #atmospheric density points [kg m-3]
       # 6.82E-05, #hydrostatic point
        3.9957E-03,
        1.9663E-03,
        1.0269E-03,
        5.6810E-04,
        3.0968E-04,
        1.6321E-04,
        8.82E-05, #70km
        3.99E-05, #75km
        1.85E-05, #80km
        6.958E-06, #86km
        3.42E-06,
        1.39E-06,
        5.60E-07,
        9.71E-08,
        2.22E-08,
        8.15E-09,
        3.83E-09,
        2.08E-09,
        1.23E-09,
        7.82E-10,
        5.19E-10,
        3.58E-10]

    rho_o_data = [ #total atmospheric oxygen [kg m-3]
        #O2 is 21% by volume, or about 23.3% by weight
        #rho_a_data[0]*0.233, #hydrostatic point
        rho_a_data[0]*0.233, #the entries up to 86km are well mixed
        rho_a_data[1]*0.233,
        rho_a_data[2]*0.233,
        rho_a_data[3]*0.233,
        rho_a_data[4]*0.233,
        rho_a_data[5]*0.233,
        rho_a_data[6]*0.233,
        rho_a_data[7]*0.233,
        rho_a_data[8]*0.233,
#        3.71E-05, #too high by a factor of 2
#        1.68E-05, #too high by a factor of 2
#        7.75E-06, #too high by a factor of 2
        1.61E-06,
        7.92E-07,
        3.21E-07,
        1.26E-07,
        2.00E-08,
        4.80E-09,
        1.96E-09,
        1.03E-09,
        6.19E-10,
        4.06E-10,
        2.83E-10,
        2.05E-10,
        1.54E-10]

    if alt < 40000: #under 80km in altitude
        #use the hydrostatic approximation in this case.
        rho_a = atmosphericDensity(altitude)
        rho_o = rho_a*0.233 #23.3% by mass for O2

    elif alt < 190000:
        #find the point and the data point above and below. Then linearly 
        #interpolate between the two to find the value.
        idx = 0
        for i in range(0,len(alt_data)-1):
            if alt > alt_data[i] and alt < alt_data[i+1]:
                idx = i
                break

        frac_low = 1 - (alt-alt_data[idx])/(alt_data[idx+1] - 
                alt_data[idx])

        rho_a = rho_a_data[idx]*frac_low + rho_a_data[idx+1]*(1-frac_low)
        rho_o = rho_o_data[idx]*frac_low + rho_o_data[idx+1]*(1-frac_low)

    return rho_a, rho_o




def velocityUpdate(theta, v_0, rho_a, rho_m, rad, dt, altitude):
    """
    Calculates the velocity magnitude of a micrometeorite as it moves through
    the atmosphere. This is based on equation 1 of Genge et al. (2016).

    Inputs:
        theta    - the angle between the velocity vector and the Earth's 
                   surface.An angle of 90 is parallel to the surface.
        v_0      - current particle velocity magnitude [m s-1]
        rho_a    - atmospheric density [kg m-3]
        rho_m    - density of micrometeorite [kg m-3]
        rad      - radius of the micrometeorite [m]
        dt       - the time step [s]
        altitude - the altitude of the micrometeorite from the Earth's center
                   (not from the surface!) [m]

    Returns:
        velocity  - the magnitude of the velocity vector [m s-2]
        new_theta - the new angle of the velocity vector 

    """

    #this scaling is from pg 11 of David's book
    gravity = gravity_0*(earth_rad/altitude)**2

    drag = 0.75*rho_a*v_0**2/(rad*rho_m)*dt
    new_velocity = v_0 - drag

    #the velocity in the vertical direction
    vel_y = cos(theta)*new_velocity + gravity*dt

    #the velocity in the horizontal direction
    vel_x = sin(theta)*new_velocity

    velocity = sqrt(vel_x**2 + vel_y**2)

    new_theta = atan(vel_x/vel_y)
    if vel_y < 0:
        #the micrometeorite is moving away from Earth, make the angle positive
        new_theta += pi

    return velocity, new_theta




def positionUpdate(altitude, velocity, theta, phi, dt):
    """
    Update the position of the micrometeorite. The position is recorded using
    the altitude and the degrees travelled around the Earth's center (phi). As
    the Earth curves under the micrometeorite the velocity vector angle (theta)
    also changes. This function returns the updated theta value after motion is
    accounted for.

    Inputs:
        altitude - the altitude of the micrometeorite from Earth's center [m]
        velocity - the velocity of the micrometeorite [m s-1]
        theta    - the angle of the velocity vector relative to the surface
        phi      - the angle of the micrometeorite around the Earth
        dt       - the time step [s]

    Returns
        new_theta - the updated angle for the velocity vector
        phi       - the updated angle around the Earth
        new_alt   - the updated altitude [m]
        
    """

    #calculate the distance travelled
    dist = velocity*dt

    #find the new altitude with law of cosines
    new_alt = sqrt(altitude**2 + dist**2 - 2.0*dist*altitude*cos(theta))

    #find the change in phi
    delta_phi = asin(dist*sin(theta)/new_alt)

    phi += delta_phi
    new_theta = theta + delta_phi
    
    return new_theta, phi, new_alt 


def atmosphericDensity(altitude):
    """
    Returns the atmospheric density at a given altitude assuming an isothermal
    atmosphere that is in hydrostatic equilibrium and well mixed.

    Inputs:
        altitude     - the distance above Earth's center [m]

    Returns:
        rho_a - atmosphere density at altitude [kg m-3]
    """

    #atmospheric constants, taken from David's book
    m_bar = 29*proton_mass #mean molecular weight of the atmosphere [kg m-3]
    scale_height = 8400 #atmospheric scale height [m]
    isothermal_temp = 288 #temperature of the atmosphere [K]
    p_sur = 1.0E5 #surface pressure [Pa]


    height = altitude - earth_rad
    if height < 0:
        #can happen on the last run
        height = 0
    pressure = p_sur*exp(-height/scale_height)
    rho_a = m_bar*pressure/(kb*isothermal_temp)
    
    return rho_a


def updateRadiusAndDensity(M_Fe, M_FeO):
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

    #densities from from the paragraph below equation 10 in Genge et al. (2016)
    rho_Fe = 7000 #liquid Fe density [kg m-3]
    rho_FeO = 4400 #liquid FeO density [kg m-3]

    volume = M_Fe/rho_Fe + M_FeO/rho_FeO
    new_rad = (3*volume/(4*pi))**(1/3)

    new_rho = (M_Fe + M_FeO)/volume

    return new_rad, new_rho


def simulateParticle(radius, velocity, theta, debug_print=False,
        dt_run = 0):
    """
    Top level function to simulate a micrometeorite.

    Inputs:
        radius   - the radius of the micrometeorite [m]
        velocity - the initial entry velocity of the micrometeorite [m s-1]
        theta    - initial entry angle of the micrometeorite [radians]
        dt_run   - do not manually use this parameter, it is a control parameter
                   used to correct for inaccurate runs

    Returns:
        radius    - the final micrometeorite radius [m]
        total_Fe  - total mass of Fe remaining [kg]
        total_FeO - total mass of FeO remaining [kg]
        max_temp  - maximum temperature of micrometeorite [K]
        stoich_O  - the resulting stoichiometry of the FeO (so number of O moles
                    to Fe moles, it will be >=1)
    """

    #store the inputs
    input_rad = radius
    input_vel = velocity
    input_the = theta
    input_debug = debug_print

    
    temp = 300 #assumed temp of micrometeorite at start [K], this doesn't
               #really matter (just can't be 0).

    rho_m = 7000.0 #micrometeorite density, starts as pure Fe [kg m-3]


    total_Fe = 4/3*pi*radius**3*rho_m #mass of Fe
    total_FeO = 0 #mass of FeO
    stoich_O = 1.0 # the number of O atoms per Fe atoms in Feo 
    Fe_melting_temp = 1809 #temperature at which Fe melts [K]
    FeO_melting_temp = 1720 #melting temp of Fe) [K]



    phi = 0 #initial position around the Earth (always starts at 0)
    altitude = 1.90E5 + earth_rad #initial altitude [m]

    #this specific heat was taken from figure 1 of Stolen et al (2015),
    #figure 2 of that paper shows c_sp as 696 though?
    c_sp = 390 #specific heat of FeO from Stolen et al. (2015) [J K-1 kg-1]
    #figure 2 in the same paper shows a c_sp of 696 [J K-1 kg-1], so try both?
    c_sp = 696 #specific heat of FeO from TODO
    c_sp_Fe = 440 #specific heat of Fe

    #latent heat of vaporization. This value is for silicates and taken from
    #love and Brownlee (1991) by Genge. Genge doesn't say that he uses a 
    #different L_v for FeO... But Fe is only slightly different (6.265E6) so 
    #it's probably ok.
    L_v = 6.050E6 #latent heat of vaporization for FeO [J kg-1] TODO: is it?
    #L_v = 6.265E6 #latent heat of vaporization for Fe [j kg-1]

    m_Fe = 0.0558 #molecular weight of Fe [kg mol-1]
    m_O = 0.016 #molecular weight of O [kg mol-1]
    m_FeO = m_Fe + m_O #molecular weight of FeO [kg mol-1]

    max_iter = 10000000
    dt = 0.01*10**(-dt_run) #time step [s]
    end_index = -1

    max_dT = 0 #max temperature change
    dt_attempts = 0 #number of times timestep has been reduced

    max_temp = 0
    alt_of_max = 0

    #storage arrays
    temps = []
    velocities =[] 
    radii = []
    altitudes =[] 
    times = []
    stoichs = [] 
    fe_fractions = [] 
       
    for i in range(0, max_iter):
        #rho_a, rho_o = US1976StandardAtmosphere(altitude)
        rho_a, rho_o = AtmosphericDensityAndOxygen(altitude)

        velocity, theta = velocityUpdate(theta, velocity, rho_a, rho_m, radius, 
                dt, altitude)
        theta, phi, altitude = positionUpdate(altitude, velocity, theta, phi, dt)

        #Genge equation 13, which is in [dynes cm-2], convert to[Pa]
        p_v_FeO = 10**(11.3-2.0126E4/temp)/10
        #p_v_FeO = exp(25.93-50390/temp) #ORL derived equation from Wang
        p_v_Fe = exp(26.5-45210/temp) #ORL derived equation from Wang


        #to read more about the Langmuir formula see this website:
        #http://www.atsunday.com/2013/07/water-evaporation-rate-per-surface-area.html?m=1
        #Genge equation 7, but the Langmuir formula has been adjusted for SI
        #this mass loss rate is in [kg s-1] of FeO
        dM_evap_dt_FeO = 4*pi*radius**2*p_v_FeO*sqrt(m_FeO/(2*pi*gas_const*temp))

        #the mass evaporation of Fe
        dM_evap_dt_Fe = 4*pi*radius**2*p_v_Fe*sqrt(m_Fe/(2*pi*gas_const*temp))
        dM_evap_dt_Fe = 0

        #the total mass lost
        dM_evap_dt = dM_evap_dt_FeO #this will be updated below


        #handle the oxidation of the Fe to FeO here
        dM_Fe_dt = 0
        dM_FeO_dt = 0 
        added_O_dt = 0 # the oxygen added once pure FeO

        #make sure there's some Fe before trying to oxidize it
        if total_Fe > 0 and temp > FeO_melting_temp:
            #equation 11, Fe lost to oxidation [kg s-1]
            dM_Fe_dt = -m_Fe/m_O*rho_o*pi*radius**2*velocity

            #equation 12, FeO growth [kg s-1]
            dM_FeO_dt = m_FeO/m_O*rho_o*pi*radius**2*velocity

            #check if there's any Fe left, remember, dM_Fe_dt is negative
            if total_Fe + dM_Fe_dt*dt < 0:
                dM_Fe_dt = -total_Fe/dt
                dM_FeO_dt = dM_Fe_dt*m_FeO/m_Fe

        elif total_Fe <= 0 and temp > FeO_melting_temp and stoich_O < 1.33: 
            #just FeO left, oxidize the liquid oxide further
            added_O_dt = rho_o*pi*radius**2*velocity
            dM_FeO_dt = added_O_dt #add the mass of the O

            #update the stoichiometry of the FeO
            Fe_moles = total_FeO/m_FeO/stoich_O
            new_O_moles = Fe_moles*stoich_O + added_O_dt*dt/m_O
            stoich_O = new_O_moles/Fe_moles


        total_Fe += dM_Fe_dt*dt #dM_Fe_dt is already negative
        total_FeO += dM_FeO_dt*dt


        #evaporate material based on the dM_evap_dt terms. Evaporate FeO first,
        #then if all FeO is lost during dt evaporate Fe to compensate
        FeO_loss = dM_evap_dt_FeO*dt
        Fe_loss = 0
        if FeO_loss > total_FeO and total_Fe > 0:
            frac = (1-total_FeO/FeO_loss)
            new_dt = frac*dt
            FeO_loss = total_FeO
            Fe_loss = dM_evap_dt_Fe*new_dt

            #set the total evaporative mass loss here
            dM_evap_dt = frac*dM_evap_dt_Fe + (1-frac)*dM_evap_dt_FeO

        total_FeO -= FeO_loss
        total_Fe -= Fe_loss
              
        #genge equation 4
        if added_O_dt == 0:
            dq_ox_dt = 3716000*dM_FeO_dt
        else:
            dq_ox_dt = 3716000*(added_O_dt*m_FeO/m_O)

        #equation 6 of Genge (2016). This has the oxidation energy considered
        #which is described by equation 14
        #NOTE we've assumed L_v is the same for Fe and FeO here
        dT_dt = 1/(radius*c_sp*rho_m)*(3*rho_a*velocity**3/8 - 
                3*L_v*dM_evap_dt/(4*pi*radius**2) - 3*sigma*temp**4 - 
                3*dq_ox_dt/(4*pi*radius**2))
        temp += dT_dt*dt

        if abs(dT_dt) > max_dT:
            max_dT = abs(dT_dt)

        if total_FeO + total_Fe > 0:
            radius, rho_m = updateRadiusAndDensity(total_Fe, total_FeO)
        else:
            radius = 0
            rho_m = 0

        if temp > max_temp:
            max_temp = float(temp) #temp was being passed by reference??
            alt_of_max = altitude - earth_rad


        if debug_print:
            temps.append(temp)
            velocities.append(velocity)
            radii.append(radius)
            altitudes.append(altitude)
            times.append(dt*i)
            stoichs.append(stoich_O)
            fe_fractions.append(total_Fe/(total_Fe + total_FeO))

        #check if the particle has started cooling significantly
        if (temp < max_temp/2 and temp < FeO_melting_temp/2) or radius == 0:
            end_index = i
            if debug_print:
                print("Early end (%d steps)"%i)
            break

    if max_dT*dt > 100 and dt_run < 4:
        """
        If the temperature changed by more than 100 K in a single step there is
        likely a problem with the time step. Typically the max dT in a single
        step should be in the range of 20-30 K. Sometimes the time step is too
        large and it causes oscillations in the simulation, which will be caught
        by this check. Decrease the time step and run again.
        """
        if debug_print:
            print("Running again. Max dT was: %0.1f"%(max_dT*dt))
        return simulateParticle(input_rad, input_vel, input_the, 
                debug_print=input_debug, dt_run=dt_run+1)

    else:
        if end_index == -1:
            print("Warning: simulation did not converge before maximum iterations reached")
            end_index += max_iter

        if debug_print:
            print("\n\n")
            print("Final radius: %0.1f [microns]"%(radius*1.0E6))
            print("Maximum temp: %0.0f [K] at %0.0f [km]"%(max_temp, 
                alt_of_max/1000))
            print("Fe mass fraction %0.2f"%(total_Fe/(total_Fe+total_FeO)))
            print("Oxygen stoichiometry: %0.2f"%(stoich_O))
            print("Max dT = %0.1f"%(max_dT*dt))
            print("Number of runs: %d"%(dt_run+1))

            plotParticleParameters(temps, velocities, radii, altitudes, times, 
                    stoichs, fe_fractions)

        return radius, total_Fe, total_FeO, max_temp, stoich_O



def compareStandardAndHydrostaticAtmospheres():

    #atmospheric constants
    m_bar = 29*proton_mass #mean molecular weight of the atmosphere [kg m-3]
    scale_height = 8400 #atmospheric scale height [m]
    isothermal_temp = 288 #temperature of the atmosphere [K]
    p_sur = 1.0E5 #surface pressure [Pa]

    altitudes = np.linspace(earth_rad+3.0E4,earth_rad+1.9E5, 20)
    altitudes2 = np.linspace(earth_rad+3.0E4,earth_rad+1.9E5, 40)

    stnd_rho = np.zeros(len(altitudes))
    stnd_ox = np.zeros_like(stnd_rho)

    hydro_rho0 = np.zeros_like(altitudes2)

    new_rho_a = np.zeros_like(altitudes2)
    new_rho_o = np.zeros_like(altitudes2)

    for i in range(0,len(altitudes)):
        alt = altitudes[i]
        rho_a, rho_o = US1976StandardAtmosphere(alt)

        stnd_rho[i] = rho_a
        stnd_ox[i] = rho_o

    for i in range(len(altitudes2)):
        alt = altitudes2[i]
        rho_a0 = atmosphericDensity(alt)
        rho_a, rho_o = AtmosphericDensityAndOxygen(alt)
        print("%3d, alt=%3.0f km: rho_a: %2.3e, rho_o: %2.3e"%(i,
            (alt-earth_rad)/1000,rho_a,rho_o))

        hydro_rho0[i] = rho_a0
        new_rho_o[i] = rho_o
        new_rho_a[i] = rho_a



    coeffs = curve_fit(USStandardAtmosFit, altitudes, stnd_rho, maxfev=10000)
    print("best param fit is:")
    print("\tA = %2.3e"%(coeffs[0][0]))
    print("\tB = %2.3e"%(coeffs[0][1]))
    print("\tC = %2.3e"%(coeffs[0][2]))
    print("\tD = %2.3e"%(coeffs[0][3]))

    fit_line = USStandardAtmosFit(altitudes2)


    altitudes = (altitudes-earth_rad)/1000 #convert to altitude in km
    altitudes2 = (altitudes2-earth_rad)/1000 #convert to altitude in km

    plt.plot(stnd_rho, altitudes,'ro', label="US Standard")
    plt.plot(stnd_ox, altitudes,'bo', label="Stnd ox")
    plt.plot(hydro_rho0, altitudes2, 'r', label="Hydrostatic")

    #plt.plot(fit_line, altitudes2, ":g", label="Best Fit")
    #plt.plot(2*fit_line*0.233, altitudes2, ":b", label="Best Fit ox")

    plt.plot(new_rho_a, altitudes2, ":g", label="New Rho A")
    plt.plot(new_rho_o, altitudes2, ":b", label="New Rho O")

    plt.gca().set_xscale("log")
    plt.xlabel(r"Atmospheric Density [kg m$^{-3}$]")
    plt.ylabel("Altitude [km]")
    plt.ylim([np.min(altitudes2),190])
    plt.legend()
    plt.show()


def plotParticleParameters(temps, velocities, rads, altitudes, times, stoichs,
        fe_fractions):
    """
    Function to plot the various parameters of the simulation.

    Inputs:
        temps      - temperatures [K]
        velocities - micrometeorite velocities [m s-1]
        rads       - micrometeorite radii [m]
        altitudes  - micrometeorite altitude above Earth's center [m]
        times      - times [s]
        stoichs    - O stoichiometry
        fe_fractions - the Fe core mass fraction
    """


    fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6,1, figsize=(6,10), sharex=True)

    ax1.plot(times,temps)
    ax1.set_ylabel("Temp. [K]")
    
    velocities = np.array(velocities)
    ax2.plot(times,velocities/1000)
    ax2.set_ylabel(r"Vel. [km s$^{-1}$]")

    rads = np.array(rads)
    ax3.plot(times,rads*(1.0E6))
    ax3.set_ylabel(r"Radius [$\mu$m]")

    ax4.plot(times,fe_fractions)
    ax4.set_ylabel("Fe Frac")

    ax5.plot(times, stoichs)
    ax5.set_ylabel("O Stoich")

    altitudes = np.array(altitudes)
    ax6.plot(times,(altitudes-earth_rad)/1000)
    ax6.set_ylabel("Alt. [km]")
    ax6.set_xlabel("Time [s]")


    plt.show()




def multithreadWrapper(args):
    """
    This function will pass the multithreaded run arguments to simulateParticle
    then return the simulation parameters.

    Input:
        args - a tuple with the form (pbar, radius, velocity, impact angle)

    Returns:
        result - a tuple with the form (radius, total_Fe, total_FeO, max_temp)
    """

    radius, velocity, theta = args
    final_radius, total_Fe, total_FeO, max_temp, stoich_O = simulateParticle(
            radius, velocity, theta)

    result = (final_radius, total_Fe, total_FeO, max_temp, stoich_O)

    return result


def printSimulationFromFiles(directory="output"):
    """
    Wrapper to print simulation results from files.
    """
    inputs = readModelDataFile(directory+"/args_array.dat")
    results = readModelDataFile(directory+"/results.dat")

    simulationPrint(inputs, results)

def simulationPrint(inputs, results):
    """
    Takes the output of the multithreaded simulation and prints it nicely.

    Inputs:
        inputs - the input array for the model run with form (radius, velocity,
                 theta)
        tuples - the results of the simulation with form (final radius, total
                 Fe, total FeO, max temperature)
    """
    for i in range(0, len(inputs)):
        radius, velocity, theta = inputs[i]
        print("\n-------------Run %d--------------"%(i))
        print("Inputs:")
        print("\tradius: %0.1f [microns]"%(radius/1.0E-6))
        print("\tvelocity: %0.1f [km s-1]"%(velocity))
        print("\timpact angle: %0.1f [degrees]"%(theta*180/pi))

        final_radius, total_Fe, total_FeO, max_temp = results[i]
        Fe_fraction = 0
        if total_FeO > 0 or total_Fe > 0:
            Fe_fraction = total_Fe/(total_Fe+total_FeO)
        print("Results:")
        print("\tfinal radius: %0.1f [microns]"%(final_radius/1.0E-6))
        print("\tFe total mass: %2.2e [kg]"%(total_Fe))
        print("\tFeO total mass: %2.2e [kg]"%(total_FeO))
        print("\tFe mass percent: %0.1f%%"%(Fe_fraction*100))
        print("\tmax temperature: %0.0f [K]"%(max_temp))


def plotMultithreadResultsRadiusVsTheta(param=3, directory="output"):
    """
    Plot the particle radius vs impact parameter for various velocities. The 
    displayed output is specified by param, which defaults to maximum temp.

    Inputs:
        param         - the chosen result to display, the options are:
                            0: final radius [microns]
                            1 and 2: remaining Fe fraction
                            3: maximum temperature [K]
    """
    #TODO implement the 5 parameters correctly!

    radii = np.array(readModelDataFile(directory+"/radii.dat"))
    velocities = np.array(readModelDataFile(directory+"/velocities.dat"))
    thetas = np.array(readModelDataFile(directory+"/thetas.dat"))
    results = readModelDataFile(directory+"/results.dat")


    #the velocities to display (well, the closest available in the dataset)
    velocities_in = np.array([12000, 14000, 18000])


    rad_theta12 = np.zeros((len(radii), len(thetas)))
    rad_theta14 = np.zeros((len(radii), len(thetas)))
    rad_theta18 = np.zeros((len(radii), len(thetas)))

    the_len = len(thetas)
    vel_len = len(velocities)

    velocity_vals = []
    for vel in velocities_in:
        index = np.abs(velocities - vel).argmin()
        velocity_vals.append(velocities[index])

    for i in range(0, len(radii)):
        for j in range(0, len(velocities)): #just 3 velocities
            for k in range(0, len(thetas)):
                if velocities[j] == velocity_vals[0]:
                    if param == 3: 
                        rad_theta12[k][i] = results[i*vel_len*the_len + 
                                j*the_len + k][param] 
                    elif param == 0:
                        rad_theta12[k][i] = results[i*vel_len*the_len + 
                                j*the_len + k][param]/1.0E-6
                    else:
                        Fe_mass = results[i*vel_len*the_len + j*the_len + 
                                k][1]
                        FeO_mass = results[i*vel_len*the_len + j*the_len + 
                                k][2]
                        rad_theta12[k][i] = Fe_mass/(Fe_mass+FeO_mass)

                if velocities[j] == velocity_vals[1]:
                    if param == 3: 
                        rad_theta14[k][i] = results[i*vel_len*the_len + 
                                j*the_len + k][param] 
                    elif param == 0:
                        rad_theta14[k][i] = results[i*vel_len*the_len + 
                                j*the_len + k][param]/1.0E-6
                    else:
                        Fe_mass = results[i*vel_len*the_len + j*the_len + 
                                k][1]
                        FeO_mass = results[i*vel_len*the_len + j*the_len + 
                                k][2]
                        rad_theta14[k][i] = Fe_mass/(Fe_mass+FeO_mass)

                if  velocities[j] == velocity_vals[2]:
                    if param == 3: 
                        rad_theta18[k][i] = results[i*vel_len*the_len + 
                                j*the_len + k][param] 
                    elif param == 0:
                        rad_theta18[k][i] = results[i*vel_len*the_len + 
                                j*the_len + k][param]/1.0E-6
                    else:
                        Fe_mass = results[i*vel_len*the_len + j*the_len + 
                                k][1]
                        FeO_mass = results[i*vel_len*the_len + j*the_len + 
                                k][2]
                        rad_theta18[k][i] = Fe_mass/(Fe_mass+FeO_mass)

    fig, (ax0,ax1,ax2) = plt.subplots(3,1, figsize=(11,8), sharex=True)
    levels = np.linspace(0, 500, 31)
    if param == 3:
        levels = [1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]
    elif param == 2 or param == 1:
        levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    CS = ax0.contour(radii/(1.0E-6), thetas*180/pi, rad_theta12, levels)
    #plt.imshow(rad_theta12, origin="lower", cmap="cool", interpolation="nearest")
    #plt.colorbar()
    #plt.gca().invert_yaxis()
    plt.clabel(CS, inline=1, fontsize=10)
    ax0.set_ylabel("Entry Angle")
    ax0.set_title(r"%0.1f [km s$^{-1}$]"%(velocity_vals[0]/1000))
    ax0.invert_yaxis()

    CS1 = ax1.contour(radii/(1.0E-6), thetas*180/pi, rad_theta14, levels)
    plt.clabel(CS1, inline=1, fontsize=10)
    ax1.set_ylabel("Entry Angle")
    ax1.set_title(r"%0.1f [km s$^{-1}$]"%(velocity_vals[1]/1000))
    ax1.invert_yaxis()

    CS2 = ax2.contour(radii/(1.0E-6), thetas*180/pi, rad_theta18, levels)
    plt.clabel(CS2, inline=1, fontsize=10)
    plt.xlabel("Radius [microns]")
    plt.ylabel("Entry Angle")
    ax2.set_title(r"%0.1f [km s$^{-1}$]"%(velocity_vals[2]/1000))
    ax2.invert_yaxis()


    plt.show()


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
        if isinstance(d, tuple):
            for item in d:
                line += "%2.10e "%item
        else:
            line += "%2.10e"%d
        line += "\n"
        file_obj.write(line)
    file_obj.close()



def runMultithreadAcrossParams(debug_print=False, output_dir="output"):
    """
    Run the simulation across the parameter ranges of initial radius, velocity,
    and impact angle (theta).

    Inputs:
        debug_print - set to true to print out model results.
        output_dir  - the directory to which the output file will be saved.
    """
    if __name__ == '__main__':

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            print("The directory \""+output_dir+"\" already exists.")
            resp = input("Overwrite files in \""+output_dir+"\"? [y/n]: ")
            if resp != "y" and resp != "Y":
                return

        rad_count = 15
        vel_count = 3 
        the_count = 15
        radii = np.linspace(5*1.0E-6, 500*1.0E-6, rad_count)
        #velocities = np.linspace(11200, 20000, vel_count)
        velocities = [12000,14000,18000]
        thetas = np.linspace(0*pi/180,80*pi/180, the_count)
        #thetas = np.array([0,45,70])*pi/180 

        length = len(radii)*len(velocities)*len(thetas)

        args_array = []
        for i in range(0, len(radii)):
            for j in range(0, len(velocities)):
                for k in range(0, len(thetas)):
                    args = (radii[i], velocities[j], thetas[k])
                    args_array.append(args)

        with Pool(cpu_count()-1) as p:
            results = list(tqdm(p.imap(multithreadWrapper, args_array), 
                total=length))
            if debug_print:
                simulationPrint(args_array, results)

            saveModelData(radii, output_dir+"/radii.dat")
            saveModelData(velocities, output_dir+"/velocities.dat")
            saveModelData(thetas, output_dir+"/thetas.dat")
            saveModelData(args_array, output_dir+"/args_array.dat")
            saveModelData(results, output_dir+"/results.dat")


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
            print("The directory \""+output_dir+"\" already exists.")
            resp = input("Overwrite files in \""+output_dir+"\"? [y/n]: ")
            if resp != "y" and resp != "Y":
                return

        thetas = impactAngleDistribution().sample(size=num_samples)
        velocities = initialVelocityDistribution().sample(size=num_samples)
        velocities = velocities*1000 #convert from [km s-1] to [m s-1]
        radii = initialMassDistribution().sample(size=num_samples)
        radii = radii/1000 #convert from [g] to [kg]
        radii = radii/7000 #convert from mass to volume [m3]
        radii = (radii*3/(4*pi))**(1/3)

        args_array = []
        for i in range(num_samples):
            args = (radii[i], velocities[i], thetas[i])
            args_array.append(args)


        with Pool(cpu_count()-1) as p:
            results = list(tqdm(p.imap(multithreadWrapper, args_array), 
                total=num_samples))

            saveModelData(args_array, output_dir+"/args_array.dat")
            saveModelData(results, output_dir+"/results.dat")


def plotInputParamsForRandomData(directory="rand_sim"):
    """
    Plot the input parameters for the randomized micrometeorite data.
    """

    inputs = readModelDataFile(directory+"/args_array.dat")
    results = readModelDataFile(directory+"/results.dat")

    final_radii = []
    core_frac = []
    Fe_atomic_percent = [] 
    O_atomic_percent = []
    stoichs = []

    m_FeO = 0.0718 #molecular weight of FeO [kg mol-1]
    m_Fe = 0.0558 #molecular weight of Fe [kg mol-1]
    m_O = 0.016 #molecular weight of O [kg mol-1]

    input_rad = []
    input_vel = []
    input_ang = []

    for i in range(len(results)):
        final_radius, total_Fe, total_FeO, max_temp, stoich_O = results[i]

        Fe_mols = total_Fe/m_Fe + total_FeO/m_FeO/stoich_O #total Fe in mols
        O_mols = total_FeO/m_FeO*stoich_O #total O in mols
        O_perc = O_mols/(Fe_mols + O_mols)

        if total_FeO <= 0 :
            input_ang.append(inputs[i][2])
            input_rad.append(inputs[i][0])
            input_vel.append(inputs[i][1])

            print("-------------------------------------------------------")
            print("Initial Parameters for Entry: %4d"%(i))
            print("Velocity: %0.1f [km s-1]"%(inputs[i][1]/1000))
            print("Radius: %0.1f [microns]"%(inputs[i][0]/1.0E-6))
            print("Angle: %0.1f [degrees]"%(inputs[i][2]*180/pi))
            print("Total FeO: %2.3e"%(total_FeO))

    #plot input radius
    ax0 = plt.subplot(131)
    (num_rad, bins_rad, p_rad) = ax0.hist(np.array(input_rad)/1.0E-6, bins=100, normed=True)
    ax0.set_title("Input Radius")
    ax0.set_ylim(0, np.max(num_rad)*1.2)

    #plot input velocity
    ax1 = plt.subplot(132)
    (num_vel, bins_vel, p_vel) = ax1.hist(np.array(input_vel)/1000, bins=100, normed=True)
    ax1.set_title("Input Velocity")
    ax1.set_ylim(0, np.max(num_vel)*1.2)

    #plot input angle
    ax2 = plt.subplot(133)
    (num_ang, bins_ang, p_ang) = ax2.hist(np.array(input_ang)*180/pi, bins=100, normed=True)
    ax2.set_title("Input Angle")
    ax2.set_ylim(0, np.max(num_ang)*1.2)

    plt.show()


def plotFractionalFeHistogram(directory="rand_sim_hires"):
    """
    Plot the fractional Fe content (non-oxidized) in a histogram.

    Inputs:
        directory - the directory to find the data in
    """

    results = readModelDataFile(directory+"/results.dat")

    fe_frac_array = []

    m_FeO = 0.0718 #molecular weight of FeO [kg mol-1]
    m_Fe = 0.0558 #molecular weight of Fe [kg mol-1]
    m_O = 0.016 #molecular weight of O [kg mol-1]

    tossed_magnetite = 0
    toss_size = 0
    pure_iron_count = 0

    has_fe = 0
    no_fe = 0


    for i in range(len(results)):
        final_radius, total_Fe, total_FeO, max_temp, stoich_O = results[i]

        Fe_mols = total_Fe/m_Fe + total_FeO/m_FeO #total Fe in mols

        #calculate the fraction of Fe in each phase
        magnetite_fraction = 0
        wustite_fraction = 0
        iron_fraction = 0
        if total_Fe == 0:
            magnetite_fraction = (stoich_O - 1.0)*3
            wustite_fraction = 1.0 - magnetite_fraction
        else:
            iron_fraction = total_Fe/m_Fe/Fe_mols
            wustite_fraction = 1.0 - iron_fraction

        if final_radius/1.0E-6 > 25 and iron_fraction < 1 and iron_fraction>0:
            #clip micrometeorites with very small radii, as they aren't 
            #in the data collection
            #clip pure Fe micrometeorites, they didn't melt

            if iron_fraction > 0:
                has_fe += 1
            else:
                no_fe += 1

            fe_frac_array.append(iron_fraction)

    genge_data = Genge_2017_Fe_Fraction()
    t_test_result = stats.ttest_ind(genge_data, fe_frac_array, equal_var=False)
    print(t_test_result)

    p_value = zStatistic(fe_frac_array, genge_data)
    print("Z-test gives p-value: %0.5f"%(p_value))

    mean = np.mean(fe_frac_array)
    std = np.std(fe_frac_array)
    print("Number of samples: %d"%(len(fe_frac_array)))
    print("Mean: %0.4f, Std: %0.4f"%(mean,std))
    print("Genge mean: %0.4f, Std: %0.4f"%(np.mean(genge_data), np.std(genge_data)))
    print("Total with Fe Core: %d (%d without)"%(has_fe, no_fe))
    plt.hist(fe_frac_array, bins=50, normed=True)
    plt.show()


def plotRandomIronPartition(directory="rand_sim"):
    """
    Plot the random data in the style of figure 4 from Genge et al 2017.

    Inputs:
        directory - the directory to find the data in
    """

    results = readModelDataFile(directory+"/results.dat")

    particle_fractions = []

    m_FeO = 0.0718 #molecular weight of FeO [kg mol-1]
    m_Fe = 0.0558 #molecular weight of Fe [kg mol-1]
    m_O = 0.016 #molecular weight of O [kg mol-1]

    tossed_magnetite = 0
    toss_size = 0
    pure_iron_count = 0

    has_fe = 0
    no_fe = 0


    for i in range(len(results)):
        final_radius, total_Fe, total_FeO, max_temp, stoich_O = results[i]

        Fe_mols = total_Fe/m_Fe + total_FeO/m_FeO #total Fe in mols

        #calculate the fraction of Fe in each phase
        magnetite_fraction = 0
        wustite_fraction = 0
        iron_fraction = 0
        if total_Fe == 0:
            magnetite_fraction = (stoich_O - 1.0)*3
            wustite_fraction = 1.0 - magnetite_fraction
        else:
            iron_fraction = total_Fe/m_Fe/Fe_mols
            wustite_fraction = 1.0 - iron_fraction

        if iron_fraction == 1:
            pure_iron_count += 1

        if stoich_O >= 1.33:
            tossed_magnetite += 1

        if final_radius/1.0E-6 > 20 and iron_fraction != 1:
            #clip micrometeorites with very small radii, as they aren't 
            #in the data collection

            if iron_fraction > 0:
                has_fe += 1
            else:
                no_fe += 1

            particle_fractions.append((iron_fraction, wustite_fraction,
                magnetite_fraction))

            

    #sort the entries by the wustite to magnetite ratio first
    particle_fractions.sort(key=lambda tup: (tup[2],tup[1]))


    print("len(partices) = %d"%(len(particle_fractions)))
    print("Tossed %d entries that were pure magnetite"%(tossed_magnetite))
    print("Pure iron mms %d"%(pure_iron_count))
    if no_fe > 0:
        print("Has iron: %d, no iron: %d, ratio: %0.2f"%(has_fe, no_fe, float(has_fe/no_fe)))

    num_entries = len(particle_fractions)
    width = 1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(num_entries):
#        print(particle_fractions[i])
#        if i == 700:
#        print(particle_fractions[i])
        iron_frac, wust_frac, mag_frac = particle_fractions[i]
        #draw the rectangles representing the fraction of each Fe phase
        if iron_frac > 0:
            r0 = Rectangle((i, 0), width, iron_frac, color="blue")
            r1 = Rectangle((i, iron_frac), width, wust_frac, color="red")
            ax.add_patch(r0)
            ax.add_patch(r1)
        else:
            r0 = Rectangle((i, 0), width, wust_frac, color="red")
            r1 = Rectangle((i, wust_frac), width, mag_frac, color="black")
            ax.add_patch(r0)
            ax.add_patch(r1)


    plt.xlim(0, num_entries)
    plt.show()



def plotRandomDataHistogram(directory="rand_sim", plot_num=1):
    """
    Plot a histogram of the randomly generated data.

    Inputs:
        directory - the directory to find the data in
    """

    results = readModelDataFile(directory+"/results.dat")

    final_radii = []
    core_frac = []
    Fe_atomic_percent = [] 
    O_atomic_percent = []
    stoichs = []

    m_FeO = 0.0718 #molecular weight of FeO [kg mol-1]
    m_Fe = 0.0558 #molecular weight of Fe [kg mol-1]
    m_O = 0.016 #molecular weight of O [kg mol-1]


    for i in range(len(results)):
        final_radius, total_Fe, total_FeO, max_temp, stoich_O = results[i]

        Fe_mols = total_Fe/m_Fe + total_FeO/m_FeO #total Fe in mols
        O_mols = total_FeO/m_FeO*stoich_O #total O in mols
        O_perc = O_mols/(Fe_mols + O_mols)


        if final_radius/1.0E-6 > 25 and total_FeO > 0 and stoich_O < 1.33:  
            #clip micrometeorites with very small radii, as they aren't 
            #in the data collection

            Fe_fraction = 0
            if total_FeO > 0 or total_Fe > 0:
                Fe_fraction = total_Fe/(total_Fe+total_FeO)
            final_radii.append(final_radius/1.0E-6) #convert from [m] to [microns]
            core_frac.append(Fe_fraction)

            Fe_atomic_percent.append(Fe_mols/(Fe_mols + O_mols)*100)
            O_atomic_percent.append(O_perc*100)

            stoichs.append(stoich_O)

    Fe_atomic_percent = np.array(Fe_atomic_percent)
    O_atomic_percent = np.array(O_atomic_percent)
    O_mean = np.mean(O_atomic_percent)
    O_std = np.std(O_atomic_percent)
    Fe_mean = np.mean(Fe_atomic_percent)
    Fe_std = np.std(Fe_atomic_percent)
    print("O %% mean: %2.2f, std: %2.2f"%(O_mean, O_std))
    print("Fe %% mean: %2.2f, std: %2.2f"%(Fe_mean, Fe_std))


#    if plot_num == 0:
    final_radii = np.array(final_radii)
    rad_mean = np.mean(final_radii)
    rad_std = np.std(final_radii)
    genge_rad = 52.5

    ax1 = plt.subplot(121)
    log_bins = np.logspace(0,3,100)
    (num, bins, p) = ax1.hist(final_radii, bins=log_bins, normed=True)
    ax1.errorbar([rad_mean], [np.max(num)*1.1], xerr=[rad_std], fmt="bo")
    ax1.errorbar([genge_rad], [np.max(num)*1.07], xerr=[0], fmt="ro")
    ax1.set_title(r"Final Radius")
    ax1.set_xlabel(r"Final Radius [$\mu$m]")
    ax1.set_ylim(0, np.max(num)*1.2)
    ax1.set_xscale("log")

#        ax2 = plt.subplot(122)
#        (num_frac, bins_frac, p_frac) = ax2.hist(core_frac, bins=100, normed=True)
#        ax2.set_title(r"Final Fe Mass Fraction")
#        ax2.set_xlabel("Fe Mass Fraction")
#        ax2.set_ylim(0, np.max(num_frac)*1.2)
#        #ax2.set_xlim(np.min(Fe_fraction), np.max(Fe_fraction))


    #plot Fe percent (comment out when plotting O)
    ax3 = plt.subplot(122)
    Fe_mean_genge = 71.77 #see genge2017ModernMicrometeoritesComp()
    Fe_std_genge = 4.09 #see genge2017ModernMicrometeoritesComp()
    (num_Fe, bins_Fe, p_Fe) = ax3.hist(Fe_atomic_percent, bins=100, normed=True)
    ax3.errorbar([Fe_mean], [np.max(num_Fe)*1.1], xerr=[Fe_std], fmt="bo")
    ax3.errorbar([Fe_mean_genge], np.max(num_Fe)*1.07, xerr=[Fe_std_genge], fmt="ro")
    ax3.set_title("Atomic Fe %")
    ax3.set_xlabel("Atomic Fe %")
    ax3.set_ylim(0, np.max(num_Fe)*1.2)

    #plot O percent (comment out when plotting Fe)
#    ax4 = plt.subplot(122)
#    (num_O, bins_O, p_O) = ax4.hist(O_atomic_percent, bins=100, normed=True)
#    O_mean_genge = 21.80
#    O_std_genge = 0.36
#    ax4.errorbar([O_mean], [np.max(num_O)*1.1], xerr=[O_std], fmt="bo")
#    ax4.errorbar([O_mean_genge], np.max(num_O)*1.07, xerr=[O_std_genge], fmt="ro")
#    ax4.set_title("Atomic O %")
#    ax4.set_xlabel("Atomic O %")
#    ax4.set_ylim(0, np.max(num_O)*1.2)



    plt.show()


def plotParameterSpace(thetas_in, directory="output", color=0):
    """
    This function will plot the model output across all parameter ranges of 
    velocity and initial radius. Multiple plots will be generated for the impact
    angle. 

    Inputs:
        thetas_in          - the theta values to use in the plot
        directory          - the directory that holds the inputs files
        color     - the parameter to show on the colorbar. The options are:
                        0: Final pure Fe of the micrometeorite [fraction]
                        1: Maximum temperature reached during entry [K]
                        2: The final stoichiometry of the oxide

    NOTE: the thetas plotted will be the closest thetas found in the data file.
    If the theta you want isn't exactly correct just rerun the data files to 
    ensure the exact value you want is included.
    """

    radii = np.array(readModelDataFile(directory+"/radii.dat"))
    velocities = np.array(readModelDataFile(directory+"/velocities.dat"))
    thetas = np.array(readModelDataFile(directory+"/thetas.dat"))
    inputs = readModelDataFile(directory+"/args_array.dat")
    results = readModelDataFile(directory+"/results.dat")

    theta_vals = []
    for theta in thetas_in:
        index = np.abs(thetas - theta).argmin()
        theta_vals.append(thetas[index])


    the_len = len(theta_vals)
    micron = 1.0E-6
    sc = None

    cm = plt.cm.get_cmap("rainbow")

    for i in range(the_len):
        ax = plt.subplot(the_len,1,i+1)
        ax.set_xlim(np.min(radii)/micron, np.max(radii)/micron)
        ax.set_ylim(np.min(velocities)/1000, np.max(velocities)/1000)
        ax.set_ylabel(r"Input Velocity [km s$^{-1}$]")
        title = r"Input Impact Angle: %0.0f$^{\degree}$"%(theta_vals[i]*180/pi)
        ax.text(0.025, 0.85, title, transform=ax.transAxes, 
                bbox=dict(facecolor="white", alpha=1.0))

        #remove the tick labels from all but the last one
        if i != the_len-1:
            ax.set_xticklabels([])


        #set the min and max z-values
        min_z = 0
        max_z = 0
        if color == 0:
            min_z = 0
            max_z = 100
        elif color == 1:
            min_z = 1000
            max_z = 3800
        elif color == 2:
            min_z = 1
            max_z = 1.3333

        for j in range(len(inputs)):
            if inputs[j][2] == theta_vals[i]:
                final_radius, total_Fe, total_FeO, max_temp, stoich_O = results[j]

                #set the color here
                z = 0
                if color == 0:
                    core_frac = total_Fe/(total_Fe+total_FeO)
                    z = core_frac*100

                elif color == 1:
                    z = max_temp
                
                elif color == 2:
                    z = stoich_O

                max_rad_size = 200*micron
                size_frac = abs(final_radius-max_rad_size)/max_rad_size
                if size_frac > 1:
                    size_frac = 1
                size = (1 - size_frac)*200 + 5

                sc = ax.scatter([inputs[j][0]/micron], [inputs[j][1]/1000],
                        c=[z], vmin=min_z, vmax=max_z, s=size, cmap=cm, 
                        edgecolor='none')

        if i==0:
            plt.scatter([-500],[-500],s=5,label=r"$\leq 5$ $[\mu m]$", c="black")
            plt.scatter([-500],[-500],s=105,label=r"$100$ $[\mu m]$", c="black")
            plt.scatter([-500],[-500],s=205,label=r"$\geq 200$ $[\mu m]$", c="black")
            plt.legend(scatterpoints=1, title="Final Radius")
        cbar = plt.colorbar(sc)
        if color == 0:
            cbar.set_label("Final Pure Fe Percent")
        elif color ==1 :
            cbar.set_label("Maximum Temperature [K]")
        elif color ==2:
            cbar.set_label("Oxide Stoichiometry")



    plt.gca().set_xlabel(r"Input Radius [$\mu$m]")
    plt.subplots_adjust(hspace=0.05)
    plt.show()




def plotParticleComparison(measured_rad, measured_core_frac, thetas_in,
        directory="output"):
    """
    This function will plot the model output across all parameter ranges of 
    velocity and initial radius. Multiple plots will be generated for the impact
    angle. The size of the points reflects the closeness to the measured radius
    (with smaller be further away) and the color bar indicates the core fraction
    of the model compared to the measured core fraction.

    Inputs:
        measured_rad       - the measured radius of a micrometeorite [m]
        measured_core_frac - the measured Fe core fraction [0-1]
        thetas_in          - the theta values to use in the plot
        directory          - the directory that holds the inputs files

    NOTE: the thetas plotted will be the closest thetas found in the data file.
    If the theta you want isn't exactly correct just rerun the data files to 
    ensure the exact value you want is included.
    """

    radii = np.array(readModelDataFile(directory+"/radii.dat"))
    velocities = np.array(readModelDataFile(directory+"/velocities.dat"))
    thetas = np.array(readModelDataFile(directory+"/thetas.dat"))
    inputs = readModelDataFile(directory+"/args_array.dat")
    results = readModelDataFile(directory+"/results.dat")

    theta_vals = []
    for theta in thetas_in:
        index = np.abs(thetas - theta).argmin()
        theta_vals.append(thetas[index])


    the_len = len(theta_vals)
    micron = 1.0E-6
    sc = None

    cm = plt.cm.get_cmap("rainbow")

    for i in range(the_len):
        ax = plt.subplot(the_len,1,i+1)
        ax.set_xlim(np.min(radii)/micron, np.max(radii)/micron)
        ax.set_ylim(np.min(velocities)/1000, np.max(velocities)/1000)
        ax.set_ylabel(r"Velocity [km s$^{-1}$]")
        title = r"Impact Angle: %0.0f$^{\degree}$"%(theta_vals[i]*180/pi)
        ax.text(0.025, 0.85, title, transform=ax.transAxes, 
                bbox=dict(facecolor="white", alpha=1.0))

        #remove the tick labels from all but the last one
        if i != the_len-1:
            ax.set_xticklabels([])


        for j in range(len(inputs)):
            if inputs[j][2] == theta_vals[i]:
                final_radius, total_Fe, total_FeO, max_temp, stoich_O = results[j]
                core_frac = total_Fe/(total_Fe+total_FeO)

                z = 0
                if measured_core_frac == 0:
                    #in case there was no observed core
                    z = core_frac*100
                else:
                    z = abs(core_frac-measured_core_frac)/measured_core_frac*100
                if z > 100:
                    z = 100

                size_frac = abs(final_radius-measured_rad)/measured_rad
                if size_frac > 1:
                    size_frac = 1
                size = (1 - size_frac)*200 + 5

                sc = ax.scatter([inputs[j][0]/micron], [inputs[j][1]/1000],
                        c=[z], vmin=0, vmax=100, s=size, cmap=cm, edgecolor='none')

        if i==0:
            plt.scatter([-500],[-500],s=5,label="100% Radius Error", c="black")
            plt.scatter([-500],[-500],s=105,label="50% Radius Error", c="black")
            plt.scatter([-500],[-500],s=205,label="0% Radius Error", c="black")
            plt.legend(scatterpoints=1)
        cbar = plt.colorbar(sc)
        cbar.set_label("Core Mass\nPercent Error")


    plt.gca().set_xlabel(r"Radius [$\mu$m]")
    plt.subplots_adjust(hspace=0.05)
    plt.show()


def testDist(dist, use_log=False):
    """
    Test the sample distribution passed in.

    Inputs:
        dist - the instance of the rv_continuous class to use
    """

    samples = dist.sample(size=10000)
    sample_min = np.min(samples)
    sample_max = np.max(samples)

    pdf_x = np.linspace(sample_min, sample_max, 500)
    pdf_y = np.zeros_like(pdf_x)

    for i in range(len(pdf_x)):
        pdf_y[i] = dist.pdf(pdf_x[i])

    if use_log:
        plt.yscale("log")
    
    (num, bins, patches) = plt.hist(samples, bins=100, normed=True)
    plt.plot(pdf_x, pdf_y, 'r')

    plt.ylim(0, np.max(num)*1.2)
    plt.xlim(sample_min, sample_max)

    plt.show()

def test():
    pts = [3.71E-05, 1.68E-05, 7.75E-06]
    alts = [70, 75, 80]

    pt_86 = 1.61E-06

    hydro_alt = 70000
    hydro_pt = atmosphericDensity(earth_rad + hydro_alt)*0.233
    print("Pressure at %0d [km] is %2.3e"%(hydro_alt/1000, hydro_pt))

    plt.plot(pts, alts, 'bo')
    plt.plot([pt_86], [86], 'ro')
    plt.plot([hydro_pt],[hydro_alt/1000], 'ro')
    plt.plot([pt_86, hydro_pt],[86, hydro_alt/1000],'r')
    plt.xscale('log')
    plt.ylim(hydro_alt/1000-5,90)
    plt.show()
   
#test()

simulateParticle(50*1.0E-6, 12000, 45*pi/180, debug_print=True)
#compareStandardAndHydrostaticAtmospheres()
#runMultithreadAcrossParams(output_dir="mod_output_2")

#generateRandomSampleData(num_samples=500, output_dir="modern_atmosphere_double")
#plotRandomDataHistogram(directory="rand_sim_hires")

#plotInputParamsForRandomData(directory="rand_sim")

#plotRandomIronPartition(directory="modern_atmosphere_double")


#plotFractionalFeHistogram(directory="modern_atmosphere_double")



#genge2017ModernMicrometeoritesComp()

#plot for Figure 1e (only one with Fe core)
#plotParticleComparison(3.2*1.0E-6, 0.95, [0,30*pi/180, 45*pi/180, 60*pi/180],
#        directory="output_1_percent_O2") 


#plot for Figure 1f, pure wustite
#plotParticleComparison(37.5*1.0E-6, 0, [0,45*pi/180, 60*pi/180], 
#        directory="mod_output") 

#plot for pure Fe
#plotParameterSpace([0, 45*pi/180, 70*pi/180], directory="mod_output",
#        color = 0) 

#plotMultithreadResultsRadiusVsTheta(param=1, directory="mod_output_2")
#printSimulationFromFiles()

#testDist(impactAngleDistribution())
#testDist(initialVelocityDistribution(), use_log=True)
#testDist(initialMassDistribution(), use_log=True)

