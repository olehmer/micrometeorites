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

from math import sin,cos,sqrt,atan,asin,pi,exp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

#Define constants here
gravity_0 = 9.8 #gravity at Earth's surface [m s-2]
earth_rad = 6.37E6 #radius of Earth [m]
kb = 1.381E-23 #Boltzmann constant [J K-1]
proton_mass = 1.67E-27 #mass of a proton [kg]
sigma = 5.67E-8 #Stefan-Boltzmann constant [W m-2 K-4]



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


def massEvaporationDerivative(rad, temp, c_sp, m_mol):
    """
    Calculate the rate of change for the micrometeorite mass (dm/dt) for a 
    particle. This is equation 7 of Genge et al. (2016).

    Inputs:
        rad   - micrometeorite radius [m]
        temp  - temperature of the micrometeorite [K]
        c_sp  - specific heat capacity [J kg-1 K-1]
        m_mol - mean molecular mass [kg]
        
    Returns:
        dm_evap_dt - the mass rate of change [kg s-1]
    """

    p_v = vaporPressure(temp)
    dm_evap_dt = -4*pi*rad**2*c_sp*p_v*sqrt(m_mol/temp)

    return dm_evap_dt

def vaporPressure(temp):
    """
    Return the vapor pressure of the micrometeorite using the Langmuir formula.
    The vapor pressure is for wustite, which assumed to be the only material
    evaporating from the micrometeorite. This is equation 13 in Genge et al. 
    (2016).

    Inputs:
        temp - the temperature of the micrometeorite [K]

    Returns:
        p_v - vapor pressure [Pa]
    """

    #TODO these constants presumably give p in Pa, but Genge doesn't say...
    #test this if the model doesn't reproduce their results

    const_A = 11.3
    const_B = 2.0126E4
    p_v = exp(const_A-const_B/temp)

    #p_v = p_v/10 #convert from [dynes cm-2] to [Pa]

    return p_v


def updateMetalAndOxideMasses(dm_evap_dt, gamma, M_Fe, M_FeO, rho_o, rad, v_0, 
        dt):
    """
    Calculate the new Fe liquid mass and new FeO liquid mass. These calculations
    use equations 11 and 12 of Genge et al. (2016).

    Inputs:
        dm_evap_dt - oxide evaporation rate of FeO [kg s-1]
        gamma      - constant fraction determining O2 reaction rate (0 to 1) 
        M_Fe       - mass of Fe in micrometeorite [kg]
        M_FeO      - mass of FeO in micrometeorite [kg]
        rho_o      - density of oxygen at altitude [kg m-3]
        rad        - micrometeorite radius [m]
        v_0        - micrometeorite velocity [m s-1]
        dt         - model time step [s]

    Returns:
        new_M_Fe   - updated Fe mass in micrometeorite [kg]
        new_M_FeO  - updated FeO mass in micrometeorite [kg]
    """

    #use mass of O and mass of Fe to find M_O from M_FeO
    M_O = (15.999/55.845)*M_FeO #oxygen mass in micrometeorite [kg]

    dm_metal_dt = -gamma*M_Fe/M_O*rho_o*pi*rad**2*v_0
    dm_oxide_dt = gamma*M_FeO/M_O*rho_o*pi*rad**2*v_0 - dm_evap_dt

    new_M_Fe = dm_metal_dt*dt
    new_M_FeO = dm_oxide_dt*dt

    return new_M_Fe, new_M_FeO

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



def updateTemperature(rad, c_sp, rho_m, rho_a, v_0, L_v, temp, dmox_dt, dt):
    """
    Calculate the derivative of temperature with respect to time. This is 
    equation 6 of Genge et al. (2016).

    Inputs:
        rad     - radius of the micrometeorite [m]
        c_sp    - specific heat [J kg-1 K-1]
        rho_m   - micrometeorite density [kg m-3]
        rho_a   - atmospheric density at micrometeorite position [kg m-3]
        v_0     - micrometeorite velocity magnitude [m s-1]
        L_v     - latent heat of vaporization [J kg-1]
        temp    - current temperature of the micrometeorite [K]
        dmox_dt - mass of oxide evaporated [kg s-1]
        dt      - simulation timestep [s]

    Returns:
        new_temp - new temperature of the micrometeorite [K]
    """
    eps = 1.0 #emissivity assumed unity
    delta_Hox = 3.716E6 #oxidation heat of formation [J kg-1]
    dqox_dx = delta_Hox*dmox_dt
    dT_dt = 1/(rad*c_sp*rho_m)*(3*rho_a*v_0**3/8-3*L_v*dmox_dt/(4*pi*rad**2)-
            3*sigma*eps*temp**4-3*dqox_dx/(4*pi*rad**2))


    new_temp = temp + dT_dt*dt #update the temperature

    return new_temp



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


def atmosphericDensity(p_sur, altitude, temp, scale_height, m_bar):
    """
    Returns the atmospheric density at a given altitude assuming an isothermal
    atmosphere that is in hydrostatic equilibrium and well mixed.

    Inputs:
        p_sur        - the surface pressure of the atmosphere [Pa]
        altitude     - the distance above Earth's center [m]
        temp         - the isothermal atmospheric temperature [K]
        scale_height - the scale height of the atmosphere [m]
        m_bar        - mean molecular weight of the atmosphere [kg]

    Returns:
        rho_a - atmosphere density at altitude [kg m-3]
    """

    height = altitude - earth_rad
    if height < 0:
        #can happen on the last run
        height = 0
    pressure = p_sur*exp(-height/scale_height)
    rho_a = m_bar*pressure/(kb*temp)

    
    return rho_a




def simulateParticle():
    """
    Top level function to simulate a micrometeorite.
    """

    #atmospheric constants, taken from David's book
    m_bar = 29*proton_mass #mean molecular weight of the atmosphere [kg m-3]
    scale_height = 8400 #atmospheric scale height [m]
    isothermal_temp = 288 #temperature of the atmosphere [K]
    p_sur = 1.0E5 #surface pressure [Pa]

    temp = isothermal_temp #assumed temp of micrometeorite at start

    #FeO properties
    c_sp = 725 #specific heat [J kg-1 K-1] from Angelo et al. (2016)
    m_mol = 0.072 #mean molecular weight [kg mol-1]
    #this is actually the latent heat for Fe and silica meteorites. Genge 
    #doesn't list the L_v used TODO does this match Genge?
    L_v = 6E6 #latent heat of vaporization [J kg-1]

    #oxygen intake constant
    gamma = 1.0



    rho_m = 7000.0 #micrometeorite density, starts as pure Fe [kg m-3]
    rad = 50*1.0E-6 #micrometeorite radius [m]
    dt = 0.05 #time step [s]

    max_iter = 1000

    v_0 = 12000.0 #initial velocity [m s-1]
    theta = 45*pi/180 #initial entry angle in radians
    phi = 0 #initial position around the Earth (always starts at 0)
    altitude = 2.08E5 + earth_rad #initial altitude [m]
    M_Fe = 4/3*pi*rad**2*rho_m #mass of Fe
    M_FeO = 0 #mass of FeO

    #arrays so hold results in
    altitudes = np.zeros(max_iter)
    phis = np.zeros(max_iter)
    temps = np.zeros(max_iter)
    rads = np.zeros(max_iter)
    velocities = np.zeros(max_iter)
    FeO_masses = np.zeros(max_iter)
    Fe_masses  = np.zeros(max_iter)
    times = np.zeros(max_iter)

    end_index = -1
       
    for i in range(0, max_iter):
        rho_a = atmosphericDensity(p_sur, altitude, isothermal_temp, 
                scale_height, m_bar)
        rho_o = rho_a*0.2 #TODO just use 20% oxygen at this point
        v_0, theta = velocityUpdate(theta, v_0, rho_a, rho_m, rad, dt, altitude)
        theta, phi, altitude = positionUpdate(altitude, v_0, theta, phi, dt)
        dm_evap_dt = massEvaporationDerivative(rad, temp, c_sp, m_mol)
        temp = updateTemperature(rad, c_sp, rho_m, rho_a, v_0, L_v, temp, 
                dm_evap_dt, dt)

        M_Fe, M_FeO = updateMetalAndOxideMasses(dm_evap_dt, gamma, M_Fe, M_FeO, 
                rho_o, rad, v_0, dt)
        rad, rho_m = updateRadiusAndDensity(M_Fe, M_FeO)


        altitudes[i] = altitude
        phis[i] = phi
        temps[i] = temp
        rads[i] = rad
        velocities[i] = v_0
        times[i] = i*dt
        Fe_masses[i] = M_Fe
        FeO_masses[i] = M_FeO

        if altitude < earth_rad:
            #the particle hit the surface, no need to continue
            end_index = i 
            break

    if end_index == -1:
        end_index = max_iter



    x_vals, y_vals = convertToCartesian(altitudes, phis, end_index)
    plotParticlePath(x_vals, y_vals)

    plotParticleParameters(temps[0:end_index+1], velocities[0:end_index+1], 
            rads[0:end_index+1], altitudes[0:end_index+1], times[0:end_index+1])


def plotParticlePath(x_vals, y_vals):
    """
    Plot the path of the particle around the Earth using the calculated 
    altitudes and phis.

    Inputs:
        altitudes - the altitudes of the particle through time [m]
        phis      - the angle of the altitude vector through time
    """

    scale = 1000 #convert to km
    earth = Circle((0,0), earth_rad/scale, alpha=0.4, color="black")
    plt.gca().add_patch(earth)

    plt.plot(x_vals/scale, y_vals/scale)

    #plt.xlim(-earth_rad*1.1, earth_rad*1.1)
    #plt.ylim(-earth_rad*1.1, earth_rad*1.1)
    plt.xlabel("Distance [km]")
    plt.ylabel("Distance [km]")
    plt.axes().set_aspect("equal")
    plt.show()


def plotParticleParameters(temps, velocities, rads, altitudes, times):
    """
    Function to plot the various parameters of the simulation.

    Inputs:
        temps      - temperatures [K]
        velocities - micrometeorite velocities [m s-1]
        rads       - micrometeorite radii [m]
        altitudes  - micrometeorite altitude above Earth's center [m]
        times      - times [s]
    """

    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, sharex=True)
    fig.set_size_inches(11,9)

    ax1.plot(times,temps)
    ax1.set_ylabel("Temp. [K]")
    
    ax2.plot(times,velocities/1000)
    ax2.set_ylabel(r"Vel. [km s$^{-1}$]")

    ax3.plot(times,rads*(1.0E6))
    ax3.set_ylabel(r"Radius [$\mu$m]")

    ax4.plot(times,(altitudes-earth_rad)/1000)
    ax4.set_ylabel("Alt. [km]")
    ax4.set_xlabel("Time [s]")

    plt.show()


def convertToCartesian(magnitudes, angles, end_index=-1):
    """
    Take an array of vector magnitudes and corresponding angle array and find
    the corresponding cartesian coordinates.

    Inputs:
        magnitudes - the vector magnitudes
        angles     - the vector angles
        end_index  - the last index to use in the input arrays

    Returns:
        x_vals - the x values of the vectors
        y_vals - the corresponding y values
    """

    length = len(magnitudes)
    if end_index != -1:
        length = end_index

    x_vals = np.zeros(length)
    y_vals = np.zeros_like(x_vals)

    
    for i in range(0,length):
        x_vals[i] = magnitudes[i]*sin(angles[i])
        y_vals[i] = magnitudes[i]*cos(angles[i])

    return x_vals, y_vals



simulateParticle()





