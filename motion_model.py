###############################################################################
# 8/27/18 - Owen Lehmer
#
# This code implements the equations of Genge et al. (2016), which follows the
# model of Love and Brownlee (1991).
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


def US1976StandardAtmosphere(altitude):
    """
    Gives the total density and the oxygen density for a given altitude from
    the 1976 US Standard Atmosphere. Assume no atmosphere above 190km, which 
    follows Love and Brownlee (1991). This function is not to be called for
    altitudes below 70 km (use the other function, which assumes hydrostatic
    equilibrium).

    Inputs:
        altitude - the micrometeorite altitude above the Earth's center [m]
    
    Returns:
        rho_a - total atmospheric density [kg m-3]
        rho_o - total oxygen density (all assumed atomic O) [kg m-3]
    """

    alt = altitude - earth_rad 
    rho_a = 0
    rho_o = 0

    alt_data = [ #altitude points [m]
        70000,
        75000,
        80000,
        85000,
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
        8.82E-05,
        3.99E-05,
        1.85E-05,
        8.22E-06,
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
        3.71E-05,
        1.68E-05,
        7.75E-06,
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


    if alt < 190000:
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

def updateRadius(rad, rho_m, dm_dt, dt):
    """
    Calculate the new micrometeorite radius. This is equation 2 of Genge et al.
    (2016).

    Inputs:
        rad   - current micrometeorite radius [m]
        rho_m - the density of the micrometeorite [kg m-3]
        dm_dt - the mass derivative with respect to time [kg s-1]
        dt    - simulation time step [s]

    Returns:
        new_r - the updated micrometeorite radius [m]
    """

    dr_dt = 1/(4*pi*rho_m*rad**2)*dm_dt
    new_r = rad + dr_dt*dt

    return new_r

def massDerivativeSilicate(rad, temp,c_sp, m_mol):
    """
    Calculate the rate of change for the micrometeorite mass (dm/dt) for a 
    silicate particle. This is equation 7 of Genge et al. (2016). However, the
    actual equation used is the m_dot equation on page labelled 33 from Love
    and Brownlee (1991). The constants given are for CGS units, so rather than
    convert their constants/formula to MKS units I'll just use the CGS formula
    and convert the result.

    Inputs:
        rad   - micrometeorite radius [m]
        temp  - temperature of the micrometeorite [K]
        c_sp  - specific heat capacity of SiO2 [J kg-1 K-1]
        m_mol - silicate mean molecular mass [kg]
        
    Returns:
        dm_dt - the mass rate of change [kg s-1]
    """
    p_v = vaporPressure(temp)
    dm_dt = -4*pi*rad**2*c_sp*p_v*sqrt(m_mol/temp) #equation 7 of Genge (2016)

    return dm_dt

def vaporPressure(temp):
    """
    Return the vapor pressure of the micrometeorite using the Langmuir formula.
    The equation given in both Genge et al (2016) and Love and Brownlee (1991)
    returns the pressure in [dynes cm-2], Genge doesn't seem to correct for the
    unit mismatch however... This function will use the correct units and
    convert the result to Pa, for use with the rest of the code in SI units.
    This is equation 8 in Genge et al. (2016).

    Inputs:
        temp - the temperature of the micrometeorite [K]

    Returns:
        p_v - vapor pressure [Pa]
    """

    const_A = 9.6
    const_B = 26700
    p_v = 10**(const_A-const_B/temp)

    p_v = p_v/10 #convert from [dynes cm-2] to [Pa]

    return p_v

def updateTemperature(rad, c_sp, rho_m, rho_a, v_0, L_v, temp, m_mol, dt):
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
        m_mol   - mean molecular weight of micrometeorite [kg]

    Returns:
        new_temp - new temperature of the micrometeorite [K]
    """
    eps = 1.0 #emissivity assumed unity
    p_v = vaporPressure(temp)
    dT_dt = 1/(rad*c_sp*rho_m)*(3/8*rho_a*v_0**3-3*c_sp*L_v*p_v*
            sqrt(m_mol/temp)-sigma*eps*temp**4)


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

    #atmospheric constants
    m_bar = 29*proton_mass #mean molecular weight of the atmosphere [kg m-3]
    scale_height = 8400 #atmospheric scale height [m]
    isothermal_temp = 288 #temperature of the atmosphere [K]
    p_sur = 1.0E5 #surface pressure [Pa]

    temp = isothermal_temp #assumed temp of micrometeorite at start
    c_sp = 680 #specific heat capacity of SiO2 [J kg-1 K-1]
    m_mol = 0.00086 #mean molecular mass [kg], converted from 45 [g mol-1]
    #NOTE: these are the values Genge used, even though the units are wrong...
    #c_sp = 4.377E-5 #from Love and Brownlee (1991), pg 33
    #m_mol = 45 #[g mol-1], from the table on pg 33 of Love and Brownlee (1991)



    rho_m = 3000.0 #micrometeorite density [kg m-3]
    rad = 50*1.0E-6 #micrometeorite radius [m]
    dt = 0.05 #time step [s]
    L_v = 6.050E6 #latent heat of vaporization [J kg-1]

    max_iter = 1000000

    v_0 = 8000.0 #initial velocity [m s-1]
    theta = 87*pi/180 #initial entry angle
    phi = 0 #initial position around the Earth (always starts at 0)
    altitude = 2.00E6 + earth_rad #initial altitude [m]

    #arrays so hold results in
    altitudes = np.zeros(max_iter)
    phis = np.zeros(max_iter)
    temps = np.zeros(max_iter)
    rads = np.zeros(max_iter)
    velocities = np.zeros(max_iter)
    times = np.zeros(max_iter)

    end_index = -1
       
    for i in range(0, max_iter):
        if altitude - earth_rad >= 70000:
            rho_a, rho_o = US1976StandardAtmosphere(altitude)
        else:
            rho_a = atmosphericDensity(p_sur, altitude, isothermal_temp, 
                    scale_height, m_bar)
            rho_o = rho_a*0.21 #just use 21% oxygen at this point


        v_0, theta = velocityUpdate(theta, v_0, rho_a, rho_m, rad, dt, altitude)
        theta, phi, altitude = positionUpdate(altitude, v_0, theta, phi, dt)
        temp = updateTemperature(rad, c_sp, rho_m, rho_a, v_0, L_v, temp, 
                m_mol, dt)
        dm_dt = massDerivativeSilicate(rad, temp, c_sp, m_mol)
        rad = updateRadius(rad, rho_m, dm_dt, dt)


        altitudes[i] = altitude
        phis[i] = phi
        temps[i] = temp
        rads[i] = rad
        velocities[i] = v_0
        times[i] = i*dt

        if altitude < earth_rad:
            #the particle hit the surface, no need to continue
            end_index = i 
            break

#        if i*dt < 150 and (i*dt)%1==0:
#            print("time: %3d, radius=%2.3e, dm_dt=%2.3e"%(i*dt, rad,dm_dt))

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





