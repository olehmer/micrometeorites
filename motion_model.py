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

def updateRadius(rho_m, r_0, dt):
    """
    Calculate the new micrometeorite radius. This is equation 2 of Genge et al.
    (2016).

    Inputs:
        rho_m - the density of the micrometeorite [kg m-3]
        r_0   - current micrometeorite radius [m]
        dt    - simulation time step [s]

    Returns:
        new_r - the updated micrometeorite radius [m]
    """

    d_mass = 0 #TODO implement the mass equation!

    new_r = 1/(4*pi*rho_m*r_0**2)*d_mass*dt 

    return new_r

def massDerivativeSilicate(temp):
    """
    Calculate the rate of change for the micrometeorite mass (dm/dt) for a 
    silicate particle. This is equation 7 of Genge et al. (2016).

    Inputs:
        temp - temperature of the micrometeorite [K]
        
    Returns:
        dm_dt - the mass rate of change [kg s-1]
    """
    c_sp = 680 #specific heat capacity of SiO2 [J kg-1 K-1]
    m_mol = 20*proton_mass #silicate mean molecular mass [kg]
    p_v = exp(9.6-26700/temp) #equation 8
    dm_dt = -4*pi*c_sp*p_v*sqrt(m_mol/temp) #equation 7
    return dm_dt



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

    rho_m = 3000.0 #micrometeorite density [kg m-3]
    rad = 1.0E-6 #micrometeorite radius [m]
    dt = 0.05 #time step [s]

    max_iter = 1000000

    v_0 = 7660.0 #initial velocity [m s-1]
    theta = 80*pi/180 #initial entry angle
    phi = 0 #initial position around the Earth (always starts at 0)
    altitude = 4.08E5 + earth_rad #initial altitude [m]

    altitudes = np.zeros(max_iter)
    phis = np.zeros(max_iter)
    end_index = -1
    
    for i in range(0, max_iter):
        rho_a = atmosphericDensity(p_sur, altitude, isothermal_temp, 
                scale_height, m_bar)
        v_0, theta = velocityUpdate(theta, v_0, rho_a, rho_m, rad, dt, altitude)
        theta, phi, altitude = positionUpdate(altitude, v_0, theta, phi, dt)

        altitudes[i] = altitude
        phis[i] = phi

        if altitude < earth_rad:
            #the particle hit the surface, no need to continue
            end_index = i 
            break

    if end_index == -1:
        end_index = max_iter


    x_vals, y_vals = convertToCartesian(altitudes, phis, end_index)
    plotParticlePath(x_vals, y_vals)


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





