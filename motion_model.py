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

from math import sin,cos,sqrt,atan,asin,pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

#Define constants here
gravity_0 = 9.8 #gravity at Earth's surface [m s-2]
earth_rad = 6.37E6 #radius of Earth [m]


def velocityUpdate(theta, v_0, rho_a, rho_m, rad, dt, altitude):
    """
    Calculates the velocity magnitude of a micrometeorite as it moves through
    the atmosphere. This is based on equation 1 of Genge et al. (2016).

    Inputs:
        theta - the angle between the velocity vector and the Earth's surface.
                An angle of 90 is parallel to the surface.
        v_0   - current particle velocity magnitude [m s-1]
        rho_a - atmospheric density [kg m-3]
        rho_m - density of micrometeorite [kg m-3]
        rad   - radius of the micrometeorite [m]
        dt    - the timestep [s]
        altitude - the altitude of the micrometeorite, for gravity [m]

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
        altitude - the altitude of the micrometeorite [m]
        velocity - the velocity of the micrometeorite [m s-1]
        theta    - the angle of the velocity vector relative to the surface
        phi      - the angle of the micrometeorite around the Earth
        dt       - the timestep [s]

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


def simulateParticle():
    """
    Top level function to simulate a micrometeorite.
    """

    rho_a = 0.0
    rho_m = 3000.0 #micrometeorite density [kg m-3]
    rad = 1.0E-6 #micrometeorite radius [m]
    dt = 10.55 #time step [s]

    max_iter = 100000

    v_0 = 18000.0 #initial velocity [m s-1]
    theta = 80*pi/180 #initial entry angle
    phi = 0 #initial position around the Earth (always starts at 0)
    altitude = 1.0E5 + earth_rad #initial altitude [m]

    altitudes = np.zeros(max_iter)
    phis = np.zeros(max_iter)
    
    for i in range(0, max_iter):
        v_0, theta = velocityUpdate(theta, v_0, rho_a, rho_m, rad, dt, altitude)
        theta, phi, altitude = positionUpdate(altitude, v_0, theta, phi, dt)

        altitudes[i] = altitude
        phis[i] = phi

    plotParticlePath(altitudes, phis)


def plotParticlePath(altitudes, phis):
    """
    Plot the path of the particle around the Earth using the calculated 
    altitudes and phis.

    Inputs:
        altitudes - the altitudes of the particle through time [m]
        phis      - the angle of the altitude vector through time
    """

    earth = Circle((0,0), earth_rad, alpha=0.4, color="black")
    plt.gca().add_patch(earth)

    x_vals, y_vals = convertToCartesian(altitudes, phis)
    plt.plot(x_vals, y_vals)

    #plt.xlim(-earth_rad*1.1, earth_rad*1.1)
    #plt.ylim(-earth_rad*1.1, earth_rad*1.1)
    plt.axes().set_aspect("equal")
    plt.show()


def convertToCartesian(magnitudes, angles):
    """
    Take an array of vector magnitudes and corresponding angle array and find
    the corresponding cartesian coordinates.

    Inputs:
        magnitudes - the vector magnitudes
        angles     - the vector angles

    Returns:
        x_vals - the x values of the vectors
        y_vals - the corresponding y values
    """

    x_vals = np.zeros(len(magnitudes))
    y_vals = np.zeros_like(x_vals)

    for i in range(0,len(x_vals)):
        x_vals[i] = magnitudes[i]*sin(angles[i])
        y_vals[i] = magnitudes[i]*cos(angles[i])

    return x_vals, y_vals



simulateParticle()





