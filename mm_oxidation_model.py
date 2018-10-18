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
import sys

#Define constants here
gravity_0 = 9.8 #gravity at Earth's surface [m s-2]
earth_rad = 6.37E6 #radius of Earth [m]
kb = 1.381E-23 #Boltzmann constant [J K-1]
proton_mass = 1.67E-27 #mass of a proton [kg]
sigma = 5.67E-8 #Stefan-Boltzmann constant [W m-2 K-4]
gas_const = 8.314 #ideal gas constant [J mol-1 K-1]


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


def atmosphericDensity(p_sur, altitude, temp, scale_height, m_bar, beta=1.0):
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
    pressure = p_sur*exp(-height/scale_height*beta)
    rho_a = m_bar*pressure/(kb*temp)

    
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




def simulateParticle(radius, velocity, theta):
    """
    Top level function to simulate a micrometeorite.

    Inputs:
        radius   - the radius of the micrometeorite [m]
        velocity - the initial entry velocity of the micrometeorite [m s-1]
        theta    - initial entry angle of the micrometeorite [radians]

    Returns:
        radius    - the final micrometeorite radius [m]
        total_Fe  - total mass of Fe remaining [kg]
        total_FeO - total mass of FeO remaining [kg]
        max_temp  - maximum temperature of micrometeorite [K]
        alt_max   - altitude of max pressure [m]
    """

    #atmospheric constants, taken from David's book
    m_bar = 29*proton_mass #mean molecular weight of the atmosphere [kg m-3]
    scale_height = 8400 #atmospheric scale height [m]
    isothermal_temp = 288 #temperature of the atmosphere [K]
    p_sur = 1.0E5 #surface pressure [Pa]

    temp = isothermal_temp #assumed temp of micrometeorite at start

    rho_m = 7000.0 #micrometeorite density, starts as pure Fe [kg m-3]


    total_Fe = 4/3*pi*radius**3*rho_m #mass of Fe
    total_FeO = 0 #mass of FeO
    Fe_metling_temp = 1809 #temperature at which Fe melts [K]
    FeO_melting_temp = 1720 #melting temp of Fe) [K]



    phi = 0 #initial position around the Earth (always starts at 0)
    altitude = 1.90E5 + earth_rad #initial altitude [m]
    M_Fe = 4/3*pi*radius**3*rho_m #mass of Fe
    M_FeO = 0 #mass of FeO

    #this specific heat was taken from figure 1 of Stolen et al (2015)
    #c_sp = 390 #specific heat of FeO from Stolen et al. (2015) [J K-1 kg-1]
    #figure 2 in the same paper shows a c_sp of 696 [J K-1 kg-1], so try both?
    #c_sp = 4.377E-5
    c_sp = 696

    #latent heat of vaporization. This value is for silicates and taken from
    #love and Brownlee (1991) by Genge. Genge doesn't say that he uses a 
    #different L_v for FeO... But Fe is only slightly different (6.265E6) so 
    #it's probably ok.
    L_v = 6.050E6 #[J kg-1]

    m_FeO = 0.0718 #molecular weight of FeO [kg mol-1]
    m_Fe = 0.0558 #molecular weight of Fe [kg mol-1]
    m_O = 0.016 #molecular weight of O [kg mol-1]

    max_iter = 3000
    dt = 0.01 #time step [s]
    end_index = -1

    max_temp = 0

    #storage arrays
    temps = np.zeros(max_iter)
    velocities = np.zeros(max_iter)
    radii = np.zeros(max_iter)
    altitudes = np.zeros(max_iter)
    times = np.zeros(max_iter)


       
    for i in range(0, max_iter):
        if altitude - earth_rad >= 70000:
            rho_a, rho_o = US1976StandardAtmosphere(altitude)
        else:
            rho_a = atmosphericDensity(p_sur, altitude, isothermal_temp, 
                    scale_height, m_bar)
            rho_o = rho_a*0.21 #just use 21% oxygen at this point
        velocity, theta = velocityUpdate(theta, velocity, rho_a, rho_m, radius, 
                dt, altitude)
        theta, phi, altitude = positionUpdate(altitude, velocity, theta, phi, dt)

        #Genge equation 13, which is in [Pa], convert to [dynes cm-2]
        #since we'll use this with the molecular weight in [g mol-1]
        p_v = 10**(11.3-2.0126E4/temp)/10

        #Genge (2016) equation 7, use cgs units then convert
        #so the result will be in [g s-1], but convert to [kg s-1]
        #dM_evap_dt = 4*pi*(radius*100)**2*c_sp*p_v*sqrt(m_FeO/temp)/1000
        dM_evap_dt = 4*pi*radius**2*c_sp*p_v*sqrt(m_FeO/(2*pi*gas_const*temp))

        #to read more about the Langmuir formula see this website:
        #http://www.atsunday.com/2013/07/water-evaporation-rate-per-surface-area.html?m=1

        dM_Fe_dt = 0
        dM_FeO_dt = 0

        #make sure there's some Fe before trying to oxidize it
        if temp > FeO_melting_temp:
            dM_FeO_dt = -dM_evap_dt 
        if total_Fe > 0 and temp > Fe_metling_temp:
            #equation 11, Fe lost to oxidation [kg s-1]
            dM_Fe_dt = -m_Fe/m_O*rho_o*pi*radius**2*velocity

            #equation 12, FeO growth [kg s-1]
            dM_FeO_dt = m_FeO/m_O*rho_o*pi*radius**2*velocity - dM_evap_dt

            #check if there's any Fe left
            if total_Fe - dM_Fe_dt*dt < 0:
                dM_Fe_dt = -total_Fe/dt
                dM_FeO_dt = dM_Fe_dt*m_FeO/m_Fe - dM_evap_dt

        total_Fe += dM_Fe_dt*dt
        total_FeO += dM_FeO_dt*dt

        #genge equation 4
        dq_ox_dt = 3716*dM_FeO_dt

        #equation 6 of Genge (2016). This has the oxidation energy considered
        #which is described by equation 14
        dT_dt = 1/(radius*c_sp*rho_m)*(3*rho_a*velocity**3/8 - 
                3*L_v*dM_evap_dt/(4*pi*radius**2) - 3*sigma*temp**4 - 
                3*dq_ox_dt/(4*pi*radius**2))
        temp += dT_dt*dt

        if total_FeO + total_Fe > 0:
            radius, rho_m = updateRadiusAndDensity(total_Fe, total_FeO)
        else:
            radius = 0
            rho_m = 0

        if temp > max_temp:
            max_temp = temp

        try:
            print("%3d: Fe: %3.0f%%, temp: %5.0f, radius: %0.1f [microns]"%(i,
                total_Fe/(total_Fe+total_FeO)*100,temp,radius/(1.0E-6)))

            if total_FeO < 0:
                print("     FeO under 0! %2.2e"%(total_FeO))
        except:
            print(total_FeO)
            print(total_Fe)
            print(radius)
            print(temp)

            break

        temps[i]=temp
        velocities[i] = velocity
        radii[i] = radius
        altitudes[i] = altitude
        times[i] = dt*i

        #check if the particle has started cooling significantly
        if temp < max_temp/2 or radius == 0:
            end_index = i
            print("Early end!")
            break

    print("\n\nFinal radius: %0.1f [microns]\nMax temperature: %0.0f[K]\nFe mass fraction: %0.2f"%(radius*1.0E6, max_temp, total_Fe/(total_Fe+total_FeO)))

    plotParticleParameters(temps[0:end_index+1], velocities[0:end_index+1], 
            radii[0:end_index+1], altitudes[0:end_index+1], times[0:end_index+1])



def compareStandardAndHydrostaticAtmospheres():

    #atmospheric constants
    m_bar = 29*proton_mass #mean molecular weight of the atmosphere [kg m-3]
    scale_height = 8400 #atmospheric scale height [m]
    isothermal_temp = 288 #temperature of the atmosphere [K]
    p_sur = 1.0E5 #surface pressure [Pa]

    altitudes = np.linspace(earth_rad+7.0E4,earth_rad+1.9E5, 20)

    stnd_rho = np.zeros(len(altitudes))
    stnd_ox = np.zeros_like(stnd_rho)

    hydro_rho0 = np.zeros_like(stnd_rho)
    hydro_rho1 = np.zeros_like(stnd_rho)
    hydro_rho2 = np.zeros_like(stnd_rho)

    hydro_ox = np.zeros_like(stnd_rho)

    for i in range(0,len(altitudes)):
        alt = altitudes[i]
        rho_a, rho_o = US1976StandardAtmosphere(alt)
        rho_o = rho_o/2

        stnd_rho[i] = rho_a
        stnd_ox[i] = rho_o

        rho_a0 = atmosphericDensity(p_sur, alt, isothermal_temp, 
                scale_height, m_bar)
        rho_o = rho_a0*0.21 #just use 21% oxygen at this point

        rho_a1 = atmosphericDensity(p_sur, alt, isothermal_temp, 
                scale_height, m_bar, beta=0.95)

        rho_a2 = atmosphericDensity(p_sur, alt, isothermal_temp, 
                scale_height, m_bar, beta=1.07)



        hydro_rho0[i] = rho_a0
        hydro_ox[i] = rho_o

        hydro_rho1[i] = rho_a1
        hydro_rho2[i] = rho_a2

    altitudes = (altitudes-earth_rad)/1000 #convert to altitude in km
    plt.plot(stnd_rho, altitudes,'ro', label="US Standard")
    plt.plot(stnd_ox, altitudes,'bo', label="Stnd ox")
    plt.plot(hydro_rho0, altitudes, 'r', label="Hydrostatic")
    #plt.plot(hydro_ox, altitudes,'b', label="Hydro ox")
    plt.plot(hydro_rho1, altitudes, 'b', label="Beta=0.95")
    plt.plot(hydro_rho2, altitudes, 'k', label="Beta=1.07")

    plt.gca().set_xscale("log")
    plt.xlabel(r"Atmospheric Density [kg m$^{-3}$]")
    plt.ylabel("Altitude [km]")
    plt.ylim([70,190])
    plt.legend()
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



simulateParticle(50*1.0E-6, 12000, 45*pi/180)
#compareStandardAndHydrostaticAtmospheres()





