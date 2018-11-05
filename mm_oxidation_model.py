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

from math import sin, cos, sqrt, atan, asin, pi, exp
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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




def simulateParticle(radius, velocity, theta, debug_print=False):
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

    #this specific heat was taken from figure 1 of Stolen et al (2015),
    #figure 2 of that paper shows c_sp as 696 though?
    #c_sp = 390 #specific heat of FeO from Stolen et al. (2015) [J K-1 kg-1]
    #figure 2 in the same paper shows a c_sp of 696 [J K-1 kg-1], so try both?
    #c_sp = 4.377E-5
    c_sp = 949.26 #specific heat of FeO from TODO
    c_sp_Fe = 440 #specific heat of Fe
    c_sp_Fe3O4 = 619.4 #specific heat of Fe3O4

    #latent heat of vaporization. This value is for silicates and taken from
    #love and Brownlee (1991) by Genge. Genge doesn't say that he uses a 
    #different L_v for FeO... But Fe is only slightly different (6.265E6) so 
    #it's probably ok.
    #L_v = 6.050E6 #latent heat of vaporization for FeO [J kg-1] TODO: is it?
    L_v = 6.265E6 #latent heat of vaporization for Fe [j kg-1]

    m_FeO = 0.0718 #molecular weight of FeO [kg mol-1]
    m_Fe = 0.0558 #molecular weight of Fe [kg mol-1]
    m_O = 0.016 #molecular weight of O [kg mol-1]

    max_iter = 10000000
    dt = 0.001 #time step [s]
    if velocity > 13000:
        dt = 0.001
    if velocity > 15000:
        dt = 0.0001
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

        #Genge equation 13, which is in [dynes cm-2], convert to[Pa]
        #p_v = 10**(11.3-2.0126E4/temp)/10
        p_v_FeO = exp(25.93-50390/temp) #ORL derived equation from Wang
        p_v_Fe = exp(26.5-45210/temp) #ORL derived equation from Wang


        #to read more about the Langmuir formula see this website:
        #http://www.atsunday.com/2013/07/water-evaporation-rate-per-surface-area.html?m=1
        #Genge equation 7, but the Langmuir formula has been adjusted for SI
        #this mass loss rate is in [kg s-1] of FeO
        dM_evap_dt_FeO = 4*pi*radius**2*p_v_FeO*sqrt(m_FeO/(2*pi*gas_const*temp))

        #the mass evaporation of Fe
        dM_evap_dt_Fe = 4*pi*radius**2*p_v_Fe*sqrt(m_Fe/(2*pi*gas_const*temp))

        #the total mass lost
        dM_evap_dt = dM_evap_dt_FeO #this will be updated below


        #handle the oxidation of the Fe to FeO here
        dM_Fe_dt = 0
        dM_FeO_dt = 0 

        #make sure there's some Fe before trying to oxidize it
        if total_Fe > 0 and temp > Fe_metling_temp:
            #equation 11, Fe lost to oxidation [kg s-1]
            dM_Fe_dt = -m_Fe/m_O*rho_o*pi*radius**2*velocity

            #equation 12, FeO growth [kg s-1]
            dM_FeO_dt = m_FeO/m_O*rho_o*pi*radius**2*velocity

            #check if there's any Fe left, remember, dM_Fe_dt is negative
            if total_Fe + dM_Fe_dt*dt < 0:
                dM_Fe_dt = -total_Fe/dt
                dM_FeO_dt = dM_Fe_dt*m_FeO/m_Fe

        total_Fe += dM_Fe_dt*dt #dM_Fe_dt is already negative
        total_FeO += dM_FeO_dt*dt


        #evaporate material based on the dM_evap_dt terms. Evaporate FeO first,
        #then if all FeO is lost during dt evaporate Fe to compensate
        FeO_loss = dM_evap_dt_FeO*dt
        Fe_loss = 0
        if FeO_loss > total_FeO:
            frac = (1-total_FeO/FeO_loss)
            new_dt = frac*dt
            FeO_loss = total_FeO
            Fe_loss = dM_evap_dt_Fe*new_dt

            #set the total evaporative mass loss here
            dM_evap_dt = frac*dM_evap_dt_Fe + (1-frac)*dM_evap_dt_FeO

        total_FeO -= FeO_loss
        total_Fe -= Fe_loss


              
        #genge equation 4
        dq_ox_dt = 3716*dM_FeO_dt

        #equation 6 of Genge (2016). This has the oxidation energy considered
        #which is described by equation 14
        #NOTE we've assumed L_v is the same for Fe and FeO here
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

        if debug_print:
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


        temps[i]=temp
        velocities[i] = velocity
        radii[i] = radius
        altitudes[i] = altitude
        times[i] = dt*i

        #check if the particle has started cooling significantly
        if temp < max_temp/2 or radius == 0:
            end_index = i
            if debug_print:
                print("Early end")
            break

    if end_index == -1:
        print("Warning: simulation did not converge before maximum iterations reached")

    if debug_print:
        print("\n\nFinal radius: %0.1f [microns]\nMax temperature: %0.0f[K]\nFe mass fraction: %0.2f"%(radius*1.0E6, max_temp, total_Fe/(total_Fe+total_FeO)))

        plotParticleParameters(temps[0:end_index+1], velocities[0:end_index+1], 
                radii[0:end_index+1], altitudes[0:end_index+1], times[0:end_index+1])

    return radius, total_Fe, total_FeO, max_temp



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
    final_radius, total_Fe, total_FeO, max_temp = simulateParticle(radius,
            velocity, theta)

    result = (final_radius, total_Fe, total_FeO, max_temp)

    return result


def printSimulationFromFiles():
    """
    Wrapper to print simulation results from files.
    """
    inputs = readModelDataFile("output/args_array.dat")
    results = readModelDataFile("output/results.dat")

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


def plotMultithreadResultsRadiusVsTheta(param=3):
    """
    Plot the particle radius vs impact parameter for various velocities. The 
    displayed output is specified by param, which defaults to maximum temp.

    Inputs:
        param         - the chosen result to display, the options are:
                            0: final radius [microns]
                            1: total Fe remaining [kg]
                            2: total FeO remaining [kg]
                            3: maximum temperature [K]
                            4: Fe mass fraction
                            5: total mass [kg]
    """
    #TODO implement the 5 parameters correctly!

    radii = np.array(readModelDataFile("output/radii.dat"))
    velocities = np.array(readModelDataFile("output/velocities.dat"))
    thetas = np.array(readModelDataFile("output/thetas.dat"))
    results = readModelDataFile("output/results.dat")


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

    fig, (ax0,ax1,ax2) = plt.subplots(3,1, sharex=True)
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

    CS1 = ax1.contour(radii/(1.0E-6), thetas*180/pi, rad_theta14, levels)
    plt.clabel(CS1, inline=1, fontsize=10)
    ax1.set_ylabel("Entry Angle")
    ax1.set_title(r"%0.1f [km s$^{-1}$]"%(velocity_vals[1]/1000))

    CS2 = ax2.contour(radii/(1.0E-6), thetas*180/pi, rad_theta18, levels)
    plt.clabel(CS2, inline=1, fontsize=10)
    plt.xlabel("Radius [microns]")
    plt.ylabel("Entry Angle")
    ax2.set_title(r"%0.1f [km s$^{-1}$]"%(velocity_vals[2]/1000))


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



def runMultithreadAcrossParams(debug_print=False):
    """
    Run the simulation across the parameter ranges of initial radius, velocity,
    and impact angle (theta).

    Inputs:
        debug_print - set to true to print out model results.
    """
    if __name__ == '__main__':
        rad_count = 30
        vel_count = 30
        the_count = 30
        radii = np.linspace(50*1.0E-6, 450*1.0E-6, rad_count)
        velocities = np.linspace(11200, 36000, vel_count)
        thetas = np.linspace(0*pi/180,80*pi/180, the_count)

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
            saveModelData(radii, "output/radii.dat")
            saveModelData(velocities, "output/velocities.dat")
            saveModelData(thetas, "output/thetas.dat")
            saveModelData(args_array, "output/args_array.dat")
            saveModelData(results, "output/results.dat")



def plotParticleComparison(measured_rad, measured_core_frac, thetas_in):
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

    NOTE: the thetas plotted will be the closest thetas found in the data file.
    If the theta you want isn't exactly correct just rerun the data files to 
    ensure the exact value you want is included.
    """

    radii = np.array(readModelDataFile("output/radii.dat"))
    velocities = np.array(readModelDataFile("output/velocities.dat"))
    thetas = np.array(readModelDataFile("output/thetas.dat"))
    inputs = readModelDataFile("output/args_array.dat")
    results = readModelDataFile("output/results.dat")

    theta_vals = []
    for theta in thetas_in:
        index = np.abs(thetas - theta).argmin()
        theta_vals.append(thetas[index])


    the_len = len(theta_vals)
    micron = 1.0E-6
    sc = None

    cm = plt.cm.get_cmap("winter")

    for i in range(the_len):
        ax = plt.subplot(the_len,1,i+1)
        ax.set_xlim(np.min(radii)/micron, np.max(radii)/micron)
        ax.set_ylim(np.min(velocities)/1000, np.max(velocities)/1000)
        ax.set_ylabel(r"Velocity [km s$^{-1}$]")
        title = r"Impact Angle: %0.0f$^{\degree}$"%(theta_vals[i]*180/pi)
        ax.text(0.025, 0.85, title, transform=ax.transAxes, 
                bbox=dict(facecolor="red", alpha=0.5))

        #remove the tick labels from all but the last one
        if i != the_len-1:
            ax.set_xticklabels([])


        for j in range(len(inputs)):
            if inputs[j][2] == theta_vals[i]:
                final_radius, total_Fe, total_FeO, max_temp = results[j]
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
                        c=z, vmin=0, vmax=100, s=size, cmap=cm, edgecolor='none')

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






#simulateParticle(450*1.0E-6, 12000, 0*pi/180, debug_print=True)
#compareStandardAndHydrostaticAtmospheres()
#runMultithreadAcrossParams()
#plotParticleComparison(3.2*1.0E-6, 0.95, [0,30*pi/180, 45*pi/180, 60*pi/180]) 
plotParticleComparison(37.5*1.0E-6, 0, [0,30*pi/180, 45*pi/180, 60*pi/180]) 
#plotMultithreadResultsRadiusVsTheta(param=0)
#printSimulationFromFiles()

