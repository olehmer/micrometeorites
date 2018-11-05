from math import pi

def getFeMassAndFeOMass(Fe_percent, O_percent, radius):
    """
    This function will calculate the mass, in kg, of Fe and FeO in a given
    micrometeorite where only atomic percents of Fe and O in the sample are
    given (as is the case in Tomkins et al. (2016) extended data table 1).

    Inputs:
        Fe_percent - the measured percent of Fe atoms in the micrometeorite
        O_percent  - the measured percent of O atoms in the micrometeorite
        radius     - the measured radius of the micrometeorite [m]
    """


    M_Fe = 0.0558 #mass of Fe [kg mol-1]
    M_FeO = 0.0718 #mass of FeO [kg mol-1]
    N_A = 6.022E23 #Avagadro's number [atoms mol-1]
    rho_Fe = 7000 #density of Fe [kg m-3]
    rho_FeO = 4400 #density of FeO [kg m-3]

    num_atoms = N_A*4*pi/3*radius**3/(M_Fe*(Fe_percent-O_percent)/rho_Fe + 
            M_FeO*O_percent/rho_FeO)

    mass_Fe = M_Fe*(Fe_percent - O_percent)*num_atoms/N_A
    mass_FeO = M_FeO*O_percent*num_atoms/N_A

    calc_rad = (3/(4*pi)*(mass_Fe/rho_Fe + mass_FeO/rho_FeO))**(1/3)

    print("Fe mass: %2.3e [kg]\nFeO mass: %2.3e [kg]"%(mass_Fe, mass_FeO))
    print("Core mass fraction: %0.2f"%(mass_Fe/(mass_Fe+mass_FeO)))
    print("Calculated radius: %0.2f [microns]"%(calc_rad/1.0E-6))


#Tomkins micrometeorite that has and FeNi core (there's only one), Figure 1e
getFeMassAndFeOMass(1-0.0343, 0.0343, 3.2*1.0E-6)

#The micrometeorite from Figure 1f, pure wustite
getFeMassAndFeOMass(0.5, 0.5, 37.5*1.0E-6)



