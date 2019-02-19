from scipy.optimize import curve_fit
from numpy import exp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

def co2_rate_const_fit():
    """
    Calculate the rate constant for Fe oxidation by CO2. The data is taken from
    table II in Abuluwefa et al (1997), but it has been converted from 
    [g cm-2 s-1 atm-1] to [kg m-2 s-1 Pa-1].
    """

    gas_const = 8.314 #[J K-1 mol-1]
    temps = [1150, 1100, 1050, 1000] #[K]
    rate_consts = [1.54E-10, 6.84E-11, 2.18E-11, 3.58E-12] #[kg m-2 s-1 Pa-1]

    def func(T, a, b):
        return a*exp(b/(gas_const*T))

    coeffs = curve_fit(func, temps, rate_consts)
    a_term = coeffs[0][0]
    b_term = coeffs[0][1] 
    print("for k=A*exp(B/RT):")
    print("\tA = %2.3e"%(a_term))
    print("\tB = %2.3e"%(b_term))
    print("k=%2.2e [kg m-2 s-1 Pa-1] at %0.1f [K]"%(func(1800, a_term, b_term),1800))

    plot_temps = np.linspace(800, 3000, 20)
    plot_consts = np.zeros_like(plot_temps)
    for i in range(0, len(plot_temps)):
        plot_consts[i] = func(plot_temps[i], a_term, b_term)

    plt.subplot(2, 1, 1)
    plt.plot(plot_temps, plot_consts)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1e}'))
    plt.xlabel("Temperature [K]")
    plt.ylabel("Rate Constant [kg m-2 s-1 Pa-1]")

    plot_consts_fit = np.zeros(4)
    for i in range(4):
        plot_consts_fit[i] = func(temps[i], a_term, b_term)
    plt.subplot(2, 1, 2)
    plt.plot(temps, rate_consts, 'ro', label="Data")
    plt.plot(temps, plot_consts_fit, label="Fit")
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1e}'))
    plt.xlabel("Temperature [k]")
    plt.ylabel("Rate Constant [kg m-2 s-1 Pa-1]")
    plt.legend()
    plt.show()



co2_rate_const_fit()

