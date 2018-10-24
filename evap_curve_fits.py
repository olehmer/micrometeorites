from scipy.optimize import curve_fit
from numpy import exp

def wustite_curve_fit():
    temps = [1633, 1684, 1741, 1783, 1881, 1968, 2065]
    evap_rate = [0.006511692, 0.016606195, 0.04509766, 0.089690957, 0.402097008,
            1.404318753, 4.619319364]

    def func(T, a, b):
        return exp(a-b/T)

    coeffs = curve_fit(func, temps, evap_rate)
    print("for p=exp(A-B/T):")
    print("\tA = %2.3e"%(coeffs[0][0]))
    print("\tB = %2.3e"%(coeffs[0][1]))


def iron_curve_fit():
    temps = [1607, 1695, 1794, 1892, 1991, 2091]
    evap_rate = [0.1778, 0.7792, 3.559, 13.33, 44.36, 131.0]

    def func(T, a, b):
        return exp(a-b/T)

    coeffs = curve_fit(func, temps, evap_rate)
    print("for p=exp(A-B/T):")
    print("\tA = %2.3e"%(coeffs[0][0]))
    print("\tB = %2.3e"%(coeffs[0][1]))


def magnetite_curve_fit():
    temps = [1608, 1693, 1792, 1890, 1989]
    evap_rate = [0.00288, 0.01519, 0.08703, 0.40869, 1.7031]

    def func(T, a, b):
        return exp(a-b/T)

    coeffs = curve_fit(func, temps, evap_rate)
    print("for p=exp(A-B/T):")
    print("\tA = %2.3e"%(coeffs[0][0]))
    print("\tB = %2.3e"%(coeffs[0][1]))


print("--------------Wustite Fit---------------")
wustite_curve_fit()
print("\n--------------Iron Fit---------------")
iron_curve_fit()
print("\n--------------Magnetite Fit---------------")
magnetite_curve_fit()
