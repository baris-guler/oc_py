import uncertainties.unumpy as un
from uncertainties import ufloat as uf
import numpy as np
from uncertainties.umath import sin
from scipy.optimize import root

def period_in_years(period,period_std,p_orb,p_orb_std):
    period = uf(period,period_std)
    p_orb = uf(p_orb,p_orb_std)
    return period*p_orb/365.242199
def sma12sini(amp,amp_std,ecc,ecc_std,omega,omega_std):
    omega = np.deg2rad(omega)
    omega_std = np.deg2rad(omega_std)
    amp = uf(amp,amp_std)
    ecc = uf(ecc,ecc_std)
    omega = uf(omega,omega_std)
    return (amp*173.15)/un.sqrt(1-(ecc**2)*un.cos(omega)**2)

def mass_func(period3,period3_std,sma12sini,sma12sini_std):
    period3 = uf(period3,period3_std)
    sma12sini = uf(sma12sini,sma12sini_std)
    return (sma12sini**3)/(period3**2)
def _m2_func(m2, m1, sini, mf):
    return (m2 * sini)**3 / (m1 + m2)**2 - mf

def mass3(mf, mf_std, m1, m1_std, m2, m2_std, inc=90, inc_std=0.01):
    """Compute the companion mass with uncertainty."""
    
    # Create uncertainties objects without astropy units
    mf = uf(mf, mf_std)  # mass function
    m1 = uf(m1, m1_std)  # primary mass
    m2 = uf(m2, m2_std)  # companion mass
    inc = uf(inc, inc_std)  # inclination in degrees

    # Convert inclination to radians and calculate sine
    sini = sin(inc * (3.141592653589793 / 180))  # Convert to radians
    # Total mass
    mT = m1 + m2

    # Solve for m2 using the root-finding function
    res = root(_m2_func, x0=10., args=(mT.nominal_value, sini.nominal_value, mf.nominal_value))
    if res.success:
        m2_value = res.x[0]
        m2_uncertainty = m2_value * (m1.std_dev / m1.nominal_value)  # Proportional uncertainty
        return uf(m2_value, m2_uncertainty)
    else:
        return np.nan


def sma12(sma12sini,sma12sini_std,inc,inc_std):
    inc = np.deg2rad(inc)
    inc_std = np.deg2rad(inc_std)
    sma12sini = uf(sma12sini,sma12sini_std)
    inc = uf(inc,inc_std)
    return sma12sini/un.sin(inc)

def sma3(sma12,sma12_std,mass1,mass1_std,mass2,mass2_std,mass3,mass3_std):
    sma12 = uf(sma12,sma12_std)
    mass1 = uf(mass1,mass1_std)
    mass2 = uf(mass2,mass2_std)
    mass3 = uf(mass3,mass3_std)
    return sma12*(mass1+mass2)/mass3
    
def ang_sep3(sma12,sma12_std,sma3,sma3_std,paralax,paralax_std):
    sma12 = uf(sma12,sma12_std)
    sma3 = uf(sma3,sma3_std)
    paralax = uf(paralax,paralax_std)
    parsec = 1./paralax
    distAU = parsec*206265.
    return 206205.*(un.arctan((sma12+sma3)/distAU))