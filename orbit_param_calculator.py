# orbit_param_calculator.py 

import numpy as np
from uncertainties import unumpy as unp

# -----------------------------------------------------------------------------
# Physical constants 
# -----------------------------------------------------------------------------
c             = 299_792_458.0        # speed of light in m/s
AU            = 1.495978707e11       # astronomical unit in meters
DAY2AU        = 86400.0 * c / AU     # convert LiTE days → AU (≈173.15)
DAYS_PER_YEAR = 365.242199           # days per Julian year

# -----------------------------------------------------------------------------
# Core functions
# -----------------------------------------------------------------------------

def period_in_years(P_LiTE, Ref_period):
    """
    Convert LiTE period from "cycles of Ref_period" to years.
    """
    return (P_LiTE * Ref_period) / DAYS_PER_YEAR


def a12sini(amp, ecc, omega):
    """
    Compute projected inner semi-major axis (a12 * sin(i)) from LiTE amplitude.
    """
    num   = amp * DAY2AU
    denom = unp.sqrt(1 - ecc**2 * unp.cos(omega)**2)
    return num / denom


def a12(a12sini_au, inc):
    """
    Recover true inner semi-major axis a12 from its projection.
    """
    return a12sini_au / unp.sin(inc)


def mass_func(period_yr, a12sini_au):
    """
    Compute the mass function f(m3) = (a12*sin(i))^3 / P^2 in M_sun.
    """
    return a12sini_au**3 / period_yr**2


def m3sini3(f_mass, m1, m2, tol: float = 1e-15, maxiter: int = 50):
    """
    Solve x = m3*sin(i) from the exact equation
        f_mass = x^3 / (m1 + m2 + x)^2
    via Newton–Raphson, so that it works with ufloats.

    Parameters
    ----------
    f_mass : float or uncertainties.uarray
        Mass function f(m3) in M_sun.
    m1, m2 : float or uncertainties.uarray
        Masses of the inner binary in M_sun.
    tol : float
        Convergence tolerance on the change in x (nominal_value).
    maxiter : int
        Maximum NR iterations.

    Returns
    -------
    x : float or uncertainties.uarray
        m3 * sin(i) in M_sun.
    """
    M12 = m1 + m2
    # initial guess ignoring m3 in denominator
    print(f_mass)
    x = (f_mass * M12**2)**(1/3)

    for _ in range(maxiter):
        # function and derivative
        g  = x**3 - f_mass * (M12 + x)**2
        gp = 3*x**2 - 2*f_mass * (M12 + x)
        dx = -g/gp
        x_new = x + dx

        # check convergence on nominal part of dx
        nv = getattr(dx, "nominal_value", dx)
        if abs(nv) < tol:
            x = x_new
            break
        x = x_new

    return x


def a3sini3(a12_au, m1, m2, m3sini3):
    """
    Compute projected tertiary semi-major axis a3*sin(i):
        a3*sin(i) = a12 * (m1 + m2) / (m3 * sin(i))
                  = a12 * (m1 + m2) / m3sini3
    """
    return a12_au * (m1 + m2) / m3sini3
