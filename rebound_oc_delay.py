import rebound
import matplotlib.pyplot as plt
import numpy as np

def simulate_oc_delay(
    m1, m2, system_inclination_deg, times_of_p, periods, eccentricities,
    omegas_deg, masses, incs, mintimes, gif=False
):
    """
    Simulate the system to calculate O-C delays. Optionally compute positions for all objects for visualization.

    Parameters:
        m1 (float): Mass of the primary star.
        m2 (float): Mass of the secondary star.
        system_inclination_deg (float): Inclination of the system in degrees.
        times_of_p (list): Times of periastron for planets.
        periods (list): Orbital periods of planets.
        eccentricities (list): Orbital eccentricities of planets.
        omegas_deg (list): Argument of periastron in degrees for planets.
        masses (list): Masses of planets.
        incs (list): Inclinations of planets.
        mintimes (list): Observation times.
        gif (bool): If True, compute and return positions for all objects.

    Returns:
        np.array: O-C delays for the given observation times.
        dict (optional): Positions of all objects if `gif=True`.
    """
    SPEED_OF_LIGHT_AU_PER_DAY = 173.144633
    system_inclination_rad = np.radians(system_inclination_deg)
    omegas = np.radians(omegas_deg)
    inclinations = [system_inclination_rad + np.radians(inc) for inc in incs]
    start_julian_date = min(mintimes)
    T_closest_planets = [jd - start_julian_date for jd in times_of_p]

    # Initialize the simulation
    sim = rebound.Simulation()
    sim.units = ('day', 'AU', 'Msun')
    sim.integrator = "WHFast"
    sim.dt = 0.5
    sim.add(m=m1 + m2)

    for i in range(len(masses)):
        sim.add(
            m=masses[i], P=periods[i], e=eccentricities[i],
            inc=inclinations[i], T=T_closest_planets[i], omega=omegas[i]
        )
    sim.move_to_com()

    # Calculate O-C values for mintimes
    calculated_oc_for_mintimes = []
    positions = {i: [] for i in range(len(sim.particles))} if gif else None  # Store positions only if gif=True

    for jd in mintimes:
        sim.integrate(jd - start_julian_date)
        light_travel_time_delay = -sim.particles[0].z / SPEED_OF_LIGHT_AU_PER_DAY
        calculated_oc_for_mintimes.append(light_travel_time_delay)

        if gif:
            # Record positions for all objects
            for i, particle in enumerate(sim.particles):
                positions[i].append((particle.x, particle.y))

    if gif:
        return np.array(calculated_oc_for_mintimes), positions
    else:
        return np.array(calculated_oc_for_mintimes)