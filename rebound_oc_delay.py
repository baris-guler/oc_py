import rebound
import matplotlib.pyplot as plt
import numpy as np

def collision_print_only(sim_pointer, collision):
    print("col") 

def simulate_oc_delay(
    m1, m2, system_inclination_deg, times_of_p, periods, eccentricities,
    omegas_deg, masses, incs, mintimes, return_pos=False, integrator="IAS15",
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
        return_pos (bool): If True, compute and return positions for all objects.

    Returns:
        np.array: O-C delays for the given observation times.
        dict (optional): Positions of all objects if `gif=True`.
    """

    # print("m1", m1)
    # print("m2", m2)
    # print("p",periods)
    # print("e",eccentricities)
    # print("o",omegas_deg)
    # print("m",masses)
    # print("i",incs)
    # print("t",times_of_p)
    # print(mintimes)
    # print(integrator)

    ecc = np.asarray(eccentricities)
    if np.any((ecc < 0) | (ecc > 1)):
        return -1e-50
    SPEED_OF_LIGHT_AU_PER_DAY = 173.144633
    omegas = np.radians(omegas_deg)
    inclinations = [np.radians(inc) for inc in incs]
    start_julian_date = min(mintimes)
    T_closest_planets = [jd - start_julian_date for jd in times_of_p]

    # Initialize the simulation
    sim = rebound.Simulation()
    sim.collision_resolve = collision_print_only
    sim.units = ('day', 'AU', 'Msun')
    sim.integrator = integrator
    sim.dt = 0.5
    sim.ri_ias15_accuracy = 1e-8 
    sim.add(m=m1 + m2)

    for i in range(len(masses)):
        sim.add(
            m=masses[i], P=periods[i], e=eccentricities[i],
            inc=inclinations[i], T=T_closest_planets[i], omega=omegas[i]
        )
    sim.move_to_com()
    # Calculate O-C values for mintimes
    calculated_oc_for_mintimes = []
    positions = {i: [] for i in range(len(sim.particles))} if return_pos else None  # Store positions only if return_pos=True

    for jd in mintimes:
        sim.integrate(jd - start_julian_date)
        light_travel_time_delay = -sim.particles[0].z / SPEED_OF_LIGHT_AU_PER_DAY
        calculated_oc_for_mintimes.append(light_travel_time_delay)

        if return_pos:
            # Record positions for all objects
            for i, particle in enumerate(sim.particles):
                positions[i].append((particle.x, particle.y))

    if return_pos:
        return np.array(calculated_oc_for_mintimes), positions
    else:
        return np.array(calculated_oc_for_mintimes)