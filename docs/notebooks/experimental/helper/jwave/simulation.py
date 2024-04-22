from jwave import FourierSeries
import jax.numpy as jnp
from typing import Callable, Any, Optional
from tqdm import tqdm
import logging

def run_simulations(compiled_simulator: Callable, 
                    domain: Any, 
                    sound_speed_map: jnp.ndarray,
                    density_map: jnp.ndarray, 
                    p0: jnp.ndarray, 
                    num_iterations: int,
                    sources: Optional[Any] = None) -> jnp.ndarray:
    """
    Run simulations for a specified number of iterations and concatenate the pressure data.

    Args:
        compiled_simulator (Callable): The compiled simulator function.
        domain (Any): The domain for the simulation.
        sound_speed_map (jnp.ndarray): The sound speed map.
        density_map (jnp.ndarray): The density map.
        p0 (jnp.ndarray): The initial pressure.
        num_iterations (int): The number of iterations to run the simulation.

    Returns:
        jnp.ndarray: The concatenated pressure data from all iterations.
    """
    pressure_combined_raw = None

    print(f"Starting simulations for {num_iterations} iterations")

    for i in tqdm(range(num_iterations), desc="Running simulations"):
        if i == 0:
            initial_pressure = p0
            current_sources = sources
        else:
            initial_pressure = pressure[-1]
            current_sources = None

        print(f"Rtunning simulation for iteration {i+1}")

        pressure = compiled_simulator(
            domain=domain,
            sound_speed_field=FourierSeries(jnp.expand_dims(sound_speed_map, -1), domain),
            density_field=FourierSeries(jnp.expand_dims(density_map, -1), domain),
            initial_pressure=initial_pressure,
            sources=current_sources
        )

        print(f"Simulation completed for iteration {i+1}")

        if pressure_combined_raw is None:
            pressure_combined_raw = pressure.on_grid
        else:
            pressure_combined_raw = jnp.concatenate((pressure_combined_raw, pressure.on_grid), axis=0)

        print(f"Pressure data concatenated for iteration {i+1}")

    print("Simulations completed")

    return pressure_combined_raw