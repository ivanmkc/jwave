import numpy as np

def calculate_max_simulation_time(*, grid_size: tuple, grid_spacing: tuple, sound_speed: float = 1540) -> float:
    """
    Calculate the maximum simulation time needed for an ultrasound simulation using k-wave.

    Args:
        grid_size (tuple): Grid size in grid points (nx, ny) for 2D or (nx, ny, nz) for 3D.
        grid_spacing (tuple): Grid spacing in meters (dx, dy) for 2D or (dx, dy, dz) for 3D.
        sound_speed (float): Speed of sound in the medium (default: 1540 m/s for soft tissue).

    Returns:
        float: The maximum simulation time in seconds.
    """
    assert len(grid_size) == len(grid_spacing), "Grid size and grid spacing dimensions must match."
    assert len(grid_size) in [2, 3], "Grid size must be either 2D or 3D."

    # Calculate the physical dimensions of the simulation domain in meters
    dimensions = [n * d for n, d in zip(grid_size, grid_spacing)]

    # Calculate the maximum distance along the diagonal
    max_distance = (sum(d ** 2 for d in dimensions)) ** 0.5

    # Calculate the time taken for the wave to travel the maximum distance
    t_max = max_distance / sound_speed

    # Calculate the maximum time needed for the simulation to capture the first echo
    t_sim = 2 * t_max

    return t_sim

def create_sensor_mask(kgrid_nx: int, kgrid_ny: int, x_stride: int, y_stride: int) -> tuple:
    """
    Create a sensor mask with specified x_stride and y_stride.
    
    Args:
        kgrid_nx (int): Number of elements in the x-dimension of the kgrid.
        kgrid_ny (int): Number of elements in the y-dimension of the kgrid.
        x_stride (int): Stride along the x-dimension for setting mask values to 1.
        y_stride (int): Stride along the y-dimension for setting mask values to 1.
    
    Returns:
        tuple: A tuple containing the following elements:
            - sensor_mask (numpy.ndarray): The sensor mask with specified strides.
            - num_x_sensor (int): Number of 1's in the x-dimension of the mask.
            - num_y_sensor (int): Number of 1's in the y-dimension of the mask.
    """
    sensor_mask = np.zeros((kgrid_nx, kgrid_ny), dtype=int)
    sensor_mask[::x_stride, ::y_stride] = 1
    
    num_x_sensor = np.sum(sensor_mask, axis=1).max()
    num_y_sensor = np.sum(sensor_mask, axis=0).max()
    
    return sensor_mask, num_x_sensor, num_y_sensor