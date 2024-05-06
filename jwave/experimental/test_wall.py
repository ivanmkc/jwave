import matplotlib.pyplot as plt
import numpy as np
import pytest

def plot_wall_properties(expected_sound_speed: np.ndarray, actual_sound_speed: np.ndarray,
                         expected_density: np.ndarray, actual_density: np.ndarray,
                         grid_spacing: tuple) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    extent = (0, actual_sound_speed.shape[1] * grid_spacing[1], 0, actual_sound_speed.shape[0] * grid_spacing[0])
    
    axes[0].imshow(expected_sound_speed, extent=extent)
    axes[0].set_title("Expected Sound Speed")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    
    axes[1].imshow(actual_sound_speed, extent=extent)
    axes[1].set_title("Actual Sound Speed")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    
    axes[2].imshow(expected_sound_speed - actual_sound_speed, extent=extent)
    axes[2].set_title("Sound Speed Diff")
    axes[2].set_xlabel("x [m]")
    axes[2].set_ylabel("y [m]")
    
    plt.tight_layout()
    plt.show()

def test_add_layer(plot: bool = False) -> None:
    """
    Test adding a single layer to the wall.
    
    Expected behavior:
    - The layer should be added at the specified wall offset.
    - The layer should have the specified thickness, sound speed, and density.
    - The rest of the medium should be filled with air properties.
    """
    grid_size = (10, 10)
    grid_spacing = (1e-3, 1e-3)
    wall_offset = 2e-3
    
    wall = Wall(grid_size=grid_size, grid_spacing=grid_spacing, wall_offset=wall_offset, background_sound_speed=1, background_density=1)
    
    wall.add_layer(thickness=3e-3, sound_speed=2, density=3)
    
    expected_sound_speed = np.ones(grid_size, dtype=float) * 1
    expected_density = np.ones(grid_size, dtype=float) * 1
    expected_sound_speed[:, 2:5] = 2
    expected_density[:, 2:5] = 3
    
    print("Expected sound speed:")
    print(expected_sound_speed)
    print("Actual sound speed:")
    print(wall.medium.sound_speed)
    
    print("Expected density:")
    print(expected_density)
    print("Actual density:")
    print(wall.medium.density)
    
    if plot:
        plot_wall_properties(expected_sound_speed=expected_sound_speed, actual_sound_speed=wall.medium.sound_speed,
                             expected_density=expected_density, actual_density=wall.medium.density,
                             grid_spacing=wall.grid_spacing)
    
    assert np.allclose(wall.medium.sound_speed[:, 2:5], 2)
    assert np.allclose(wall.medium.density[:, 2:5], 3)

def test_add_stud(plot: bool = False) -> None:
    """
    Test adding a single stud to the wall.
    
    Expected behavior:
    - The stud should be added at the specified position and have the specified dimensions.
    - The stud should have the specified sound speed and density.
    - The rest of the medium should be filled with air properties.
    """
    grid_size = (10, 10)
    grid_spacing = (1e-3, 1e-3)
    wall_offset = 2e-3
    
    wall = Wall(grid_size=grid_size, grid_spacing=grid_spacing, wall_offset=wall_offset, background_sound_speed=1, background_density=1)
    
    x_start = 4e-3
    x_end = 6e-3
    depth_start = 2e-3
    depth_end = 8e-3
    wall.add_stud(x_start=x_start, x_end=x_end, depth_start=depth_start, depth_end=depth_end, sound_speed=4, density=5)
    
    expected_sound_speed = np.ones(grid_size, dtype=float) * 1
    expected_density = np.ones(grid_size, dtype=float) * 1
    stud_start_x = int(x_start / 1e-3)
    stud_end_x = int(x_end / 1e-3)
    stud_start_depth = int(depth_start / 1e-3)
    stud_end_depth = int(depth_end / 1e-3)
    expected_sound_speed[stud_start_x:stud_end_x, stud_start_depth:stud_end_depth] = 4
    expected_density[stud_start_x:stud_end_x, stud_start_depth:stud_end_depth] = 5
    
    print("Expected sound speed:")
    print(expected_sound_speed)
    print("Actual sound speed:")
    print(wall.medium.sound_speed)
    
    print("Expected density:")
    print(expected_density)
    print("Actual density:")
    print(wall.medium.density)
    
    if plot:
        plot_wall_properties(expected_sound_speed=expected_sound_speed, actual_sound_speed=wall.medium.sound_speed,
                             expected_density=expected_density, actual_density=wall.medium.density,
                             grid_spacing=wall.grid_spacing)
    
    assert np.allclose(wall.medium.sound_speed[stud_start_x:stud_end_x, stud_start_depth:stud_end_depth], 4)
    assert np.allclose(wall.medium.density[stud_start_x:stud_end_x, stud_start_depth:stud_end_depth], 5)

def test_add_air_filled_vertical_pipe(plot: bool = False) -> None:
    """
    Test adding an air-filled vertical pipe to the wall.
    
    Expected behavior:
    - The pipe should be added at the specified position.
    - The pipe should have the specified outer diameter and wall thickness.
    - The fluid inside the pipe should have the specified sound speed and density.
    - The pipe wall should have the specified sound speed and density.
    - The rest of the medium should be filled with air properties.
    """
    grid_size = (10, 10)
    grid_spacing = (1e-3, 1e-3)
    wall_offset = 2e-3
    
    wall = Wall(grid_size=grid_size, grid_spacing=grid_spacing, wall_offset=wall_offset, background_sound_speed=1, background_density=1)
    
    pos = (5e-3, 5e-3)

    wall.add_vertical_pipe(pos=pos, outer_diameter=3e-3,
                           wall_thickness=1e-3, fluid_sound_speed=2, fluid_density=2,
                           pipe_wall_sound_speed=3, pipe_wall_density=3)

    outer_pipe_mask, inner_pipe_mask, pipe_wall_mask = wall._create_vertical_pipe_masks(
        pos_vector=Vector(pos) / wall.grid_spacing,
        outer_radius=3e-3 / (2 * 1e-3),
        inner_radius=(3e-3 - 2 * 1e-3) / (2 * 1e-3),
    )

    expected_sound_speed = np.ones(grid_size, dtype=float) * 1
    expected_density = np.ones(grid_size, dtype=float) * 1
    expected_sound_speed[inner_pipe_mask] = 2
    expected_density[inner_pipe_mask] = 2
    expected_sound_speed[pipe_wall_mask] = 3
    expected_density[pipe_wall_mask] = 3

    print("Expected sound speed:")
    print(expected_sound_speed)
    print("Actual sound speed:")
    print(wall.medium.sound_speed)
    
    print("Expected density:")
    print(expected_density)
    print("Actual density:")
    print(wall.medium.density)
    
    if plot:
        plot_wall_properties(expected_sound_speed=expected_sound_speed, actual_sound_speed=wall.medium.sound_speed,
                             expected_density=expected_density, actual_density=wall.medium.density,
                             grid_spacing=wall.grid_spacing)

    assert np.allclose(wall.medium.sound_speed, expected_sound_speed)
    assert np.allclose(wall.medium.density, expected_density)
    
def test_add_water_filled_vertical_pipe(plot: bool = False) -> None:
    """
    Test adding a water-filled vertical pipe to the wall.
    
    Expected behavior:
    - The pipe should be added at the specified position.
    - The pipe should have the specified outer diameter and wall thickness.
    - The fluid inside the pipe should have the specified sound speed and density.
    - The pipe wall should have the specified sound speed and density.
    - The rest of the medium should be filled with air properties.
    """
    grid_size = (10, 10)
    grid_spacing = (1e-3, 1e-3)
    wall_offset = 2e-3
    
    wall = Wall(grid_size=grid_size, grid_spacing=grid_spacing, wall_offset=wall_offset, background_sound_speed=1, background_density=1)
    
    pos = (5e-3, 5e-3)

    wall.add_vertical_pipe(pos=pos, outer_diameter=3e-3,
                           wall_thickness=1e-3, fluid_sound_speed=4, fluid_density=4,
                           pipe_wall_sound_speed=5, pipe_wall_density=5)

    outer_pipe_mask, inner_pipe_mask, pipe_wall_mask = wall._create_vertical_pipe_masks(
        pos_vector=Vector(pos) / wall.grid_spacing,
        outer_radius=3e-3 / (2 * 1e-3),
        inner_radius=(3e-3 - 2 * 1e-3) / (2 * 1e-3),
    )

    expected_sound_speed = np.ones(grid_size, dtype=float) * 1
    expected_density = np.ones(grid_size, dtype=float) * 1
    expected_sound_speed[inner_pipe_mask] = 4
    expected_density[inner_pipe_mask] = 4
    expected_sound_speed[pipe_wall_mask] = 5
    expected_density[pipe_wall_mask] = 5

    print("Expected sound speed:")
    print(expected_sound_speed)
    print("Actual sound speed:")
    print(wall.medium.sound_speed)
    
    print("Expected density:")
    print(expected_density)
    print("Actual density:")
    print(wall.medium.density)
    
    if plot:
        plot_wall_properties(expected_sound_speed=expected_sound_speed, actual_sound_speed=wall.medium.sound_speed,
                             expected_density=expected_density, actual_density=wall.medium.density,
                             grid_spacing=wall.grid_spacing)
            
    assert np.allclose(wall.medium.sound_speed, expected_sound_speed)
    assert np.allclose(wall.medium.density, expected_density)
    
def test_composite_wall(plot: bool = False) -> None:
    """
    Test creating a composite wall with multiple layers, studs, and pipes.
    
    Expected behavior:
    - The layers should be added sequentially, with each layer starting at the offset of the previous layer.
    - The studs should be added at the specified positions and have the specified dimensions and properties.
    - The vertical pipes should be added at the specified positions, with the specified diameters, wall thicknesses, and properties.
    - The resulting medium should have the correct sound speed and density values based on the added components.
    """
    grid_size = (10, 10)
    grid_spacing = (1e-3, 1e-3)
    wall_offset = 1e-3
    
    wall = Wall(grid_size=grid_size, grid_spacing=grid_spacing, wall_offset=wall_offset, background_sound_speed=1, background_density=1)
    
    # Add layers
    wall.add_layer(thickness=2e-3, sound_speed=2, density=2)
    wall.add_layer(thickness=2e-3, sound_speed=3, density=3)
    
    # Add studs
    stud_positions = [(3e-3, 4e-3), (7e-3, 8e-3)]
    for x_start, x_end in stud_positions:
        wall.add_stud(x_start=x_start, x_end=x_end, depth_start=2e-3, depth_end=8e-3, sound_speed=4, density=4)
    
    # Add vertical pipes
    air_filled_pipe_pos = (2e-3, 2e-3)
    water_filled_pipe_pos = (7e-3, 7e-3)
    
    wall.add_vertical_pipe(pos=air_filled_pipe_pos, outer_diameter=2e-3,
                           wall_thickness=1e-3, fluid_sound_speed=5, fluid_density=5,
                           pipe_wall_sound_speed=6, pipe_wall_density=6)
    
    wall.add_vertical_pipe(pos=water_filled_pipe_pos, outer_diameter=2e-3,
                           wall_thickness=1e-3, fluid_sound_speed=7, fluid_density=7,
                           pipe_wall_sound_speed=8, pipe_wall_density=8)
    
    # Create expected sound speed and density arrays
    expected_sound_speed = np.ones(grid_size, dtype=float) * 1
    expected_density = np.ones(grid_size, dtype=float) * 1
    
    # Apply layer properties
    layer_offset = int(wall_offset / 1e-3)
    expected_sound_speed[:, layer_offset:layer_offset+2] = 2
    expected_density[:, layer_offset:layer_offset+2] = 2
    layer_offset += 2

    expected_sound_speed[:, layer_offset:layer_offset+2] = 3
    expected_density[:, layer_offset:layer_offset+2] = 3

    # Apply stud properties
    for x_start, x_end in stud_positions:
        stud_start_x = int(x_start / 1e-3)
        stud_end_x = int(x_end / 1e-3)
        stud_start_depth = int(2e-3 / 1e-3)
        stud_end_depth = int(8e-3 / 1e-3)
        expected_sound_speed[stud_start_x:stud_end_x, stud_start_depth:stud_end_depth] = 4
        expected_density[stud_start_x:stud_end_x, stud_start_depth:stud_end_depth] = 4

    # Apply pipe properties
    air_filled_outer_pipe_mask, air_filled_inner_pipe_mask, air_filled_pipe_wall_mask = wall._create_vertical_pipe_masks(
        pos_vector=Vector(air_filled_pipe_pos) / wall.grid_spacing,
        outer_radius=2e-3 / (2 * 1e-3),
        inner_radius=(2e-3 - 2 * 1e-3) / (2 * 1e-3),
    )

    water_filled_outer_pipe_mask, water_filled_inner_pipe_mask, water_filled_pipe_wall_mask = wall._create_vertical_pipe_masks(
        pos_vector=Vector(water_filled_pipe_pos) / wall.grid_spacing,
        outer_radius=2e-3 / (2 * 1e-3),
        inner_radius=(2e-3 - 2 * 1e-3) / (2 * 1e-3),
    )
    
    expected_sound_speed[air_filled_inner_pipe_mask] = 5
    expected_density[air_filled_inner_pipe_mask] = 5
    expected_sound_speed[air_filled_pipe_wall_mask] = 6
    expected_density[air_filled_pipe_wall_mask] = 6

    expected_sound_speed[water_filled_inner_pipe_mask] = 7
    expected_density[water_filled_inner_pipe_mask] = 7
    expected_sound_speed[water_filled_pipe_wall_mask] = 8
    expected_density[water_filled_pipe_wall_mask] = 8

    print("Expected sound speed:")
    print(expected_sound_speed)
    print("Actual sound speed:")
    print(wall.medium.sound_speed)

    print("Expected density:")
    print(expected_density)
    print("Actual density:")
    print(wall.medium.density)

    if plot:
        plot_wall_properties(expected_sound_speed=expected_sound_speed, actual_sound_speed=wall.medium.sound_speed,
                             expected_density=expected_density, actual_density=wall.medium.density,
                             grid_spacing=wall.grid_spacing)

    assert np.allclose(wall.medium.sound_speed, expected_sound_speed)
    assert np.allclose(wall.medium.density, expected_density)    
    
def test_multiple_layers(plot: bool = False) -> None:
    """
    Test adding multiple layers to the wall.
    
    Expected behavior:
    - The layers should be added sequentially, with each layer starting at the offset of the previous layer.
    - Each layer should have the specified thickness, sound speed, and density.
    - The background medium should be filled with the specified background sound speed and density.
    """
    grid_size = (10, 10)
    grid_spacing = (1e-3, 1e-3)
    wall_offset = 1e-3
    background_sound_speed = 1
    background_density = 1

    wall = Wall(grid_size=grid_size, grid_spacing=grid_spacing, wall_offset=wall_offset,
                background_sound_speed=background_sound_speed, background_density=background_density)

    # Add layers
    wall.add_layer(thickness=2e-3, sound_speed=2, density=2)
    wall.add_layer(thickness=3e-3, sound_speed=3, density=3)
    wall.add_layer(thickness=1e-3, sound_speed=4, density=4)

    # Create expected sound speed and density arrays
    expected_sound_speed = np.ones(grid_size, dtype=float) * background_sound_speed
    expected_density = np.ones(grid_size, dtype=float) * background_density

    # Apply layer properties
    layer_offset = int(wall_offset / 1e-3)
    expected_sound_speed[:, layer_offset:layer_offset+2] = 2
    expected_density[:, layer_offset:layer_offset+2] = 2
    layer_offset += 2

    expected_sound_speed[:, layer_offset:layer_offset+3] = 3
    expected_density[:, layer_offset:layer_offset+3] = 3
    layer_offset += 3

    expected_sound_speed[:, layer_offset:layer_offset+1] = 4
    expected_density[:, layer_offset:layer_offset+1] = 4

    print("Expected sound speed:")
    print(expected_sound_speed)
    print("Actual sound speed:")
    print(wall.medium.sound_speed)

    print("Expected density:")
    print(expected_density)
    print("Actual density:")
    print(wall.medium.density)

    if plot:
        plot_wall_properties(expected_sound_speed=expected_sound_speed, actual_sound_speed=wall.medium.sound_speed,
                             expected_density=expected_density, actual_density=wall.medium.density,
                             grid_spacing=wall.grid_spacing)

    assert np.allclose(wall.medium.sound_speed, expected_sound_speed)
    assert np.allclose(wall.medium.density, expected_density)  
    
def test_add_layer_out_of_bounds():
    """
    Test adding a layer that exceeds the grid size.
    Expected behavior:
    - Raises an AssertionError when the layer thickness exceeds the grid size.
    """
    grid_size = (10, 10)
    grid_spacing = (1e-3, 1e-3)
    wall_offset = 5e-3

    wall = Wall(grid_size=grid_size, grid_spacing=grid_spacing, wall_offset=wall_offset,
                background_sound_speed=1, background_density=1)

    with pytest.raises(AssertionError, match="Layer exceeds grid size"):
        wall.add_layer(thickness=6e-3, sound_speed=2, density=2)
        
def test_add_stud_out_of_bounds():
    """
    Test adding a stud that is out of bounds.
    Expected behavior:
    - Raises an AssertionError when the stud position is out of bounds.
    """
    grid_size = (10, 10)
    grid_spacing = (1e-3, 1e-3)
    wall_offset = 2e-3

    wall = Wall(grid_size=grid_size, grid_spacing=grid_spacing, wall_offset=wall_offset,
                background_sound_speed=1, background_density=1)

    with pytest.raises(AssertionError, match="Stud x-position out of bounds"):
        wall.add_stud(x_start=11e-3, x_end=12e-3, depth_start=2e-3, depth_end=8e-3, sound_speed=4, density=4)

    with pytest.raises(AssertionError, match="Stud depth-position out of bounds"):
        wall.add_stud(x_start=3e-3, x_end=4e-3, depth_start=11e-3, depth_end=12e-3, sound_speed=4, density=4)    
        
def test_add_vertical_pipe_out_of_bounds():
    """
    Test adding a vertical pipe that is out of bounds.
    Expected behavior:
    - Raises an AssertionError when the pipe position is out of bounds.
    """
    grid_size = (10, 10)
    grid_spacing = (1e-3, 1e-3)
    wall_offset = 2e-3

    wall = Wall(grid_size=grid_size, grid_spacing=grid_spacing, wall_offset=wall_offset,
                background_sound_speed=1, background_density=1)

    with pytest.raises(AssertionError, match="Pipe position out of bounds"):
        wall.add_vertical_pipe(pos=(-1e-3, 5e-3), outer_diameter=2e-3,
                               wall_thickness=1e-3, fluid_sound_speed=2, fluid_density=2,
                               pipe_wall_sound_speed=3, pipe_wall_density=3)

    with pytest.raises(AssertionError, match="Pipe position out of bounds"):
        wall.add_vertical_pipe(pos=(5e-3, 11e-3), outer_diameter=2e-3,
                               wall_thickness=1e-3, fluid_sound_speed=2, fluid_density=2,
                               pipe_wall_sound_speed=3, pipe_wall_density=3)