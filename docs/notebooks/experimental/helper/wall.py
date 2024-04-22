import numpy as np
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils import mapgen
from copy import deepcopy
import numpy as np
from kwave.utils.signals import tone_burst
from kwave.utils.conversion import cart2grid
from typing import Tuple, List
import numpy as np

class Wall:
    def __init__(self, grid_size: tuple, grid_spacing: tuple, wall_offset: float,
                 background_sound_speed: float, background_density: float):
        """
        Initialize the Wall object.

        :param grid_size: Size of the grid (tuple)
        :param grid_spacing: Spacing of the grid (tuple)
        :param wall_offset: Offset of the wall in meters (float)
        :param background_sound_speed: Background sound speed (float)
        :param background_density: Background density (float)
        """
        self.grid_size = Vector(grid_size)
        self.grid_spacing = Vector(grid_spacing)
        self.kgrid = kWaveGrid(self.grid_size, self.grid_spacing)
        self.wall_offset = int(wall_offset / self.grid_spacing[0])
        self.background_sound_speed = background_sound_speed
        self.background_density = background_density
        self.layers: List[Tuple[int, float, float]] = []
        self.objects: List[Tuple[str, Tuple, float, float]] = []

    @property
    def layer_thickness(self) -> float:
        """
        Total thickness of all layers in meters with offset, considering each layer's pixels converted to meters.
        """
        total_pixels = self.wall_offset + np.sum([layer_pixels for (layer_pixels, _, _) in self.layers])
        return total_pixels * self.grid_spacing[0]  # Convert total pixels back to meters

    def add_layer(self, thickness: float, sound_speed: float, density: float) -> None:
        layer_pixels = int(np.ceil(thickness / self.grid_spacing[0]))
        assert layer_pixels >= 0, "Layer thickness must be non-negative"
        assert layer_pixels + self.wall_offset <= self.grid_size[0], "Layer exceeds grid size"
        self.layers.append((layer_pixels, sound_speed, density))

    def add_stud(self, x_start: float, x_end: float, depth_start: float, depth_end: float,
                 sound_speed: float, density: float) -> None:
        """
        Add a stud to the wall.

        :param x_start: Start position of the stud in the x-direction in meters (float)
        :param x_end: End position of the stud in the x-direction in meters (float)
        :param depth_start: Start position of the stud in the depth-direction in meters (float)
        :param depth_end: End position of the stud in the depth-direction in meters (float)
        :param sound_speed: Sound speed of the stud (float)
        :param density: Density of the stud (float)
        """
        stud_start_x = int(x_start / self.grid_spacing[0])
        stud_end_x = int(x_end / self.grid_spacing[0])
        stud_start_depth = int(depth_start / self.grid_spacing[1])
        stud_end_depth = int(depth_end / self.grid_spacing[1])
        assert stud_start_x >= 0 and stud_end_x <= self.grid_size[0], "Stud x-position out of bounds"
        assert stud_start_depth >= 0 and stud_end_depth <= self.grid_size[1], "Stud depth-position out of bounds"
        self.objects.append(("stud", (stud_start_x, stud_end_x, stud_start_depth, stud_end_depth), sound_speed, density))

    def add_vertical_pipe(self, pos: Tuple[float, float], outer_diameter: float,
                          wall_thickness: float, fluid_sound_speed: float, fluid_density: float,
                          pipe_wall_sound_speed: float, pipe_wall_density: float) -> None:
        """
        Add a vertical pipe to the wall.

        :param pos: Position of the pipe in meters (tuple)
        :param outer_diameter: Outer diameter of the pipe in meters (float)
        :param wall_thickness: Wall thickness of the pipe in meters (float)
        :param fluid_sound_speed: Sound speed of the fluid inside the pipe (float)
        :param fluid_density: Density of the fluid inside the pipe (float)
        :param pipe_wall_sound_speed: Sound speed of the pipe wall (float)
        :param pipe_wall_density: Density of the pipe wall (float)
        """
        pos_vector = Vector(pos) / self.grid_spacing
        outer_radius = outer_diameter / (2 * self.grid_spacing[0])
        inner_radius = (outer_diameter - 2 * wall_thickness) / (2 * self.grid_spacing[0])
        assert pos_vector[0] >= 0 and pos_vector[0] < self.grid_size[0], "Pipe position out of bounds"
        assert pos_vector[1] >= 0 and pos_vector[1] < self.grid_size[1], "Pipe position out of bounds"
        self.objects.append(("vertical_pipe", (pos_vector, outer_radius, inner_radius),
                             fluid_sound_speed, fluid_density, pipe_wall_sound_speed, pipe_wall_density))

    def _create_vertical_pipe_masks(self, pos_vector: Vector, outer_radius: float, inner_radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create masks for the outer pipe, inner pipe, and pipe wall of a vertical pipe.

        :param pos_vector: Position of the pipe (Vector)
        :param outer_radius: Outer radius of the pipe (float)
        :param inner_radius: Inner radius of the pipe (float)
        :return: Tuple of masks for the outer pipe, inner pipe, and pipe wall (tuple of ndarrays)
        """
        outer_pipe_mask = mapgen.make_disc(self.grid_size, pos_vector, outer_radius)
        inner_pipe_mask = mapgen.make_disc(self.grid_size, pos_vector, inner_radius)
        pipe_wall_mask = np.logical_xor(outer_pipe_mask, inner_pipe_mask)
        return outer_pipe_mask, inner_pipe_mask, pipe_wall_mask
    
    @property
    def medium(self) -> kWaveMedium:
        """
        Generate the medium for the wall.

        :return: kWaveMedium object representing the wall medium
        """
        sound_speed = np.ones(self.grid_size) * self.background_sound_speed
        density = np.ones(self.grid_size) * self.background_density

        # Add layers sequentially
        layer_offset = self.wall_offset
        for layer_pixels, layer_sound_speed, layer_density in self.layers:
            sound_speed[:, layer_offset:layer_offset+layer_pixels] = layer_sound_speed
            density[:, layer_offset:layer_offset+layer_pixels] = layer_density
            layer_offset += layer_pixels

        # Add objects on top of layers
        for obj_type, obj_data, *obj_properties in self.objects:
            if obj_type == "stud":
                stud_start_x, stud_end_x, stud_start_depth, stud_end_depth = obj_data
                sound_speed[stud_start_x:stud_end_x, stud_start_depth:stud_end_depth] = obj_properties[0]
                density[stud_start_x:stud_end_x, stud_start_depth:stud_end_depth] = obj_properties[1]
            elif obj_type == "vertical_pipe":
                pos_vector, outer_radius, inner_radius = obj_data
                fluid_sound_speed, fluid_density, pipe_wall_sound_speed, pipe_wall_density = obj_properties
                outer_pipe_mask, inner_pipe_mask, pipe_wall_mask = self._create_vertical_pipe_masks(
                    pos_vector, outer_radius, inner_radius)
                sound_speed[inner_pipe_mask] = fluid_sound_speed
                density[inner_pipe_mask] = fluid_density
                sound_speed[pipe_wall_mask] = pipe_wall_sound_speed
                density[pipe_wall_mask] = pipe_wall_density

        return kWaveMedium(sound_speed=sound_speed, density=density)