# This file is part of j-Wave.
#
# j-Wave is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# j-Wave is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with j-Wave. If not, see <https://www.gnu.org/licenses/>.

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxdf import Field, FourierSeries
from jaxdf.geometry import Domain
from jaxdf.mods import Module
from jaxdf.operators import dot_product, functional
from jaxtyping import Array
from plum import parametric

from jwave.logger import logger

Number = Union[float, int]


@parametric
class Medium(Module):
    """_summary_

    Args:
        eqx (_type_): _description_

    Raises:
        ValueError: _description_
        TypeError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    domain: Domain
    sound_speed: Union[Array, Field, float]
    density: Union[Array, Field, float]
    attenuation: Union[Array, Field, float]
    pml_size: float = eqx.field(default=20.0, static=True)

    def __init__(self,
                 domain: Domain,
                 sound_speed: Union[Array, Field, float] = 1.0,
                 density: Union[Array, Field, float] = 1.0,
                 attenuation: Union[Array, Field, float] = 1.0,
                 pml_size: float = 20.0):
        self.domain = domain

        # Check if any input is an Array and none are subclasses of Field
        inputs_are_arrays = [
            isinstance(x, Array) and not jnp.isscalar(x)
            for x in [sound_speed, density, attenuation]
        ]
        inputs_are_fields = [
            issubclass(type(x), Field)
            for x in [sound_speed, density, attenuation]
        ]

        if any(inputs_are_arrays) and any(inputs_are_fields):
            raise ValueError(
                "Ambiguous inputs for Medium: cannot mix Arrays and Field subclasses."
            )

        if all(inputs_are_arrays):
            logger.warning(
                "All inputs are Arrays. This is not recommended for performance reasons. Consider using Fields instead."
            )

        self.sound_speed = sound_speed
        self.density = density
        self.attenuation = attenuation

        # Converting if needed
        for field_name in ["sound_speed", "density", "attenuation"]:
            # Convert to Fourier Series if it is a jax Array and is not a scalar
            if isinstance(
                    self.__dict__[field_name],
                    Array) and not jnp.isscalar(self.__dict__[field_name]):
                #logger.info(f"Converting {field_name}, which is an Array, to a FourierSeries before storing it in the Medium object.")
                self.__dict__[field_name] = FourierSeries(
                    self.__dict__[field_name], domain)

        # Other parameters
        self.pml_size = pml_size

    def __check_init__(self):
        # Check that all domains are the same
        for field in [self.sound_speed, self.density, self.attenuation]:
            if isinstance(field, Field):
                assert self.domain == field.domain, "The domain of all fields must be the same as the domain of the Medium object."

    @classmethod
    def __init_type_parameter__(self, t: type):
        """Check whether the type parameters is valid."""
        if issubclass(t, Field):
            return t
        else:
            raise TypeError(
                f"The type parameter of a Medium object must be a subclass of Field. Got {t}"
            )

    @property
    def max_sound_speed(self):
        """
        Calculate and return the maximum sound speed.

        This property uses the `sound_speed` method/function and applies the `amax`
        function from JAX's numpy (jnp) library to find the maximum sound speed value.

        Returns:
            The maximum sound speed value.
        """
        return functional(self.sound_speed)(jnp.amax)

    @property
    def min_sound_speed(self):
        """
        Calculate and return the minimum sound speed.

        This property uses the `sound_speed` method/function and applies the `amin`
        function from JAX's numpy (jnp) library to find the minimum sound speed value.

        Returns:
            The minimum sound speed value.
        """
        return functional(self.sound_speed)(jnp.amin)

    @property
    def max_density(self):
        """
        Calculate and return the maximum density.

        This property uses the `density` method/function and applies the `amax`
        function from JAX's numpy (jnp) library to find the maximum density value.

        Returns:
            The maximum density value.
        """
        return functional(self.density)(jnp.amax)

    @property
    def min_density(self):
        """
        Calculate and return the minimum density.

        This property uses the `density` method/function and applies the `amin`
        function from JAX's numpy (jnp) library to find the minimum density value.

        Returns:
            The minimum density value.
        """
        return functional(self.density)(jnp.amin)

    @property
    def max_attenuation(self):
        """
        Calculate and return the maximum attenuation.

        This property uses the `attenuation` method/function and applies the `amax`
        function from JAX's numpy (jnp) library to find the maximum attenuation value.

        Returns:
            The maximum attenuation value.
        """
        return functional(self.attenuation)(jnp.amax)

    @property
    def min_attenuation(self):
        """
        Calculate and return the minimum attenuation.

        This property uses the `attenuation` method/function and applies the `amin`
        function from JAX's numpy (jnp) library to find the minimum attenuation value.

        Returns:
            The minimum attenuation value.
        """
        return functional(self.attenuation)(jnp.amin)

    @classmethod
    def __infer_type_parameter__(self, *args, **kwargs):
        """Inter the type parameter from the arguments. Defaults to FourierSeries if
        the parameters are all floats"""
        # Reconstruct kwargs from args
        keys = self.__init__.__code__.co_varnames[1:]
        extra_kwargs = dict(zip(keys, args))
        kwargs.update(extra_kwargs)

        # Get fields types
        field_inputs = ["sound_speed", "density", "attenuation"]
        input_types = []
        for field_name in field_inputs:
            if field_name in kwargs:
                field = kwargs[field_name]

                if isinstance(field, Field):
                    input_types.append(type(field))

        # Keep only unique
        input_types = set(input_types)

        has_fields = len(input_types) > 0
        if not has_fields:
            return FourierSeries

        # Check that there are no more than one field type
        if len(input_types) > 1:
            raise ValueError(
                f"All fields must be of the same type or scalars for a Medium object. Got {input_types}"
            )

        return input_types.pop()

    @classmethod
    def __le_type_parameter__(self, left, right):
        assert len(left) == 1 and len(
            right) == 1, "Medium type parameters can't be tuples."
        return issubclass(left[0], right[0])

    @property
    def int_pml_size(self) -> int:
        r"""Returns the size of the PML layer as an integer"""
        return int(self.pml_size)


def points_on_circle(
        n: int,
        radius: float,
        centre: Tuple[float, float],
        cast_int: bool = True,
        angle: float = 0.0,
        max_angle: float = 2 * np.pi) -> Tuple[List[float], List[float]]:
    """
    Generate points on a circle.

    Args:
        n (int): Number of points.
        radius (float): Radius of the circle.
        centre (tuple): Centre coordinates of the circle (x, y).
        cast_int (bool, optional): If True, points will be rounded and converted to integers. Default is True.
        angle (float, optional): Starting angle in radians. Default is 0.
        max_angle (float, optional): Maximum angle to reach in radians. Default is 2*pi (full circle).

    Returns:
        x, y (tuple): Lists of x and y coordinates of the points.
    """
    angles = np.linspace(0, max_angle, n, endpoint=False)
    x = (radius * np.cos(angles + angle) + centre[0]).tolist()
    y = (radius * np.sin(angles + angle) + centre[1]).tolist()
    if cast_int:
        x = list(map(int, x))
        y = list(map(int, y))
    return x, y


def unit_fibonacci_sphere(
        samples: int = 128) -> List[Tuple[float, float, float]]:
    """
    Generate evenly distributed points on the surface
    of a unit sphere using the Fibonacci Sphere method.

    From https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

    Args:
        samples (int, optional): The number of points to generate.
            Default is 128.

    Returns:
        points (list): A list of tuples representing the (x, y, z)
            coordinates of the points on the sphere.
    """
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))    # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2    # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)    # radius at y
        theta = phi * i    # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
    return points


def fibonacci_sphere(
        n: int,
        radius: float,
        centre: Union[Tuple[float, float, float], np.ndarray],
        cast_int: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate evenly distributed points on the surface of
    a sphere using the Fibonacci Sphere method.

    Args:
    n (int): The number of points to generate.
    radius (float): The radius of the sphere.
    centre (tuple or np.ndarray): The (x, y, z) coordinates of
        the center of the sphere.
    cast_int (bool, optional): If True, points will be rounded
        and converted to integers. Default is True.

    Returns:
    x, y, z (tuple): The x, y, and z coordinates of the points on the sphere.
    """
    points = unit_fibonacci_sphere(n)
    points = np.array(points)
    points = points * radius + centre
    if cast_int:
        points = points.astype(int)
    return points[:, 0], points[:, 1], points[:, 2]


def circ_mask(N: Tuple[int, int], radius: float,
              centre: Union[List[float], Tuple[float, float]]) -> np.ndarray:
    """
    Generate a 2D binary mask representing a circle within a 2D grid.

    The mask is an ndarray of size N with 1s inside the circle (defined by a given
    centre and radius) and 0s outside.

    Args:
        N (Tuple[int, int]): The shape of the output mask (size of the grid).
            It should be in the format (x_size, y_size).
        radius (float): The radius of the circle.
        centre (Union[List[float], Tuple[float, float]]): The coordinates of
            the centre of the circle in the format (x, y).

    Returns:
        mask (np.ndarray): The 2D mask as a numpy ndarray of integers.
            The shape of the mask is N. Values inside the circle are 1, and values
            outside the circle are 0.
    """
    x, y = np.mgrid[0:N[0], 0:N[1]]
    dist_from_centre = np.sqrt((x - centre[0])**2 + (y - centre[1])**2)
    mask = (dist_from_centre < radius).astype(int)
    return mask


def sphere_mask(
        N: Tuple[int, int, int], radius: float,
        centre: Union[List[float], Tuple[float, float, float]]) -> np.ndarray:
    """
    Generate a 3D binary mask representing a sphere within a 3D grid.

    The mask is an ndarray of size N with 1s inside the sphere (defined by a given
    centre and radius) and 0s outside.

    Args:
        N (Tuple[int, int, int]): The shape of the output mask (size of the grid).
            It should be in the format (x_size, y_size, z_size).
        radius (float): The radius of the sphere.
        centre (Union[List[float], Tuple[float, float, float]]): The coordinates of the
            centre of the sphere in the format (x, y, z).

    Returns:
        mask (np.ndarray): The 3D mask as a numpy ndarray of integers. The shape of
            the mask is N. Values inside the sphere are 1, and values outside the
            sphere are 0.
    """
    x, y, z = np.mgrid[0:N[0], 0:N[1], 0:N[2]]
    dist_from_centre = np.sqrt((x - centre[0])**2 + (y - centre[1])**2 +
                               (z - centre[2])**2)
    mask = (dist_from_centre < radius).astype(int)
    return mask


@register_pytree_node_class
class Sources:
    r"""Sources structure

    Attributes:
      positions (Tuple[List[int]): source positions
      signals (List[jnp.ndarray]): source signals
      dt (float): time step
      domain (Domain): domain

    !!! example

      ```python
      x_pos = [10,20,30,40]
      y_pos = [30,30,30,30]
      signal = jnp.sin(jnp.linspace(0,10,100))
      signals = jnp.stack([signal]*4)
      sources = geometry.Source(positions=(x_pos, y_pos), signals=signals)
      ```
    """
    positions: Tuple[np.ndarray]
    signals: Tuple[jnp.ndarray]
    dt: float
    domain: Domain

    def __init__(self, positions, signals, dt, domain):
        self.positions = positions
        self.signals = signals
        self.dt = dt
        self.domain = domain

    def tree_flatten(self):
        children = (self.signals, self.dt)
        aux = (self.domain, self.positions)
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux, children):
        signals, dt = children
        domain, positions = aux
        a = cls(positions, signals, dt, domain)
        return a

    def to_binary_mask(self, N):
        r"""
        Convert sources to binary mask
        Args:
          N (Tuple[int]): grid size

        Returns:
          jnp.ndarray: binary mask
        """
        mask = jnp.zeros(N)
        for i in range(len(self.positions[0])):
            mask = mask.at[self.positions[0][i], self.positions[1][i]].set(1)
        return mask > 0

    def on_grid(self, n):

        src = jnp.zeros(self.domain.N)
        if len(self.signals) == 0:
            return src

        idx = n.astype(jnp.int32)
        signals = self.signals[:, idx]
        src = src.at[self.positions].add(signals)
        return jnp.expand_dims(src, -1)

    @staticmethod
    def no_sources(domain):
        return Sources(positions=([], []), signals=([]), dt=1.0, domain=domain)


@register_pytree_node_class
class DistributedTransducer:
    def __init__(self, mask, center_pos, signal=jnp.array([]), domain=None, is_active=True):
        """
        Initialize the DistributedTransducer.

        Args:
            mask: The mask representing the transducer element.
            signal: The signal for the transducer element (default: empty array).
            domain: The computational domain (default: None).
            is_active: The active state of the transducer element (default: True).
            center_pos: The center position of the transducer element (default: None).
        """
        self.mask = mask
        self.center_pos = center_pos        
        self.signal = signal
        self.domain = domain if domain is not None else mask.domain
        self.is_active = is_active

    def tree_flatten(self):
        """
        Flatten the DistributedTransducer for PyTree compatibility.
        """
        children = (self.mask, self.center_pos)
        aux = (self.signal, self.domain, self.is_active)
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """
        Unflatten the DistributedTransducer for PyTree compatibility.
        """
        mask, center_pos = children
        signal, domain, is_active = aux
        return cls(mask, center_pos, signal, domain, is_active)


    def __call__(self, p: Field, u: Field, rho: Field):
        """
        Compute the output for the transducer element.

        Args:
            p: The pressure field.
            u: The velocity field (not used in this implementation).
            rho: The density field (not used in this implementation).

        Returns:
            The output of the transducer element.
        """
        
        if not self.is_active:
            return 0
        
        return dot_product(self.mask, p)

    
    def set_is_active(self, is_active):
        """
        Set the is_active state for the transducer element.

        Args:
            is_active: The state to be set for the element.

        Returns:
            A new instance of DistributedTransducer with the updated is_active state.
        """
        return DistributedTransducer(self.mask, self.center_pos, self.signal, self.domain, is_active)

    def set_signal(self, signal):
        """
        Set the signal for the transducer element.

        Args:
            signal: The signal to be set for the element.

        Returns:
            A new instance of DistributedTransducer with the updated signal.
        """
        return DistributedTransducer(self.mask, self.center_pos, signal, self.domain, self.is_active)

    def on_grid(self, n):
        """
        Compute the wavefield produced by the transducer element on the grid.

        Args:
            n: The time index for the signal.

        Returns:
            The wavefield produced by the transducer element.
        """
        
        if not self.is_active:
            return jnp.expand_dims(jnp.zeros(self.domain.N), -1)
    
        idx = n.astype(jnp.int32)
        signal = self.signal[:, idx]
                
        return signal * self.mask.on_grid

@register_pytree_node_class
class TransducerArray:
    def __init__(
        self,
        domain: Domain,
        num_elements: int,
        element_width: int,
        element_height: int = 1,
        element_depth: int = 1,
        element_spacing: int = 0,
        position: Tuple[int] = (1,),
        radius: float = float("inf"),
        sound_speed: Optional[float] = None,
        focus_distance: float = float("inf"),
        steering_angle: float = 0.0,
        signal: jnp.ndarray = None,
        dt: float = None,
    ):
        """
        Initialize the TransducerArray.

        Args:
            domain: The computational domain.
            num_elements: The number of transducer elements.
            element_width: The width of each element in grid points. In the x direction.
            element_height: The length of each element in grid points (default: 1). In the y direction.
            element_depth: The height of each element in grid points (default: 1). In the z direction.
            element_spacing: The spacing between elements in grid points (default: 0). In the x direction.
            position: The position of the corner of the transducer array in grid points (default: (1,)).
            radius: The radius of curvature of the transducer array in meters (default: inf).
            sound_speed: The assumed homogeneous sound speed in meters per second for beamforming purposes (default: None).
            focus_distance: The focus distance in meters for beamforming (default: inf).
            steering_angle: The steering angle in degrees for beamforming (default: 0.0).
            signal: The signal to be set for all elements (default: None).
            dt: The time step of the simulation in seconds (default: None).
        """
        self.domain = domain
        self.num_elements = num_elements
        self.element_width = element_width
        self.element_height = element_height
        self.element_depth = element_depth
        self.element_spacing = element_spacing

        assert len(position) == self.domain.ndim, f"Position dimensionality {len(position)} must match domain dimensionality {self.domain.ndim}"
        self.position = position

        self.radius = radius

        if not np.isinf(self.radius):
            raise NotImplementedError("A finite radius is not currently supported")

        assert sound_speed is not None, "sound_speed must be provided"
        assert dt is not None, "dt must be provided"

        self.sound_speed = sound_speed
        self.focus_distance = focus_distance
        self.steering_angle = steering_angle
        self.dt = dt

        self.elements = self._create_elements(signal)
    
    def tree_flatten(self):
        """
        Flatten the TransducerArray for PyTree compatibility.
        """
        children = (self.elements,)
        aux = (self.domain, self.num_elements, self.element_width, self.element_height,
               self.element_depth, self.element_spacing, self.position,
               self.radius, self.sound_speed, self.focus_distance, self.steering_angle, self.dt)
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """
        Unflatten the TransducerArray for PyTree compatibility.
        """
        elements = children[0]
        domain, num_elements, element_width, element_height, element_depth, element_spacing, position, radius, sound_speed, focus_distance, steering_angle, dt = aux
        transducer_array = cls(domain, num_elements, element_width, element_height, element_depth,
                               element_spacing, position, radius, sound_speed, focus_distance,
                               steering_angle, None, dt)
        transducer_array.elements = elements
        return transducer_array

    def _create_elements(self, signal=None) -> List[DistributedTransducer]:
        """
        Create instances of DistributedTransducer for each element and apply beamforming delays.

        Args:
            signal: The signal to be set for all elements (default: None).

        Returns:
            A list of DistributedTransducer instances representing each element.
        """
        elements = []
        center_offset = (self.num_elements - 1) * (self.element_width + self.element_spacing) / 2

        element_dimensions = (self.element_width, self.element_height, self.element_depth)[:self.domain.ndim]

        center_pos_list = []
        element_pos_list = []
        for element_index in range(self.num_elements):
            # Calculate the position of the current element based on the domain dimensions
            element_pos = list(self.position)
            element_pos[0] += (self.element_width + self.element_spacing) * element_index - center_offset
            element_pos_list.append(element_pos)
            
            center_pos = tuple((element_pos[i] + element_dimensions[i]) / 2 for i in range(self.domain.ndim))
            center_pos_list.append(center_pos)

        delays_in_s = TransducerArray.calculate_beamforming_delays(
            source_positions=jnp.array(center_pos_list),
            target_point=self.target_point,
            sound_speed=self.sound_speed,
            dx=self.domain.dx[0], 
            dy=self.domain.dx[1]            
        )
        
        for element_index in range(self.num_elements):
            element_pos = element_pos_list[element_index]
            
            # Create a mask for the current element
            mask = jnp.zeros(self.domain.N)

            # Set the mask values to 1.0 for the current element based on the domain dimensions
            slices = tuple(
                slice(int(element_pos[i] / self.domain.dx[i]), int((element_pos[i] + element_dimensions[i]) / self.domain.dx[i]))
                for i in range(self.domain.ndim)
            )
            mask = mask.at[slices].set(1.0)

            center_pos = center_pos_list[element_index]
            delay_in_s = delays_in_s[element_index]

            # Add an extra dimension to the mask and convert it to a FourierSeries
            mask = jnp.expand_dims(mask, -1)
            mask = FourierSeries(mask, self.domain)
                
            # print(f"delay_in_s: {delay_in_s}")
            
            # Apply beamforming delay to the signal
            delay_samples = int(delay_in_s / self.dt)
            # print(f"delay_samples: {delay_samples}")
            delayed_signal = jnp.roll(signal, delay_samples) if signal is not None else None

            element = DistributedTransducer(mask, center_pos, delayed_signal, self.domain, True)            
            elements.append(element)

        return elements

    @staticmethod
    def calculate_beamforming_delays(
        source_positions: np.ndarray, 
        target_point: np.ndarray, 
        sound_speed: float, 
        dx: float, 
        dy: float
    ) -> np.ndarray:
        """
        Calculate the beamforming delays for the signal sources based on the target point.

        Parameters:
        - source_positions: Array of signal source coordinates with shape (num_sources, 2). In meters.
        - target_point: Array representing the target point coordinates with shape (2,). In meters.
        - sound_speed: Speed of sound in the medium.
        - dx: Grid spacing in the x-direction.
        - dy: Grid spacing in the y-direction.

        Returns:
        - delays: Array of beamforming delays for each signal source with shape (num_sources,) in seconds.
        """
        # Calculate distances from signal sources to the target point
        distances = jnp.sqrt(((source_positions[:, 0] - target_point[0]))**2 +
                            ((source_positions[:, 1] - target_point[1]))**2)

        # Calculate sound wave travel times
        times = distances / sound_speed

        # Normalize the times by subtracting the minimum time
        delays = times - np.min(times)

        return delays

    @property
    def target_point(self) -> jnp.ndarray:
        """
        Calculate the target point position based on the transducer center, steering angle, and distance from the center.

        Returns:
            jnp.ndarray: The target point position as a numpy array [target_x, target_y]. In meters.
        """
        # Convert the steering angle from degrees to radians
        steering_angle_rad = jnp.deg2rad(self.steering_angle)

        # Calculate the target point position
        # TODO: Handle 3D case
        target_point = jnp.array([
            self.position[0] - jnp.sin(steering_angle_rad) * self.focus_distance, 
            self.position[1] + jnp.cos(steering_angle_rad) * self.focus_distance
        ])

        return target_point
    
    def scan_line(self, sensor_data: jnp.ndarray) -> jnp.ndarray:
        """
        Apply beamforming to the sensor data to form a single scan line.

        Args:
            sensor_data: A 2D array of shape (time_samples, num_elements) containing the sensor data.

        Returns:
            A 1D array representing the formed scan line.
        """
        time_samples, num_elements = sensor_data.shape

        delays_in_s = TransducerArray.calculate_beamforming_delays(
            source_positions=np.array([element.center_pos for element in self.elements if element.is_active]),
            target_point=self.target_point,
            sound_speed=self.sound_speed,
            dx=self.domain.dx[0],
            dy=self.domain.dx[1]
        )

        assert all(delay >= 0 for delay in delays_in_s), "All delays must be non-negative"
        assert len(delays_in_s) == num_elements, "Number of delays must match the number of elements"

        # Calculate the beamforming delays in samples
        delay_in_samples = jnp.round(delays_in_s / self.dt).astype(int)

        # Find the maximum delay
        max_delay = jnp.max(delay_in_samples)

        # Subtract the delays from the maximum delay to get the relative delays
        relative_delays = max_delay - delay_in_samples

        # Determine the required size for the padded_data
        padded_shape = (time_samples + max_delay, num_elements)

        # Create the padded_data matrix with the required size
        padded_data = jnp.zeros(padded_shape, dtype=sensor_data.dtype)

        # Copy the sensor_data into the padded_data matrix based on the relative delays
        for i in range(num_elements):
            padded_data = padded_data.at[relative_delays[i]:relative_delays[i]+time_samples, i].set(sensor_data[:, i])

        # Apply beamforming delays to each element by slicing the padded_data
        shifted_data = padded_data[:time_samples + max_delay, :]

        # Sum the shifted data along the element axis to get the beamformed data
        beamformed_data = jnp.sum(shifted_data, axis=1)

        return beamformed_data

    def scan_line_vectorized(self, sensor_data, return_debug=False):
        """
        Apply beamforming to the sensor data to form a single scan line using vectorized operations.
        Optionally return debugging information.

        Args:
            sensor_data: A 2D array of shape (time_samples, num_elements) containing the sensor data.
            return_debug: Boolean to indicate if debug information should be returned.

        Returns:
            A 1D array representing the formed scan line, optionally returns debugging info.
        """
        time_samples, num_elements = sensor_data.shape

        # Calculate the beamforming delays in samples
        delays_samples = jnp.round(self._calculate_beamforming_delays() / self.dt).astype(int)

        # Construct an array of indices to gather data after applying delays
        row_indices = jnp.arange(time_samples)[:, None] - delays_samples[None, :]
        # Clip the indices so they are within valid range
        row_indices = jnp.clip(row_indices, 0, time_samples - 1)

        # Gather the data using the constructed indices, each column corresponds to delayed data for each sensor
        delayed_data = sensor_data[row_indices, jnp.arange(num_elements)]

        # Sum across the columns to perform the beamforming
        beamformed_data = jnp.sum(delayed_data, axis=1)

        if return_debug:
            return beamformed_data, row_indices
        return beamformed_data

    

    def set_active_elements(self, active_elements: Union[List[Union[bool, int]], np.ndarray]) -> None:
        """
        Set the active/inactive status of the transducer elements.

        Args:
            active_elements: A boolean or integer array or list indicating the active/inactive status of each element.
                             True or 1 for active, False or 0 for inactive.
        """
        assert len(active_elements) == self.num_elements, "The length of active_elements must match the total number of elements."

        for i, active in enumerate(active_elements):
            if isinstance(active, int):
                assert active in [0, 1], "Integer values for active_elements must be either 0 or 1."
                self.elements[i] = self.elements[i].set_is_active(bool(active))
            else:
                self.elements[i] = self.elements[i].set_is_active(active)

    def __call__(self, p: Field, u: Field, rho: Field):
        """
        Compute the output for each transducer element.

        Args:
            p: The pressure field.
            u: The velocity field (not used in this implementation).
            rho: The density field (not used in this implementation).

        Returns:
            An array of outputs for each transducer element.
        """
        element_outputs = []
        for element in self.elements:
            if element.is_active:
                element_output = element(p, u, rho)
                element_outputs.append(element_output)
        return jnp.array(element_outputs)

    def set_signal(self, signal):
        """
        Set the same signal for all transducer elements.

        Args:
            signal: The signal to be set for all elements.
        """
        for idx, element in enumerate(self.elements):
            self.elements[idx] = element.set_signal(signal)

    def on_grid(self, n):
        """
        Compute the wavefield produced by the transducer array on the grid.

        Args:
            n: The time index for the signals.

        Returns:
            The wavefield produced by the transducer array.
        """
        wavefield = jnp.expand_dims(jnp.zeros(self.domain.N), -1)
        
        for element in self.elements:
            wavefield += element.on_grid(n)
        
        return wavefield     
    
    def get_segmentation_mask(self) -> np.array:
        """
        Generates a segmentation mask of the transducer elements on the grid.

        Returns:
            An array where each transducer's footprint is marked with its index+1 and non-transducer areas are 0.
        """
        # Create an empty mask with the same shape as the domain
        segmentation_mask = np.zeros(self.domain.N)

        # Iterate over each element and set its index in the segmentation mask
        for idx, element in enumerate(self.elements):
            mask = element.mask.on_grid  # Assuming the mask is stored in a JAX array within each DistributedTransducer
            indices = jnp.where(mask == 1)[:-1]
            segmentation_mask[indices] = idx + 1

        return segmentation_mask

@dataclass
class TimeHarmonicSource:
    r"""TimeHarmonicSource dataclass

    Attributes:
      domain (Domain): domain
      amplitude (Field): The complex amplitude field of the sources
      omega (float): The angular frequency of the sources

    """
    amplitude: Field
    omega: Union[float, Field]
    domain: Domain

    def on_grid(self, t=0.0):
        r"""Returns the complex field corresponding to the
        sources distribution at time $t$.
        """
        return self.amplitude * jnp.exp(1j * self.omega * t)

    @staticmethod
    def from_point_sources(domain, x, y, value, omega):
        src_field = jnp.zeros(domain.N, dtype=jnp.complex64)
        src_field = src_field.at[x, y].set(value)
        return TimeHarmonicSource(src_field, omega, domain)


@register_pytree_node_class
class Sensors:
    r"""Sensors structure

    Attributes:
      positions (Tuple[List[int]]): sensors positions

    !!! example

      ```python
      x_pos = [10,20,30,40]
      y_pos = [30,30,30,30]
      sensors = geometry.Sensors(positions=(x_pos, y_pos))
      ```
    """

    positions: Tuple[tuple]

    def __init__(self, positions):
        self.positions = positions

    def tree_flatten(self):
        children = None
        aux = (self.positions, )
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux, children):
        positions = aux[0]
        return cls(positions)

    def to_binary_mask(self, N):
        r"""
        Convert sensors to binary mask
        Args:
          N (Tuple[int]): grid size

        Returns:
          jnp.ndarray: binary mask
        """
        mask = jnp.zeros(N)
        for i in range(len(self.positions[0])):
            mask = mask.at[self.positions[0][i], self.positions[1][i]].set(1)
        return mask > 0

    def __call__(self, p: Field, u: Field, rho: Field):
        r"""Returns the values of the field u at the sensors positions.
        Args:
          u (Field): The field to be sampled.
        """
        if len(self.positions) == 1:
            return p.on_grid[self.positions[0]]
        elif len(self.positions) == 2:
            return p.on_grid[self.positions[0],
                             self.positions[1]]    # type: ignore
        elif len(self.positions) == 3:
            return p.on_grid[self.positions[0], self.positions[1],
                             self.positions[2]]    # type: ignore
        else:
            raise ValueError(
                "Sensors positions must be 1, 2 or 3 dimensional. Not {}".
                format(len(self.positions)))


def bli_function(x0: jnp.ndarray,
                 x: jnp.ndarray,
                 n: int,
                 include_imag: bool = False) -> jnp.ndarray:
    """
    The function used to compute the band limited interpolation function.

    Args:
        x0 (jnp.ndarray): Position of the sensors along the axis.
        x (jnp.ndarray): Grid positions.
        n (int): Size of the grid
        include_imag (bool): Include the imaginary component?

    Returns:
        jnp.ndarray: The values of the function at the grid positions.
    """
    dx = jnp.where(
        (x - x0[:, None]) == 0, 1,
        x - x0[:, None])    # https://github.com/google/jax/issues/1052
    dx_nonzero = (x - x0[:, None]) != 0

    if n % 2 == 0:
        y = jnp.sin(jnp.pi * dx) / \
            jnp.tan(jnp.pi * dx / n) / n
        y -= jnp.sin(jnp.pi * x0[:, None]) * jnp.sin(jnp.pi * x) / n
        if include_imag:
            y += 1j * jnp.cos(jnp.pi * x0[:, None]) * jnp.sin(jnp.pi * x) / n
    else:
        y = jnp.sin(jnp.pi * dx) / \
            jnp.sin(jnp.pi * dx / n) / n

    # Deal with case of precisely on grid.
    y = y * jnp.all(dx_nonzero, axis=1)[:, None] + (1 - dx_nonzero) * (
        ~jnp.all(dx_nonzero, axis=1)[:, None])
    return y


@register_pytree_node_class
class BLISensors:
    """ Band-limited interpolant (off-grid) sensors.

    Args:
        positions (Tuple of List of float): Sensor positions.
        n (Tuple of int): Grid size.

    Attributes:
        positions (Tuple[jnp.ndarray]): Sensor positions
        n (Tuple[int]): Grid size.
    """

    positions: Tuple[jnp.ndarray]
    n: Tuple[int]

    def __init__(self, positions: Tuple[jnp.ndarray], n: Tuple[int]):
        self.positions = positions
        self.n = n

        # Calculate the band-limited interpolant weights if not provided.
        x = jnp.arange(n[0])[None]
        self.bx = jnp.expand_dims(bli_function(positions[0], x, n[0]),
                                  axis=range(2, 2 + len(n)))

        if len(n) > 1:
            y = jnp.arange(n[1])[None]
            self.by = jnp.expand_dims(bli_function(positions[1], y, n[1]),
                                      axis=range(2, 2 + len(n) - 1))
        else:
            self.by = None

        if len(n) > 2:
            z = jnp.arange(n[2])[None]
            self.bz = jnp.expand_dims(bli_function(positions[2], z, n[2]),
                                      axis=range(2, 2 + len(n) - 2))
        else:
            self.bz = None

    def tree_flatten(self):
        children = self.positions,
        aux = self.n,
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)

    def __call__(self, p: Field, u, v):
        r"""Returns the values of the field p at the sensors positions.
        Args:
          p (Field): The field to be sampled.
        """
        if len(self.positions) == 1:
            # 1D
            pw = jnp.sum(p.on_grid[None] * self.bx, axis=1)
            return pw
        elif len(self.positions) == 2:
            # 2D
            pw = jnp.sum(p.on_grid[None] * self.bx, axis=1)
            pw = jnp.sum(pw * self.by, axis=1)
            return pw
        elif len(self.positions) == 3:
            # 3D
            pw = jnp.sum(p.on_grid[None] * self.bx, axis=1)
            pw = jnp.sum(pw * self.by, axis=1)
            pw = jnp.sum(pw * self.bz, axis=1)
            return pw
        else:
            raise ValueError(
                "Sensors positions must be 1, 2 or 3 dimensional. Not {}".
                format(len(self.positions)))


@register_pytree_node_class
class TimeAxis:
    r"""Temporal vector to be used for acoustic
    simulation based on the pseudospectral method of
    [k-Wave](http://www.k-wave.org/)
    Attributes:
      dt (float): time step
      t_end (float): simulation end time
    """
    dt: float
    t_end: float

    def __init__(self, dt, t_end):
        self.dt = dt
        self.t_end = t_end

    def tree_flatten(self):
        children = (None, )
        aux = (self.dt, self.t_end)
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux, children):
        dt, t_end = aux
        return cls(dt, t_end)

    @property
    def Nt(self):
        r"""Returns the number of time steps"""
        return np.ceil(self.t_end / self.dt)

    def to_array(self):
        r"""Returns the time-axis as an array"""
        out_steps = jnp.arange(0, self.Nt, 1)
        return out_steps * self.dt

    @staticmethod
    def from_medium(medium: Medium, cfl: float = 0.3, t_end=None):
        r"""Construct a `TimeAxis` object from `kGrid` and `Medium`
        Args:
          grid (kGrid):
          medium (Medium):
          cfl (float, optional):  The [CFL number](http://www.k-wave.org/). Defaults to 0.3.
          t_end ([float], optional):  The final simulation time. If None,
              it is automatically calculated as the time required to travel
              from one corner of the domain to the opposite one.
        """
        dt = cfl * min(medium.domain.dx) / functional(medium.sound_speed)(
            np.max)
        if t_end is None:
            t_end = np.sqrt(
                sum((x[-1] - x[0])**2
                    for x in medium.domain.spatial_axis)) / functional(
                        medium.sound_speed)(np.min)
        return TimeAxis(dt=float(dt), t_end=float(t_end))
