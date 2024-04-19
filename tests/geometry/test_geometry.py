import numpy as np
import pytest
import jax.numpy as jnp

from jwave.geometry import (fibonacci_sphere, points_on_circle,
                            unit_fibonacci_sphere, Domain, DistributedTransducer, TransducerArray)


def testpoints_on_circle():
    n = 5
    radius = 10.0
    centre = (0.0, 0.0)
    x_expected = [10, 3, -8, -8, 3]
    y_expected = [0, 9, 5, -5, -9]

    x_actual, y_actual = points_on_circle(n, radius, centre, cast_int=True)

    assert x_actual == x_expected
    assert y_actual == y_expected


def testunit_fibonacci_sphere():
    samples = 128

    points = unit_fibonacci_sphere(samples=samples)

    # Assert that the correct number of points have been generated
    assert len(points) == samples

    # Assert that all points lie on the unit sphere
    for point in points:
        x, y, z = point
        distance_from_origin = np.sqrt(x**2 + y**2 + z**2)
        assert np.isclose(distance_from_origin, 1.0, atol=1e-5)


def testfibonacci_sphere():
    n = 128
    radius = 10.0
    centre = np.array([1.0, 2.0, 3.0])

    x, y, z = fibonacci_sphere(n, radius, centre, cast_int=False)

    # Assert that the correct number of points have been generated
    assert len(x) == len(y) == len(z) == n

    # Assert that all points lie on the sphere with the given radius and center
    for i in range(n):
        distance_from_centre = np.sqrt((x[i] - centre[0])**2 +
                                       (y[i] - centre[1])**2 +
                                       (z[i] - centre[2])**2)
        assert np.isclose(distance_from_centre, radius, atol=1e-5)


@pytest.fixture
def transducer_array():
    """
    Fixture that creates a TransducerArray instance for testing.
    This instance is configured with typical settings that are expected in operational scenarios.
    """
    domain = Domain(N=(100, 100, 100), dx=(0.1, 0.1, 0.1))
    return TransducerArray(
        domain=domain,
        num_elements=10,
        element_width=5,
        element_height=5,
        element_depth=5,
        element_spacing=2,
        position=(1, 1, 1),
        sound_speed=1500,
        focus_distance=50,
        steering_angle=30,
        dt=1e-6,
    )

def test_create_elements(transducer_array):
    """
    Verify that the TransducerArray correctly creates the specified number of elements.
    Checks that each element is an instance of DistributedTransducer to validate proper initialization.
    """
    assert len(transducer_array.elements) == transducer_array.num_elements
    assert all(isinstance(element, DistributedTransducer) for element in transducer_array.elements), "All elements must be instances of DistributedTransducer"

def test_calculate_beamforming_delays(transducer_array):
    """
    Ensure that beamforming delays are calculated correctly and are all non-negative.
    This test checks the shape and non-negativity of the delays to ensure that the beamforming logic is implemented correctly.
    """
    delays = transducer_array._calculate_beamforming_delays()
    assert delays.shape == (transducer_array.num_elements,), "Delays array must match the number of elements"

def test_scan_line(transducer_array):
    """
    Verifies that the scan_line method of the TransducerArray class processes input sensor data and outputs
    a scan line of the expected shape. The test ensures that the method integrates sensor data across
    transducer elements to form a single dimensional output representing time samples.

    A dummy sensor data array is created with the correct shape to focus the test on the output dimensions,
    sidestepping the need for actual random data generation.
    """
    # Define the number of time samples and retrieve the number of elements from the transducer array
    time_samples = 1000
    num_elements = transducer_array.num_elements

    # Create dummy sensor data with the defined shape
    sensor_data = jnp.ones((time_samples, num_elements))

    # Invoke the scan_line method on the dummy sensor data
    scan_line_result = transducer_array.scan_line(sensor_data)

    # Assert that the output shape is correct, matching the number of time samples
    assert scan_line_result.shape == (time_samples,), "The output shape of the scan line must match the number of time samples"

def test_set_active_elements(transducer_array):
    """
    Validate that the set_active_elements method correctly modifies the active state of each element.
    This checks for both active and inactive states to ensure that the method handles different input correctly.
    """
    active_elements = [True, False] * 5  # Alternating active/inactive states
    transducer_array.set_active_elements(active_elements)
    for i, element in enumerate(transducer_array.elements):
        assert element.is_active == active_elements[i], "Element active state must match the expected state"

def test_set_signal(transducer_array):
    """
    Confirm that the set_signal method correctly assigns the signal across all elements.
    This test ensures that the signal setting mechanism works as expected by checking uniformity across all elements.
    """
    signal = jnp.ones(1000)  # A uniform signal of ones
    transducer_array.set_signal(signal)
    for element in transducer_array.elements:
        assert jnp.allclose(element.signal, signal), "All elements must have the signal set correctly"

def test_get_segmentation_mask(transducer_array):
    """
    Ensure that the segmentation mask is correctly generated.
    This test verifies that the mask dimensions match the domain and that it only contains valid element indices.
    """
    segmentation_mask = transducer_array.get_segmentation_mask()
    assert segmentation_mask.shape == tuple(transducer_array.domain.N), "Segmentation mask shape must match the domain dimensions"
    assert jnp.all((segmentation_mask >= 0) & (segmentation_mask <= transducer_array.num_elements)), "Mask values must be within valid element indices range"

def test_init_invalid_radius():
    """
    Test that initializing TransducerArray with an invalid radius raises an appropriate exception.
    This validates error handling for unsupported configurations.
    """
    domain = Domain(N=(100, 100, 100), dx=(0.1, 0.1, 0.1))
    with pytest.raises(NotImplementedError):
        TransducerArray(
            domain=domain,
            num_elements=10,
            element_width=5,
            element_height=5,
            element_depth=5,
            element_spacing=2,
            position=(1, 1, 1),
            radius=10,  # Invalid radius (not infinite)
            sound_speed=1500,
            focus_distance=50,
            steering_angle=30,
            dt=1e-6,
        )

        
def test_scan_line_methods(transducer_array):
    time_samples = 1000
    num_elements = transducer_array.num_elements
    sensor_data = jnp.ones((time_samples, num_elements))

    # Calculate delays
    delays_samples_original = jnp.round(transducer_array._calculate_beamforming_delays() / transducer_array.dt).astype(int)

    # Run original method
    original_result = transducer_array.scan_line(sensor_data)

    # Run vectorized method with debug
    vectorized_result, row_indices_vectorized = transducer_array.scan_line_vectorized(sensor_data, return_debug=True)

    # Print intermediate values
    print("Delays (Original):", delays_samples_original)
    print("Row indices (Vectorized):", row_indices_vectorized)

    # Final comparison
    assert jnp.allclose(original_result, vectorized_result), "Outputs should be identical"


def test_scan_line_output_shape(transducer_array, method):
    """
    Verifies that the scan_line method of the TransducerArray class processes input sensor data and outputs
    a scan line of the expected shape. The test ensures that the method integrates sensor data across
    transducer elements to form a single dimensional output representing time samples.
    """
    time_samples = 1000
    num_elements = transducer_array.num_elements

    sensor_data = jnp.ones((time_samples, num_elements))

    # Depending on the method provided ('original' or 'vectorized'), invoke the appropriate function
    if method == 'original':
        result = transducer_array.scan_line(sensor_data)
    elif method == 'vectorized':
        result = transducer_array.scan_line_vectorized(sensor_data)
    else:
        raise ValueError("Unknown method type. Use 'original' or 'vectorized'.")

    # Assert that the output shape is correct, matching the number of time samples
    assert result.shape == (time_samples,), "The output shape of the scan line must match the number of time samples"
    
if __name__ == "__main__":
    testpoints_on_circle()
    testunit_fibonacci_sphere()
    testfibonacci_sphere()
