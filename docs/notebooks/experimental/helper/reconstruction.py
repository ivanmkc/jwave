import numpy as np

def smooth_shift(x: int, total_len: int, shift_len: int) -> np.ndarray:
    """
    Generates a 1-D numpy array where values smoothly shift from 0 to 1, centered at `x`.
    The shift is contained within the interval [x - shift_len/2, x + shift_len/2].

    Args:
        x (int): The center position of the shift in the array.
        total_len (int): The total length of the resulting array.
        shift_len (int): The length of the interval over which the transition occurs.

    Returns:
        ndarray: A 1-D numpy array with values smoothly transitioning from 0 to 1 around `x`.
    """
    # Create a 1-D array of indices from 0 to total_len-1
    indices = np.linspace(0, total_len - 1, total_len)
    
    # Calculate the center point of the shift
    center = x
    
    # Calculate the effective width of the shift
    # The width is adjusted for the sigmoid to effectively make the transition within the specified range
    width = shift_len / 10  # The divisor controls the steepness. Smaller values make it steeper.
    
    # Create a smooth shift function using the sigmoid function
    # Clipping the input to the exponential to avoid overflow
    z = -(indices - center) / width
    z = np.clip(z, -10, 10)  # Clipping z to avoid overflow in the exponential function
    shift = 1 / (1 + np.exp(z))
    
    return shift

def trim_scan_lines(scan_lines: np.ndarray, input_signal: np.ndarray, max_delay_in_samples: int) -> np.ndarray:
    """
    Process ultrasound scan line data by trimming unnecessary early data around the center of the input signal 
    and applying a Tukey window to smooth the transitions in the trimmed signal. This approach helps focus on 
    the ultrasound echoes and reduces signal processing artifacts.

    Args:
        scan_lines (np.ndarray): 2D array of ultrasound scan line data with shape (num_time_samples, num_scan_lines).
        input_signal (np.ndarray): The input signal array, expected to be non-padded and 1D.

    Returns:
        np.ndarray: The processed scan line data with Tukey window applied, emphasizing the useful echo signals.

    Raises:
        AssertionError: If the dimensions of the input data are not as expected.

    Example:
        >>> processed_data = process_ultrasound_data(scan_lines, input_signal)
    """

    # Ensure the input signal and scan lines are of correct dimensions
    assert input_signal.ndim == 1, "input_signal must be a 1D array"
    assert scan_lines.ndim == 2, "scan_lines must be a 2D array (num_time_samples, num_scan_lines)"

    num_time_samples = np.shape(scan_lines)[0]
    scan_line_win = smooth_shift(x=max_delay_in_samples+len(input_signal), total_len=num_time_samples, shift_len=len(input_signal)/2)
    
    # Apply the window to the trimmed scan line data
    scan_lines = scan_lines * scan_line_win[:, np.newaxis]

    return scan_lines[max_delay_in_samples:]