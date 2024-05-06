import numpy as np
from typing import List

def generate_active_elements(number_scan_lines: int, transducer_number_elements: int, window_size: int) -> np.ndarray:
    """
    Generate the active elements matrix for a linear array transducer.

    :param number_scan_lines: The number of scan lines.
    :param transducer_number_elements: The number of elements in the transducer.
    :param window_size: The size of the active window (must be an odd number).
    :return: A 2D numpy array representing the active elements for each scan line.
    """
    if number_scan_lines <= 0 or transducer_number_elements <= 0 or window_size <= 0:
        raise ValueError("number_scan_lines, transducer_number_elements, and window_size must be positive integers.")

    if window_size > transducer_number_elements:
        raise ValueError("window_size must be less than or equal to transducer_number_elements.")

    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd number.")

    active_elements = np.zeros((number_scan_lines, transducer_number_elements), dtype=int)

    window_centers = np.linspace(0, transducer_number_elements - 1, number_scan_lines, dtype=int)
    half_window_size = window_size // 2

    for scan_index, center in enumerate(window_centers):
        start_element = max(0, center - half_window_size)
        end_element = min(center + half_window_size + 1, transducer_number_elements)
        active_elements[scan_index, start_element:end_element] = 1

    return active_elements
