import numpy as np
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm
import psutil
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.image import AxesImage
import math

mm_scale = 1e-3
    
def plot_segmentation_mask(segmentation_mask, colormap='viridis', xlabel='X', ylabel='Y', title='Segmentation Mask', extent=[]):
    """
    Plots a 2D segmentation mask with different colors for each class.

    Args:
        segmentation_mask (numpy.ndarray): A 2D array representing the segmentation mask.
        colormap (str, optional): The name of the colormap to use. Default is 'viridis'.
        xlabel (str, optional): The label for the x-axis. Default is 'X'.
        ylabel (str, optional): The label for the y-axis. Default is 'Y'.
        title (str, optional): The title of the plot. Default is 'Segmentation Mask'.

    Returns:
        None
    """
    # Get the unique class labels from the segmentation mask
    class_labels = np.unique(segmentation_mask)

    # Create a custom colormap
    num_classes = len(class_labels)
    cmap = plt.cm.get_cmap(colormap, num_classes)
    colors = [cmap(i) for i in range(num_classes)]

    # Create a mapping between class labels and colors
    class_color_map = dict(zip(class_labels, colors))

    # Create a RGB image based on the segmentation mask and class colors
    rgb_image = np.zeros((*segmentation_mask.shape, 3))
    for label in class_labels:
        mask = segmentation_mask == label
        rgb_image[mask] = class_color_map[label][:3]  # Assign the corresponding color to the class

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Display the RGB image
    ax.imshow(rgb_image.transpose(1, 0, 2), extent=extent)

    # Create a custom legend
    legend_elements = [plt.Line2D([0], [0], marker='s', color='w', label=f'Class {label}',
                                  markerfacecolor=class_color_map[label], markersize=10)
                       for label in class_labels]
    ax.legend(handles=legend_elements, loc='upper right')

    # Set the axis labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Display the plot
    plt.show()
    

def process_hdr(pressure_data: np.ndarray, window_size: int = 5, batch_size: int = None, chunk_size: int = None) -> np.ndarray:
    """
    Apply HDR-like processing to pressure data.

    This function applies a High Dynamic Range (HDR) like processing to the input pressure data.
    The motivation behind this processing is to enhance the visibility of local variations in the
    pressure data by adapting to the local dynamic range.

    The mechanism of the HDR-like conversion involves the following steps:
    1. Compute the rolling mean and standard deviation of the pressure data using a sliding window.
       This captures the local average and variability of the pressure values.
    2. Standardize the pressure data by subtracting the local mean and dividing by the local standard
       deviation. This centers the data around zero and normalizes the local variability.
    3. Apply a nonlinear scaling to the standardized data, such as a logarithmic function. This
       compresses the dynamic range and enhances the visibility of local variations.

    The HDR-like processing is performed in chunks and batches to handle large datasets efficiently.
    The chunk size is automatically determined based on the available CPU memory, and the batch size
    is set to the chunk size if not provided.

    Parameters:
    - pressure_data (numpy.ndarray): The input pressure data array.
    - window_size (int): The size of the sliding window for computing local statistics (default: 5).
    - batch_size (int): The size of each batch for processing (default: None, set to chunk_size).
    - chunk_size (int): The size of each chunk for processing (default: None, automatically detected).

    Returns:
    - hdr_like_data (numpy.ndarray): The HDR-like processed pressure data array.
    """
    print("Processing HDR-like effect on pressure data...")
    
    # Get the dimensions of the pressure_data array
    time_steps, x_size, y_size = pressure_data.shape
    print(f"Data dimensions: time_steps={time_steps}, x_size={x_size}, y_size={y_size}")
    
    # Automatically detect available CPU memory and set chunk_size if not provided
    if chunk_size is None:
        mem = psutil.virtual_memory()
        free_mem = mem.available
        chunk_size = max(1, int(free_mem * 0.8 / (pressure_data.nbytes / time_steps)))
        print(f"Automatically detected chunk_size based on CPU memory: {chunk_size}")
    else:
        print(f"Using provided chunk_size: {chunk_size}")
    
    # Set batch_size to chunk_size if not provided
    if batch_size is None:
        batch_size = chunk_size
        print(f"Using chunk_size as batch_size: {batch_size}")
    else:
        print(f"Using provided batch_size: {batch_size}")
    
    # Create an empty array to store the HDR-like processed data
    print("Creating empty array for HDR-like processed data...")
    hdr_like_data = np.zeros_like(pressure_data)
    
    # Process the data in chunks along the time axis
    print("Processing data in chunks...")
    num_chunks = (time_steps + chunk_size - 1) // chunk_size
    with tqdm(total=num_chunks, unit='chunk') as progress_bar:
        for i in range(0, time_steps, chunk_size):
            print(f"\nProcessing chunk {i // chunk_size + 1} of {num_chunks}")
            chunk = pressure_data[i:i+chunk_size]
            print(f"Chunk shape: {chunk.shape}")
            
            # Process the chunk in batches
            print("Processing chunk in batches...")
            num_batches = (chunk.shape[0] + batch_size - 1) // batch_size
            for j in range(0, chunk.shape[0], batch_size):
                print(f"Processing batch {j // batch_size + 1} of {num_batches}")
                batch = chunk[j:j+batch_size]
                print(f"Batch shape: {batch.shape}")
                
                # Compute rolling mean and standard deviation for the current batch
                print("Computing rolling mean and standard deviation...")
                rolling_mean = uniform_filter1d(batch, size=window_size, axis=0, mode='reflect')
                rolling_std = np.sqrt(uniform_filter1d((batch - rolling_mean)**2, size=window_size, axis=0, mode='reflect'))
                
                # Scale the data based on local standard deviations (standardize)
                print("Standardizing the data...")
                epsilon = 1e-8
                standardized_data = (batch - rolling_mean) / (rolling_std + epsilon)
                
                # Apply a nonlinear scaling (e.g., logarithmic to mimic HDR effect)
                # print("Applying nonlinear scaling...")
                # hdr_like_data[i+j:i+j+batch_size] = np.log1p(np.abs(standardized_data))
                hdr_like_data[i+j:i+j+batch_size] = standardized_data
            
            progress_bar.update(1)
    
    print("\nHDR-like processing completed.")
    return hdr_like_data    
    
def create_pressure_animation(pressure_data: np.ndarray, domain_size: tuple, dt: float, num_steps_to_skip: int = 10, fps: int = 10, output_file: str = 'pressure_field_animation.mp4', swap_y_with_z: bool = False) -> None:
    """
    Creates an animation of the pressure field over time.

    Args:
        pressure_data (np.ndarray): The pressure data as a 3D NumPy array with dimensions (time, y, x) or (time, z, x) if swap_y_with_z is True.
        domain_size (tuple): A tuple representing the size of the domain in meters (x, y).
        dt (float): The time step size.
        num_steps_to_skip (int, optional): The number of time steps to skip between frames in the animation. Default is 10.
        fps (int, optional): The frames per second for the animation. Default is 10.
        output_file (str, optional): The output file path for the animation. Default is 'pressure_field_animation.mp4'.
        swap_y_with_z (bool, optional): Flag indicating whether to swap y with z. Default is False.

    Returns:
        None

    The function creates an animation of the pressure field over time using the provided pressure data.
    It visualizes the pressure field at each time step, skipping frames based on the specified num_steps_to_skip.
    The animation is saved as a video file at the specified output file path.

    Note:
        - The pressure data is expected to be in units of Pascals (Pa).
        - The animation uses a logarithmic scale for pressure visualization.
        - The colormap used for the pressure field is 'viridis'.
        - The animation may take some time to render, especially for large datasets.
        - The progress of the animation rendering is displayed using a progress bar.
    """
    # Find the minimum and maximum values across all time steps
    p_min = np.min(pressure_data)
    p_max = np.max(pressure_data)

    # Create a figure and a single plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Initialize the plot
    def init_plot():
        ax.clear()
        return ()

    # Update function for the animation
    def update(frame: int):
        time_step = frame * num_steps_to_skip
        p_at_time_step = pressure_data[time_step]

        x_vec_mm = np.array([0, domain_size[0]]) * mm_scale
        if swap_y_with_z:
            y_or_z_vec_mm = np.array([0, domain_size[1]]) * mm_scale
            y_or_z_label = 'z-position [mm]'
        else:
            y_or_z_vec_mm = np.array([0, domain_size[1]]) * mm_scale
            y_or_z_label = 'y-position [mm]'

        ax.clear()
        im = ax.imshow(1e-6 * p_at_time_step,
                       extent=[x_vec_mm[0], x_vec_mm[-1], y_or_z_vec_mm[0], y_or_z_vec_mm[-1]],
                       v_min=p_min,
                       v_max=p_max,
                       cmap=plt.get_cmap('viridis'))
        ax.set_xlabel('x-position [mm]')
        ax.set_ylabel(y_or_z_label)
        ax.set_title(f'Pressure Field (Time Step {time_step}, t = {time_step * dt})')
        return im,

    # Create the animation
    total_frames = len(pressure_data) // num_steps_to_skip
    ani = animation.FuncAnimation(fig, update, frames=total_frames, init_func=init_plot, blit=True)

    # Save the animation as a movie file with progress indicator
    writer = animation.FFMpegWriter(fps=fps)
    with tqdm(total=total_frames, unit='frame', desc='Rendering movie') as pbar:
        def progress_callback(current_frame, total_frames):
            pbar.update(1)
        ani.save(output_file, writer=writer, progress_callback=progress_callback)

def plot_pressure_fields(pressure_data: np.ndarray, domain_size: tuple, time_axis_dt: float, num_cols: int = 3, num_images: int = 9) -> None:
    """
    Plots pressure fields at different time steps in a grid of subplots.

    Args:
        pressure_data (np.ndarray): The pressure data as a NumPy array with dimensions (time, y, x).
        domain_size (tuple): A tuple representing the size of the domain in meters (x, y).
        time_axis_dt (float): The time step size in seconds.
        num_cols (int, optional): The number of columns in the subplot grid. Default is 3.
        num_images (int, optional): The total number of images to plot. Default is 9.

    Returns:
        None

    Notes:
        - The function assumes that the pressure data is provided in pascals (Pa).
        - The pressure fields are displayed using a logarithmic scale (1e-6 * pressure).
        - The x and y axes are labeled in millimeters.
        - The subplot titles indicate the time step and corresponding time value.
        - The function uses a non-linear transformation (`transform_pressure`) to modify the pressure values before plotting.
          You can customize this transformation as needed.
    """
    # Calculate the number of rows based on num_images and num_cols
    num_rows = math.ceil(num_images / num_cols)
    max_time_step = len(pressure_data) - 1

    # Generate evenly spaced time steps
    time_steps = np.linspace(0, max_time_step, num_images, dtype=int)

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))
    axes = axes.ravel()

    # Apply a non-linear transformation to the pressure values
    def transform_pressure(p):
        return p

    # Iterate over time steps and plot pressure field for each step
    for i, time_step in enumerate(time_steps):
        p_at_time_step = pressure_data[time_step]

        # Convert grid coordinates from meters to millimeters
        x_vec_mm = np.array([0, domain_size[0]]) * mm_scale
        z_vec_mm = np.array([0, domain_size[1]]) * mm_scale

        # Apply the non-linear transformation to the pressure values
        p_transformed = transform_pressure(p_at_time_step)

        im = axes[i].imshow(1e-6 * p_transformed, extent=[x_vec_mm[0], x_vec_mm[-1], z_vec_mm[0], z_vec_mm[-1]], aspect=None)

        axes[i].set_xlabel('y-position [mm]')
        axes[i].set_ylabel('x-position [mm]')
        axes[i].set_title(f'Pressure Field (Time Step {time_step}, t = {time_step * time_axis_dt})')

    plt.tight_layout()
    plt.show()