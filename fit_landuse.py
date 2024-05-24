#%%
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import scipy.ndimage

#%%
# Example usage
shapefile_path = 'Beijing_landuse\\Beijing-shp\\shape\\railways.shp'

shapefile_path = 'Beijing_landuse\\Beijing-shp\\shape\\railways.shp'

# Load your shapefile into a GeoDataFrame
gdf = gpd.read_file(shapefile_path)

# Plotting
fig, ax = plt.subplots()  # Create a figure and a set of subplots
gdf.plot(ax=ax)           # Plot the GeoDataFrame on these axes
ax.set_axis_off()         # Turn off the axis

plt.savefig('test.tif')                # Display the plot

#%%
tiff_paths = ['spatialProxy\\SpatialProxy_distance_to_railways.tif','test.tif']

def load_tiff_to_numpy(tiff_path):
    # Open the TIFF file
    with rasterio.open(tiff_path) as src:
        # Read the raster data from the first band
        array_data = src.read(1)  # Change the band number if needed
        
        # Optionally, you can read all bands into a single array
        # If the TIFF has multiple bands and you want to load them all:
        # array_data = src.read()

    return array_data

og_array = load_tiff_to_numpy(tiff_paths[0])
new_array = load_tiff_to_numpy(tiff_paths[1])

#%%

og_array = np.where(og_array == 0, 1, 0)
counts = np.bincount(new_array.flatten())
most_frequent = np.argmax(counts)

# Step 2: Create the new array where the most frequent number is 0, others are 1
new_array = np.where(new_array == most_frequent, 0, 1)

# %%
def compare_arrays(big_array, small_array, x, y, zoom_factors, angle):
    # Calculate the dimensions of the small array

    big_array = resize_and_rotate_array(big_array, zoom_factors, angle)

    rows_small, cols_small = small_array.shape
    
    # Cut the big array to match the size of the small array
    cut_array = big_array[x:x + rows_small, y:y + cols_small]
    
    # Create a difference array based on the conditions described
    # Initialize the difference array
    difference_array = np.zeros_like(small_array)

    # Compute differences
    difference_array[(small_array == 0) & (cut_array == 1)] = 1  # Small is 0, Cut is 1
    difference_array[(small_array == 1) & (cut_array == 0)] = 2  # Small is 1, Cut is 0
    difference_array[(small_array == 1) & (cut_array == 1)] = 3 
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Old map', 'New map', 'Map overlay']
    arrays = [cut_array, small_array, difference_array]

    for ax, arr, title in zip(axes, arrays, titles):
        im = ax.imshow(arr, cmap='viridis', vmin=0, vmax=3)
        ax.set_title(title)
        ax.axis('off')
    
    # Add a color bar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label('Value')
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['Both 0', 'Absent in new, presents in Old', 'Present in New, Absent in Old', 'Both Present'])
    
    plt.savefig('compare.png')

def resize_and_rotate_array(input_array, zoom_factors, angle):
    """
    Resize and rotate an input array.
    
    Args:
    input_array (np.array): The array to transform.
    zoom_factors (tuple or list): Zoom factors for each dimension.
    angle (float): Rotation angle in degrees (counterclockwise).
    
    Returns:
    np.array: The transformed array.
    """
    
    # Use scipy.ndimage.zoom to resize the array
    resized_array = scipy.ndimage.zoom(input_array, zoom_factors, order=0)  # order=0 for nearest-neighbor interpolation
    
    # Use scipy.ndimage.rotate to rotate the resized array
    rotated_array = scipy.ndimage.rotate(resized_array, angle, reshape=False, order=0, mode='reflect')
    
    return rotated_array




compare_arrays(og_array, new_array, 542, 170, 0.549, -7)
# %%
