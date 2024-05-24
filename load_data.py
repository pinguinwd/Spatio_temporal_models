#%%
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import scipy.ndimage
import os

#%%
def load_tiff_to_numpy(tiff_path):
    """
    Loads a TIFF file and returns its data as a NumPy array.

    Parameters:
    tiff_path (str): The path to the TIFF file.

    Returns:
    numpy.ndarray: The raster data from the TIFF file.
    """
    # Open the TIFF file
    with rasterio.open(tiff_path) as src:
        # Read the raster data from the first band
        array_data = src.read(1)  # Change the band number if needed

    return array_data

def shape_path_to_array(shape_paths):
    """
    Takes a list of file paths leading to TIFF files and returns a list of NumPy arrays.

    Parameters:
    shape_paths (list of str): List of file paths to the TIFF files.

    Returns:
    list of numpy.ndarray: List of NumPy arrays containing the raster data from each TIFF file.
    """
    arrays = []
    for path in shape_paths:
        array_data = load_tiff_to_numpy(path)
        arrays.append(array_data)
    
    return arrays

def plot_shapefile_and_save(shapefile_path, output_path):
    """
    Plots a shapefile and saves the plot as a TIFF file.

    Parameters:
    shapefile_path (str): The path to the shapefile.
    output_path (str): The path to save the output TIFF file. Default is 'test.tif'.
    """
    # Load your shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)

    # Plotting
    fig, ax = plt.subplots()  # Create a figure and a set of subplots
    gdf.plot(ax=ax)           # Plot the GeoDataFrame on these axes
    ax.set_axis_off()         # Turn off the axis

    plt.savefig(output_path)  # Save the plot as a TIFF file

def shape_path_to_array(shape_paths):
    base_path_input = 'C:\\Users\\wanne\\OneDrive\\Documents\\STM\\Spatio_temporal_models\\Beijing_landuse\\Beijing-shp\\shape\\'
    base_path_output = 'C:\\Users\\wanne\\OneDrive\\Documents\\STM\\Spatio_temporal_models\\spatialProxy_extra\\SpatialProxy_'
    arrays = []
    for path in shape_paths:
        file_path = base_path_input + path + '.shp'
        output_path = base_path_output + path + '.tif'
        plot_shapefile_and_save(file_path, output_path)

def generate_year_dict(folder_path):
    data_dict = {}
    # Iterate over each file in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            # Extract the year from the filename
            year = filename.split('_')[1].split('.')[0]
            
            # Full path to the file
            file_path = os.path.join(folder_path, filename)
            
            # Open the dataset
            with rasterio.open(file_path) as dataset:
                # Read the first band
                band1 = dataset.read(1)
                
                # Store the data in the dictionary
                data_dict[f'data_{year}'] = band1
    return data_dict


def generate_features_dict(folder_path, name):
    proxy_dict = {}
    # Iterate over each file in the directory
    for filename in os.listdir(folder_path):
        if filename.startswith(name) and filename.endswith('.tif'):
            # Extract KEYVALUE from the filename
            # Remove prefix "SpatialProxy_" and suffix ".tif"
            keyvalue = filename[len(name):-len('.tif')]
            
            # Full path to the file
            file_path = os.path.join(folder_path, filename)
            
            # Open the dataset
            with rasterio.open(file_path) as dataset:
                # Read the first band
                band1 = dataset.read(1)
                
                # Store the data in the dictionary with KEYVALUE as the key
                proxy_dict[keyvalue] = band1
    return(proxy_dict)

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

def alter_map(og_dictionary, new_dictionary):
    # Sample values from the new_dictionary to define small_array
    small_array = next(iter(new_dictionary.values()))
    
    # Extract small array dimensions
    rows_small, cols_small = small_array.shape

    # Parameters for transformation
    x, y, zoom_factors, angle = 542, 170, 0.549, -7

    # Process each array in the og_dictionary
    modified_dict = {}
    for key, big_array in og_dictionary.items():
        # Resize and rotate the array
        big_array_transformed = resize_and_rotate_array(big_array, zoom_factors, angle)
        # Cut the array
        cut_array = big_array_transformed[x:x + rows_small, y:y + cols_small]
        # Store in the modified dictionary
        modified_dict[key] = cut_array

    return modified_dict


def generate_data(landuse, shapefiles):

    data_path = 'BJ_urbanTS'
    feature_path = 'spatialProxy'
    feature_path_extra = 'spatialProxy_extra'

    file_name = 'SpatialProxy_'

    data_dict = generate_year_dict(data_path)
    feature_dict = generate_features_dict(feature_path, file_name)

    landuse_dict = {}

    if landuse == True:
        shape_path_to_array(shapefiles)
        new_feature_dict = generate_features_dict(feature_path_extra, file_name)
        data_dict = alter_map(data_dict, new_feature_dict)
        feature_dict = alter_map(feature_dict, new_feature_dict)
        
        landuse_dict = {}
        for key in shapefiles:
            # Check if the key is present in new_feature_dict
            if key in new_feature_dict:
                # Add the key-value pair to landuse_dict
                landuse_dict[key] = new_feature_dict[key]
                # Remove the key-value pair from new_feature_dict
                del new_feature_dict[key]


    return data_dict, feature_dict, landuse_dict

