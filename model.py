#%%
import os
import rasterio
import numpy as np
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
import numpy as np

#%%
# Path to the folder containing the TIFF files
folder_path = 'BJ_urbanTS'

# Dictionary to store the data, with keys as 'data_year'
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

# Now you have a dictionary `data_dict` where each key is 'data_year' and the value is the band1 data.

# Path to the folder containing the TIFF files
folder_path = 'spatialProxy'

# Dictionary to store the data, with keys as 'KEYVALUE'
proxy_dict = {}

# Iterate over each file in the directory
for filename in os.listdir(folder_path):
    if filename.startswith('SpatialProxy_') and filename.endswith('.tif'):
        # Extract KEYVALUE from the filename
        # Remove prefix "SpatialProxy_" and suffix ".tif"
        keyvalue = filename[len('SpatialProxy_'):-len('.tif')]
        
        # Full path to the file
        file_path = os.path.join(folder_path, filename)
        
        # Open the dataset
        with rasterio.open(file_path) as dataset:
            # Read the first band
            band1 = dataset.read(1)
            
            # Store the data in the dictionary with KEYVALUE as the key
            proxy_dict[keyvalue] = band1

# %%
#Define the logistic CA model

#calculate suitability
def calculate_suitability(i,j):
    return 1 / (1 + np.exp(-linear_predictor(i,j)))

def neighborhood_density(array, i, j):
    """
    Calculate the density of 1s around a specific cell in a numpy array,
    excluding the cell itself from the calculation.

    :param array: numpy array
    :param i: row index of the cell
    :param j: column index of the cell
    :return: density of 1s in the neighborhood
    """
    # Define the size of the neighborhood around the cell
    neighborhood_size = 5  # 5x5 neighborhood

    # Calculate the start and end indices for rows and columns
    row_start = max(0, i - 2)
    row_end = min(array.shape[0], i + 3)
    col_start = max(0, j - 2)
    col_end = min(array.shape[1], j + 3)

    # Extract the neighborhood
    neighborhood = array[row_start:row_end, col_start:col_end]

    # Calculate the number of 1s and exclude the central cell if it's in the range
    if 0 <= i - row_start < neighborhood.shape[0] and 0 <= j - col_start < neighborhood.shape[1]:
        # Temporarily set the central cell to zero for counting 1s
        original_value = neighborhood[i - row_start, j - col_start]
        neighborhood[i - row_start, j - col_start] = 0
        num_ones = np.sum(neighborhood == 1)
        # Restore the original value
        neighborhood[i - row_start, j - col_start] = original_value
        # Decrease total_cells by 1 since the central cell is excluded
        total_cells = neighborhood.size - 1
    else:
        num_ones = np.sum(neighborhood == 1)
        total_cells = neighborhood.size

    # Calculate the density of 1s
    density = num_ones / total_cells if total_cells > 0 else 0

    return density


def linear_predictor(i, j, coefficients):
    # Start with the intercept
    linear_pred = coefficients['b0']
    
    # Iterate over each coefficient and corresponding proxy key
    for idx, key in enumerate(proxy_dict.keys(), start=1):
        # Ensure the coefficient for this key exists
        coeff_key = f'b{idx}'
        if coeff_key in coefficients:
            # Add to the linear predictor the product of coefficient and the value at (i, j) in the proxy array
            linear_pred += coefficients[coeff_key] * proxy_dict[key][i, j]
        else:
            raise KeyError(f"Coefficient {coeff_key} not found in coefficients dictionary.")
    
    return linear_pred

def stochastic_perturbation():
    alpha = 1
    lambda_val = np.random.uniform(low=1e-10, high=1)
    return 1 + np.log(lambda_val) * alpha

def land_constraint(i,j):
    return 1

def development_probability(i, j, array, coefficients):
    pg_ij = linear_predictor(i,j, coefficients)
    omega_ij = neighborhood_density(array, i, j)
    ra_ij = stochastic_perturbation()
    land_ij = land_constraint(i,j)
    return pg_ij * omega_ij * ra_ij * land_ij


# %%
#First we need to figure out the different coefficients and keep our fingers crossed that they correspond to those in the paper

#data_dict['data_2013'] = data_dict['data_2013'][:-1, :-1]  # Remove the last row and column

# Get arrays from start and end years
start_array = data_dict['data_1984']
end_array = data_dict['data_2013']
proxy_array = proxy_dict['distance_to_Tiananmen_Square']

# Define the no-data value
nodata_value = -3e+38

# Create individual valid masks for each proxy array and combine them
all_valid_mask = np.ones_like(start_array, dtype=bool)  # Start with all True
for key, array in proxy_dict.items():
    # Create a mask where data is valid (above no-data value)
    valid_mask = (array > nodata_value)
    # Combine with the global valid mask
    all_valid_mask &= valid_mask  # Update the overall valid mask with logical AND

# Now apply this combined mask to filter out indices where all conditions are met
# Update your primary valid_mask to include this all_valid_mask
primary_valid_mask = ((start_array == 0) & ((end_array == 1) | (end_array == 0)) & all_valid_mask)

# Find indices where primary_valid_mask is True
valid_indices = np.transpose(np.nonzero(primary_valid_mask))

# Sample 10,000 random indices from valid_indices
if len(valid_indices) < 5000:
    raise ValueError("Not enough valid cells to sample.")
sampled_indices = valid_indices[np.random.choice(len(valid_indices), size=5000, replace=False)]

# Collect changes (0 if unchanged, 1 if changed to urban)
changes = (start_array[sampled_indices[:, 0], sampled_indices[:, 1]] != 
           end_array[sampled_indices[:, 0], sampled_indices[:, 1]]).astype(int)
# Initialize features matrix and response vector
num_predictors = len(proxy_dict)
features = np.zeros((5000, num_predictors))
response = changes

# Populate features matrix
for i, key in enumerate(proxy_dict.keys()):
    features[:, i] = proxy_dict[key][sampled_indices[:, 0], sampled_indices[:, 1]]

# Create and train the logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(features, response)

# Model is now trained and can predict urbanization likelihood

# Accessing the coefficients
coefficients = model.coef_[0]  # This will give you the array of coefficients for the features
intercept = model.intercept_[0]  # This gives you the intercept term

# Print coefficients and intercept
print("Intercept:", intercept)
print("Coefficients:")
for i, coef in enumerate(coefficients):
    print(f"Coefficient for {list(proxy_dict.keys())[i]}: {coef}")
# %%
