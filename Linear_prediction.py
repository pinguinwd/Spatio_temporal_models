#%%
import numpy as np
from sklearn.linear_model import LogisticRegression
from fastkml import kml
from shapely.geometry import shape
from shapely.geometry import Polygon
from load_data import *

# %%
def find_coef_and_intercept(data_dict, proxy_dict, startyear, endyear):
    #First we need to figure out the different coefficients and keep our fingers crossed that they correspond to those in the paper

    if not (data_dict['data_2013'].shape == data_dict['data_2012'].shape):
        data_dict['data_2013'] = data_dict['data_2013'][:-1, :-1]  # Remove the last row and column

    first_value = 'data_' + str(startyear)
    last_value = 'data_' + str(endyear)

    # Get arrays from start and end years
    start_array = data_dict[first_value]
    end_array = data_dict[last_value]

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

    return coefficients, intercept


