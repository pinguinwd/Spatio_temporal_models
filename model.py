# %%
import numpy as np
from load_data import *
from Linear_prediction import *
#%%
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


def linear_predictor(i, j, coefficients, intercept, proxy_dict):
    # Start with the intercept
    linear_pred = intercept
    
    # Iterate over each coefficient and corresponding proxy key
    for idx, key in enumerate(proxy_dict.keys()):
        # Ensure we do not exceed the number of coefficients
        if idx < len(coefficients):
            # Add to the linear predictor the product of coefficient and the value at (i, j) in the proxy array
            linear_pred += coefficients[idx] * proxy_dict[key][i, j]
        else:
            raise IndexError(f"Index {idx} out of range for coefficients list.")
    
    return linear_pred

def stochastic_perturbation():
    alpha = 1
    lambda_val = np.random.uniform(low=1e-10, high=1)
    return 1 + np.log(lambda_val) * alpha

def land_constraint(i, j, land_use_dict):
    # If land_use_dict is empty, return 1
    if not land_use_dict:
        return 1
    
    # Iterate through each array in the land_use_dict
    for key, array in land_use_dict.items():
        # Check if the value at (i, j) is not 255
        if array[i, j] != 255:
            return 0

    # If all values in all arrays at (i, j) are 255, return 1
    return 1

def development_probability(i, j, array, proxy_dict, land_use_dict, coefficients, intercept):
    pg_ij = linear_predictor(i,j, coefficients, intercept, proxy_dict)
    omega_ij = neighborhood_density(array, i, j)
    ra_ij = stochastic_perturbation()
    land_ij = land_constraint(i,j, land_use_dict)
    return pg_ij * omega_ij * ra_ij * land_ij

def calculate_overall_accuracy(predicted_array, actual_array):
    # Calculate the number of correctly classified cells
    correctly_classified = np.sum(predicted_array == actual_array)
    # Calculate the total number of cells
    total_cells = predicted_array.size
    # Calculate overall accuracy
    overall_accuracy = correctly_classified / total_cells
    return overall_accuracy

def calculate_figure_of_merit(predicted_array, actual_array):
    # Identify changes between actual and predicted arrays
    change_predicted = (predicted_array != actual_array).astype(int)
    change_actual = (actual_array != np.roll(actual_array, shift=1, axis=0)).astype(int) | \
                    (actual_array != np.roll(actual_array, shift=1, axis=1)).astype(int)
    
    # Calculate areas
    A = np.sum((change_predicted == 1) & (change_actual == 1))  # Correctly predicted change
    B = np.sum((change_predicted == 0) & (change_actual == 1))  # Observed change but predicted non-change
    C = np.sum((change_predicted == 1) & (change_actual == 0))  # Predicted change but no change occurred
    D = np.sum((change_predicted == 0) & (change_actual == 0))  # Correctly predicted non-change
    
    # Calculate figure of merit
    if A + B + C + D == 0:
        figure_of_merit = 0
    else:
        figure_of_merit = A / (A + B + C + D)
    
    return figure_of_merit

def run_model(data_dict, proxy_dict, land_use_dict, first_year, last_year, coefficients, intercept):
    first_value = 'data_' + str(first_year)
    last_value = 'data_' + str(last_year)

    # Get arrays from start and end years
    start_array = data_dict[first_value]
    end_array = data_dict[last_value]

    necessary_development = np.sum(end_array) - np.sum(start_array)

    rows, columns = start_array.shape

    evaluation_list = []

    # Iterate over all cells in the start_array
    for i in range(rows):
        for j in range(columns):
            # Check the conditions
            if feature_dict['distance_to_Tiananmen_Square'][i, j] > 0 and start_array[i, j] == 0:
                # Evaluate the cell using development_probability function
                evaluation_score = development_probability(i, j, start_array, proxy_dict, land_use_dict, coefficients, intercept)

                # Store the score along with its coordinates
                evaluation_list.append((evaluation_score, (i, j)))

    # Sort the evaluations based on the scores in descending order
    evaluation_list.sort(reverse=True, key=lambda x: x[0])

    # Determine the number of cells to develop
    num_cells_to_develop = int(necessary_development)

    # Change the necessary amount of cells from 0 to 1 based on the highest evaluation scores
    predicted_array = start_array.copy()
    for idx in range(num_cells_to_develop):
        score, (i, j) = evaluation_list[idx]
        predicted_array[i, j] = 1

    oa = calculate_overall_accuracy(predicted_array, end_array)
    fom = calculate_figure_of_merit(predicted_array, end_array)

    return predicted_array, oa, fom
#%%

shapefiles = ['natural', 'waterways']
data_dict, feature_dict, landuse_dict = generate_data(True, shapefiles)

first_year = 1984
last_year = 2013

coefficients, intercept = find_coef_and_intercept(data_dict, feature_dict, first_year, last_year)
predicted_array, oa, fom = run_model(data_dict, feature_dict, landuse_dict, first_year, last_year, coefficients, intercept)
# %%
