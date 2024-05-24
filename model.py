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

def run_model(data_dict, feat_dict, landuse_dict, first_year, last_year):
    #calculate how much cells need development
    #take the array from first year
    #calculate probability of development for each cell
    #develop x cells with highest probability 
    #calculate overal accuracy and FoM
    #return predicted array and oa and fom
    first_value = 'data_' + str(startyear)
    last_value = 'data_' + str(endyear)

    # Get arrays from start and end years
    start_array = data_dict[first_value]
    end_array = data_dict[last_value]
    pass


