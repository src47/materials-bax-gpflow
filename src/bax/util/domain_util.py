"""
Utilities for domains (search spaces).
"""

import numpy as np


def unif_random_sample_domain(domain, n=1):
    """Draws a sample uniformly at random from domain (a list of tuple bounds)."""
    list_of_arr_per_dim = [np.random.uniform(dom[0], dom[1], n) for dom in domain]
    list_of_list_per_sample = [list(l) for l in np.array(list_of_arr_per_dim).T]
    return list_of_list_per_sample

# def unif_random_sample_domain_composition(domain, n=1):
#     """Draws a sample uniformly at random from domain (a list of tuple bounds)."""
#     list_of_arr_per_dim = [np.random.uniform(dom[0], dom[1], n) for dom in domain]
    
#     list_of_arr_per_dim_constraint = []
    
#     for i in range(len(list_of_arr_per_dim)):
#         if (list_of_arr_per_dim[i][0] + list_of_arr_per_dim[i][1]) <= 1.0:
#             list_of_arr_per_dim_constraint.append(list_of_arr_per_dim[i])
    
#     list_of_list_per_sample = [list(l) for l in np.array(list_of_arr_per_dim_constraint).T]
#     return list_of_list_per_sample

def project_to_domain(x, domain):
    """Project x, a list of scalars, to be within domain (a list of tuple bounds)."""

    # Assume input x is either a list or 1d numpy array
    assert isinstance(x, list) or isinstance(x, np.ndarray)
    if isinstance(x, np.ndarray):
        assert len(x.shape) == 1
        x_is_list = False
    else:
        x_is_list = True

    # Project x to be within domain
    x_arr = np.array(x).reshape(-1)
    min_list = [tup[0] for tup in domain]
    max_list = [tup[1] for tup in domain]
    x_arr_clip = np.clip(x_arr, min_list, max_list)

    # Convert to original type (either list or keep as 1d numpy array)
    x_return = list(x_arr_clip) if x_is_list else x_arr_clip

    return x_return
