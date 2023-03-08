import numpy as np
from numpy.linalg import norm

def rbf_kernel(u, v, sigma):
    return np.exp(-norm(u - v) ** 2 / (2 * sigma ** 2))