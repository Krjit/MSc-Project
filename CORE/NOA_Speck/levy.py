import numpy as np
from scipy.special import gamma

def levy(n, m, beta):
    num = gamma(1 + beta) * np.sin(np.pi * beta / 2) # used for Numerator
    den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2) # used for Denominator
    sigma_u = (num / den) ** (1 / beta) # Standard deviation
    u = np.random.normal(0, sigma_u, size=(n, m))
    v = np.random.normal(0, 1, size=(n, m))
    z = u / np.abs(v) ** (1 / beta)
    return z
