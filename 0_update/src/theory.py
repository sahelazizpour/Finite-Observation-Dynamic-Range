# First part: Mean-field solution for T->inf
##################
import numpy as np

def mean_field_activity(lam, mu, h, dt=1):
    return mu*(1-np.exp(-h*dt))/(1-lam*(1-mu)-lam*mu*np.exp(-h*dt))