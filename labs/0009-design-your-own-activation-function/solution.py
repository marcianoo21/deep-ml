# --- cell 1 ---
import numpy as np

def activation(x):
    '''
    Apply an activation function element-wise.
    
    Args:
        x: numpy array of any shape (raw neuron outputs)
    
    Returns:
        numpy array of same shape (activated outputs)
    
    Requirements:
        - Must be non-linear (not just returning x)
        - Must work on arrays of any shape
        - Must be deterministic
    '''
  
    
    result = np.where(x>0, x**(1.5/2), -np.log(2) + np.log(1 + np.exp(x)))
    
    return result