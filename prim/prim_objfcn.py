from __future__ import absolute_import, division

import numpy as np
from prim.exceptions import PRIMError

def lenient1(y_old, y_new):
    r'''
    the default objective function used by prim, instead of the original
    objective function, This function can cope with continuous, integer, 
    and categorical uncertainties. The basic idea is that the gain in mean
    is divided by the loss in mass. 
    
    .. math::
        
        obj = \frac
             {\text{ave} [y_{i}\mid x_{i}\in{B-b}] - \text{ave} [y\mid x\in{B}]}
             {|n(y_{i})-n(y)|}
    
    where :math:`B-b` is the set of candidate new boxes, :math:`B` 
    the old box and :math:`y` are the y values belonging to the old 
    box. :math:`n(y_{i})` and :math:`n(y)` are the cardinality of 
    :math:`y_{i}` and :math:`y` respectively. So, this objective 
    function looks for the difference between  the mean of the old 
    box and the new box, divided by the change in the  number of 
    data points in the box. This objective function offsets a problem 
    in case of categorical data where the normal objective function often 
    results in boxes mainly based on the categorical data.  
    
    TODO:: seems to be identical to 14.3 in friedman and fisher
    
    '''
    mean_old = np.mean(y_old)
    
    if y_new.shape[0] > 0:
        mean_new = np.mean(y_new)
    else:
        mean_new = 0
        
    obj = 0
    if mean_old != mean_new:
        if y_old.shape[0] > y_new.shape[0]:
            obj = (mean_new-mean_old) / (y_old.shape[0]-y_new.shape[0])
        elif y_old.shape[0] < y_new.shape[0]:
            obj = (mean_new-mean_old) / (y_new.shape[0]-y_old.shape[0])
        else:
            raise PRIMError('''mean is different {} vs {}, while shape is the same,
                                   this cannot be the case'''.format(mean_old, mean_new))
    return obj

def lenient2(y_old, y_new):
    '''
    
    friedman and fisher 14.6
    
    
    '''
    mean_old = np.mean(y_old)
    
    if y_new.shape[0]>0:
        mean_new = np.mean(y_new)
    else:
        mean_new = 0
        
    obj = 0
    if mean_old != mean_new:
        if y_old.shape==y_new.shape:
            raise PRIMError('''mean is different {} vs {}, while shape is the same,
                                   this cannot be the case'''.format(mean_old, mean_new))
        
        change_mean = mean_new - mean_old
        change_mass = abs(y_old.shape[0]-y_new.shape[0])
        mass_new = y_new.shape[0]
            
        obj = mass_new * change_mean / change_mass
            
    return obj

def original(y_old, y_new):
    ''' The original objective function: the mean of the data inside the 
    box'''
    
    if y_new.shape[0] > 0:
        return np.mean(y_new)
    else:
        return -1    

