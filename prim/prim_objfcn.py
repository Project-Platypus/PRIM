# The PRIM module for Python is a standalone version of the Patient Rule
# Induction Method (PRIM) algorithm implemented in the EMA Workbench by Jan
# Kwakkel, which is itself derived from the sdtoolkit R package developed by
# RAND Corporation.  This standalone version of PRIM was created and maintained
# by David Hadka.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import, division

import numpy as np
from .exceptions import PrimError

def lenient1(y_old, y_new):
    """The default objective function (peeling criteria) used by PRIM.
    
    The basic idea is that the gain in mean is divided by the loss in mass.
    
    .. math::
        
        obj = \frac
             {\text{ave} [y_{i}\mid x_{i}\in{B-b}] - \text{ave} [y\mid x\in{B}]}
             {|n(y_{i})-n(y)|}
    
    where :math:`B-b` is the set of candidate new boxes, :math:`B`  the old box
    and :math:`y` are the y values belonging to the old box.  :math:`n(y_{i})`
    and :math:`n(y)` are the cardinality of :math:`y_{i}` and :math:`y`
    respectively. So, this objective function looks for the difference between
    the mean of the old box and the new box, divided by the change in the
    number of data points in the box. This objective function offsets a problem
    in case of categorical data where the normal objective function often
    results in boxes mainly based on the categorical data.  This is based on
    equation 14.5 in Friedman and Fisher (1999).
    
    This function can cope with continuous, integer, and categorical
    uncertainties.
    
    Parameters
    ----------
    y_old : ndarray
        the y values in the original box
    y_new : ndarray
        the y values in the new box
            
    Returns
    -------
    the score, where values closer to inf are preferred
    """
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
            raise PrimError('''mean is different {} vs {}, while shape is the same,
                                   this cannot be the case'''.format(mean_old, mean_new))
    return obj

def lenient2(y_old, y_new):
    """An alternative objective function (peeling criteria) for PRIM.
    
    Based on equation 14.6 in Friedman and Fishesr (1999), this peeling criteria
    minimizes the mean in the peeled subbox.
    
    Parameters
    ----------
    y_old : ndarray
        the y values in the original box
    y_new : ndarray
        the y values in the new box
            
    Returns
    -------
    the score, where values closer to inf are preferred
    """
    mean_old = np.mean(y_old)
    
    if y_new.shape[0]>0:
        mean_new = np.mean(y_new)
    else:
        mean_new = 0
        
    obj = 0
    if mean_old != mean_new:
        if y_old.shape==y_new.shape:
            raise PrimError('''mean is different {} vs {}, while shape is the same,
                                   this cannot be the case'''.format(mean_old, mean_new))
        
        change_mean = mean_new - mean_old
        change_mass = abs(y_old.shape[0]-y_new.shape[0])
        mass_new = y_new.shape[0]
            
        obj = mass_new * change_mean / change_mass
            
    return obj

def original(y_old, y_new):
    """The original objective function (peeling criteria) used by PRIM.
    
    Given by equation 14.1 in Friedman and Fisher (1999), this objective
    function seeks to maximize the improvement in the peeled subbox.
    
    Parameters
    ----------
    y_old : ndarray
        the y values in the original box
    y_new : ndarray
        the y values in the new box
            
    Returns
    -------
    the score, where values closer to inf are preferred
    """
    if y_new.shape[0] > 0:
        return np.mean(y_new)
    else:
        return -1    
