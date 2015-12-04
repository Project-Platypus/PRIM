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

import copy
import math
import numpy as np
from .scenario_discovery_util import in_box

def get_quantile(data, quantile):
    """Computes the given quantile of the dataset.
    
    Finds the given quantile of the dataset but handles repeated values.  If the
    quantile lies between identical values, we need to narrow it by raising or
    lowering the threshold.

    Parameters
    ----------
    data : ndarray
        dataset for which quantile is needed
    quantile : float
        the desired quantile
    
    Returns
    -------
    the value at the given quantile
    """
    assert quantile > 0
    assert quantile < 1

    if isinstance(data, np.ma.MaskedArray):
        data = data.compressed()
    
    data = np.sort(data)

    i = (len(data)-1)*quantile
    index_lower = int(math.floor(i))
    index_upper = int(math.ceil(i))
    
    value = 0

    if quantile > 0.5:
        # upper
        while (data[index_lower] == data[index_upper]) and (index_lower>0):
            index_lower -= 1
            
        value = (data[index_lower]+data[index_upper])/2
    else:
        # lower
        while (data[index_lower] == data[index_upper]) and (index_upper<len(data)-1):
            index_upper += 1
            
        value = (data[index_lower]+data[index_upper])/2

    return value

def real_peel(prim, box, name):
    """Performs peel operation on a real-valued column.
    
    Peels along the upper and lower dimension of the given real-valued column,
    returning two new candidate boxes.
    
    Parameters
    ----------
    prim : a Prim instance
    box : a PrimBox instance
    name : str
        the name of the column being peeled
    
    Returns
    -------
    list of tuples storing the rows retained in the new peeling trajectory and
    the new box limits, or an empty list if no peels were possible
    """
    # get the values within the box
    x = prim.x[box.yi][name]

    # peel from the lower and upper dimension
    peels = []
    
    for direction in ['upper', 'lower']:
        if not np.any(np.isnan(x)):
            peel_alpha = prim.peel_alpha
            index = 0
            
            if direction == 'upper':
                peel_alpha = 1-peel_alpha
                index = 1
            
            box_peel = get_quantile(x, peel_alpha)
            
            if direction == 'lower':
                logical = x >= box_peel
                indices = box.yi[logical]
            elif direction == 'upper':
                logical = x <= box_peel
                indices = box.yi[logical]
                
            temp_box = copy.deepcopy(box._box_lims[-1])
            temp_box[name][index] = box_peel
            peels.append((indices, temp_box))
        else:
            return []

    return peels

def discrete_peel(prim, box, name):
    """Performs peel operation on a discrete-valued column.
    
    Peels along the upper and lower dimension of the given discrete-valued
    column, returning two new candidate boxes.
    
    Parameters
    ----------
    prim : a Prim instance
    box : a PrimBox instance
    name : str
        the name of the column being peeled
    
    Returns
    -------
    list of tuples storing the rows retained in the new peeling trajectory and
    the new box limits, or an empty list if no peels were possible
    """
    # get the values within the box and the box limits
    x = prim.x[box.yi][name]
    limits = box._box_lims[-1][name]
    
    # peel from the lower and upper dimension
    peels = []
    
    for direction in ['upper', 'lower']:
        peel_alpha = prim.peel_alpha
        index = 0
        
        if direction == 'upper':
            peel_alpha = 1-peel_alpha
            index = 1
        
        box_peel = get_quantile(x, peel_alpha)
        box_peel = int(box_peel)

        # determine logical associated with peel value            
        if direction == 'lower':
            if box_peel == limits[0]:
                logical = (x > limits[0]) &\
                          (x <= limits[1])
            else:
                logical = (x >= box_peel) &\
                          (x <= limits[1])
        elif direction == 'upper':
            if box_peel == limits[1]:
                logical = (x < limits[1]) &\
                          (x >= limits[0])
            else:
                logical = (x <= box_peel) &\
                          (x >= limits[0])

        # determine value of new limit given logical
        if x[logical].shape[0] == 0:
            if direction == 'upper':
                new_limit = np.max(x)
            else:
                new_limit = np.min(x)
        else:
            if direction =='upper':
                new_limit = np.max(x[logical])
            else:
                new_limit = np.min(x[logical])            
        
        indices = box.yi[logical] 
        temp_box = copy.deepcopy(box._box_lims[-1])
        temp_box[name][index] = new_limit
        peels.append((indices, temp_box))

    return peels

def categorical_peel(prim, box, name):
    """Performs peel operation on a categorical (factor) column.
    
    Returns new candidate boxes resulting from peeling each individual category.
    
    Parameters
    ----------
    prim : a Prim instance
    box : a PrimBox instance
    name : str
        the name of the column being peeled
    
    Returns
    -------
    list of tuples storing the rows retained in the new peeling trajectory and
    the new box limits, or an empty list if no peels were possible
    """
    # get the values within the box and the categories contained within the box
    x = prim.x[box.yi][name]
    entries = box._box_lims[-1][name][0]
    
    # peel each category contained in the box
    peels = []
    
    if len(entries) > 1:
        # can only peel if there is more than one category
        for entry in entries:
            temp_box = np.copy(box._box_lims[-1])
            peel = copy.deepcopy(entries)
            peel.discard(entry)
            temp_box[name][:] = peel
            
            if type(list(entries)[0]) not in (str, float, int):
                bools = []   
                             
                for element in list(x):
                    if element != entry:
                        bools.append(True)
                    else:
                        bools.append(False)
                        
                logical = np.asarray(bools, dtype=bool)
            else:
                logical = x != entry
                
            indices = box.yi[logical]
            peels.append((indices, temp_box))

    return peels

def real_paste(prim, box, name):
    """Performs paste operation on a real-valued column.
    
    Pastes to the upper and lower dimension of the given real-valued column,
    returning two new candidate boxes.
    
    Parameters
    ----------
    prim : a Prim instance
    box : a PrimBox instance
    name : str
        the name of the column being pasted
    
    Returns
    -------
    list of tuples storing the rows retained in the new peeling trajectory and
    the new box limits, or an empty list if no pastes were possible
    """
    x = prim.x[prim.yi_remaining]
    limits = box._box_lims[-1]
    init_limits = prim._box_init

    pastes = []
    for direction in ['lower', 'upper']:
        box_paste = np.copy(limits)
        paste_box = np.copy(limits) # box containing data candidate for pasting
        
        if direction == 'upper':
            paste_box[name][0] = paste_box[name][1]
            paste_box[name][1] = init_limits[name][1]
            
            indices = in_box(x, paste_box)
            data = x[indices][name]
            
            if data.shape[0] > 0:
                paste_value = get_quantile(data, prim.paste_alpha)
            else:
                paste_value = init_limits[name][1]
                
            assert paste_value >= limits[name][1]
        elif direction == 'lower':
            paste_box[name][0] = init_limits[name][0]
            paste_box[name][1] = box_paste[name][0]
            
            indices = in_box(x, paste_box)
            data = x[indices][name]
            
            
            if data.shape[0] > 0:
                paste_value = get_quantile(data, 1-prim.paste_alpha)
            else:
                paste_value = init_limits[name][0]
                
            assert paste_value <= limits[name][0]

        dtype = box_paste.dtype.fields[name][0]
        
        if dtype==np.int32:
            paste_value = np.int(paste_value)
        
        box_paste[name][1 if direction == 'upper' else 0] = paste_value
        indices = in_box(x, box_paste)
        indices = prim.yi_remaining[indices]
        
        pastes.append((indices, box_paste))

    return pastes

def categorical_paste(prim, box, name):
    """Performs paste operation on a categorical (factor) column.
    
    Returns new candidate boxes resulting from pasting each individual category
    back into the box limits.
    
    Parameters
    ----------
    prim : a Prim instance
    box : a PrimBox instance
    name : str
        the name of the column being peeled
    
    Returns
    -------
    list of tuples storing the rows retained in the new peeling trajectory and
    the new box limits, or an empty list if no peels were possible
    """
    x = prim.x[prim.yi_remaining]
    limits = box._box_lims[-1]
    
    c_in_b = limits[name][0]
    c_t = prim._box_init[name][0]
    
    pastes = []
    
    if len(c_in_b) < len(c_t):
        possible_cs = c_t - c_in_b
        
        for entry in possible_cs:
            paste = copy.deepcopy(c_in_b)
            paste.add(entry)
            
            box_paste = np.copy(limits)
            box_paste[name][:] = paste
            
            indices = in_box(x, box_paste)
            indices = prim.yi_remaining[indices]
            
            pastes.append((indices, box_paste))
            
    return pastes
