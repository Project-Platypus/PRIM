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

from __future__ import absolute_import, print_function, division

import six
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as recfunctions

def get_sorted_box_lims(boxes, box_init):
    '''Sort the uncertainties for each box in boxes based on a normalization
    given box_init. Unrestricted dimensions are dropped. The sorting is based
    on the normalization of the first box in boxes. 
    
    Parameters
    ----------
    boxes : list of numpy structured arrays
    box_init : numpy structured array
    
    Returns
    -------
    tuple 
        with the sorted boxes, and the list of restricted uncertainties
    
    '''
        
    # determine the uncertainties that are being restricted
    # in one or more boxes
    uncs = set()
    for box in boxes:
        us  = determine_restricted_dims(box, box_init).tolist()
        uncs = uncs.union(us)
    uncs = np.asarray(list(uncs))

    # normalize the range for the first box
    box_lim = boxes[0]
    nbl = normalize(box_lim, box_init, uncs)
    box_size = nbl[:,1]-nbl[:,0]
    
    # sort the uncertainties based on the normalized size of the 
    # restricted dimensions
    uncs = uncs[np.argsort(box_size)]
    box_lims = [box for box in boxes]
    
    return box_lims, uncs.tolist()

def make_box(x):
    '''
    Make a box that encompasses all the data
    
    Parameters
    ----------
    x : structured numpy array
    '''
    # get the types in the order they appear in the numpy array
    types = [(v[1], k, v[0].name) for k, v in six.iteritems(x.dtype.fields)]
    types = sorted(types)
    
    # convert any bool types to object to store set(False, True)
    ntypes = [(k, 'object' if t == 'bool' else t) for (_, k, t) in types]
    
    # create box limits
    box = np.zeros((2, ), ntypes)
    names = recfunctions.get_names(x.dtype)
    
    for name in names:
        dtype = box.dtype.fields.get(name)[0]
        values = x[name]
        
        if isinstance(values, np.ma.MaskedArray):
            values = values.compressed()
        
        if dtype == 'object':
            try:
                values = set(values)
                box[name][:] = values
            except TypeError as e:
                logging.getLogger(__name__).warning("{} has unhashable values".format(name))
                raise e
        else:
            box[name][0] = np.min(values, axis=0)
            box[name][1] = np.max(values, axis=0)
               
    return box  

def normalize(box_lim, box_init, uncertainties):
    '''Normalize the given box lim to the unit interval derived
    from box init for the specified uncertainties.
    
    Categorical uncertainties are normalized based on fractionated. So
    value specifies the fraction of categories in the box_lim. 
    
    Parameters
    ----------
    box_lim : a numpy structured array.
    box_init :  a numpy structured array.
    uncertainties : list of strings
                    valid names of columns that exist in both structured 
                    arrays.

    Returns
    -------
    ndarray
        a numpy array of the shape (2, len(uncertainties) with the box limits.
    '''
    # normalize the range for the first box
    norm_box_lim = np.zeros((len(uncertainties), box_lim.shape[0]))
    
    for i, u in enumerate(uncertainties):
        dtype = box_lim.dtype.fields[u][0]
        if dtype ==np.dtype(object):
            nu = len(box_lim[u][0])/ len(box_init[u][0])
            nl = 0
        else:
            lower, upper = box_lim[u]
            dif = (box_init[u][1]-box_init[u][0])
            a = 1/dif
            b = -1 * box_init[u][0] / dif
            nl = a * lower + b
            nu = a * upper + b
        norm_box_lim[i, :] = nl, nu
    return norm_box_lim

def determine_restricted_dims(box_lims, box_init):
    '''
    
    determine which dimensions of the given box are restricted compared 
    to compared to the initial box that contains all the data
    
    Parameters
    ----------
    box_lims : structured numpy array
               a specific box limit
    box_init : structured numpy array
               the initial box containing all data points
    
    '''

    logical = compare(box_init, box_lims)
    u = np.asarray(recfunctions.get_names(box_lims.dtype), 
                   dtype=object)
    dims = u[logical==False]
    return dims

def determine_nr_restricted_dims(box_lims, box_init):
    '''
    
    determine the number of restriced dimensions of a box given
    compared to the inital box that contains all the data
    
    Parameters
    ----------
    box_lims : structured numpy array
               a specific box limit
    box_init : structured numpy array
               the initial box containing all data points
    
    '''

    return determine_restricted_dims(box_lims, box_init).shape[0]

def compare(a, b):
    '''compare two boxes, for each dimension return True if the
    same and false otherwise'''
    dtypesDesc = a.dtype.descr
    logical = np.ones((len(dtypesDesc,)), dtype=np.bool)
    for i, entry in enumerate(dtypesDesc):
        name = entry[0]
        logical[i] = logical[i] &\
                    (a[name][0] == b[name][0]) &\
                    (a[name][1] == b[name][1])
    return logical

def setup_figure(uncs):
    '''
    
    helper function for creating the basic layout for the figures that
    show the box lims.
    
    '''
    nr_unc = len(uncs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # create the shaded grey background
    rect = mpl.patches.Rectangle((0, -0.5), 1, nr_unc+1.5,
                                 alpha=0.25,  
                                 facecolor="#C0C0C0",
                                 edgecolor="#C0C0C0")
    ax.add_patch(rect)
    ax.set_xlim(xmin=-0.2, xmax=1.2)
    ax.set_ylim(ymin= -0.5, ymax=nr_unc-0.5)
    ax.yaxis.set_ticks([y for y in range(nr_unc)])
    ax.xaxis.set_ticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(uncs[::-1]) 
    return fig, ax

def in_box(x, boxlim):
    '''
     
    returns the indices of the data points that are within the 
    box_lims.
    
    Parameters
    ----------
    x : numpy structured array
    boxlim : numpy structured array
    
    
    Returns
    -------
    ndarray
        valid numpy indices on x
    
    '''
    logical = np.ones(x.shape[0], dtype=np.bool)
    
    dims = recfunctions.get_names(boxlim.dtype)

    for name in dims:
        value = x.dtype.fields.get(name)[0]
        
        if value == 'object' or value == 'bool':
            entries = boxlim[name][0]
            l = np.ones( (x.shape[0], len(entries)), dtype=np.bool)
            for i,entry in enumerate(entries):
                if type(list(entries)[0]) not in (str, float, int):
                    bools = []                
                    for element in list(x[name]):
                        if element == entry:
                            bools.append(True)
                        else:
                            bools.append(False)
                    l[:, i] = np.asarray(bools, dtype=bool)
                else:
                    l[:, i] = x[name] == entry
            l = np.any(l, axis=1)
            logical = logical & l
        else:
            logical = logical & (boxlim[name][0] <= x[name] )&\
                                        (x[name] <= boxlim[name][1])                
    
    indices = np.where(logical==True)
    
    assert len(indices)==1
    indices = indices[0]
    
    return indices
