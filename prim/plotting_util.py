'''

Plotting utility functions

'''
from __future__ import (absolute_import, division,
                        unicode_literals)

import logging

# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

COLOR_LIST = ['b',
              'g',
              'r',
              'c',
              'm',
              'y',
              'k',
              'b',
              'g',
              'r',
              'c',
              'm',
              'y',
              'k'
                ]


def do_text_ticks_labels(ax, i, j, field1, field2, ylabels, outcomes_to_show):
    '''
    
    Helper function for setting the tick labels on the axes correctly on and 
    off

    Parameters
    ----------
    ax : axes
    i : int
    j : int
    field1 : str
    field2 : str
    ylabels : dict, optional
    outcomes_to_show : str
    
    
    '''
    
    #text and labels
    if i == j:
        #only plot the name in the middle
        if ylabels:
            text = ylabels[field1]
        else:
            text = field1
        ax.text(0.5, 0.5, text,
                horizontalalignment='center',
                verticalalignment='center',
                transform = ax.transAxes)  
    
    # are we at the end of the row?
    if i != len(outcomes_to_show)-1:
        #xaxis off
        ax.set_xticklabels([])
    else:
        if ylabels:
            try:
                ax.set_xlabel(ylabels.get(field2))
            except KeyError:
                logging.getLogger(__name__).info("no label specified for "+field2)
        else:
            ax.set_xlabel(field2) 
    
    # are we at the end of the column?
    if j != 0:
        #yaxis off
        ax.set_yticklabels([])
    else:
        if ylabels:
            try:
                ax.set_ylabel(ylabels.get(field1))
            except KeyError:
                logging.getLogger(__name__).info("no label specified for "+field1) 
        else:
            ax.set_ylabel(field1)   
