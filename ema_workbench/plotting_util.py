'''

Plotting utility functions

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
import six

import copy
import logging

import numpy as np


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec 

from .exceptions import PRIMError

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
'''Default color list'''

TIGHT = False
'''Parameter controlling whether tight layout from matplotlib should be used'''

TIME = "TIME"
'''Default key for time'''

ENVELOPE = 'envelope'
'''constant for plotting envelopes'''

LINES = 'lines'
'''constant for plotting lines'''

ENV_LIN = "env_lin"
'''constant for plotting envelopes with lines'''

KDE = 'kde'
'''constant for plotting density as a kernel density estimate'''

HIST = 'hist'
'''constant for plotting density as a histogram'''

BOXPLOT = 'boxplot'
'''constant for plotting density as a boxplot'''

VIOLIN = 'violin'
'''constant for plotting density as a violin plot, which combines a
Gaussian density estimate with a boxplot'''

# used for legend
LINE = 'line'
PATCH = 'patch'
SCATTER = 'scatter'

#see http://matplotlib.sourceforge.net/users/customizing.html for details
#mpl.rcParams['savefig.dpi'] = 600
#mpl.rcParams['axes.formatter.limits'] = (-5, 5)
#mpl.rcParams['font.family'] = 'serif'
#mpl.rcParams['font.serif'] = 'Times New Roman'
#mpl.rcParams['font.size'] = 12.0

def make_legend(categories,
                ax,
                ncol=3,
                legend_type=LINE,
                alpha=1):
    '''
    Helper function responsible for making the legend

    Parameters
    ----------
    categories : str or tuple
                 the categories in the legend
    ax : axes instance 
         the axes with which the legend is associated
    ncol : int
           the number of columns to use
    legend_type : {LINES, SCATTER, PATCH}
                  whether the legend is linked to lines, patches, or scatter 
                  plots
    alpha : float
            the alpha of the artists
    
    '''
    
    some_identifiers = []
    labels = []
    for i, category in enumerate(categories):
        if legend_type == LINE:    
            artist = plt.Line2D([0,1], [0,1], color=COLOR_LIST[i], 
                                alpha=alpha) #TODO
        elif legend_type == SCATTER:
#             marker_obj = mpl.markers.MarkerStyle('o')
#             path = marker_obj.get_path().transformed(
#                              marker_obj.get_transform())
#             artist  = mpl.collections.PathCollection((path,),
#                                         sizes = [20],
#                                         facecolors = COLOR_LIST[i],
#                                         edgecolors = 'k',
#                                         offsets = (0,0)
#                                         )
            # TODO work arround, should be a proper proxyartist for scatter legends
            artist = mpl.lines.Line2D([0],[0], linestyle="none", c=COLOR_LIST[i], marker = 'o')

        elif legend_type == PATCH:
            artist = plt.Rectangle((0,0), 1,1, edgecolor=COLOR_LIST[i],
                                   facecolor=COLOR_LIST[i], alpha=alpha)

        some_identifiers.append(artist)
        
        if type(category) == tuple:
            label =  '%.2f - %.2f' % category 
        else:
            label = category
        
        labels.append(str(label))
    
    ax.legend(some_identifiers, labels, ncol=ncol,
                      loc=3, borderaxespad=0.1,
                      mode='expand', bbox_to_anchor=(0., 1.1, 1., .102))


        
def filter_scalar_outcomes(outcomes):
    '''
    Helper function that removes non time series outcomes from all the 
    outcomes.

    Parameters
    ----------
    outcomes : dict
    
    Returns
    -------
    dict
        the filtered outcomes
    
    
    '''
    outcomes_to_remove = []
    for key, value in outcomes.items():
        if len(value.shape) <2:
            outcomes_to_remove.append(key)
            logging.getLogger(__name__).info("%s not shown because it is not time series data" %key)
    [outcomes.pop(entry) for entry in outcomes_to_remove]
    return outcomes


def determine_time_dimension(outcomes):
    '''
    helper function for determining or creating time dimension

    
    Parameters
    ----------
    outcomes : dict
    
    Returns
    -------
    ndarray
    
    
    '''

    time = None
    try:
        time = outcomes['TIME']
        time = time[0, :]
        outcomes.pop('TIME')
    except KeyError:
        values = iter(outcomes.values())
        for value in values:
            if len(value.shape)==2:
                time =  np.arange(0, value.shape[1])
                break
    if time is None:
        logging.getLogger(__name__).info("no time dimension found in results")
    return time, outcomes    


def group_results(experiments, outcomes, group_by, grouping_specifiers,
                  grouping_labels):
    '''
    Helper function that takes the experiments and results and returns a list 
    based on groupings. Each element in the dictionary contains the experiments 
    and results for a particular group, the key is the grouping specifier.
    
    Parameters
    ----------
    experiments : recarray
    outcomes : dict
    group_by : str
               The column in the experiments array to which the grouping 
               specifiers apply. If the name is'index' it is assumed that the 
               grouping specifiers are valid indices for numpy.ndarray.
    grouping_specifiers : iterable
                    An iterable of grouping specifiers. A grouping 
                    specifier is a unique identifier in case of grouping by 
                    categorical uncertainties. It is a tuple in case of 
                    grouping by a parameter uncertainty. In this cose, the code
                    treats the tuples as half open intervals, apart from the 
                    last entry, which is treated as closed on both sides.  
                    In case of 'index', the iterable should be a dictionary 
                    with the name for each group as key and the value being a 
                    valid index for numpy.ndarray.
    
    Returns 
    dict
        A dictionary with the experiments and results for each group, the 
        grouping specifier is used as key
             
    ..note:: In case of grouping by parameter uncertainty, the list of 
             grouping specifiers is sorted. The traversal assumes half open
             intervals, where the upper limit of each interval is open, except 
             for the last interval which is closed.
    
    '''
    groups = {}
    if group_by != 'index':
        column_to_group_by = experiments[group_by]
    
    for label, specifier in zip(grouping_labels, grouping_specifiers):
        if isinstance(specifier, tuple):
            # the grouping is a continuous uncertainty
            lower_limit, upper_limit = specifier
            
            #check whether it is the last grouping specifier
            if grouping_specifiers.index(specifier) ==\
                len(grouping_specifiers)-1:
                #last case
                
                logical = (column_to_group_by>=lower_limit) &\
                           (column_to_group_by<=upper_limit)
            else:
                logical = (column_to_group_by>=lower_limit) &\
                           (column_to_group_by<upper_limit)
        elif group_by =='index':
            # the grouping is based on indices
            logical = specifier
        else:
            # the grouping is an integer or categorical uncertainty
            logical = column_to_group_by==specifier
        
        group_outcomes = {}
        for key, value in outcomes.items():
            value = value[logical]
            group_outcomes[key] = value
        groups[label] = (experiments[logical], group_outcomes)
        
    return groups


def make_continuous_grouping_specifiers(array, nr_of_groups=5):
    '''
    Helper function for discretesizing a continuous array. By default, the 
    array is split into 5 equally wide intervals.
    
    Parameters
    ----------
    array : ndarray
            a 1-d array that is to be turned into discrete intervals.
    nr_of_groups : int, optional
    
    Returns
    -------
    list of tuples 
        list of tuples with the lower and upper bound of the intervals. 
    
    .. note:: this code only produces intervals. :func:`group_results` uses
              these intervals in half-open fashion, apart from the last 
              interval: [a, b), [b,c), [c,d]. That is, both the end point
              and the start point of the range of the continuous array are 
              included.
    
    '''
    
    minimum = np.min(array)
    maximum = np.max(array)
    step = (maximum-minimum)/nr_of_groups
    a = [(minimum+step*x, minimum+step*(x+1)) for x in range(nr_of_groups)]
    assert a[0][0] == minimum
    assert a[-1][1] == maximum
    return a


def prepare_pairs_data(results, 
                        outcomes_to_show=None,
                        group_by=None,
                        grouping_specifiers=None,
                        point_in_time=-1,
                        filter_scalar=True):
    '''
    
    Parameters
    ----------
    results : tuple
    outcomes_to_show : list of str, optional
    group_by : str, optional
    grouping_specifiers : iterable, optional
    point_in_time : int, optional
    filter_scalar : bool, optional
       
    '''
    if isinstance(outcomes_to_show, six.string_types):
        raise PRIMError("for pair wise plotting, more than one outcome needs to be provided")
    
    outcomes, outcomes_to_show, time, grouping_labels = prepare_data(results, 
                                                        outcomes_to_show,
                                                        group_by,
                                                        grouping_specifiers,
                                                        filter_scalar)

    def filter_outcomes(outcomes, point_in_time):
        new_outcomes = {}
        for key, value in outcomes.items():
            if len(value.shape)==2:
                new_outcomes[key] = value[:, point_in_time]
            else:
                new_outcomes[key] = value
        return new_outcomes
    
    if point_in_time:
        if point_in_time != -1:
            point_in_time = np.where(time==point_in_time)
        
        if group_by:
            new_outcomes = {}
            for key, value in outcomes.items():
                new_outcomes[key] = filter_outcomes(value, point_in_time)
            outcomes = new_outcomes
        else:
            outcomes = filter_outcomes(outcomes, point_in_time)
    return outcomes, outcomes_to_show, grouping_labels 


def prepare_data(results,
                 outcomes_to_show=None,
                 group_by=None,
                 grouping_specifiers=None,
                 filter_scalar=True):
    '''
    
    Parameters
    ----------
    results : tuple
    outcomes_to_show : list of str, optional
    group_by : str, optional
    grouping_specifiers : iterable, optional
    filter_scalar : bool, optional
    
    '''

    #unravel results
    experiments, outcomes = results

    temp_outcomes = {}

    # remove outcomes that are not to be shown
    if outcomes_to_show:
        if isinstance(outcomes_to_show, six.string_types):
            outcomes_to_show = [outcomes_to_show]
            
        for entry in outcomes_to_show:
            temp_outcomes[entry] = copy.deepcopy(outcomes[entry])

    time, outcomes = determine_time_dimension(outcomes)

    # filter the outcomes to exclude scalar values
    if filter_scalar:
        outcomes = filter_scalar_outcomes(outcomes)
    if not outcomes_to_show:
        outcomes_to_show = outcomes.keys()
        
    # group the data if desired
    if group_by:
        if not grouping_specifiers:
            #no grouping specifier, so infer from the data
            if group_by=='index':
                raise PRIMError("no grouping specifiers provided while trying to group on index")
            else:
                column_to_group_by = experiments[group_by]
                if column_to_group_by.dtype == np.object:
                    grouping_specifiers = set(column_to_group_by)
                else:
                    grouping_specifiers = make_continuous_grouping_specifiers(column_to_group_by, 
                                                        grouping_specifiers)
            grouping_labels = grouping_specifiers = sorted(grouping_specifiers)
        else:
            if isinstance(grouping_specifiers, six.string_types):
                grouping_specifiers = [grouping_specifiers]
                grouping_labels = grouping_specifiers
            elif isinstance(grouping_specifiers, dict):
                grouping_labels = sorted(grouping_specifiers.keys())
                grouping_specifiers = [grouping_specifiers[key] for key in 
                                       grouping_labels]
            else:
                grouping_labels = grouping_specifiers
                
        
        outcomes = group_results(experiments, outcomes, group_by,\
                                 grouping_specifiers, grouping_labels)
        
        new_outcomes = {}
        for key, value in outcomes.items():
            new_outcomes[key] = value[1]
        outcomes = new_outcomes
    else:
        grouping_labels=[]

    return outcomes, outcomes_to_show, time, grouping_labels


def do_titles(ax, titles, outcome):
    '''
    Helper function for setting the title on an ax
    
    Parameters
    ----------
    ax : axes instance
    titles : dict
             a dict which maps outcome names to titles
    outcome : str
              the outcome plotted in the ax.
    
    '''
    
    if isinstance(titles, dict):
        if not titles:
            ax.set_title(outcome)
        else:
            try:
                ax.set_title(titles[outcome])
            except KeyError:
                logging.getLogger(__name__).warning("key error in do_titles, no title provided for `%s`" % (outcome))
                ax.set_title(outcome)


def do_ylabels(ax, ylabels, outcome):
    '''
    Helper function for setting the y labels on an ax

    Parameters
    ----------
    ax : axes instance
    titles : dict
             a dict which maps outcome names to y labels
    outcome : str
              the outcome plotted in the ax.
    
    '''
    
    if isinstance(ylabels, dict):
        if not ylabels:
            ax.set_ylabel(outcome)
        else:
            try:
                ax.set_ylabel(ylabels[outcome])
            except KeyError:
                logging.getLogger(__name__).warning("key error in do_ylabels, no ylabel provided for `%s`" % (outcome))
                ax.set_ylabel(outcome)    


def make_grid(outcomes_to_show, density=False):
    '''
    Helper function for making the grid that specifies the size and location
    of the various axes. 

    Parameters
    ----------
    outcomes_to_show : list of str
                       the list of outcomes to show
    density: boolean : bool, optional
    
    '''

    
    # make the plotting grid
    if density:
        grid = gridspec.GridSpec(len(outcomes_to_show), 2,
                                 width_ratios = [4, 1])
    else:
        grid = gridspec.GridSpec(len(outcomes_to_show), 1) 
    grid.update(wspace = 0.1,
                hspace = 0.4)
    
    figure = plt.figure()
    return figure, grid

