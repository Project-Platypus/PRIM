'''

Plotting utility functions

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)


import matplotlib.pyplot as plt
import matplotlib as mpl

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




