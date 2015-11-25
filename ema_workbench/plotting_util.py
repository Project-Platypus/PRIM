'''

Plotting utility functions

'''
from __future__ import (absolute_import, division,
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

# used for legend
LINE = 'line'
PATCH = 'patch'
SCATTER = 'scatter'

# from matplotlib import cbook

# class DataCursor(object):
#     """A simple data cursor widget that displays the x,y location of a
#     matplotlib artist when it is selected."""
#     def __init__(self, artists, tolerance=5, offsets=(-20, 20), 
#                  template='x: %0.2f\ny: %0.2f', display_all=False):
#         """Create the data cursor and connect it to the relevant figure.
#         "artists" is the matplotlib artist or sequence of artists that will be 
#             selected. 
#         "tolerance" is the radius (in points) that the mouse click must be
#             within to select the artist.
#         "offsets" is a tuple of (x,y) offsets in points from the selected
#             point to the displayed annotation box
#         "template" is the format string to be used. Note: For compatibility
#             with older versions of python, this uses the old-style (%) 
#             formatting specification.
#         "display_all" controls whether more than one annotation box will
#             be shown if there are multiple axes.  Only one will be shown
#             per-axis, regardless. 
#         """
#         self.template = template
#         self.offsets = offsets
#         self.display_all = display_all
#         if not cbook.iterable(artists):
#             artists = [artists]
#         self.artists = artists
#         self.axes = tuple(set(art.axes for art in self.artists))
#         self.figures = tuple(set(ax.figure for ax in self.axes))
# 
#         self.annotations = {}
#         for ax in self.axes:
#             self.annotations[ax] = self.annotate(ax)
# 
#         for artist in self.artists:
#             artist.set_picker(tolerance)
#         for fig in self.figures:
#             fig.canvas.mpl_connect('pick_event', self)
# 
#     def annotate(self, ax):
#         """Draws and hides the annotation box for the given axis "ax"."""
#         annotation = ax.annotate(self.template, xy=(0, 0), ha='right',
#                 xytext=self.offsets, textcoords='offset points', va='bottom',
#                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=1.0),
#                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
#                 )
#         annotation.set_visible(False)
#         return annotation
# 
#     def __call__(self, event):
#         """Intended to be called through "mpl_connect"."""
#         # Rather than trying to interpolate, just display the clicked coords
#         # This will only be called if it's within "tolerance", anyway.
#         selected = None
#         
#         if isinstance(event.artist, mpl.collections.PathCollection):
#             print event.artist.get_paths()
#             
#         print selected
#         x, y = event.mouseevent.xdata, event.mouseevent.ydata
#         annotation = self.annotations[event.artist.axes]
#         if x is not None:
#             if not self.display_all:
#                 # Hide any other annotation boxes...
#                 for ann in self.annotations.values():
#                     ann.set_visible(False)
#             # Update the annotation in the current axis..
#             annotation.xy = x, y
#             annotation.set_text(self.template % (x, y))
#             annotation.set_visible(True)
#             event.canvas.draw()

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




