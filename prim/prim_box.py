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

from __future__ import absolute_import, division, print_function

import copy
import logging
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mpldatacursor
from scipy.stats import binom
from operator import itemgetter
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1 import host_subplot
from .exceptions import PrimError
from .plotting_util import pairwise_labels, format_number
from .scenario_discovery_util import (in_box, normalize,
        determine_nr_restricted_dims, determine_restricted_dims)

try:
    import mpld3
except ImportError:
    logging.getLogger(__name__).info("mpld3 library not found, some functionality will be disabled")
    global mpld3
    mpld3 = None
    
def indent(lines, amount, ch=' '):
    padding = amount * ch
    return padding + ('\n'+padding).join(lines.split('\n'))

class PrimBoxProperty(object):
    """Generate read-only attributes for a PrimBox."""
    
    def __init__(self, attribute, doc=None):
        self.attribute = attribute
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.peeling_trajectory[self.attribute][obj._cur_box]

    def __set__(self, obj, value):
        raise AttributeError("can't set attribute")

    def __delete__(self, obj):
        raise AttributeError("can't delete attribute")

class PrimBox(object):
    '''A class that holds information over a specific box 
    
    Attributes
    ----------
    coverage : float
               coverage of currently selected box
    density : float
               density of currently selected box
    mean : float
           mean of currently selected box
    res_dim : int
              number of restricted dimensions of currently selected box
    mass : float
           mass of currently selected box 
    peeling_trajectory : pandas dataframe
                         stats for each box in peeling trajectory
    box_lims : list
               list of box lims for each box in peeling trajectory

    
    by default, the currently selected box is the last box on the peeling
    trajectory, unless this is changed via :meth:`PrimBox.select`.
    
    '''
    
    coverage = PrimBoxProperty('coverage')
    density = PrimBoxProperty('density')
    mean = PrimBoxProperty('mean')
    res_dim = PrimBoxProperty('res dim')
    mass = PrimBoxProperty('mass')
    
    def __init__(self, prim, box_lims, indices):
        """Create a new PrimBox object.
        
        Parameters
        ----------
        prim : Prim object
            the Prim object which created this box
        box_lims : recarray
            the initial box limits
        indices : ndarray
            the indices in the dataset
        """
        
        self.prim = prim
        
        # peeling and pasting trajectory
        columns = ['coverage', 'density', 'mean', 'res dim', 'mass']
        self.peeling_trajectory = pd.DataFrame(columns=columns)
        
        self._box_lims = []
        self._cur_box = -1
        self._frozen = False
        
        # add the given box limits to the peeling/pasting trajectory
        self.update(box_lims, indices)
        
    def __len__(self):
        """Returns the number pf peeling/pasting trajectories."""
        return len(self.peeling_trajectory)
        
    def __str__(self):
        message = "".join(["Box %d [Peeling Trajectory %d]\n",
                           "    Stats\n",
                           "        Coverage: %f\n",
                           "        Density:  %f\n",
                           "        Mass:     %f\n",
                           "        Res Dim:  %f\n",
                           "        Mean:     %f\n",
                           "    Limits\n",
                           "%s"])
        
        return message % (len(self.prim._boxes),
                          self._cur_box,
                          self.coverage,
                          self.density,
                          self.mass,
                          self.res_dim,
                          self.mean,
                          indent(str(self.limits), 8))
        
    @property
    def stats(self):
        """Returns the statistics for the current peeling/pasting trajectory."""
        return {"coverage" : self.coverage,
                "density" : self.density,
                "mean" : self.mean,
                "res dim" : self.res_dim,
                "mass" : self.mass}
    
    @property
    def limits(self):
        stats = self.peeling_trajectory.iloc[self._cur_box].to_dict()
        stats['restricted_dim'] = stats['res dim']

        qp_values = self._calculate_quasi_p(self._cur_box)
        
        uncs = [(key, value) for key, value in qp_values.items()]
        uncs.sort(key=itemgetter(1))
        uncs = [uncs[0] for uncs in uncs]
        
        box_lim = pd.DataFrame(np.zeros((len(uncs), 3)), 
                               index=uncs, 
                               columns=['min', 'max', 'qp values'])
        
        for unc in uncs:
            values = self._box_lims[self._cur_box][unc][:]
            box_lim.loc[unc] = [values[0], values[1], qp_values[unc]]
             
        return box_lim

    def show_details(self, fig=None):
        """Detail plot for the current peeling/pasting trajectory.
        
        Generates a plot showing the details of the current peeling/pasting
        trajectory.
        
        Returns
        -------
        the Matplotlib figure
        """
        i = self._cur_box
        qp_values = self._calculate_quasi_p(i)
        uncs = [(key, value) for key, value in qp_values.items()]
        uncs.sort(key=itemgetter(1))
        uncs = [uncs[0] for uncs in uncs]
        n = len(uncs)
        
        if fig is not None:
            plt.figure(fig.number)
            plt.clf()
        else:
            fig = plt.figure(figsize=(12, 6))
        
        outer_grid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)
          
        ax0 = plt.Subplot(fig, outer_grid[0], frame_on=False)
        ax0.xaxis.set_visible(False)
        ax0.yaxis.set_visible(False)
        ax0.set_title("Box Coverage Plot")
        fig.add_subplot(ax0)
          
        inner_grid = gridspec.GridSpecFromSubplotSpec(n, n,
            subplot_spec=outer_grid[0], wspace=0.1, hspace=0.1)  
          
        self.show_scatter(grid=inner_grid)
          
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1,
            subplot_spec=outer_grid[1], wspace=0.0, hspace=0.0)
          
        ax1 = plt.Subplot(fig, inner_grid[0])
          
        fig.add_subplot(ax1)
        self._show_limits()
          
        ax2 = plt.Subplot(fig, inner_grid[1], frame_on=False)
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        fig.add_subplot(ax2)
           
        ax2.add_table(plt.table(cellText=[["Coverage", "%0.1f%%" % (100*self.peeling_trajectory['coverage'][i])],
                                          ["Density", "%0.1f%%" % (100*self.peeling_trajectory['density'][i])],
                                          ["Mass", "%0.1f%%" % (100*self.peeling_trajectory['mass'][i])],
                                          ["Res Dim", "%d" % self.peeling_trajectory['res dim'][i]],
                                          ["Mean", "%0.2f" % self.peeling_trajectory['mean'][i]]],
                                cellLoc='center',
                                colWidths=[0.3, 0.7],
                                loc='center'))
        ax2.set_title("Statistics", y=0.7)
        
        def show_next(event):
            i = (self._cur_box + 1) % self.peeling_trajectory.shape[0]
            self.select(i)
            self.show_details(fig=event.canvas.figure)
            
        def show_prev(event):
            i = (self._cur_box - 1) % self.peeling_trajectory.shape[0]
            self.select(i)
            self.show_details(fig=event.canvas.figure)

        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        self._bnext = Button(axnext, "Next")
        self._bprev = Button(axprev, "Prev")
        self._bnext.on_clicked(show_next)
        self._bprev.on_clicked(show_prev)
        
        plt.subplots_adjust(top=0.85)
        plt.draw()
        
        return fig
        
    def _show_limits(self):
        i = self._cur_box
        qp_values = self._calculate_quasi_p(i)
        uncs = [(key, value) for key, value in qp_values.items()]
        uncs.sort(key=itemgetter(1))
        uncs = [uncs[0] for uncs in uncs]
        
        box_lim_init = self.prim._box_init
        box_lim = self._box_lims[i]
        norm_box_lim = normalize(box_lim, box_lim_init, uncs)
        
        left = []
        height = []
        bottom = []
        
        for i, v in enumerate(uncs):
            if (self.prim.x.dtype.fields[v][0].name == 'bool' or
                    self.prim.x.dtype.fields[v][0].name == 'object'):
                left.append(i)
                height.append(-1)
                bottom.append(0)
            else:
                left.append(i)
                height.append(norm_box_lim[i][1]-norm_box_lim[i][0])
                bottom.append(norm_box_lim[i][0])
        
        plt.bar(left, 
                height,
                width = 0.6,
                bottom = bottom,
                align="center",
                color='r',
                alpha=0.6)
        
        plt.ylim(0, 1)
        plt.xticks(left, uncs)
        plt.tick_params(axis='y',
                        which='both',
                        right='off',
                        left='off',
                        labelleft='off')
        
        for i, v in enumerate(uncs):
            if (self.prim.x.dtype.fields[v][0].name == 'bool' or
                    self.prim.x.dtype.fields[v][0].name == 'object'):
                elements = sorted(list(box_lim_init[v][0]))
                selected_elements = box_lim[v][0]
                length = len(elements)
                
                x = [v / (length+1) for v in range(1, length+1)]

                for xi, label in zip(x, elements):
                    if label in selected_elements:
                        plt.text(i, xi, label, horizontalalignment='center',
                                 verticalalignment='center', color='b')
                    else:
                        plt.text(i, xi, label, horizontalalignment='center',
                                 verticalalignment='center', color="#cccccc")
        
        fig = plt.gcf()
        ax = plt.gca()
        
        for i, unc in enumerate(uncs):
            if (self.prim.x.dtype.fields[v][0].name != 'bool' and
                    self.prim.x.dtype.fields[v][0].name != 'object'):
                # be sure to convert the normalized bounds back to original
                l, u = box_lim[unc]
                
                ax.text(i - 0.15,
                        norm_box_lim[i][0],
                        format_number(l),
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        color='k')
                
                ax.text(i + 0.15,
                        norm_box_lim[i][1],
                        format_number(u),
                        horizontalalignment='center',
                        verticalalignment='top',
                        color='k')
            
        ax.set_title("Restricted Dimensions")
            
        return fig
        
    def select(self, i):
        """Selects the given peeling/pasting trajectory.
        
        Updates this PrimBox object to the given peeling/pasting trajectory.
        Subsequent calls to this object will reflect the data and statistics
        for the given trajectory.
        
        Parameters
        ----------
        i : int
            the index of the peeling/pasting trajectory
        """
        if self._frozen:
            raise PrimError("box has been frozen because PRIM has found at least one more recent box")
        
        indices = in_box(self.prim.x[self.prim.yi_remaining], 
                         self._box_lims[i])
        self.yi = self.prim.yi_remaining[indices]
        self._cur_box = i

    def drop_restriction(self, name):
        """Drops any restrictions on the specified variable.
        
        Removes any restrictions imposed by this box on the given variable.
        This creates a new PrimBox object that is appended to the
        peeling/pasting trajectory.
        
        Parameters
        ----------
        name : str
            the name of the variable
        """
        new_box_lim = copy.deepcopy(self._box_lims[self._cur_box])
        new_box_lim[name][:] = self._box_lims[0][name][:]
        
        indices = in_box(self.prim.x[self.prim.yi_remaining], 
                         new_box_lim)
        indices = self.prim.yi_remaining[indices]
        
        self.update(new_box_lim, indices)
        
    def update(self, box_lims, indices):
        """Updates this box with new limits.
        
        Adds the given limits to the peeling/pasting trajectory for this box
        and selects this trajectory.  
        
        Parameters
        ----------
        box_lims: recarray
            the new box limits
        indices: ndarray
            the indices of y that are inside the box
        """
        self.yi = indices
        y = self.prim.y[self.yi]

        self._box_lims.append(box_lims)
        coi = self.prim.determine_coi(self.yi)

        stats = {"coverage" : coi/self.prim.t_coi, 
                "density" : coi/y.shape[0],  
                "mean" : np.mean(y),
                "res dim" : determine_nr_restricted_dims(
                        self._box_lims[-1], 
                        self.prim._box_init),
                "mass" : y.shape[0]/self.prim.n}
        
        self.peeling_trajectory = self.peeling_trajectory.append(
                pd.DataFrame([stats]), 
                ignore_index=True)
        
        self._cur_box = len(self.peeling_trajectory)-1
        
    def show_ppt(self):
        """Plot of peeling and pasting trajectory statistics.
        
        Produces a plot of the peeling and pasting trajectory statistics,
        including the mean, mass, coverage, density, and number of restricted
        dimensions.
        
        Returns
        -------
        the Matplotlib figure
        """
        ax = host_subplot(111)
        ax.set_xlabel("Peeling and Pasting Trajectory")
        ax.set_ylabel("Mean / Mass / Coverage / Density")
        
        par = ax.twinx()
        par.set_ylabel("# of Restricted Dimensions")
            
        linewidth = 2.0
        ax.plot(self.peeling_trajectory['mean'], linewidth=linewidth)
        ax.plot(self.peeling_trajectory['mass'], linewidth=linewidth)
        ax.plot(self.peeling_trajectory['coverage'], linewidth=linewidth)
        ax.plot(self.peeling_trajectory['density'], linewidth=linewidth)
        par.plot(self.peeling_trajectory['res dim'], linewidth=linewidth)
        
        ax.grid(True, which='both')
        ax.set_ylim(0, 1)
        
        fig = plt.gcf()
        
        # reduce the height of the plot so the legend has enough room
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height*0.2, box.width, box.height*0.8])
        
        # create the legend
        ax.legend(['Mean', 'Mass', 'Coverage', 'Density', "Restricted Dimensions"],
                  ncol=3,
                  loc=9,
                  borderaxespad=0.1,
                  bbox_to_anchor=(0.5, -0.2))

        return fig
    
    def show_tradeoff(self):
        """Plot the tradeoff between coverage and density.
        
        Generates a plot of the tradeoff between coverage and density for the
        peeling/pasting trajectories.  Color is used to denote the number of
        restricted dimensions.
        
        Returns
        -------
        the Matplotlib figure
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        
        # setup the color map for coloring the number of restricted dimensions
        cmap = mpl.cm.YlGnBu_r #@UndefinedVariable
        boundaries = np.arange(-0.5, 
                               max(self.peeling_trajectory['res dim'])+1.5, 
                               step=1)
        ncolors = cmap.N
        norm = mpl.colors.BoundaryNorm(boundaries, ncolors)
        
        # plot the tradeoff
        p = ax.scatter(self.peeling_trajectory['coverage'], 
                       self.peeling_trajectory['density'], 
                       c=self.peeling_trajectory['res dim'], 
                       norm=norm,
                       cmap=cmap,
                       picker=True)

        ax.set_ylabel('Density')
        ax.set_xlabel('Coverage')
        ax.set_ylim(0, 1.2)
        ax.set_xlim(0, 1.2)
        
        ticklocs = np.arange(0, 
                             max(self.peeling_trajectory['res dim'])+1, 
                             step=1)
        cb = fig.colorbar(p, spacing='uniform', ticks=ticklocs, drawedges=True)
        cb.set_label("# of Restricted Dimensions")
        
        # enable mouse interaction
        def handle_click(event):
            if hasattr(event, "ind"):
                i = event.ind[0]
                self.select(i)
                self.show_details().show()
            
        def formatter(**kwargs):
            i = kwargs.get("ind")[0]
            data = self.peeling_trajectory.ix[i]
            return (("Box %d\n" +
                     "Coverage: %2.1f%%\n" +
                     "Density: %2.1f%%\n" +
                     "Mass: %2.1f%%\n" +
                     "Res Dim: %d") % (i,
                                       100*data["coverage"],
                                       100*data["density"],
                                       100*data["mass"],
                                       data["res dim"]))
        
        mpldatacursor.datacursor(formatter=formatter, hover=True)
        fig.canvas.mpl_connect('pick_event', handle_click)
        
        # enable tooltips on IPython Notebook
        if mpld3:
            css = """
            table {
              border-collapse: collapse;
            }
            
            th {
              background-color:  rgba(255,255,255,0.95);
            }
            
            td {
              background-color: rgba(255,255,255,0.95);
            }
            
            table, th, td {
              font-family:Tahoma, Tahoma, sans-serif;
              font-size: 16px;
              border: 1px solid black;
              text-align: right;
            }
            """   
            
            labels = []
            columns_to_include = ['coverage','density', 'mass', 'res dim']
            frmt = lambda x: '{:.2f}'.format( x )
            
            for i in range(len(self.peeling_trajectory['coverage'])):
                label = self.peeling_trajectory.ix[[i], columns_to_include]
                label.columns = ["Coverage", "Density", "Mass", "Res. Dim."]
                label = label.T
                label.columns = ["Box {0}".format(i)]
                labels.append(str(label.to_html(float_format=frmt)))       
    
            tooltip = mpld3.plugins.PointHTMLTooltip(p, labels, voffset=10, 
                                                     hoffset=10, css=css)  
            mpld3.plugins.connect(fig, tooltip)        
        
        return fig
    
    def show_scatter(self, grid=None):
        """Shows restricted dimensions overlay on scatter plot.
        
        Generates a plot showing the data points, with the cases of interest
        colored red, and the restricted dimensions overlayed as black
        rectangles.
        
        Returns
        -------
        the Matplotlib figure
        """   
        fig = self._pairwise_scatter(
                self.prim.x[self.prim.yi_remaining],
                self.prim.y[self.prim.yi_remaining],
                self._box_lims[self._cur_box], 
                determine_restricted_dims(
                        self._box_lims[self._cur_box], 
                        self.prim._box_init),
                grid = grid)
        
        title = "Peeling/Pasting Trajectory %d" % self._cur_box
        fig.suptitle(title, fontsize=16)
        fig.canvas.set_window_title(title)
        return fig
    
    def _pairwise_scatter(self, x, y, box_lim, restricted_dims, grid=None):
        ''' helper function for pair wise scatter plotting
        
        Parameters
        ----------
        x : numpy structured array
            the experiments
        y : numpy array
            the outcome of interest
        box_lim : numpy structured array
            a boxlim
        restricted_dims : list of strings
            list of uncertainties that define the boxlims
        
        '''
        restricted_dims = list(restricted_dims)
        combis = [(field1, field2) for field1 in restricted_dims\
                                   for field2 in restricted_dims]
    
        if not grid:
            grid = gridspec.GridSpec(len(restricted_dims), len(restricted_dims))                             
            grid.update(wspace = 0.1,
                        hspace = 0.1)    
            figure = plt.figure()
        else:
            figure = plt.gcf()
        
        for field1, field2 in combis:
            i = restricted_dims.index(field1)
            j = restricted_dims.index(field2)
            ax = figure.add_subplot(grid[i,j])  
            
            # scatter points
            for n in [0,1]:
                x_n = x[y==n]        
                x_1 = x_n[field2]
                x_2 = x_n[field1]
                
                if field1 == field2 and not len(restricted_dims) == 1:
                    ec = 'white'
                elif n == 0:
                    ec = 'b'
                else:
                    ec = 'r'
                    
                # map categorical data onto the axes (bool is already mapped to
                # 0/1)
                if x.dtype.fields[field2][0].name == 'object':
                    elements = sorted(list(self.prim._box_init[field2][0]))
                    length = len(elements)
                    x_1 = [elements.index(v) / (length-1) for v in x_1]
                    
                if x.dtype.fields[field1][0].name == 'object':
                    elements = sorted(list(self.prim._box_init[field1][0]))
                    length = len(elements)
                    x_2 = [elements.index(v) / (length-1) for v in x_2]
                
                ax.scatter(x_1, x_2, facecolor=ec, edgecolor=ec, s=10)
                
                # provide appropriate labels for categorical data
                if x.dtype.fields[field1][0].name == 'bool':
                    ax.set_yticklabels(["False", "True"])
                    ax.set_yticks([0, 1])
                elif x.dtype.fields[field1][0].name == 'object':
                    elements = sorted(list(self.prim._box_init[field1][0]))
                    length = len(elements)
                    ax.set_yticklabels(elements)
                    ax.set_yticks([elements.index(v) / (length-1) for v in elements])
                    
                if x.dtype.fields[field2][0].name == 'bool':
                    ax.set_xticklabels(["False", "True"])
                    ax.set_xticks([0, 1])
                elif x.dtype.fields[field2][0].name == 'object':
                    elements = sorted(list(self.prim._box_init[field2][0]))
                    length = len(elements)
                    ax.set_xticklabels(elements)
                    ax.set_xticks([elements.index(v) / (length-1) for v in elements])
                
            ax.autoscale(tight=True)
    
            # draw boxlim
            if field1 != field2 or len(restricted_dims) == 1:
                x_1 = box_lim[field2]
                x_2 = box_lim[field1]

                if ((x.dtype.fields.get(field1)[0].name == 'bool' or
                        x.dtype.fields.get(field1)[0].name == 'object') and
                    (x.dtype.fields.get(field2)[0].name == 'bool' or
                        x.dtype.fields.get(field2)[0].name == 'object')):
                    elements1 = sorted(list(self.prim._box_init[field1][0]))
                    length1 = len(elements1)
                    selected_elements1 = box_lim[field1][0]
                    
                    elements2 = sorted(list(self.prim._box_init[field2][0]))
                    length2 = len(elements2)
                    selected_elements2 = box_lim[field2][0]
                    
                    for e in selected_elements1:
                        for f in selected_elements2:
                            v1 = elements1.index(e) / (length1-1)
                            v2 = elements2.index(f) / (length2-1)
                            for n in [0,1]:
                                ax.plot([v1 - 0.05, v1 + 0.05],
                                        [v2 + (0.05 if n == 0 else -0.05)]*2,
                                        c='k', linewidth=3)
                                ax.plot([v1 + (0.05 if n == 0 else -0.05)]*2,
                                        [v2 - 0.05, v2 + 0.05],
                                        c='k', linewidth=3)
                elif (x.dtype.fields.get(field2)[0].name == 'bool' or
                        x.dtype.fields.get(field2)[0].name == 'object'):
                    elements = sorted(list(self.prim._box_init[field2][0]))
                    length = len(elements)
                    selected_elements = box_lim[field2][0]
                    
                    for e in selected_elements:
                        v = elements.index(e) / (length-1)
                        for n in [0,1]:
                            ax.plot([v - 0.05, v + 0.05], [x_2[n], x_2[n]],
                                    c='k', linewidth=3)
                            ax.plot([v + (0.05 if n == 0 else -0.05)]*2, x_2,
                                    c='k', linewidth=3)
                elif (x.dtype.fields.get(field1)[0].name == 'bool' or
                        x.dtype.fields.get(field1)[0].name == 'object'):
                    elements = sorted(list(self.prim._box_init[field1][0]))
                    length = len(elements)
                    selected_elements = box_lim[field1][0]
                    
                    for e in selected_elements:
                        v = elements.index(e) / (length-1)
                        for n in [0,1]:
                            ax.plot([x_1[n], x_1[n]], [v - 0.05, v + 0.05],
                                    c='k', linewidth=3)
                            ax.plot(x_1, [v + (0.05 if n == 0 else -0.05)]*2,
                                    c='k', linewidth=3)
                else:
                    for n in [0,1]:
                        ax.plot(x_1, [x_2[n], x_2[n]], c='k', linewidth=3)
                        ax.plot([x_1[n], x_1[n]], x_2, c='k', linewidth=3)
                
            # update labels and ticks
            if len(restricted_dims) > 1:
                pairwise_labels(ax, i, j, field1, field2, None, restricted_dims)
                
        return figure

    def _calculate_quasi_p(self, i):
        """Calculates quasi-p values as discussed in Bryant and Lempert (2010).
        
        This is a one sided binomial test.
        
        Parameters
        ----------
        i : int
            the specific box in the peeling trajectory for which the quasi-p 
            values are to be calculated.
        
        Returns
        -------
        the quasi-p value
        """
        box_lim = self._box_lims[i]
        restricted_dims = list(determine_restricted_dims(
                box_lim,
                self.prim._box_init))
        
        # total nr. of cases in box
        Tbox = self.peeling_trajectory['mass'][i] * self.prim.n 
        
        # total nr. of cases of interest in box
        Hbox = self.peeling_trajectory['coverage'][i] * self.prim.t_coi  
        
        qp_values = {}
        
        for u in restricted_dims:
            temp_box = copy.deepcopy(box_lim)
            temp_box[u] = self._box_lims[0][u]
            
            indices = in_box(self.prim.x[self.prim.yi_remaining], 
                             temp_box)
            indices = self.prim.yi_remaining[indices]
            
            # total nr. of cases in box with one restriction removed
            Tj = indices.shape[0]  
            
            # total nr. of cases of interest in box with one restriction 
            # removed
            Hj = np.sum(self.prim.y[indices])
            
            p = Hj/Tj
            
            Hbox = int(Hbox)
            Tbox = int(Tbox)
            
            qp = binom.sf(Hbox-1, Tbox, p)
            qp_values[u] = qp
            
        return qp_values
    