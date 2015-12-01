# Copyright 2015 David Hadka
#
# This file is part of the PRIM module.
#
# PRIM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PRIM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PRIM.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import, division

import copy
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import mpldatacursor
from operator import itemgetter
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1 import host_subplot
from prim.exceptions import PRIMError
from prim import pairs_plotting
from prim import scenario_discovery_util as sdutil
from prim.plotting_util import make_legend

try:
    import mpld3
except ImportError:
    logging.getLogger(__name__).info("mpld3 library not found, some functionality will be disabled")
    global mpld3
    mpld3 = None
    
def _pair_wise_scatter(x,y, box_lim, restricted_dims, grid=None):
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
            
            ax.scatter(x_1, x_2, facecolor=ec, edgecolor=ec, s=10)
            
        ax.autoscale(tight=True)

        # draw boxlim
        if field1 != field2 or len(restricted_dims) == 1:
            x_1 = box_lim[field2]
            x_2 = box_lim[field1]
    
            for n in [0,1]:
                ax.plot(x_1,
                        [x_2[n], x_2[n]], c='k', linewidth=3)
                ax.plot([x_1[n], x_1[n]],
                        x_2, c='k', linewidth=3)
            
#       #reuse labeling function from pairs_plotting
        if len(restricted_dims) > 1:
            pairs_plotting.do_text_ticks_labels(ax, i, j, field1, field2, None, 
                                            restricted_dims)
            
    return figure

class CurEntry(object):
    '''a descriptor for the current entry on the peeling and pasting 
    trajectory'''
    
    def __init__(self, name):
        self.name = name
        
    def __get__(self, instance, owner):
        print instance.peeling_trajectory[self.name]
        return instance.peeling_trajectory[self.name][instance._cur_box]
    
    def __set__(self, instance, value):
        raise PRIMError("this property cannot be assigned to")

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
    
    coverage = CurEntry('coverage')
    density = CurEntry('density')
    mean = CurEntry('mean')
    res_dim = CurEntry('res dim')
    mass = CurEntry('mass')
    
    _frozen=False
    
    def __init__(self, prim, box_lims, indices):
        '''init 
        
        Parameters
        ----------
        prim : Prim instance
        box_lims : recarray
        indices : ndarray
        
        
        '''
        
        self.prim = prim
        
        # peeling and pasting trajectory
        colums = ['coverage', 'density', 'mean', 'res dim', 'mass']
        self.peeling_trajectory = pd.DataFrame(columns=colums)
        
        self.box_lims = []
        self._cur_box = -1
        
        # indices van data in box
        self.update(box_lims, indices)

    def __getattr__(self, name):
        '''
        used here to give box_lim same behaviour as coverage, density, mean
        res_dim, and mass. That is, it will return the box lim associated with
        the currently selected box. 
        '''
        
        if name=='box_lim':
            return self.box_lims[self._cur_box]
        else:
            raise AttributeError

    def inspect(self, i=None, style='table'):
        '''
        
        Write the stats and box limits of the user specified box to standard 
        out. if i is not provided, the last box will be printed
        
        Parameters
        ----------
        i : int, optional
            the index of the box, defaults to currently selected box
        style : {'table', 'graph'}
                the style of the visualization
        
        '''
        if i == None:
            i = self._cur_box
        
        stats = self.peeling_trajectory.iloc[i].to_dict()
        stats['restricted_dim'] = stats['res dim']
        print stats

        qp_values = self._calculate_quasi_p(i)
        uncs = [(key, value) for key, value in qp_values.items()]
        uncs.sort(key=itemgetter(1))
        uncs = [uncs[0] for uncs in uncs]
        
        if style == 'table':
            return self._inspect_table(i, uncs, qp_values)
        elif style == 'graph':
            return self._inspect_graph(i, uncs, qp_values)
        else:
            raise ValueError("style must be one of graph or table")
            
    def _inspect_table(self, i, uncs, qp_values):
        '''Helper function for visualizing box statistics in 
        table form'''
        #make the descriptive statistics for the box
        i = 19
        print(self.peeling_trajectory.iloc[i])
        print()
        
        # make the box definition
        columns = pd.MultiIndex.from_product([['box {}'.format(i)],
                                              ['min', 'max', 'qp values']])
        box_lim = pd.DataFrame(np.zeros((len(uncs), 3)), 
                               index=uncs, 
                               columns=columns)
        
        for unc in uncs:
            values = self.box_lims[i][unc][:]
            box_lim.loc[unc] = [values[0], values[1], qp_values[unc]]
             
        print(box_lim)
        print()
        

        
    def show_box_details(self, fig=None):
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
          
        self.show_pairs_scatter(grid=inner_grid)
          
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1,
            subplot_spec=outer_grid[1], wspace=0.0, hspace=0.0)
          
        ax1 = plt.Subplot(fig, inner_grid[0])
          
        fig.add_subplot(ax1)
        self.show_box()
        ax1.set_title("Restricted Dimensions", y=1.08)
          
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
        
        def next(event):
            i = (self._cur_box + 1) % self.peeling_trajectory.shape[0]
            self.select(i)
            self.show_box_details(fig=event.canvas.figure)
            
        def prev(event):
            i = (self._cur_box - 1) % self.peeling_trajectory.shape[0]
            self.select(i)
            self.show_box_details(fig=event.canvas.figure)

        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, "Next")
        self.bprev = Button(axprev, "Prev")
        self.bnext.on_clicked(next)
        self.bprev.on_clicked(prev)
        
        plt.subplots_adjust(top=0.85)
        plt.draw()
        
        return fig
        
        
    def show_box(self, ax=None):
        i = self._cur_box
        qp_values = self._calculate_quasi_p(i)
        uncs = [(key, value) for key, value in qp_values.items()]
        uncs.sort(key=itemgetter(1))
        uncs = [uncs[0] for uncs in uncs]
        
        box_lim_init = self.prim.box_init
        box_lim = self.box_lims[i]
        norm_box_lim =  sdutil._normalize(box_lim, box_lim_init, uncs)
        
        left = []
        height = []
        bottom = []
        
        for i, _ in enumerate(uncs):
            left.append(i)
            height.append(norm_box_lim[i][1]-norm_box_lim[i][0])
            bottom.append(norm_box_lim[i][0])
        
        plt.bar(left, 
                height,
                width = 0.6,
                bottom = bottom,
                align="center")
        plt.ylim(0, 1)
        plt.xticks(left, uncs)
        plt.tick_params(axis='y',
                        which='both',
                        right='off',
                        left='off',
                        labelleft='off')
        
        fig = plt.gcf()
        ax = plt.gca()
        
        for i, _ in enumerate(uncs):
            ax.text(i - 0.15,
                    norm_box_lim[i][0], "%0.2f" % norm_box_lim[i][0],
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    color='w')
            
            ax.text(i + 0.15,
                    norm_box_lim[i][1], "%0.2f" % norm_box_lim[i][1],
                    horizontalalignment='center',
                    verticalalignment='top',
                    color='w')
            
        return fig
        
    def _inspect_graph(self,  i, uncs, qp_values):
        '''Helper function for visualizing box statistics in 
        graph form'''        
        
        # normalize the box lims
        # we don't need to show the last box, for this is the 
        # box_init, which is visualized by a grey area in this
        # plot.
        box_lim_init = self.prim.box_init
        box_lim = self.box_lims[i]
        norm_box_lim =  sdutil._normalize(box_lim, box_lim_init, uncs)
        
        fig, ax = sdutil._setup_figure(uncs)

        for j, u in enumerate(uncs):
            # we want to have the most restricted dimension
            # at the top of the figure
            xj = len(uncs) - j - 1

            self.prim._plot_unc(box_lim_init, xj, j, 0, norm_box_lim, box_lim, 
                                u, ax)

            # new part
            dtype = box_lim_init[u].dtype
            
            props = {'facecolor':'white',
                     'edgecolor':'white',
                     'alpha':0.25}
            y = xj

        
            if dtype == object:
                pass
                elements = sorted(list(box_lim_init[u][0]))
                max_value = (len(elements)-1)
                values = box_lim[u][0]
                x = [elements.index(entry) for entry in 
                     values]
                x = [entry/max_value for entry in x]
                
                for xi, label in zip(x, values):
                    ax.text(xi, y-0.1, label, ha='center', va='center',
                           bbox=props, color='blue', fontweight='normal')

            else:
                props = {'facecolor':'white',
                         'edgecolor':'white',
                         'alpha':0.25}
    
                # plot limit text labels
                x = norm_box_lim[j][0]
    
                if not np.allclose(x, 0):
                    label = "{: .2g}".format(self.box_lims[i][u][0])
                    ax.text(x-0.01, y, label, ha='right', va='center',
                           bbox=props, color='blue', fontweight='normal')
    
                x = norm_box_lim[j][1]
                if not np.allclose(x, 1):
                    label = "{: .2g}".format(self.box_lims[i][u][1])
                    ax.text(x+0.01, y, label, ha='left', va='center',
                           bbox=props, color='blue', fontweight='normal')

                # plot uncertainty space text labels
                x = 0
                label = "{: .2g}".format(box_lim_init[u][0])
                ax.text(x-0.01, y, label, ha='right', va='center',
                       bbox=props, color='black', fontweight='normal')
    
                x = 1
                label = "{: .2g}".format(box_lim_init[u][1])
                ax.text(x+0.01, y, label, ha='left', va='center',
                       bbox=props, color='black', fontweight='normal')
                
            # set y labels
            labels = ["{} ({:.2g})".format(u, qp_values[u]) for u in uncs]
            labels = labels[::-1]
            ax.set_yticklabels(labels)

            # remove x tick labels
            ax.set_xticklabels([])

            # add table to the left
            coverage = '{:.3g}'.format(self.peeling_trajectory['coverage'][i])
            density = '{:.3g}'.format(self.peeling_trajectory['density'][i])
            
            ax.table(cellText=[[coverage], [density]],
                    colWidths = [0.1]*2,
                    rowLabels=['coverage', 'density'],
                    colLabels=None,
                    loc='right',
                    bbox=[1.1, 0.9, 0.1, 0.1])
        
            #plt.tight_layout()
        return fig
        
    def select(self, i):
        '''        
        select an entry from the peeling and pasting trajectory and update
        the prim box to this selected box.
        
        Parameters
        ----------
        i : int
            the index of the box to select.
        
        '''
        if self._frozen:
            raise PRIMError("""box has been frozen because PRIM has found 
                                at least one more recent box""")
        
        indices = sdutil._in_box(self.prim.x[self.prim.yi_remaining], 
                                 self.box_lims[i])
        self.yi = self.prim.yi_remaining[indices]
        self._cur_box = i

    def drop_restriction(self, uncertainty):
        '''
        drop the restriction on the specified dimension. That is, replace
        the limits in the chosen box with a new box where for the specified 
        uncertainty the limits of the initial box are being used. The resulting
        box is added to the peeling trajectory.
        
        Parameters
        ----------
        uncertainty : str
        
        '''
        
        new_box_lim = copy.deepcopy(self.box_lim)
        new_box_lim[uncertainty][:] = self.box_lims[0][uncertainty][:]
        indices = sdutil._in_box(self.prim.x[self.prim.yi_remaining], 
                                 new_box_lim)
        indices = self.prim.yi_remaining[indices]
        self.update(new_box_lim, indices)
        
    def update(self, box_lims, indices):
        '''
        
        update the box to the provided box limits.
        
        Parameters
        ----------
        box_lims: numpy recarray
                  the new box_lims
        indices: ndarray
                 the indices of y that are inside the box
      
        '''
        self.yi = indices
        
        y = self.prim.y[self.yi]

        self.box_lims.append(box_lims)

        coi = self.prim.determine_coi(self.yi)

        data = {'coverage':coi/self.prim.t_coi, 
                'density':coi/y.shape[0],  
                'mean':np.mean(y),
                'res dim':sdutil._determine_nr_restricted_dims(self.box_lims[-1], 
                                                              self.prim.box_init),
                'mass':y.shape[0]/self.prim.n}
        new_row = pd.DataFrame([data])
        self.peeling_trajectory = self.peeling_trajectory.append(new_row, 
                                                             ignore_index=True)
        
        self._cur_box = len(self.peeling_trajectory)-1
        
    def show_ppt(self):
        '''show the peeling and pasting trajectory in a figure'''
        
        ax = host_subplot(111)
        ax.set_xlabel("peeling and pasting trajectory")
        
        par = ax.twinx()
        par.set_ylabel("nr. restricted dimensions")
            
        ax.plot(self.peeling_trajectory['mean'], label="mean")
        ax.plot(self.peeling_trajectory['mass'], label="mass")
        ax.plot(self.peeling_trajectory['coverage'], label="coverage")
        ax.plot(self.peeling_trajectory['density'], label="density")
        par.plot(self.peeling_trajectory['res dim'], label="restricted dims")
        ax.grid(True, which='both')
        ax.set_ylim(ymin=0,ymax=1)
        
        fig = plt.gcf()
        
        make_legend(['mean', 'mass', 'coverage', 'density', 'restricted_dim'],
                    ax, ncol=5, alpha=1)
        return fig
    
    def formatter(self, **kwargs):
        i = kwargs.get("ind")[0]
        data = self.peeling_trajectory.ix[i]
        label = "Box %d\nCoverage: %2.1f%%\nDensity: %2.1f%%\nMass: %2.1f%%\nRes Dim: %d" % (i, 100*data["coverage"], 100*data["density"], 100*data["mass"], data["res dim"])
        return label
    
    def handle_click(self, event):
        #if event.mouseevent.dblclick:
        i = event.ind[0]
        self.select(i)
            
        if event.mouseevent.button == 1:
            self.show_box_details().show()
    
    def show_tradeoff(self):
        '''Visualize the trade off between coverage and density. Color is used
        to denote the number of restricted dimensions.'''
       
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        
        cmap = mpl.cm.YlGnBu_r #@UndefinedVariable
        boundaries = np.arange(-0.5, 
                               max(self.peeling_trajectory['res dim'])+1.5, 
                               step=1)
        ncolors = cmap.N
        norm = mpl.colors.BoundaryNorm(boundaries, ncolors)
        
        p = ax.scatter(self.peeling_trajectory['coverage'], 
                       self.peeling_trajectory['density'], 
                       c=self.peeling_trajectory['res dim'], 
                       norm=norm,
                       cmap=cmap,
                       picker=True)

        ax.set_ylabel('density')
        ax.set_xlabel('coverage')
        ax.set_ylim(ymin=0, ymax=1.2)
        ax.set_xlim(xmin=0, xmax=1.2)
        
        mpldatacursor.datacursor(formatter=self.formatter, hover=True)
        fig.canvas.mpl_connect('pick_event', self.handle_click)
        
        ticklocs = np.arange(0, 
                             max(self.peeling_trajectory['res dim'])+1, 
                             step=1)
        cb = fig.colorbar(p, spacing='uniform', ticks=ticklocs, drawedges=True)
        cb.set_label("nr. of restricted dimensions")
        
        # make the tooltip tables
        if mpld3:
            # Define some CSS to control our custom labels
            css = """
            table
            {
              border-collapse: collapse;
            }
            th
            {
              background-color:  rgba(255,255,255,0.95);
            }
            td
            {
              background-color: rgba(255,255,255,0.95);
            }
            table, th, td
            {
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
    
    def show_pairs_scatter(self, grid=None):
        '''
        
        make a pair wise scatter plot of all the restricted dimensions
        with color denoting whether a given point is of interest or not
        and the boxlims superimposed on top.
        
        '''   
        fig = _pair_wise_scatter(self.prim.x[self.prim.yi_remaining], self.prim.y[self.prim.yi_remaining], self.box_lim, 
                           sdutil._determine_restricted_dims(self.box_lim, 
                                                        self.prim.box_init),
                            grid = grid)
        
        title = "Box %d" % self._cur_box
        fig.suptitle(title, fontsize=16)
        fig.canvas.set_window_title(title)
        return fig
    
    def write_ppt_to_stdout(self):
        '''write the peeling and pasting trajectory to stdout'''
        print(self.peeling_trajectory)
        print("\n")

    def _calculate_quasi_p(self, i):
        '''helper function for calculating quasi-p values as discussed in 
        Bryant and Lempert (2010). This is a one sided  binomial test. 
        
        Parameters
        ----------
        i : int
            the specific box in the peeling trajectory for which the quasi-p 
            values are to be calculated.
        
        '''
        from scipy.stats import binom
        
        box_lim = self.box_lims[i]
        restricted_dims = list(sdutil._determine_restricted_dims(box_lim,
                                                           self.prim.box_init))
        print restricted_dims
        
        # total nr. of cases in box
        Tbox = self.peeling_trajectory['mass'][i] * self.prim.n 
        
        # total nr. of cases of interest in box
        Hbox = self.peeling_trajectory['coverage'][i] * self.prim.t_coi  
        
        qp_values = {}
        
        for u in restricted_dims:
            temp_box = copy.deepcopy(box_lim)
            temp_box[u] = self.box_lims[0][u]
        
            indices = sdutil._in_box(self.prim.x[self.prim.yi_remaining], 
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

    def _format_stats(self, nr, stats):
        '''helper function for formating box stats'''
        row = self.stats_format.format(nr,**stats)
        return row
    