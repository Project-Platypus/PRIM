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

import six
import operator
import logging
import functools
import numpy as np
import numpy.lib.recfunctions as rf
import pandas as pd
from .exceptions import PrimError
from .prim_box import PrimBox
from .prim_ops import (real_peel, discrete_peel, categorical_peel,
        real_paste, categorical_paste)
from .prim_objfcn import lenient1
from .scenario_discovery_util import (make_box, compare, get_sorted_box_lims,
        determine_nr_restricted_dims, determine_restricted_dims)

class Prim(object):
    """Patient Rule Induction Method.
    
    This implementation of PRIM is designed for scenario discovery.  It is based
    on the following paper:
    
        Friedman and Fisher (1999).  "Bump Hunting in High-Dimensional Data."
        Statistics and Computing, 9(2):123-143.

    This implementation of PRIM is datatype aware, so discrete and categorical
    variables will be handled appropriately.  Internally, this implementation
    uses Numpy.  The class-level dicts PEEL_OPERATIONS and PASTE_OPERATIONS map
    each Numpy datatype to a peeling or pasting operation appropriate for that
    datatype.   
    """
    
    PEEL_OPERATIONS = {'object' : categorical_peel,
                       'bool' : categorical_peel,
                       'int8' : discrete_peel,
                       'int16' : discrete_peel,
                       'int32' : discrete_peel,
                       'int64' : discrete_peel,
                       'uint8' : discrete_peel,
                       'uint16' : discrete_peel,
                       'uint32' : discrete_peel,
                       'uint64' : discrete_peel,
                       'float16' : real_peel,
                       'float32' : real_peel,
                       'float64' : real_peel}

    PASTE_OPERATIONS = {'object' : categorical_paste,
                        'bool' : categorical_paste,
                        'int8' : discrete_peel,
                        'int16' : real_paste,
                        'int32' : real_paste,
                        'int64' : real_paste,
                        'uint8' : discrete_peel,
                        'uint16' : real_paste,
                        'uint32' : real_paste,
                        'uint64' : real_paste,
                        'float16' : real_paste,
                        'float32' : real_paste,
                        'float64' : real_paste} 
    
    def __init__(self, 
                 x,
                 y, 
                 threshold = None, 
                 threshold_type = ">",
                 obj_func = lenient1, 
                 peel_alpha = 0.05, 
                 paste_alpha = 0.05,
                 mass_min = 0.05, 
                 include = None,
                 exclude = None,
                 coi = None):
        """Creates a new PRIM object.
        
        The PRIM object maintains the current state of the PRIM algorithm,
        recording the PRIM boxes found thus far, the remaining (uncaptured)
        cases of interest in the dataset, and provides methods for finding the
        next PRIM box and viewing statistics.
        
        Parameters
        ----------
        x : a matrix-like object (pandas.DataFrame, numpy.recarray, etc.)
            the independent variables
        y : a list-like object, the column name (str), or callable
            the dependent variable either provided as a list-like object
            classifying the data into cases of interest (e.g., False/True),
            a list-like object storing the raw variable value (in which case
            a threshold must be given), a string identifying the dependent
            variable in x, or a function called on each row of x to compute the
            dependent variable
        threshold : float
            threshold for identifying cases of interest
        threshold_type : str
            comparison operator used whwen identifying cases of interest
        obj_func : callable (default: lenient1)
            a function that computes the objective function (peeling criteria)
        peel_alpha : float (default: 0.05) 
            parameter controlling the peeling stage
        paste_alpha : float (default: 0.05)
            parameter controlling the pasting stage
        mass_min : float (default: 0.05)
            minimum mass of a box
        include : list of str
            the names of variables included in the PRIM analysis
        exclude : list of str
            the names of variables excluded from the PRIM analysis
        coi : str or list of str
            if y contains strings, coi identifies which string is the case of
            interest
        """
        
        # Ensure the input x is a numpy matrix/array
        if isinstance(x, pd.DataFrame):
            x = x.to_records(index=False)
        elif isinstance(x, np.ma.MaskedArray):
            pass
        else:
            x = pd.DataFrame(x).to_records(index=False)
            
        # if y is a string or function, compute the actual response value
        # otherwise, ensure y is a numpy matrix/array
        if isinstance(y, six.string_types):
            key = y
            y = x[key]
            
            if exclude:
                exclude = list(exclude) + [key]
            else:
                exclude = [key]
        elif six.callable(y):
            fun = y
            y = np.apply_along_axis(fun, 0, x)
        elif isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        elif isinstance(y, np.ma.MaskedArray):
            pass
        else:
            y = np.asarray(y)
            
        # convert include/exclude arguments to lists if they are strings
        if include and isinstance(include, six.string_types):
            include = [include]
            
        if exclude and isinstance(exclude, six.string_types):
            exclude = [exclude]     
            
        # include or exclude columns from the analysis
        if include:
            if isinstance(include, six.string_types):
                include = [include]

            drop_names = set(rf.get_names(x.dtype))-set(include)
            x = rf.drop_fields(x, drop_names, asrecarray=True)
        
        if exclude:
            if isinstance(exclude, six.string_types):
                exclude = [exclude]

            drop_names = set(exclude) 
            x = rf.drop_fields(x, drop_names, asrecarray=True)
            
        # apply the threshold if 
        if threshold:
            if six.callable(threshold):
                y = np.apply_along_axis(threshold, 0, y)
            else:
                # The syntax for threshold_type is "x <op> <threshold>", e.g.,
                # "x > 0.5".  However, partial only supports positional
                # arguments for built-in operators.  Thus, we must assign the
                # threshold to the first position and use a different operator.
                # For example, "x > 0.5" must be evaluated as "0.5 < x".
                OPERATORS = {"<=" : operator.ge,
                             ">=" : operator.le,
                             "<" : operator.gt,
                             ">" : operator.lt,
                             "=" : operator.eq}
                
                op = OPERATORS[threshold_type]
                y = np.apply_along_axis(functools.partial(op, threshold), 0, y)
                
        # validate inputs
        if len(y.shape) > 1:
            raise PrimError("y is not a 1-d array")
        
        unique_y = np.unique(y)
        
        if unique_y.shape[0] > 2:
            raise PrimError("y must contain only two values (0/1 or False/True)")
        
        if ((unique_y.shape[0] == 2 and (False not in unique_y or True not in unique_y)) or
                (False not in unique_y and True not in unique_y)):
            if coi is None:
                raise PrimError("y must contain only two values (0/1 or False/True)")
            else:
                if not hasattr(coi, "__iter__") and not isinstance(coi, six.string_types):
                    coi = [coi]
                y = np.asarray([1 if yi in coi else 0 for yi in y])
            
        # store the parameters       
        self.x = x
        self.y = y
        self.paste_alpha = paste_alpha
        self.peel_alpha = peel_alpha
        self.mass_min = mass_min
        self.threshold = threshold 
        self.threshold_type = threshold_type
        self.obj_func = obj_func
       
        # set the indices
        self.yi = np.arange(0, self.y.shape[0])
       
        # how many data points do we have
        self.n = self.y.shape[0]
        
        # how many cases of interest do we have?
        self.t_coi = self.determine_coi(self.yi)
        
        # initial box that contains all data
        self._box_init = make_box(self.x)
    
        # make a list in which the identified boxes can be put
        self._boxes = []
        
        # set yi_remaining to all y values
        self._update_yi_remaining()
    
    @property
    def stats(self):
        """Returns the statistics for all PRIM boxes found thus far.
        
        Returns the statistics for all PRIM boxes found and selected using
        :meth:`find_box`.
        
        Returns
        -------
        a Pandas DataFrame storing the stats for each box
        """
        stats = [box.stats for box in self._boxes]
        index = pd.Index(['Box %d' % (i+1) for i in range(len(stats))])
        return pd.DataFrame(stats, index=index)
    
    @property
    def limits(self):
        """Returns the limits for all PRIM boxes found thus far.
        
        Returns the lower and upper limits for all PRIM boxes found and selected
        using :meth:`find_box`.
        
        Returns
        -------
        a Pandas DataFrame storing the limits for each box
        """
        # collect the box limits into an array
        box_lims = [box._box_lims[box._cur_box] for box in self._boxes]
        
        if not box_lims:
            box_lims = [self._box_init]
        elif not np.all(compare(box_lims[-1], self._box_init)):
            box_lims.append(self._box_init)
            
        # determine the restricted dimensions
        nr_boxes = len(box_lims)
        box_lims, uncs = get_sorted_box_lims(box_lims, make_box(self.x))
        
        # determine the data type for the bounds
        dtype = float
        index = ["box {}".format(i+1) for i in range(nr_boxes)]
        for value in box_lims[0].dtype.fields.values():
            if value[0] == object:
                dtype = object
                break
                
        # create the data frame
        columns = pd.MultiIndex.from_product([index,
                                              ['min', 'max',]])
        df_boxes = pd.DataFrame(np.zeros((len(uncs), nr_boxes*2)),
                               index=uncs,
                               dtype=dtype,
                               columns=columns)

        for i, box in enumerate(box_lims):
            for unc in uncs:
                values = box[unc][:]
                values = pd.Series(values, 
                                   index=['min', 'max'])
                df_boxes.ix[unc][index[i]] = values  
                 
        return df_boxes 
    
    def perform_pca(self, subsets=None, exclude=set()):
        '''
        
        WARNING:: code still needs to be tested!!!
        
        Pre-process the data by performing a pca based rotation on it. 
        This effectively turns the algorithm into PCA-PRIM as described
        in `Dalal et al (2013) <http://www.sciencedirect.com/science/article/pii/S1364815213001345>`_
        
        Parameters
        ----------
        subsets: dict, optional 
                 expects a dictionary with group name as key and a list of 
                 uncertainty names as values. If this is used, a constrained 
                 PCA-PRIM is executed 
                 
                ..note:: the list of uncertainties should not contain 
                         categorical uncertainties. 
        exclude : list of str, optional 
                  the uncertainties that should be excluded from the rotation
        
        '''
        
        #transform experiments to numpy array
        dtypes = self.x.dtype.fields
        object_dtypes = [key for key, value in dtypes.items() 
                         if value[0]==np.dtype(object)]
        
        #get experiments of interest
        # TODO this assumes binary classification!!!!!!!
        logical = self.y>=self.threshold
        
        # if no subsets are provided all uncertainties with non dtype object 
        # are in the same subset, the name of this is r, for rotation
        if not subsets:
            subsets = {"r":[key for key, value in dtypes.items() 
                            if value[0].name!=np.dtype(object)]}
        else:
            # remove uncertainties that are in exclude and check whether 
            # uncertainties occur in more then one subset
            seen = set()
            for key, value in subsets.items():
                value = set(value) - set(exclude)
    
                subsets[key] = list(value)
                if (seen & value):
                    raise PrimError("uncertainty occurs in more then one subset")
                else:
                    seen = seen | set(value)
            
        #prepare the dtypes for the new rotated experiments recarray
        new_dtypes = []
        for key, value in subsets.items():
            self._assert_dtypes(value, dtypes)
            
            # the names of the rotated columns are based on the group name 
            # and an index
            [new_dtypes.append((str("{}_{}".format(key, i)), float)) for i 
             in range(len(value))]
        
        #add the uncertainties with object dtypes to the end
        included_object_dtypes = set(object_dtypes)-set(exclude)
        [new_dtypes.append((name, object)) for name in included_object_dtypes]
        
        #make a new empty recarray
        rotated_experiments = np.empty((self.x.shape[0],), dtype=new_dtypes)
        
        #put the uncertainties with object dtypes already into the new recarray 
        for name in included_object_dtypes :
            rotated_experiments[name] = self.x[name]
        
        #iterate over the subsets, rotate them, and put them into the new 
        # recarray
        shape = 0
        for key, value in subsets.items():
            shape += len(value) 
        rotation_matrix = np.zeros((shape,shape))
        column_names = []
        row_names = []
        
        j = 0
        for key, value in subsets.items():
            data = self._rotate_subset(value, self.x, logical)
            subset_rotation_matrix, subset_experiments = data 
            rotation_matrix[j:j+len(value), j:j+len(value)] = subset_rotation_matrix
            [row_names.append(entry) for entry in value]
            j += len(value)
            
            for i in range(len(value)):
                name = "%s_%s" % (key, i)
                rotated_experiments[name] = subset_experiments[:,i]
                [column_names.append(name)]
        
        self.rotation_matrix = rotation_matrix
        self.column_names = column_names
        self.row_names = row_names
        
        self.x = np.ma.array(rotated_experiments)
        self._box_init = make_box(self.x)
    
    def find_box(self):
        """Execute one iteration of the PRIM algorithm.
        
        Finds the next PRIM box starting from the current state of PRIM.  All
        previous boxes are frozen and can no longer be modified.
        
        Returns
        -------
        the PRIM box, or None if there are no cases of interest remaining
        """
        logger = logging.getLogger(__name__)
        
        # set the indices
        self._update_yi_remaining()
        
        # make boxes already found immutable 
        for box in self._boxes:
            box._frozen = True
        
        if self.yi_remaining.shape[0] == 0:
            logger.info("no data remaining")
            return
        
        # log how much data and how many coi are remaining
        logger.info("%d points remaining, containing %d cases of interest" %
                    (self.yi_remaining.shape[0],
                     self.determine_coi(self.yi_remaining)))
        
        # make a new box that contains all the remaining data points
        box = PrimBox(self, self._box_init, self.yi_remaining[:])
        
        #  perform peeling phase
        logger.debug("peeling started")
        box = self._peel(box)
        logger.debug("peeling completed")

        # perform pasting phase       
        logger.debug("pasting started") 
        box = self._paste(box)
        logger.debug("pasting completed")
        
        logger.info("mean: %f, mass: %f, coverage: %f, density: %f restricted_dimensions: %d" %
                    (box.mean, box.mass, box.coverage, box.density, box.res_dim))
        
        self._boxes.append(box)
        return box
    
    def find_all(self):
        """Runs the entire PRIM algorithm to find all boxes.
        
        Iteratively calls :meth:`find_box` to generate all PRIM boxes.  This
        method is provided for convenience and will select the last peeling
        trajectory for every box.  Calling :meth:`find_box` is recommended as
        it allows interactive selection of the desired peeling trajectory.
        
        Returns
        -------
        list of PRIM boxes
        """
        boxes = []
        box = self.find_box()
        
        while box is not None:
            boxes.append(box)
            box = self.find_box()
            
        return boxes

    def determine_coi(self, indices):
        '''        
        Given a set of indices on y, how many cases of interest are there in 
        this set.
        
        Parameters
        ----------
        indices: ndarray
                 a valid index for y

        Returns
        ------- 
        int
            the number of cases of interest.
        
        Raises
        ------
        ValueError 
            if threshold_type is not either ABOVE or BELOW

        '''
        
        y = self.y[indices]
        coi = y[y == True].shape[0]
        return coi
    
    def _update_yi_remaining(self):
        '''
        
        Update yi_remaining in light of the state of the boxes associated
        with this prim instance.
        
        '''
        
        # set the indices
        logical = np.ones(self.yi.shape[0],dtype=np.bool )
        for box in self._boxes:
            logical[box.yi] = False
        self.yi_remaining = self.yi[logical]
    
    def _peel(self, box):
        """Performs the peeling phase of the PRIM algorithm.
        
        Delegates peeling to a specific peel method based on the data type.
        
        Parameters
        ----------
        box : a PrimBox instance
              the original box
        
        Returns
        -------
        a new box resulting from the peel operation, or the original box if
        no peeling was performed
        """
        #identify all possible peels
        possible_peels = []
        
        for entry in self.x.dtype.descr:
            name = entry[0]
            dtype = self.x.dtype.fields.get(name)[0].name
            
            if dtype not in Prim.PEEL_OPERATIONS:
                raise PrimError("no peel operation defined for type %s" % dtype)
            
            peels = Prim.PEEL_OPERATIONS[dtype](self, box, name)
            possible_peels += peels
            
        # if there are no peels identified, return the unchanged box
        if len(possible_peels) == 0:
            return box

        # determine the scores for each peel in order to identify the next
        # candidate box
        scores = []
        
        for i, box_lim in possible_peels:
            obj = self.obj_func(self.y[box.yi], self.y[i])
            non_res_dim = len(self.x.dtype.descr)-\
                          determine_nr_restricted_dims(box_lim, 
                                                       self._box_init)
            score = (obj, non_res_dim, box_lim, i)
            scores.append(score)

        scores.sort(key=operator.itemgetter(0,1), reverse=True)
        obj_score, non_res_dim, box_new, indices = scores[0]
        
        # if the best peel results in an improvement, return the peel;
        # otherwise return the unchanged box
        mass_old = box.yi.shape[0]/self.n
        mass_new = self.y[indices].shape[0]/self.n
        
        if mass_new >= self.mass_min and mass_new < mass_old and obj_score > 0:
            box.update(box_new, indices)
            return self._peel(box)
        else:
            return box
    
    def _paste(self, box):
        ''' Executes the pasting phase of the PRIM. Delegates pasting to data 
        type specific helper methods.'''
        
        x = self.x[self.yi_remaining]
        res_dim = determine_restricted_dims(box._box_lims[-1],
                                            self._box_init)
        
        #identify all possible pastes
        possible_pastes = []
        
        for u in res_dim:
            logging.getLogger(__name__).info("pasting "+u)
            dtype = self.x.dtype.fields.get(u)[0].name
            
            if dtype not in Prim.PASTE_OPERATIONS:
                raise PrimError("no paste operation defined for type %s" % dtype)
            
            pastes = Prim.PASTE_OPERATIONS[dtype](self, box, u)
            [possible_pastes.append(entry) for entry in pastes] 
            
        # if there are no pastes identified, return the unchanged box
        if not possible_pastes:
            return box
    
        # determine the scores for each peel in order to identify the next
        # candidate box
        scores = []
        for entry in possible_pastes:
            i, box_lim = entry
            obj = self.obj_func(self.y[box.yi], self.y[i])
            non_res_dim = len(x.dtype.descr)-\
                          determine_nr_restricted_dims(box_lim,
                                                       self._box_init)
            score = (obj, non_res_dim, box_lim, i)
            scores.append(score)

        scores.sort(key=operator.itemgetter(0,1), reverse=True)
        obj, _, box_new, indices = scores[0]
        
        # if the best paste results in an improvement, return the paste;
        # otherwise return the unchanged box
        mass_old = box.yi.shape[0]/self.n
        mass_new = self.y[indices].shape[0]/self.n
        
        mean_old = np.mean(self.y[box.yi])
        mean_new = np.mean(self.y[indices])
        
        if mass_new >= self.mass_min and mass_new > mass_old and obj > 0 and \
                mean_new > mean_old:
            box.update(box_new, indices)
            return self._paste(box)
        else:
            return box
    
    def _assert_dtypes(self, keys, dtypes):
        '''
        helper fucntion that checks whether none of the provided keys has
        a dtype object as value.
        '''
        
        for key in keys:
            if dtypes[key][0] == np.dtype(object):
                raise PrimError("%s has dtype object and can thus not be rotated" %key)
        return True

    def _rotate_subset(self, value, orig_experiments, logical): 
        '''
        rotate a subset
        
        Parameters
        ----------
        value : list of str
        orig_experiment : numpy structured array
        logical : boolean array
        
        '''
        list_dtypes = [(name, "<f8") for name in value]
        
        #cast everything to float
        drop_names = set(rf.get_names(orig_experiments.dtype)) - set(value)
        orig_subset = rf.drop_fields(orig_experiments, drop_names, 
                                               asrecarray=True)
        subset_experiments = orig_subset.astype(list_dtypes).view('<f8').reshape(orig_experiments.shape[0], len(value))
 
        #normalize the data
        mean = np.mean(subset_experiments,axis=0)
        std = np.std(subset_experiments, axis=0)
        std[std==0] = 1 #in order to avoid a devision by zero
        subset_experiments = (subset_experiments - mean)/std
        
        #get the experiments of interest
        experiments_of_interest = subset_experiments[logical]
        
        #determine the rotation
        rotation_matrix =  self._determine_rotation(experiments_of_interest)
        
        #apply the rotation
        subset_experiments = np.dot(subset_experiments,rotation_matrix)
        return rotation_matrix, subset_experiments

    def _determine_rotation(self, experiments):
        '''
        Determine the rotation for the specified experiments
        
        '''
        covariance = np.cov(experiments.T)
        
        eigen_vals, eigen_vectors = np.linalg.eig(covariance)
    
        indices = np.argsort(eigen_vals)
        indices = indices[::-1]
        eigen_vectors = eigen_vectors[:,indices]
        eigen_vals = eigen_vals[indices]
        
        #make the eigen vectors unit length
        for i in range(eigen_vectors.shape[1]):
            eigen_vectors[:,i] / np.linalg.norm(eigen_vectors[:,i]) * np.sqrt(eigen_vals[i])
            
        return eigen_vectors

