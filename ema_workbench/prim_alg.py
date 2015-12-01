'''

A scenario discovery oriented implementation of PRIM.

The implementation of prim provided here is datatype aware, so
categorical variables will be handled appropriately. It also uses a 
non-standard objective function in the peeling and pasting phase of the
algorithm. This algorithm looks at the increase in the mean divided 
by the amount of data removed. So essentially, it uses something akin
to the first order derivative of the original objective function. 

The implementation is designed for interactive use in combination with the
ipython notebook. 

'''
from __future__ import (absolute_import, division,
                        unicode_literals)
import six

from operator import itemgetter
import operator
import copy
import math
import logging
import functools

import numpy as np
import numpy.lib.recfunctions as rf

import pandas as pd
from ema_workbench.exceptions import PRIMError
from ema_workbench.prim_box import PrimBox
from ema_workbench import scenario_discovery_util as sdutil

# Created on 22 feb. 2013
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

LENIENT2 = 'lenient2'
LENIENT1 = 'lenient1'
ORIGINAL = 'original'

PRECISION = '.2f'


def get_quantile(data, quantile):
    '''
    quantile calculation modeled on the implementation used in sdtoolkit

    Parameters
    ----------
    data : nd array like 
           dataset for which quantile is needed
    quantile : float
               the desired quantile
    
    '''
    assert quantile>0
    assert quantile<1
 
    data = np.sort(data)
    print type(data)
    
    i = (len(data)-1)*quantile
    index_lower =  int(math.floor(i))
    index_higher = int(math.ceil(i))
    
    value = 0

    if quantile > 0.5:
        # upper
        while (data[index_lower] == data[index_higher]) & (index_lower>0):
            index_lower -= 1
        value = (data[index_lower]+data[index_higher])/2
    else:
        #lower
        while (data[index_lower] == data[index_higher]) & \
              (index_higher<len(data)-1):
            index_higher += 1
        value = (data[index_lower]+data[index_higher])/2

    return value
    

class Prim(sdutil.OutputFormatterMixin):
    '''Patient rule induction algorithm
    
    The implementation of Prim is tailored to interactive use in the context
    of scenario discovery

    Parameters
    ----------
    x : structured array
        the independent variables
    y : 1d ndarray
        the dependent variable
    threshold : float
                the coverage threshold that a box has to meet
    obj_function : {LENIENT1, LENIENT2, ORIGINAL}
                   the objective function used by PRIM. Defaults to a lenient 
                   objective function based on the gain of mean divided by the 
                   loss of mass. 
    peel_alpha : float, optional 
                 parameter controlling the peeling stage (default = 0.05). 
    paste_alpha : float, optional
                  parameter controlling the pasting stage (default = 0.05).
    mass_min : float, optional
               minimum mass of a box (default = 0.05). 
    threshold_type : {ABOVE, BELOW}
                     whether to look above or below the threshold value
  
        
    See also
    --------
    :mod:`cart`
    
    
    '''
    
    message = "{0} points remaining, containing {1} cases of interest"
    
    def __init__(self, 
                 x,
                 y, 
                 threshold=None, 
                 threshold_type=">",
                 obj_function=LENIENT1, 
                 peel_alpha=0.05, 
                 paste_alpha=0.05,
                 mass_min=0.05, 
                 include=None,
                 exclude=None):
        
        # Ensure the input x is a numpy matrix/array
        if isinstance(x, pd.DataFrame):
            x = x.to_records(index=False)
        else:
            x = np.asarray(x)
            
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
            print y   
        elif isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
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
                OPERATORS = {"<" : operator.ge,
                             ">" : operator.le,
                             "<=" : operator.gt,
                             ">=" : operator.lt,
                             "=" : operator.eq}
                op = OPERATORS[threshold_type]
                y = np.apply_along_axis(functools.partial(op, threshold), 0, y)
                
        # validate inputs
        if len(y.shape) > 1:
            raise PRIMError("y is not a 1-d array")
        
        unique_y = np.unique(y)
        
        if (unique_y.shape[0] > 2 or 
                (unique_y.shape[0] == 2 and (False not in unique_y or
                                             True not in unique_y)) or
                (False not in unique_y and True not in unique_y)):
            raise PRIMError("y must contain only two values (0/1 or False/True)")
            
        # store the parameters       
        self.x = x
        self.y = y
        self.paste_alpha = paste_alpha
        self.peel_alpha = peel_alpha
        self.mass_min = mass_min
        self.threshold = threshold 
        self.threshold_type = threshold_type
        self.obj_func = self._obj_functions[obj_function]
       
        # set the indices
        self.yi = np.arange(0, self.y.shape[0])
       
        # how many data points do we have
        self.n = self.y.shape[0]
        
        # how many cases of interest do we have?
        self.t_coi = self.determine_coi(self.yi)
        
        # initial box that contains all data
        self.box_init = sdutil._make_box(self.x)
    
        # make a list in which the identified boxes can be put
        self._boxes = []
        
        self._update_yi_remaining()
    
    @property
    def boxes(self):
        boxes = [box.box_lim for box in self._boxes]
        
        if not boxes:
            return [self.box_init]
        elif not np.all(sdutil._compare(boxes[-1], self.box_init)):
                boxes.append(self.box_init)
        return boxes 
    
    @property
    def stats(self):
        stats = []
        items = ['coverage','density', 'mass', 'res_dim']
        for box in self._boxes:
            stats.append({key: getattr(box, key) for key in items})
        return stats
    
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
                    raise PRIMError("uncertainty occurs in more then one subset")
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
        self.box_init = sdutil._make_box(self.x)
    
    def find_box(self):
        '''Execute one iteration of the PRIM algorithm. That is, find one
        box, starting from the current state of Prim.'''
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
        logger.info(self.message.format(self.yi_remaining.shape[0],
                                 self.determine_coi(self.yi_remaining)))
        
        # make a new box that contains all the remaining data points
        box = PrimBox(self, self.box_init, self.yi_remaining[:])
        
        #  perform peeling phase
        box = self._peel(box)
        logger.debug("peeling completed")

        # perform pasting phase        
        box = self._paste(box)
        logger.debug("pasting completed")
        
        message = "mean: {0}, mass: {1}, coverage: {2}, density: {3} restricted_dimensions: {4}"
        message = message.format(box.mean,
                                 box.mass,
                                 box.coverage,
                                 box.density,
                                 box.res_dim)

        logger.info(message)
        self._boxes.append(box)
        return box
#         if (self.threshold_type==ABOVE) &\
#            (box.mean >= self.threshold):
#             logger.info(message)
#             self._boxes.append(box)
#             return box
#         elif (self.threshold_type==BELOW) &\
#            (box.mean <= self.threshold):
#             logger.info(message)
#             self._boxes.append(box)
#             return box
#         else:
#             # make a dump box
#             logger.info('box does not meet threshold criteria, value is {}, returning dump box'.format(box.mean))
#             box = PrimBox(self, self.box_init, self.yi_remaining[:])
#             self._boxes.append(box)
#             return box

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
        '''
        
        Executes the peeling phase of the PRIM algorithm. Delegates peeling
        to data type specific helper methods.

        '''
    
        mass_old = box.yi.shape[0]/self.n

        x = self.x[box.yi]
       
        #identify all possible peels
        possible_peels = []
        for entry in x.dtype.descr:
            u = entry[0]
            dtype = x.dtype.fields.get(u)[0].name
            peels = self._peels[dtype](self, box, u, x)
            [possible_peels.append(entry) for entry in peels] 
        if not possible_peels:
            # there is no peel identified, so return box
            return box

        # determine the scores for each peel in order
        # to identify the next candidate box
        scores = []
        for entry in possible_peels:
            i, box_lim = entry
            obj = self.obj_func(self, self.y[box.yi],  self.y[i])
            non_res_dim = len(x.dtype.descr)-\
                          sdutil._determine_nr_restricted_dims(box_lim, 
                                                              self.box_init)
            score = (obj, non_res_dim, box_lim, i)
            scores.append(score)

        scores.sort(key=itemgetter(0,1), reverse=True)
        entry = scores[0]
        
        
        obj_score = entry[0]
        box_new, indices = entry[2:]
        
        mass_new = self.y[indices].shape[0]/self.n
       
        if (mass_new >= self.mass_min) &\
           (mass_new < mass_old)&\
           (obj_score>0):
            box.update(box_new, indices)
            return self._peel(box)
        else:
            #else return received box
            return box
    
    
    def _real_peel(self, box, u, x):
        '''
        
        returns two candidate new boxes, peel along upper and lower dimension
        
        Parameters
        ----------
        box : a PrimBox instance
        u : str
            the uncertainty for which to peel
        
        Returns
        -------
        tuple
            two box lims and the associated indices
        
        '''

        peels = []
        for direction in ['upper', 'lower']:
            
            if not np.any(np.isnan(x[u])):
                peel_alpha = self.peel_alpha
            
                i=0
                if direction=='upper':
                    peel_alpha = 1-self.peel_alpha
                    i=1
                
                box_peel = get_quantile(x[u], peel_alpha)
                if direction=='lower':
                    logical = x[u] >= box_peel
                    indices = box.yi[logical]
                if direction=='upper':
                    logical = x[u] <= box_peel
                    indices = box.yi[logical]
                temp_box = copy.deepcopy(box.box_lims[-1])
                temp_box[u][i] = box_peel
                peels.append((indices, temp_box))
            else:
                return []
    
        return peels
    
    def _discrete_peel(self, box, u, x):
        '''
        
        returns two candidate new boxes, peel along upper and lower dimension
        
        Parameters
        ----------
        box : a PrimBox instance
        u : str
            the uncertainty for which to peel
        
        Returns
        -------
        tuple
            two box lims and the associated indices
        
        '''
        peels = []
        for direction in ['upper', 'lower']:
            peel_alpha = self.peel_alpha
        
            i=0
            if direction=='upper':
                peel_alpha = 1-self.peel_alpha
                i=1
            
            box_peel = get_quantile(x[u], peel_alpha)
            box_peel = int(box_peel)

            # determine logical associated with peel value            
            if direction=='lower':
                if box_peel == box.box_lims[-1][u][i]:
                    logical = (x[u] > box.box_lims[-1][u][i]) &\
                              (x[u] <= box.box_lims[-1][u][i+1])
                else:
                    logical = (x[u] >= box_peel) &\
                              (x[u] <= box.box_lims[-1][u][i+1])
            if direction=='upper':
                if box_peel == box.box_lims[-1][u][i]:
                    logical = (x[u] < box.box_lims[-1][u][i]) &\
                              (x[u] >= box.box_lims[-1][u][i-1])
                else:
                    logical = (x[u] <= box_peel) &\
                              (x[u] >= box.box_lims[-1][u][i-1])

            # determine value of new limit given logical
            if x[logical].shape[0] == 0:
                if direction == 'upper':
                    new_limit = np.max(x[u])
                else:
                    new_limit = np.min(x[u])
            else:
                if direction =='upper':
                    new_limit = np.max(x[u][logical])
                else:
                    new_limit = np.min(x[u][logical])            
            
            indices= box.yi[logical] 
            temp_box = copy.deepcopy(box.box_lims[-1])
            temp_box[u][i] = new_limit
            peels.append((indices, temp_box))
    
        return peels
    
    def _categorical_peel(self, box, u, x):
        '''
        
        returns candidate new boxes for each possible removal of a single 
        category. So. if the box[u] is a categorical variable with 4 
        categories, this method will return 4 boxes. 
        
        Parameters
        ----------
        box : a PrimBox instance
        u : str
            the uncertainty for which to peel
        
        Returns
        -------
        tuple
            a list of box lims and the associated indices
        
        '''
        entries = box.box_lims[-1][u][0]
        
        if len(entries) > 1:
            peels = []
            for entry in entries:
                temp_box = np.copy(box.box_lims[-1])
                peel = copy.deepcopy(entries)
                peel.discard(entry)
                temp_box[u][:] = peel
                
                if type(list(entries)[0]) not in (str, float, 
                                                  int):
                    bools = []                
                    for element in list(x[u]):
                        if element != entry:
                            bools.append(True)
                        else:
                            bools.append(False)
                    logical = np.asarray(bools, dtype=bool)
                else:
                    logical = x[u] != entry
                indices = box.yi[logical]
                peels.append((indices,  temp_box))
            return peels
        else:
            # no peels possible, return empty list
            return []

    def _paste(self, box):
        ''' Executes the pasting phase of the PRIM. Delegates pasting to data 
        type specific helper methods.'''
        
        x = self.x[self.yi_remaining]
        
        mass_old = box.yi.shape[0]/self.n
        
        res_dim = sdutil._determine_restricted_dims(box.box_lims[-1],
                                                    self.box_init)
        
        possible_pastes = []
        for u in res_dim:
            logging.getLogger(__name__).info("pasting "+u)
            dtype = self.x.dtype.fields.get(u)[0].name
            pastes = self._pastes[dtype](self, box, u)
            [possible_pastes.append(entry) for entry in pastes] 
        if not possible_pastes:
            # there is no peel identified, so return box
            return box
    
        # determine the scores for each peel in order
        # to identify the next candidate box
        scores = []
        for entry in possible_pastes:
            i, box_lim = entry
            obj = self.obj_func(self, self.y[box.yi],  self.y[i])
            non_res_dim = len(x.dtype.descr)-\
                          sdutil._determine_nr_restricted_dims(box_lim,
                                                              self.box_init)
            score = (obj, non_res_dim, box_lim, i)
            scores.append(score)

        scores.sort(key=itemgetter(0,1), reverse=True)
        entry = scores[0]
        obj, _, box_new, indices = entry
        mass_new = self.y[indices].shape[0]/self.n
        
        mean_old = np.mean(self.y[box.yi])
        mean_new = np.mean(self.y[indices])
        
        if (mass_new >= self.mass_min) &\
           (mass_new > mass_old) &\
           (obj>0) &\
           (mean_new>mean_old):
            box.update(box_new, indices)
            return self._paste(box)
        else:
            #else return received box
            return box

    def _real_paste(self, box, u):
        '''
        
        returns two candidate new boxes, pasted along upper and lower 
        dimension
        
        Parameters
        ----------
        box : a PrimBox instance
        u : str
            the uncertainty for which to peel
        
        Returns
        -------
        tuple
            two box lims and the associated indices
       
        '''

        pastes = []
        for i, direction in enumerate(['lower', 'upper']):
            box_paste = np.copy(box.box_lims[-1])
            paste_box = np.copy(box.box_lims[-1]) # box containing data candidate for pasting
            
            if direction == 'upper':
                paste_box[u][0] = paste_box[u][1]
                paste_box[u][1] = self.box_init[u][1]
                indices = sdutil._in_box(self.x[self.yi_remaining], paste_box)
                data = self.x[self.yi_remaining][indices][u]
                
                paste_value = self.box_init[u][i]
                if data.shape[0] > 0:
                    paste_value = get_quantile(data, self.paste_alpha)
                    
                assert paste_value >= box.box_lims[-1][u][i]
                    
            elif direction == 'lower':
                paste_box[u][0] = self.box_init[u][0]
                paste_box[u][1] = box_paste[u][0]
                
                indices = sdutil._in_box(self.x[self.yi_remaining], paste_box)
                data = self.x[self.yi_remaining][indices][u]
                
                paste_value = self.box_init[u][i]
                if data.shape[0] > 0:
                    paste_value = get_quantile(data, 1-self.paste_alpha)
           
                if not paste_value <= box.box_lims[-1][u][i]:
                    print("{}, {}".format(paste_value, box.box_lims[-1][u][i]))
            
            
            dtype = box_paste.dtype.fields[u][0]
            if dtype==np.int32:
                paste_value = np.int(paste_value)
            
            box_paste[u][i] = paste_value
            indices = sdutil._in_box(self.x[self.yi_remaining], box_paste)
            indices = self.yi_remaining[indices]
            
            pastes.append((indices, box_paste))
    
        return pastes        
            
    def _categorical_paste(self, box, u):
        '''
        
        Return a list of pastes, equal to the number of classes currently
        not on the box lim. 
        
        Parameters
        ----------
        box : a PrimBox instance
        u : str
            the uncertainty for which to peel
        
        Returns
        -------
        tuple
            a list of box lims and the associated indices
        
        
        '''
        box_lim = box.box_lims[-1]
        
        c_in_b = box_lim[u][0]
        c_t = self.box_init[u][0]
        
        if len(c_in_b) < len(c_t):
            pastes = []
            possible_cs = c_t - c_in_b
            for entry in possible_cs:
                box_paste = np.copy(box_lim)
                paste = copy.deepcopy(c_in_b)
                paste.add(entry)
                box_paste[u][:] = paste
                indices = sdutil._in_box(self.x[self.yi_remaining], box_paste)
                indices = self.yi_remaining[indices]
                pastes.append((indices, box_paste))
            return pastes
        else:
            # no pastes possible, return empty list
            return []
    
    def _lenient1_obj_func(self, y_old, y_new):
        r'''
        the default objective function used by prim, instead of the original
        objective function, This function can cope with continuous, integer, 
        and categorical uncertainties. The basic idea is that the gain in mean
        is divided by the loss in mass. 
        
        .. math::
            
            obj = \frac
                 {\text{ave} [y_{i}\mid x_{i}\in{B-b}] - \text{ave} [y\mid x\in{B}]}
                 {|n(y_{i})-n(y)|}
        
        where :math:`B-b` is the set of candidate new boxes, :math:`B` 
        the old box and :math:`y` are the y values belonging to the old 
        box. :math:`n(y_{i})` and :math:`n(y)` are the cardinality of 
        :math:`y_{i}` and :math:`y` respectively. So, this objective 
        function looks for the difference between  the mean of the old 
        box and the new box, divided by the change in the  number of 
        data points in the box. This objective function offsets a problem 
        in case of categorical data where the normal objective function often 
        results in boxes mainly based on the categorical data.  
        
        TODO:: seems to be identical to 14.3 in friedman and fisher
        
        '''
        mean_old = np.mean(y_old)
        
        if y_new.shape[0]>0:
            mean_new = np.mean(y_new)
        else:
            mean_new = 0
            
        obj = 0
        if mean_old != mean_new:
            if y_old.shape[0] > y_new.shape[0]:
                obj = (mean_new-mean_old)/(y_old.shape[0]-y_new.shape[0])
            elif y_old.shape[0] < y_new.shape[0]:
                obj = (mean_new-mean_old)/(y_new.shape[0]-y_old.shape[0])
            else:
                raise PRIMError('''mean is different {} vs {}, while shape is the same,
                                       this cannot be the case'''.format(mean_old, mean_new))
        return obj
    
    def _lenient2_obj_func(self, y_old, y_new):
        '''
        
        friedman and fisher 14.6
        
        
        '''
        mean_old = np.mean(y_old)
        
        if y_new.shape[0]>0:
            mean_new = np.mean(y_new)
        else:
            mean_new = 0
            
        obj = 0
        if mean_old != mean_new:
            if y_old.shape==y_new.shape:
                raise PRIMError('''mean is different {} vs {}, while shape is the same,
                                       this cannot be the case'''.format(mean_old, mean_new))
            
            change_mean = mean_new - mean_old
            change_mass = abs(y_old.shape[0]-y_new.shape[0])
            mass_new = y_new.shape[0]
                
            obj = mass_new * change_mean / change_mass
                
        return obj
    
    def _original_obj_func(self, y_old, y_new):
        ''' The original objective function: the mean of the data inside the 
        box'''
        
        if y_new.shape[0]>0:
            return np.mean(y_new)
        else:
            return -1    

    def _assert_dtypes(self, keys, dtypes):
        '''
        helper fucntion that checks whether none of the provided keys has
        a dtype object as value.
        '''
        
        for key in keys:
            if dtypes[key][0] == np.dtype(object):
                raise PRIMError("%s has dtype object and can thus not be rotated" %key)
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

    _peels = {'object': _categorical_peel,
              'int64': _discrete_peel,
               'int32': _discrete_peel,
               'float64': _real_peel}

    _pastes = {'object': _categorical_paste,
               'int32': _real_paste,
               'int64': _real_paste,
               'float64': _real_paste}

    # dict with the various objective functions available
    _obj_functions = {LENIENT2 : _lenient2_obj_func,
                      LENIENT1 : _lenient1_obj_func,
                      ORIGINAL: _original_obj_func}    
