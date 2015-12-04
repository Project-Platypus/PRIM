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
from __future__ import division

import copy
import unittest
import numpy as np
import pandas as pd
from prim import Prim
from prim.prim_box import PrimBox
from prim.prim_ops import get_quantile, categorical_peel, categorical_paste
import numpy.lib.recfunctions as recfunctions

class TestPrimInitMethod(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.df = pd.DataFrame(np.random.rand(1000, 3),
                              columns=["x1", "x2", "x3"])
        cls.response = cls.df["x1"]*cls.df["x2"] + 0.3*cls.df["x3"]
        
        p = Prim(cls.df, cls.response, threshold=0.5, threshold_type=">")
        box = p.find_box()
        cls.expected_output = str(box)
        
    def test_init_numpy(self):
        df = TestPrimInitMethod.df.to_records(index=False)
        response = TestPrimInitMethod.response.values
        p = Prim(df, response, threshold=0.5, threshold_type=">")
        box = p.find_box()
        output = str(box)
        
        self.assertEqual(output, TestPrimInitMethod.expected_output)
        
    def test_function(self):
        p = Prim(TestPrimInitMethod.df,
                 lambda x : x["x1"]*x["x2"] + 0.3*x["x3"],
                 threshold=0.5,
                 threshold_type=">")
        box = p.find_box()
        output = str(box)
        
        self.assertEqual(output, TestPrimInitMethod.expected_output)
        
    def test_threshold(self):
        p = Prim(TestPrimInitMethod.df,
                 lambda x : x["x1"]*x["x2"] + 0.3*x["x3"] > 0.5)
        box = p.find_box()
        output = str(box)
        
        self.assertEqual(output, TestPrimInitMethod.expected_output)
        
    def test_string(self):
        df = copy.deepcopy(TestPrimInitMethod.df)
        df["y"] = df["x1"]*df["x2"] + 0.3*df["x3"]
        
        p = Prim(df, "y", threshold=0.5, threshold_type=">")
        box = p.find_box()
        output = str(box)
        
        self.assertEqual(output, TestPrimInitMethod.expected_output)
        
class TestPrimBox(unittest.TestCase):
    
    def test_init(self):
        x = np.array([(0,1,2),
                      (2,5,6),
                      (3,2,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float),
                            ('c', np.float)])
        y = np.array([0,1,2])
        
        prim_obj = Prim(x, y, threshold=0.8)
        box = PrimBox(prim_obj, prim_obj._box_init, prim_obj.yi)

        self.assertTrue(box.peeling_trajectory.shape==(1,5))
    
    def test_select(self):
        x = np.array([(0,1,2),
                      (2,5,6),
                      (3,2,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float),
                            ('c', np.float)])
        y = np.array([1,1,0])
        
        prim_obj = Prim(x, y, threshold=0.8)
        box = PrimBox(prim_obj, prim_obj._box_init, prim_obj.yi)

        new_box_lim = np.array([(0,1,1),
                                (2,5,6)], 
                                dtype=[('a', np.float),
                                       ('b', np.float),
                                       ('c', np.float)])
        indices = np.array([0,1], dtype=np.int)
        box.update(new_box_lim, indices)
        
        box.select(0)
        self.assertTrue(np.all(box.yi==prim_obj.yi))
    
    def test_update(self):
        x = np.array([(0,1,2),
                      (2,5,6),
                      (3,2,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float),
                            ('c', np.float)])
        y = np.array([1,1,0])
        
        prim_obj = Prim(x, y, threshold=0.8)
        box = PrimBox(prim_obj, prim_obj._box_init, prim_obj.yi)

        new_box_lim = np.array([(0,1,1),
                                (2,5,6)], 
                                dtype=[('a', np.float),
                                       ('b', np.float),
                                       ('c', np.float)])
        indices = np.array([0,1], dtype=np.int)
        box.update(new_box_lim, indices)

        self.assertEqual(box.peeling_trajectory['mean'][1], 1)
        self.assertEqual(box.peeling_trajectory['coverage'][1], 1)
        self.assertEqual(box.peeling_trajectory['density'][1], 1)
        self.assertEqual(box.peeling_trajectory['res dim'][1], 1)
        self.assertEqual(box.peeling_trajectory['mass'][1], 2/3)
    
    def test_drop_restriction(self):
        x = np.array([(0,1,2),
                      (2,5,6),
                      (3,2,1)], 
                     dtype=[('a', np.float),
                            ('b', np.float),
                            ('c', np.float)])
        y = np.array([1,1,0])
         
        prim_obj = Prim(x, y, threshold=0.8)
        box = PrimBox(prim_obj, prim_obj._box_init, prim_obj.yi)
 
        new_box_lim = np.array([(0,1,1),
                                (2,2,6)], 
                                dtype=[('a', np.float),
                                       ('b', np.float),
                                       ('c', np.float)])
        indices = np.array([0,1], dtype=np.int)
        box.update(new_box_lim, indices)
         
        box.drop_restriction('b')
         
        correct_box_lims = np.array([(0,1,1),
                                     (2,5,6)], 
                                    dtype=[('a', np.float),
                                           ('b', np.float),
                                           ('c', np.float)]) 
               
        box_lims = box._box_lims[-1]
        names = recfunctions.get_names(correct_box_lims.dtype)
        
        for entry in names:
            lim_correct = correct_box_lims[entry]
            lim_box = box_lims[entry]
            for i in range(len(lim_correct)):
                self.assertEqual(lim_correct[i], lim_box[i])
         
        self.assertEqual(box.peeling_trajectory['mean'][2], 1)
        self.assertEqual(box.peeling_trajectory['coverage'][2], 1)
        self.assertEqual(box.peeling_trajectory['density'][2], 1)
        self.assertEqual(box.peeling_trajectory['res dim'][2], 1)
        self.assertEqual(box.peeling_trajectory['mass'][2], 2/3)

    def test_quantile(self):
        data = np.ma.array([x for x in range(10)])
        self.assertTrue(get_quantile(data, 0.9)==8.5)
        self.assertTrue(get_quantile(data, 0.95)==8.5)
        self.assertTrue(get_quantile(data, 0.1)==0.5)
        self.assertTrue(get_quantile(data, 0.05)==0.5)
        
        data = np.ma.array(data = [1])
        self.assertTrue(get_quantile(data, 0.9)==1)
        self.assertTrue(get_quantile(data, 0.95)==1)
        self.assertTrue(get_quantile(data, 0.1)==1)
        self.assertTrue(get_quantile(data, 0.05)==1)
        
        data = np.ma.array([1,1,2,3,4,5,6,7,8,9,9])
        self.assertTrue(get_quantile(data, 0.9)==8.5)
        self.assertTrue(get_quantile(data, 0.95)==8.5)
        self.assertTrue(get_quantile(data, 0.1)==1.5)
        self.assertTrue(get_quantile(data, 0.05)==1.5)        
        
        data = np.ma.array([1,1,2,3,4,5,6,7,8,9,9, np.NAN], 
                           mask=[0,0,0,0,0,0,0,0,0,0,0,1])
        self.assertTrue(get_quantile(data, 0.9)==8.5)
        self.assertTrue(get_quantile(data, 0.95)==8.5)
        self.assertTrue(get_quantile(data, 0.1)==1.5)
        self.assertTrue(get_quantile(data, 0.05)==1.5)   
        
    def test_box_init(self):
        # test init box without NANS
        x = np.array([(0,1,2),
                      (2,5,6),
                      (3,2,7)], 
                     dtype=[('a', np.float),
                            ('b', np.float),
                            ('c', np.float)])
        y = np.array([0,1,2])
         
        prim_obj = Prim(x, y, threshold=0.5)
        box_init = prim_obj._box_init
         
        # some test on the box
        self.assertTrue(box_init['a'][0]==0)
        self.assertTrue(box_init['a'][1]==3)
        self.assertTrue(box_init['b'][0]==1)
        self.assertTrue(box_init['b'][1]==5)
        self.assertTrue(box_init['c'][0]==2)
        self.assertTrue(box_init['c'][1]==7)  
  
        # test init box with NANS
        x = np.array([(0,1,2),
                      (2,5,np.NAN),
                      (3,2,7)], 
                     dtype=[('a', np.float),
                            ('b', np.float),
                            ('c', np.float)])
        y = np.array([0,1,2])
         
        x = np.ma.array(x)
        x['a'] = np.ma.masked_invalid(x['a'])
        x['b'] = np.ma.masked_invalid(x['b'])
        x['c'] = np.ma.masked_invalid(x['c'])
          
        prim_obj = Prim(x, y, threshold=0.5)
        box_init = prim_obj._box_init
        
        # some test on the box
        self.assertTrue(box_init['a'][0]==0)
        self.assertTrue(box_init['a'][1]==3)
        self.assertTrue(box_init['b'][0]==1)
        self.assertTrue(box_init['b'][1]==5)
        self.assertTrue(box_init['c'][0]==2)
        self.assertTrue(box_init['c'][1]==7)  
         
        # heterogenous without NAN
        dtype = [('a', np.float),('b', np.int), ('c', np.object)]
        x = np.empty((10, ), dtype=dtype)
         
        x['a'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.7, 0.8, 0.9, 1.0]
        x['b'] = [0,1,2,3,4,5,6,7,8,9]
        x['c'] = ['a','b','a','b','a','a','b','a','b','a', ]
         
        prim_obj = Prim(x, y, threshold=0.5)
        box_init = prim_obj._box_init
          
        # some test on the box
        self.assertTrue(box_init['a'][0]==0.1)
        self.assertTrue(box_init['a'][1]==1.0)
        self.assertTrue(box_init['b'][0]==0)
        self.assertTrue(box_init['b'][1]==9)
        self.assertTrue(box_init['c'][0]==set(['a','b']))
        self.assertTrue(box_init['c'][1]==set(['a','b'])) 
  
        # heterogenous with NAN
        dtype = [('a', np.float),('b', np.int), ('c', np.object)]
        x = np.empty((10, ), dtype=dtype)
         
        x[:] = np.NAN
        x['a'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.7, 0.8, np.NAN, 1.0]
        x['b'] = [0,1,2,3,4,5,6,7,8,9]
        x['c'] = ['a','b','a','b',np.NAN,'a','b','a','b','a', ]
         
        x = np.ma.array(x)
        x['a'] = np.ma.masked_invalid(x['a'])
        x['b'] = np.ma.masked_invalid(x['b'])
        x['c'][4] = np.ma.masked
         
        prim_obj = Prim(x, y, threshold=0.5)
        box_init = prim_obj._box_init
          
        # some test on the box
        self.assertTrue(box_init['a'][0]==0.1)
        self.assertTrue(box_init['a'][1]==1.0)
        self.assertTrue(box_init['b'][0]==0)
        self.assertTrue(box_init['b'][1]==9)
        self.assertTrue(box_init['c'][0]==set(['a','b']))
        self.assertTrue(box_init['c'][1]==set(['a','b']))
        
    def test_categorical_peel(self):
        dtype = [('a', np.float),('b', np.object)]
        x = np.empty((10, ), dtype=dtype)
        
        x['a'] = np.random.rand(10,)
        x['b'] = ['a','b','a','b','a','a','b','a','b','a']
        y = np.random.randint(0,2, (10,))
        y = y.astype(np.int)
        
        prim_obj = Prim(x, y, threshold=0.8)
        box_lims = np.array([(0, set(['a','b'])),
                             (1, set(['a','b']))], dtype=dtype )
        box = PrimBox(prim_obj, box_lims, prim_obj.yi)
        
        u = 'b'
        peels = categorical_peel(prim_obj, box, u)
        
        self.assertEquals(len(peels), 2)
        
        for peel in peels:
            pl  = peel[1][u]
            self.assertEquals(len(pl[0]), 1)
            self.assertEquals(len(pl[1]), 1)
        

    def test_categorical_paste(self):
        dtype = [('a', np.float),('b', np.object)]
        x = np.empty((10, ), dtype=dtype)
        
        x['a'] = np.random.rand(10,)
        x['b'] = ['a','b','a','b','a','a','b','a','b','a']
        y = np.random.randint(0,2, (10,))
        y = y.astype(np.int)
        
        prim_obj = Prim(x, y, threshold=0.8)
        box_lims = np.array([(0, set(['a'])),
                             (1, set(['a']))], dtype=dtype )
        
        yi = np.where(x['b']=='a')
        
        box = PrimBox(prim_obj, box_lims, yi)
        
        u = 'b'
        pastes = categorical_paste(prim_obj, box, u)
        
        self.assertEquals(len(pastes), 1)
        
        for paste in pastes:
            indices, box_lims = paste
            self.assertEquals(indices.shape[0], 10)
            self.assertEqual(box_lims[u][0], set(['a','b']))
            