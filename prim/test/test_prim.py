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

import unittest
import prim
import numpy as np
import pandas as pd

class TestInit(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.df = pd.DataFrame(np.random.rand(1000, 3),
                              columns=["x1", "x2", "x3"])
        cls.response = cls.df["x1"]*cls.df["x2"] + 0.3*cls.df["x3"]
        
        p = prim.Prim(cls.df, cls.response, threshold=0.5, threshold_type=">")
        box = p.find_box()
        cls.expected_output = str(box)
        
    def test_numpy(self):
        df = TestInit.df.to_records(index=False)
        response = TestInit.response.values
        p = prim.Prim(df, response, threshold=0.5, threshold_type=">")
        box = p.find_box()
        output = str(box)
        
        self.assertEqual(output, TestInit.expected_output)
        
    def test_function(self):
        p = prim.Prim(TestInit.df,
                      lambda x : x["x1"]*x["x2"] + 0.3*x["x3"],
                      threshold=0.5,
                      threshold_type=">")
        box = p.find_box()
        output = str(box)
        
        self.assertEqual(output, TestInit.expected_output)
        
    def test_threshold(self):
        p = prim.Prim(TestInit.df,
                      lambda x : x["x1"]*x["x2"] + 0.3*x["x3"] > 0.5)
        box = p.find_box()
        output = str(box)
        
        self.assertEqual(output, TestInit.expected_output)
        