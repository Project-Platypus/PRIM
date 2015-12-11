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

"""Patient Rule Induction Method (PRIM) for Python.

The Patient Rule Induction Method (PRIM) is a rule induction method that when
given a set of points, which a subset marked as "cases of interest", finds
limits for each variable that contain the cases of interest.  These limits are
often referred to as a box.  For example, PRIM might identify the bounds
0.25 <= x1 <= 0.75 and 10 <= x2 <= 15 to indicate that any point satisfying
these conditions is likely a case of interest.

PRIM uses two measures to quantify the quality of a box.  Coverage measures the
percentage of cases of interest contained within a box.  Density measures the
percentage of observations within a box that are cases of interest.  Ideally,
both coverage and density should be 100%, but these values are often smaller
in practice.  This implementation of PRIM provides a graphical way to view the
tradeoff between coverage and density and select a box that achieves the
desired values.

Credits
-------
This code was originally implemented in the EMA Workbench by Jan Kwakkel, which
is itself derived from the sdtoolkit R package developed by RAND Corporation.
This standalone version of PRIM was created and maintained by David Hadka.

License
-------
Released under the GNU General Public License, version 3 or later.  Copyright
is retained by the respective authors.
"""

from __future__ import absolute_import

from .exceptions import PrimError
from .prim_alg import Prim
from .prim_objfcn import original, lenient1, lenient2

__all__ = ["PrimError", "Prim", "original", "lenient1", "lenient2"]