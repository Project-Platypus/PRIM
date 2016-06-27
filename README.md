Patient Rule Induction Method for Python
========================================

This module implements the Patient Rule Induction Method (PRIM) for scenario
discovery in Python.  This is a standalone version of the PRIM algorithm
implemented in the [EMA Workbench](https://github.com/quaquel/EMAworkbench) by
Jan Kwakkel, which is based on the
[sdtoolkit](https://cran.r-project.org/web/packages/sdtoolkit/index.html) R
package developed by RAND Corporation.  All credit goes to Jan Kwakkel for
developing the original code.  This standalone version of PRIM was created and
maintained by David Hadka.

Licensed under the GNU General Public License, version 3 or later.

[![Build Status](https://travis-ci.org/Project-Platypus/Platypus.svg?branch=master)](https://travis-ci.org/Project-Platypus/Platypus)
[![PyPI](https://img.shields.io/pypi/v/PRIM.svg)](https://pypi.python.org/pypi/PRIM)
[![PyPI](https://img.shields.io/pypi/dm/PRIM.svg)](https://pypi.python.org/pypi/PRIM)

Installation
------------

The PRIM module is available on PyPI and can be installed using `easy_install`:

    easy_install prim
    
or PIP:

    pip install prim
    
The latest development version can be installed by cloning this repository
and running:

    python setup.py install
    
This module requires several dependencies, including `numpy`, `pandas`,
`mpldatacursor`, `six`, and `scipy`.  If experiencing trouble installing
these dependencies, we recommend using [Anaconda Python](https://www.continuum.io/downloads).

Usage
-----

Below shows the interactive use of the PRIM module for finding the first box.
In this example, we are interested in cases where the response is greater
than 0.5 (as indicated by the `threshold` and `threshold_type` arguments).
After creating the `Prim` object, we invoke `find_box()` to find
the first box containing cases of interest followed by `box.show_tradeoff()`
to display the tradeoff between coverage and density for each peeling/pasting
trajectory.

```python

    import prim
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame(np.random.rand(1000, 3), columns=["x1", "x2", "x3"])
    response = df["x1"]*df["x2"] + 0.2*df["x3"]
    
    p = prim.Prim(df, response, threshold=0.5, threshold_type=">")
    
    box = p.find_box()
    box.show_tradeoff()
    
    plt.show()
```

You can interact with the tradeoff plot by hovering the mouse over points
to view the stats, as shown below.

![Tradeoff plot](https://github.com/MOEAFramework/PRIM/blob/master/docs/images/screenshot1.png)

Clicking a point shows additional details in a separate window.

![Details view](https://github.com/MOEAFramework/PRIM/blob/master/docs/images/screenshot2.png)

This module extends EMA Workbench's support for categorical data by allowing the
categorical data to be plotted in the pairwise scatter plot:

![Categorical data](https://github.com/MOEAFramework/PRIM/blob/master/docs/images/screenshot3.png)

Also note the Prev / Next buttons on this window allowing navigation to adjacent
peeling trajectories without having to return to the tradeoff plot.

Differences from EMA Workbench
------------------------------

While this code is derived from the PRIM implementation in the EMA Workbench,
there are a number of differences.

1. Redesigned several figures to add interactivity within Matplotlib's native
   viewer, whereas it previously required an IPython Notebook.  The new
   `box.show_details()` plot combines several of the original plots.

2. Removed the `box.inspect` method.  To display the box details in tabular
   form, simply call `print(box)`.  To display the details graphically,
   call `box.show_details()`.
   
3. Most `box` methods no longer accept an index argument to select the
   peeling trajectory.  Instead, you first call `box.select(i)` to select
   the i-th peeling trajectory, then call any other method to see the details
   of the trajectory (`print(box)`, `box.show_details()`,
   `box.stats`, etc.).
   
4. Improved init method for `Prim`.  It can now handle a variety of inputs,
   such as Pandas data frames, Numpy matrices, or any list-like object.  The
   `setup_prim` function was removed and its functionality is now part of
   the init method.
   
5. Converted the peel, paste, and objective function (peeling criteria) methods,
   which were previously part of the `Prim` class, into separate functions.
   This is intended to allow extensibility.
   
6. Supports additional Numpy datatypes.  This include support for non-standard
   int, unsigned int, and float types (e.g., int16 and float32) in addition
   to boolean values (which is treated as categorical type).  Complex types are
   not supported.
   
7. The pairwise scatter plot can now display categorical data and overlay the
   box limits.

8. Include `setup.py` script to automate installing module.
