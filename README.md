Patient Rule Induction Method for Python
========================================

This module implements the Patient Rule Induction Method (PRIM) for scenario
discovery in Python.  This is a standalone version of the PRIM algorithm
implemented in the [EMA Workbench](https://github.com/quaquel/EMAworkbench) by
Jan Kwakkel, which is itself derived from the
[sdtoolkit](https://cran.r-project.org/web/packages/sdtoolkit/index.html) R
package developed by RAND Corporation.  This standalone version of PRIM was
created and maintained by David Hadka.

Licensed under the GNU General Public License, version 3 or later.


Usage
-----

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
   `box.stats`, etc.)
   
4. Improved init method for `Prim`.  It can now handle a variety of inputs,
   such as Pandas data frames, Numpy matrices, or any list-like object.  The
   `setup_prim` function was removed and its functionality is now part of
   the init method.
   
5. Convert peel, paste, and objective function (peeling criteria) methods into
   functions.  This is intended to allow extensibility.
   
6. Support additional Numpy datatypes (only complex types are not supported).

7. Include `setup.py` script to automate installing module
