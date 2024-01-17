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

<a href="https://github.com/Project-Platypus/PRIM"><img alt="GitHub Actions status" src="https://github.com/Project-Platypus/PRIM/workflows/Tests/badge.svg?branch=master&event=push"></a>
[![PyPI](https://img.shields.io/pypi/v/PRIM.svg)](https://pypi.python.org/pypi/PRIM)
[![PyPI](https://img.shields.io/pypi/dm/PRIM.svg)](https://pypi.python.org/pypi/PRIM)

### Installation

To install the latest PRIM release, run the following command:

```
    pip install prim
```

To install the latest development version of PRIM, run the following commands:

```
    pip install -U build setuptools
    git clone https://github.com/Project-Platypus/PRIM.git
    cd PRIM
    python -m build
    python -m pip install --editable .
```

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
