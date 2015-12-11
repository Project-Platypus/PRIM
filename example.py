from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import prim

df = pd.DataFrame(np.random.rand(1000, 3), columns=["x1", "x2", "x3"])

p = prim.Prim(df,
              lambda x : x["x1"]*x["x2"] + 0.3*x["x3"],
              threshold=0.5,
              threshold_type=">")
box = p.find_box()
box.show_tradeoff()
plt.show()