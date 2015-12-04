from __future__ import print_function

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import prim

if os.path.exists("pickle.dat"):
    df = pd.read_pickle("pickle.dat")
else:
    df = pd.DataFrame(np.random.rand(1000, 3), columns=["x1", "x2", "x3"])
    df.to_pickle("pickle.dat")
    
#df['x3'] = ["a" if x < 0.25 else "b" if x < 0.75 else "c" for x in df['x3']]

#p = prim.Prim(df, lambda x : x["x1"]*x["x2"] + 0.3*(x["x3"] == "b"), threshold=0.5, threshold_type=">")
p = prim.Prim(df, lambda x : x["x1"]*x["x2"] + 0.3*x["x3"], threshold=0.5, threshold_type=">")
box = p.find_box()
print(box)
box.show_tradeoff()
plt.show()