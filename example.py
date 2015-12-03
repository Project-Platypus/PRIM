import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import prim

logging.basicConfig(level=logging.DEBUG)

if os.path.exists("pickle.dat"):
    df = pd.read_pickle("pickle.dat")
else:
    df = pd.DataFrame(np.random.rand(1000, 3), columns=["x1", "x2", "x3"])
    df.to_pickle("pickle.dat")
    
df['x3'] = df['x3'] > 0.5
#print df.dtype.fields
#print df[0:10]

p = prim.Prim(df, lambda x : x["x1"]*x["x2"] - 0.5*x["x3"], threshold=0.5, threshold_type=">")
box = p.find_box()
print box
