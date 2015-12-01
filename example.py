import os
import numpy as np
import pandas as pd
from ema_workbench import prim_alg
import matplotlib.pyplot as plt
import logging
import ema_workbench

logging.basicConfig(level=logging.INFO)

if os.path.exists("pickle.dat"):
    df = pd.read_pickle("pickle.dat")
else:
    df = pd.DataFrame(np.random.rand(1000, 3), columns=["x1", "x2", "x3"])
    df.to_pickle("pickle.dat")
    
response = df["x1"] * df["x2"] + .2*df["x3"] > 0.5

p = prim_alg.Prim(df, lambda x : x["x1"]*x["x2"] + 0.2*x["x3"], threshold=0.5, peel_alpha=0.1)
box = p.find_box()
#box.inspect()
#print
#print "a:", box.inspect()
#print box.box_lims
box.show_tradeoff()
# box.select(box._cur_box)
# box.show_box_details()
# box.show_box_details()
# box = p.find_box()
# box.show_box_details()
#box.show_box_details()
#fig = box.show_pairs_scatter()
#fig.set_size_inches((12, 12))
#fig.savefig("scatter.png")
#box.inspect(style="graph")
plt.show()


# policy = {"pollution_limit" : [0.06]*100}
# SOWs = sample_lhs(model, 1000)
# results = evaluate(model, fix(SOWs, policy))
# df = pd.DataFrame(results)
# metric = np.asarray([1 if v["reliability"] > 0.95 else 0 for v in results])
# for lever in model.levers.keys():
#     df.drop(lever, axis=1, inplace=True)
# for response in model.responses:
#     df.drop(response.name, axis=1, inplace=True)
# prim_alg = Prim(df.to_records(), metric, threshold=0.8, peel_alpha=0.1)
# box1 = prim_alg.find_box()
# box1.show_tradeoff().savefig("tradeoff.png")
# fig = box1.show_pairs_scatter()
# fig.set_size_inches((12, 12))
# fig.savefig("scatter.png")

