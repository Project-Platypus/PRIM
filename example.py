import numpy as np
import pandas as pd
from ema_workbench.analysis import prim

df = pd.DataFrame(np.random.rand(1000, 2), columns=["x1", "x2"])
print df
response = df["x1"] * df["x2"] > 0.5
print response

p = prim.Prim(df.to_records(), response.values, threshold=0.8, peel_alpha=0.1)
box = p.find_box()
box.show_tradeoff().savefig("tradeoff.png")
fig = box.show_pairs_scatter()
fig.set_size_inches((12, 12))
fig.savefig("scatter.png")


# policy = {"pollution_limit" : [0.06]*100}
# SOWs = sample_lhs(model, 1000)
# results = evaluate(model, fix(SOWs, policy))
# df = pd.DataFrame(results)
# metric = np.asarray([1 if v["reliability"] > 0.95 else 0 for v in results])
# for lever in model.levers.keys():
#     df.drop(lever, axis=1, inplace=True)
# for response in model.responses:
#     df.drop(response.name, axis=1, inplace=True)
# prim = Prim(df.to_records(), metric, threshold=0.8, peel_alpha=0.1)
# box1 = prim.find_box()
# box1.show_tradeoff().savefig("tradeoff.png")
# fig = box1.show_pairs_scatter()
# fig.set_size_inches((12, 12))
# fig.savefig("scatter.png")

