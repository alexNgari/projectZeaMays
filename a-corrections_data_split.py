#%%
import os
import sklearn as sk
import numpy as np

#%%
DATADIR = "/home/ngari/Dev/projectzeamays/data/clean/final"
DESTINATION = "/home/ngari/Dev/projectzeamays/data/clean/final_test"

#%%
faw = np.array(os.listdir(f"{DATADIR}/faw"))
healthy = np.array(os.listdir(f"{DATADIR}/healthy"))
zinc = np.array(os.listdir(f"{DATADIR}/zinc_def"))

print(faw.shape, healthy.shape, zinc.shape)

# %%
faw_test = np.random.choice(faw, size = int(faw.shape[0]*0.25), replace=False)
healthy_test = np.random.choice(healthy, int(healthy.shape[0] * 0.25), False)
zinc_test = np.random.choice(zinc, int(zinc.shape[0]*0.25), False)

# %%
for img in zinc_test:
  source = os.path.join(os.path.join(DATADIR, f"zinc_def/{img}"))
  dest = os.path.join(DESTINATION, f"zinc_def/{img}")
  os.rename(source, dest)

# %%
for img in faw_test:
  source = os.path.join(os.path.join(DATADIR, f"faw/{img}"))
  dest = os.path.join(DESTINATION, f"faw/{img}")
  os.rename(source, dest)


# %%
for img in healthy_test:
  source = os.path.join(os.path.join(DATADIR, f"healthy/{img}"))
  dest = os.path.join(DESTINATION, f"healthy/{img}")
  os.rename(source, dest)

# %%
