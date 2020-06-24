#%%
import os
import pandas as pd
from skimage import io as io

#%%
PATH = os.getcwd()
DIRS = [ dir for dir in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, dir))]
DIRS

#%%
with open(os.path.join(PATH, "resolutions.txt"), "a") as f:
  f.write("image, height, width\n")

#%%
with open(os.path.join(PATH, "resolutions.txt"), "a") as f:
  for dir in DIRS:
    for img_name in os.listdir(dir):
      img = io.imread(os.path.join(PATH, f"{dir}/{img_name}"))
      f.write(f"{img_name}, {img.shape[0]}, {img.shape[1]}\n")
    

# %%
# img = io.imread(os.path.join(PATH, "images_boom/DSC00964_3.JPG"))
# img.shape

# %%
