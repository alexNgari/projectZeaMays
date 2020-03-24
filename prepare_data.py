"""
Script to prepare dataset.
"""
#%%
import os
from src.data_prep.remove_duplicates import find_duplicates, delete_files
from src.data_prep.resize_images import resize_images

#%%
DATA_DIR = f'{os.getcwd()}/Data/'
SOURCE_DIR = 'images_boom'
SIZE = 256

#%%
DUPLICATES = find_duplicates(DATA_DIR)
print(DUPLICATES)
