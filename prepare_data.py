"""
Script to prepare dataset.
"""
#%%
import os
from src.preprocessing.raw_transforms import convert_annotations, find_duplicates, delete_files
from src.preprocessing.raw_transforms import resize_images, group_images

#%%
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
ANNOTATION_DIR = os.path.join(ROOT_DIR, 'Annotations')
SOURCE_DIRS = [dir_name for dir_name in os.listdir(DATA_DIR) if not 'resized' in dir_name]
SIZE = (256, 256)

#%%
convert_annotations(ANNOTATION_DIR)

#%%
DUPLICATES = find_duplicates(DATA_DIR)
print(DUPLICATES)

# %%
delete_files(DUPLICATES)

#%%
for source_dir in SOURCE_DIRS:
    source_dir = os.path.join(DATA_DIR, source_dir)
    destination_dir = os.path.join(DATA_DIR, source_dir+'_resized')
    # print(source_dir, destination_dir)
    resize_images(source_dir, destination_dir, SIZE)

# %%
IMAGES_DIR = os.path.join(DATA_DIR, 'NLB_stuff_resized')
DESTINATION_DIR = os.path.join(DATA_DIR, 'NLB_stuff_separated')
group_images(IMAGES_DIR, ANNOTATION_DIR, DESTINATION_DIR)

#%%
