"""
Methods to put different classes of images in different folders.
Might be useful if we ever need to port some data from one dataset to another.
Uses the annotation to group the different classes.
"""
#%%
import os
import shutil
import pandas as pd
from src.useful_stuff.timer import time_this

#%%
@time_this
def group_images(images_dir, annotation_dir, destination_dir):
    """
    Put different classes in different folders.
    """
    if not os.path.isdir(destination_dir):
        os.makedirs(destination_dir, 0o755)
    for item in os.scandir(images_dir):
        if item.is_dir():
            print(f'Found subdirectory: {item.path}')
            new_images_dir = item.path
            leaf_dir = item.path.split('/')[-1]
            new_destination = os.path.join(destination_dir, leaf_dir)
            group_images(new_images_dir, annotation_dir, new_destination)
        else:
            break
    annotation_file = os.path.join(annotation_dir, f"{images_dir.split('/')[-1]}.csv")
    if not os.path.isfile(annotation_file):
        return
    print(f'Copying files into {destination_dir}')
    sick_path = os.path.join(destination_dir, 'NLB')
    healthy_path = os.path.join(destination_dir, 'healthy')
    paths = [sick_path, healthy_path]
    for path in paths:
        try:
            os.makedirs(path, 0o755)
        except OSError:
            print(f'{path} already exists.')
    del paths
    print(f'Reading annotations from {annotation_file}')
    annotations = pd.read_csv(annotation_file)
    for image, label in annotations.itertuples(index=False):
        source_image = os.path.join(images_dir, image)
        dest_image = ''
        if label:
            dest_image = os.path.join(sick_path, image)
        else:
            dest_image = os.path.join(healthy_path, image)
        shutil.copy(source_image, dest_image)
    return

# group_images('/home/ngari/Dev/projectzeamays/Data/NLB_stuff_resized')

# %%
