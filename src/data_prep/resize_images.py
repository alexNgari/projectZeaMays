#%%
"""
Resize images to 256x256 for training models.
"""
import os
import cv2
from src.useful_stuff.timer import time_this

# %%
@time_this
def resize_images(data_dir, source_dir, dimensions):
    """
    Resizes images: data_dir is the parent folder of the folder containing source
    images. source_dir is the actual folder, and the dimensions are the target
    dimensions.
    """
    source_dir = f'{data_dir}/{source_dir}'
    destination_dir = f'{source_dir}_resized'
    os.chdir(data_dir)
    for file_name in os.listdir(source_dir):
        abs_file_name = f'{source_dir}/{file_name}'
        if os.path.isfile(abs_file_name):
            image = cv2.imread(abs_file_name, cv2.IMREAD_COLOR)
            # print(image.shape)
            image = cv2.resize(image, dimensions)
            # print(image.shape)
            cv2.imwrite(os.path.join(destination_dir, file_name), image)

# %%
if __name__ == '__main__':
    DATA_DIR = f'{os.getcwd()}/Data/'
    SOURCE_DIR = 'images_boom'
    SIZE = 256
    resize_images(DATA_DIR, SOURCE_DIR, (SIZE, SIZE))


# %%
