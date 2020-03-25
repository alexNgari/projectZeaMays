#%%
"""
Resize images to 256x256 for training models.
"""
import os
import cv2
from src.useful_stuff.timer import time_this

# %%
@time_this
def resize_images(source_dir, destination_dir, dimensions):
    """
    Resizes images: data_dir is the parent folder of the folder containing source
    images. source_dir is the actual folder, and the dimensions are the target
    dimensions.
    """
    # source_dir = os.path.join(data_dir, source_dir)
    # print(source_dir)
    # destination_dir = os.path.join(data_dir, destination_dir)
    # print(destination_dir)
    if not os.path.isdir(destination_dir):
        os.mkdir(destination_dir, 0o755)
    # os.chdir(data_dir)
    for file_name in os.listdir(source_dir):
        abs_file_name = f'{source_dir}/{file_name}'
        if os.path.isfile(abs_file_name):
            image = cv2.imread(abs_file_name, cv2.IMREAD_COLOR)
            print(image.shape)
            if image.shape != (*dimensions, 3):
                image = cv2.resize(image, dimensions)
            cv2.imwrite(os.path.join(destination_dir, file_name), image)
        else:
            new_source_dir = os.path.join(source_dir, file_name)
            new_destination_dir = os.path.join(destination_dir, file_name)
            resize_images(new_source_dir, new_destination_dir, dimensions)

# %%
if __name__ == '__main__':
    DATA_DIR = f'{os.getcwd()}/Data'
    SOURCE_DIR = os.path.join(DATA_DIR, 'test_dir')
    DESTINATION_DIR = os.path.join(DATA_DIR, 'test_dir_resized')
    SIZE = (256, 254)
    resize_images(SOURCE_DIR, DESTINATION_DIR, SIZE)


# %%
