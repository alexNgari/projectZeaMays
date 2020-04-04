"""
Basic transformations performed on raw images and annotations to generate
a workable raw-image dataset. (Workable: organised, reduced size)
"""

#%%
import os
import shutil
import hashlib
import numpy as np
import pandas as pd
import cv2
from src.useful_stuff.timer import time_this

#%%
"""
Removes all duplicate files in the subject directory.
"""
@time_this
def find_duplicates(data_dir):
    """
    Return a list of absolute paths to all duplicate files in directory.
    """
    duplicates = []
    im_hashes = dict()

    def append_stuff(directory, im_hashes):
        """
        Return a list of absolute paths to all duplicate files in directory.
        """
        print(directory)
        for file_name in os.listdir(directory):
            file_name = f'{directory}/{file_name}'
            if os.path.isfile(file_name):
                with open(file_name, 'rb') as file:
                    file_hash = hashlib.md5(file.read()).hexdigest()
                try:
                    temp = im_hashes[file_hash]
                    del temp
                    duplicates.append(file_name)
                    print(file_name.split('/')[-1])
                except KeyError:
                    im_hashes[file_hash] = file_name
            else:
                append_stuff(file_name, im_hashes)
    append_stuff(data_dir, im_hashes)
    return duplicates

@time_this
def delete_files(file_names_list):
    """
        Deletes all the files in the list of file names passed.
        The list should contain absolute paths.
    """
    for file in file_names_list:
        os.remove(file)

#%%
"""
Methods to put different classes of images in different folders.
Might be useful if we ever need to port some data from one dataset to another.
Uses the annotation to group the different classes.
"""
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

# %%
"""
Resize images to 256x256 for training models.
"""
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

#%%
"""
Convert the annotations to show simply whether the maize plant is infected with NLB or not
instead of lines along lesions
"""
def read_file(filename):
    """
    Read unprocessed csv file.
    """
    return pd.read_csv(filename)

def clean_data(dataframe):
    """
    Drop duplicate images and remove useless columns.
    """
    dataframe.drop_duplicates('image', inplace=True, ignore_index=True)
    dataframe.drop(['user', 'day', 'month', 'year', 'hour', 'minute'], axis='columns', inplace=True)

def replace_annotations(dataframe):
    """
    Create 'NLBPresent' column and remove coordinates.
    """
    dataframe['NLBPresent'] = np.where(np.sum(dataframe.iloc[:, 1:], axis=1) != 0, 1, 0)
    dataframe.drop(['x1', 'y1', 'x2', 'y2'], axis='columns', inplace=True)

def save_file(dataframe, filename):
    """
    Save Processed file.
    """
    dataframe.to_csv(filename, index=False)

@time_this
def convert_annotations(annotations_directory):
    """
    Convert annotations from lesion detection to binary image classification format.
    """
    all_files = os.listdir(annotations_directory)
    filenames = [filename for filename in all_files if filename.endswith('.csv')]
    if filenames:
        for filename in filenames:
            filename = os.path.join(annotations_directory, filename)
            print(f'Converting {filename}...')
            dataframe = read_file(filename)
            clean_data(dataframe)
            replace_annotations(dataframe)
            if filename.__contains__('boom'):
                dataframe['image'] = dataframe['image'].astype(str)+'.JPG'
            print('Saving file...')
            save_file(dataframe, f'images{filename[4:]}')
            print('Operation completed successfully.')
        print(f'All operations completed successfully. \n Converted {len(filenames)} files.')
    else:
        print('Run script in the directory where your CSVs are!')
