"""
Convert the annotations to show simply whether the maize plant is infected with NLB or not
instead of lines along lesions
"""

import os
import pandas as pd
import numpy as np
from src.useful_stuff.timer import time_this


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



if __name__ == '__main__':
    convert_annotations(os.path.dirname(os.path.abspath(__file__)))
