'''
Convert the annotations to show simply whether the maize plant is infected with NLB or not
instead of lines along lesions
'''

import os
import pandas as pd
import numpy as np


def readCSV(fileName):
    return pd.read_csv(fileName)

''' Drop duplicate images ans remove useless columns '''
def cleanData(dataFrame):
    dataFrame.drop_duplicates('image', inplace=True, ignore_index=True)
    dataFrame.drop(['user', 'day', 'month', 'year', 'hour', 'minute'], axis='columns', inplace=True)

''' Create 'NLBPresent' column and remove coordinates '''
def replaceAnnotations(dataFrame):
    dataFrame['NLBPresent'] = np.where(np.sum(dataFrame.iloc[:, 1:], axis=1)!=0, 1, 0)
    dataFrame.drop(['x1', 'y1', 'x2', 'y2'], axis='columns', inplace=True)

def saveFile(dataFrame, fileName):
    dataFrame.to_csv(fileName, index=False)

def convertAnnotations():
    theDirectory = os.getcwd()
    fileNames = [filename for filename in os.listdir(theDirectory) if filename.endswith('.csv')]
    if fileNames:
        for fileName in fileNames:
            print(f'Converting {fileName}...')
            dataFrame = readCSV(fileName)
            cleanData(dataFrame)
            replaceAnnotations(dataFrame)
            print('Saving file...')
            saveFile(dataFrame, f'{fileName[:-4]}_classification.csv')
            print('Operation completed successfully.')
        
        print(f'All operations completed successfully. \n Converted {len(fileNames)} files.')
    else:
        print('Run script in the directory where your CSVs are!')


convertAnnotations()