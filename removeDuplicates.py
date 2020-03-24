#%%
import os
import hashlib


def findDuplicates(pathFromRoot):
    projectRootPath = os.getcwd()
    duplicates = []
    imHashes = dict()

    def appendStuff(directory, duplicates, imHashes):
        print(directory)
        for fileName in os.listdir(directory):
            fileName = f'{directory}/{fileName}'
            if os.path.isfile(fileName):
                with open(fileName, 'rb') as f:
                    fileHash = hashlib.md5(f.read()).hexdigest()
                try:
                    temp = imHashes[fileHash]
                    del temp
                    duplicates.append(fileName)
                    print(fileName.split('/')[-1])
                except KeyError:
                    imHashes[fileHash] = fileName
            else:
                appendStuff(fileName, duplicates, imHashes)
    
    appendStuff(projectRootPath+pathFromRoot, duplicates, imHashes)
    return duplicates

def deleteFiles(fileNamesList):
    for file in fileNamesList:
        os.remove(file)

#%%
if __name__ == "__main__":
    duplicates = findDuplicates(r'/Data/MeruUniversity')
    print(len(duplicates))

#%%
    # deleteFiles(duplicates)

# %%
