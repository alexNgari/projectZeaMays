#%%
"""
Removes all duplicate files in the subject directory.
"""
import os
import hashlib


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
                with open(file_name, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
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

def delete_files(file_names_list):
    """
        Deletes all the files in the list of file names passed.
        The list should contain absolute paths.
    """
    for file in file_names_list:
        os.remove(file)

#%%
if __name__ == "__main__":
    DUPLICATES = find_duplicates(r'/Data/MeruUniversity')
    print(len(DUPLICATES))

#%%
    # delete_files(DUPLICATES)

# %%
