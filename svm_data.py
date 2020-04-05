"""
Prepare data for svm
"""

#%%
import os
import numpy as np
import pandas as pd
from src.preprocessing.dataset_stuff import get_colour_features, get_texture_features

#%%
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data/clean/final_segmented')
CLASS_LABELS = [dir for dir in os.listdir(DATA_DIR)]
DATA_DIRS = [os.path.join(DATA_DIR, f'{dir}') for dir in CLASS_LABELS]
DATA_DIRS

#%%
FAW_SIZE = len(os.listdir(os.path.join(DATA_DIR, f'{CLASS_LABELS[0]}')))
HEALTHY_SIZE = len(os.listdir(os.path.join(DATA_DIR, f'{CLASS_LABELS[1]}')))
ZINC_SIZE = len(os.listdir(os.path.join(DATA_DIR, f'{CLASS_LABELS[2]}')))
[FAW_SIZE, HEALTHY_SIZE, ZINC_SIZE]

#%%
COLUMNS = ['bm', 'gm', 'rm', 'bs', 'gs', 'rs', 'ctr', 'dsl', 'hm', 'en', 'cr', 'ASM', 'lbl']
faw = pd.DataFrame(np.empty((FAW_SIZE, 13)), columns=COLUMNS)
healthy = pd.DataFrame(np.empty((HEALTHY_SIZE, 13)), columns=COLUMNS)
zinc_def = pd.DataFrame(np.empty((ZINC_SIZE, 13)), columns=COLUMNS)
dfs = [faw, healthy, zinc_def]

#%%
for i, directory in enumerate(DATA_DIRS):
    for j, image in enumerate(os.listdir(directory)):
        image_path = os.path.join(directory, image)
        dfs[i].loc[j, :'rs'] = get_colour_features(image_path)
        dfs[i].loc[j, 'ctr':'ASM'] = get_texture_features(image_path)
        dfs[i].loc[j, 'lbl'] = i

faw.head(5)

#%%
faw = faw.sample(frac=1)
faw_train = faw.iloc[0:int(np.ceil(0.75*FAW_SIZE))]
faw_test = faw.iloc[int(np.ceil(0.75*FAW_SIZE)):]
faw_train = faw_train.sample(frac=HEALTHY_SIZE/FAW_SIZE, replace=True)
len(faw_train)

#%%
healthy = healthy.sample(frac=1)
healthy_train = healthy.iloc[0:int(np.ceil(0.75*HEALTHY_SIZE))]
healthy_test = healthy.iloc[int(np.ceil(0.75*HEALTHY_SIZE)):]
healthy_train = healthy_train.sample(frac=1)
len(healthy_train)

#%%
zinc_def = zinc_def.sample(frac=1)
zinc_train = zinc_def.iloc[0:int(np.ceil(0.75*ZINC_SIZE))]
zinc_test = zinc_def.iloc[int(np.ceil(0.75*ZINC_SIZE)):]
zinc_train = zinc_train.sample(frac=HEALTHY_SIZE/ZINC_SIZE, replace=True)
len(zinc_train)

#%%
train = pd.concat([faw_train, healthy_train, zinc_train], ignore_index=True)
train = train.sample(frac=1)
test = pd.concat([faw_test, healthy_test, zinc_test], ignore_index=True)
test = test.sample(frac=1)
print(len(train), len(test))

# %%
train = train.astype({'lbl': 'int'})
test = test.astype({'lbl': 'int'})

#%%
train.iloc[:, :-1] = (train.iloc[:, :-1] - train.iloc[:, :-1].mean())/train.iloc[:, :-1].std()
train.describe()

#%%
test.iloc[:, :-1] = (test.iloc[:, :-1] - test.iloc[:, :-1].mean())/test.iloc[:, :-1].std()
test.describe()

#%%
train.to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)
test.to_csv(os.path.join(DATA_DIR, 'test.csv'), index=False)


# %%
