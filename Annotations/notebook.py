#%%
import pandas as pd

#%%     # Read csv
handheldData = pd.read_csv('NLB_handheld.csv')
print(handheldData.head(10))

# %%    # Drop duplicate images ans remove useless columns
handheldData.drop_duplicates('image', inplace=True, ignore_index=True)
handheldData.drop(['user', 'day', 'month', 'year', 'hour', 'minute'], axis='columns', inplace=True)
print(handheldData.head(10))

#%%     # Create 'NLB' column and remove coordinates
import numpy as np
handheldData['NLBPresent'] = np.where(np.sum(handheldData.iloc[:, 1:], axis=1)!=0, 1, 0)
handheldData.drop(['x1', 'y1', 'x2', 'y2'], axis='columns', inplace=True)
# print(handheldData.head(10))

# %%    # Save file
handheldData.to_csv('NLB__classification.csv', index=False)

# %%
