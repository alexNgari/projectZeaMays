#%%
import pandas as pd

#%%     # Read csv
boom = pd.read_csv('images_boom0.csv')
print(boom.head(10))

# %%    # Drop duplicate images ans remove useless columns
# boom.drop_duplicates('image', inplace=True, ignore_index=True)
# boom.drop(['user', 'day', 'month', 'year', 'hour', 'minute'], axis='columns', inplace=True)
# print(boom.head(10))

#%%     # Create 'NLB' column and remove coordinates
# import numpy as np
# boom['NLBPresent'] = np.where(np.sum(boom.iloc[:, 1:], axis=1)!=0, 1, 0)
# boom.drop(['x1', 'y1', 'x2', 'y2'], axis='columns', inplace=True)
# print(boom.head(10))
boom['image'] = boom['image'].astype(str)+'.JPG'
print(boom.head(10))

# %%    # Save file
boom.to_csv('images_boom.csv', index=False)

# %%
