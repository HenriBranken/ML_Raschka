import pandas as pd
import numpy as np
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']
])

df.columns = ['color', 'size', 'price', 'classlabel']

from_size_to_int_dict = {'XL': 3,
                         'L': 2,
                         'M': 1}
df['size'] = df['size'].map(from_size_to_int_dict)

from_label_to_int_dict = {label: idx for idx, label in
                          enumerate(np.unique(df['classlabel']))}
df['classlabel'] = df['classlabel'].map(from_label_to_int_dict)
print(df)
