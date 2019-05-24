import pandas as pd
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
print(df)
