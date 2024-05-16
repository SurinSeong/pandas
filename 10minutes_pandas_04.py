# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # Reshaping

# ## Stack

arrays = [
    ['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
    ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two'],
]
index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df2 = df[:4]
df2

stacked = df2.stack(future_stack=True)
stacked

stacked.unstack()

stacked.unstack(1)

stacked.unstack(0)

# ### stack()

df_single_level_cols = pd.DataFrame(
                                    [[0, 1], [2, 3]],
                                    index=['cat', 'dog'],
                                    columns=['weight', 'height']
                                    )
df_single_level_cols

df_single_level_cols.stack(future_stack=True)

multicol1 = pd.MultiIndex.from_tuples([('weight', 'kg'),
                                       ('weight', 'pounds')])
df_multi_level_cols1 = pd.DataFrame([[1, 2], [2, 4]],
                                    index=['cat', 'dog'],
                                    columns=multicol1)
df_multi_level_cols1

df_multi_level_cols1.stack(future_stack=True)

# missing values
multicol2 = pd.MultiIndex.from_tuples([('weight', 'kg'),
                                       ('height', 'm')])
df_multi_level_cols2 = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]],
                                    index=['cat', 'dog'],
                                    columns=multicol2)
df_multi_level_cols2

df_multi_level_cols2.stack(future_stack=True)

# prescribing the level(s) to be stacked
df_multi_level_cols2.stack(0, future_stack=True)

df_multi_level_cols2.stack([0, 1], future_stack=True)

# ### unstack()

index = pd.MultiIndex.from_tuples([('one', 'a'), ('one', 'b'),
                                   ('two', 'a'), ('two', 'b')])
s = pd.Series(np.arange(1.0, 5.0), index=index)
s

s.unstack(level=-1)

s.unstack(level=0)

df = s.unstack(level=0)
df.unstack()

# ## Pivot tables

df = pd.DataFrame(
    {
        'A' : ['one', 'one', 'two', 'three'] * 3,
        'B' : ['A', 'B', 'C'] * 4,
        'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
        'D' : np.random.randn(12),
        'E' : np.random.randn(12),
    }
)
df












