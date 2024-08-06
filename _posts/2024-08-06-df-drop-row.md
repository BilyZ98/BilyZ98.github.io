---
layout: post
title: python dataframe drop row  
date: 2024-08-04 07:59:00-0400
description:  
tags:  python  
categories: python
featured: false
---




To drop rows from a Pandas DataFrame where a key only appears once, you can use the `groupby` and `transform` methods. Here's a simple example:

```python
import pandas as pd

# Sample DataFrame
data = {'ID': [1, 1, 2, 3, 3, 4],
        'Value': [10, 20, 30, 40, 50, 60]}
df = pd.DataFrame(data)

# Drop rows where 'ID' appears only once
df = df[df.groupby('ID')['ID'].transform('count') > 1]

print(df)
```

This code will keep only the rows where the 'ID' appears more than once¹².

Would you like more details or help with a different aspect of your DataFrame?


