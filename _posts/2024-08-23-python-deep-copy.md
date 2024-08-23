---
layout: post
title: Difference between deep copy and shallow copy in python   
date: 2024-08-20 07:59:00-0400
description:  
tags:   python  
categories: 
featured: false
---




### Shallow Copy
- **Definition**: A shallow copy creates a new object, but inserts references into it to the objects found in the original.
- **Behavior**: Only the top-level object is copied. If the original object contains other objects (like lists within lists), the references to these objects are copied, not the objects themselves.
- **Use Case**: Suitable for simple data structures or when you don't need to copy nested objects.
- **Example**:
  ```python
  import copy

  original = [1, [2, 3], 4]
  shallow_copy = copy.copy(original)

  shallow_copy[1][0] = 'changed'
  print(original)  # Output: [1, ['changed', 3], 4]
  ```

### Deep Copy
- **Definition**: A deep copy creates a new object and recursively copies all objects found in the original.
- **Behavior**: Both the top-level object and all nested objects are copied. Changes to the copied object do not affect the original object.
- **Use Case**: Necessary when you need to copy complex data structures with nested objects.
- **Example**:
  ```python
  import copy

  original = [1, [2, 3], 4]
  deep_copy = copy.deepcopy(original)

  deep_copy[1][0] = 'changed'
  print(original)  # Output: [1, [2, 3], 4]
  ```

### Summary
- **Shallow Copy**: Copies the structure of the object, but not the elements within nested objects.
- **Deep Copy**: Copies the structure and all elements within nested objects, creating a completely independent copy.

Understanding these differences helps you choose the right type of copy for your specific needs. If you have any more questions or need further clarification, feel free to ask!
