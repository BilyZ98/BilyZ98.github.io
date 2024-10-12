grey line for y axis. 
```
import matplotlib.pyplot as plt

# Create a plot
plt.figure()

# Add solid horizontal lines with light grey color at y = 20, y = 40, y = 60
plt.axhline(y=20, color='lightgrey', linestyle='-')
plt.axhline(y=40, color='lightgrey', linestyle='-')
plt.axhline(y=60, color='lightgrey', linestyle='-')

# Display the plot
plt.show()

```

At the back of the bar chart
```
import matplotlib.pyplot as plt
import numpy as np

# Sample data for bar chart
x = np.arange(5)
y = [10, 30, 50, 70, 90]

# Create a plot
fig, ax = plt.subplots()

# Plot the bar chart
ax.bar(x, y)

# Add solid horizontal lines with light grey color at y = 20, y = 40, y = 60
ax.axhline(y=20, color='lightgrey', linestyle='-', zorder=0)
ax.axhline(y=40, color='lightgrey', linestyle='-', zorder=0)
ax.axhline(y=60, color='lightgrey', linestyle='-', zorder=0)

# Display the plot
plt.show()

```


